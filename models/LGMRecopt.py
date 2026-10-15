"""LGMRec optimization and its TRAIN-only behavior/hypergraph operators.

Behavior construction adapts MSCA; see UPSTREAM_SOURCE below.
"""
import hashlib
import json
import math
from numbers import Integral

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from common.abstract_recommender import GeneralRecommender

IMPLEMENTATION_VERSION = 'lgmrec-opt-standalone-anchor-v1'


class LGMRecOpt(GeneralRecommender):
    """Standalone implementation of the selected sparse_anchor_nohcl method.

    Algorithmically derived from LGMRec and MSCA-style behavior propagation;
    no dependency on the LGMRec model class and no alternate model branches.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        if config['protocol'] != 'train_only':
            raise ValueError('Only train_only is supported')
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_ui_layers = config['n_ui_layers']
        self.n_mm_layer = config['n_mm_layers']
        self.n_hyper_layer = config['n_hyper_layer']
        self.alpha = config['alpha']
        self.reg_weight = config['reg_weight']
        self.behavior_eta = float(config['behavior_eta'])
        if self.embedding_dim != self.feat_embed_dim:
            raise ValueError('ID and modality embedding dimensions must match')
        if self.n_hyper_layer < 1 or self.alpha <= 0:
            raise ValueError('Anchor hypergraph requires positive layers and alpha')
        if not math.isfinite(self.behavior_eta) or self.behavior_eta < 0:
            raise ValueError('behavior_eta must be finite and non-negative')
        if self.v_feat is None or self.t_feat is None:
            raise ValueError('Both modality features are required')
        if not torch.isfinite(self.v_feat).all() or not torch.isfinite(self.t_feat).all():
            raise ValueError('Features must be finite')
        self.n_nodes = self.n_users + self.n_items
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        # Rebuild TRAIN graphs at construction; follow .to() without storing
        # dataset-specific graphs in checkpoints.
        self.register_buffer('adj', self.scipy_matrix_to_sparse_tenser(
            self.interaction_matrix, (self.n_users, self.n_items)), persistent=False)
        degree, norm_adj = self.get_norm_adj_mat()
        self.register_buffer('norm_adj', norm_adj, persistent=False)
        self.register_buffer('num_inters',
            torch.FloatTensor(1.0 / (degree + 1e-7)).to(self.device), persistent=False)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
        self.item_image_trs = nn.Parameter(nn.init.xavier_uniform_(
            torch.zeros(self.v_feat.shape[1], self.feat_embed_dim)))
        # Consume historical initialization draws only, so matched seeds retain
        # the selected model's initialization. No prototype parameters survive.
        nn.init.xavier_uniform_(torch.empty(self.v_feat.shape[1], config['init_reference_width']))
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
        self.item_text_trs = nn.Parameter(nn.init.xavier_uniform_(
            torch.zeros(self.t_feat.shape[1], self.feat_embed_dim)))
        nn.init.xavier_uniform_(torch.empty(self.t_feat.shape[1], config['init_reference_width']))
        self.behavior_view = MSCABehaviorView(
            self.interaction_matrix, self.device, topk=config['behavior_topk'],
            minimum=config['behavior_minimum'], block_size=config['behavior_block_size'])
        self.sparse_hypergraph = SparseGroupHypergraph(
            self.behavior_view.structural_adjacency, layers=self.n_hyper_layer)
        self.sparse_hypergraph.metadata.update(
            train_graph_fingerprint=self.behavior_view.cache_fingerprint,
            user_aggregation='train_mean', routing='deterministic', incidence_dropout=False)
        self.pre_epoch_processing()

    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse_coo_tensor(
            i,
            data,
            shape,
            device=self.device,
        ).coalesce()


    def get_norm_adj_mat(self):
        inter_M = self.interaction_matrix
        users = inter_M.row
        items = inter_M.col + self.n_users
        rows = np.concatenate([users, items])
        columns = np.concatenate([items, users])
        values = np.ones(rows.shape[0], dtype=np.float32)
        # Build the symmetric user-item graph exclusively through SciPy's
        # public sparse API. New SciPy releases removed dok_matrix._update.
        A = sp.coo_matrix(
            (values, (rows, columns)),
            shape=(self.n_nodes, self.n_nodes),
            dtype=np.float32,
        )
        A.sum_duplicates()
        A.data.fill(1.0)
        # norm adj matrix
        sumArr = A.sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.asarray(sumArr).reshape(-1) + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D.dot(A).dot(D)
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        return np.asarray(sumArr), self.scipy_matrix_to_sparse_tenser(
            L,
            torch.Size((self.n_nodes, self.n_nodes)),
        )


    def cge(self):
        current = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        layers = [current]
        for _ in range(self.n_ui_layers):
            current = torch.sparse.mm(self.norm_adj, current)
            layers.append(current)
        return torch.stack(layers, dim=1).mean(dim=1)

    def mge(self, projected_items):
        users = torch.sparse.mm(self.adj, projected_items) * self.num_inters[:self.n_users]
        features = torch.cat((users, projected_items), dim=0)
        for _ in range(self.n_mm_layer):
            features = torch.sparse.mm(self.norm_adj, features)
        return features

    def forward(self):
        collaborative = self.cge()
        visual = torch.mm(self.image_embedding.weight, self.item_image_trs)
        visual_local = self.mge(visual)
        textual = torch.mm(self.text_embedding.weight, self.item_text_trs)
        textual_local = self.mge(textual)
        local = collaborative + (F.normalize(visual_local) + F.normalize(textual_local))
        global_views = []
        collaborative_items = collaborative[self.n_users:]
        for features in (visual, textual):
            items = self.sparse_hypergraph(collaborative_items, features)
            users = torch.sparse.mm(self.adj, items) * self.num_inters[:self.n_users]
            global_views.extend((users, items))
        uv, iv, ut, it = global_views
        visual_global = torch.cat((uv, iv), dim=0)
        textual_global = torch.cat((ut, it), dim=0)
        combined = local + self.alpha * F.normalize(visual_global + textual_global)
        users, items = torch.split(combined, [self.n_users, self.n_items], dim=0)
        residual = self.behavior_view(self.item_id_embedding.weight)
        ru, ri = torch.split(residual, [self.n_users, self.n_items], dim=0)
        return users + self.behavior_eta * ru, items + self.behavior_eta * ri

    def calculate_loss(self, interaction):
        if interaction.ndim != 2 or interaction.size(0) != 3:
            raise ValueError('Expected one TRAIN negative per positive')
        uid, positive, negative = interaction.long()
        users, items = self.forward()
        u, p, n = users[uid], items[positive], items[negative]
        ranking = self.bpr_loss(u, p, n)
        penalty = self.reg_weight * self.reg_loss(u, p, n)
        if self.training:
            for key, value in {'loss_bpr': ranking, 'loss_reg': penalty}.items():
                self._sums[key] = self._sums.get(key, 0.) + float(value.detach()) * uid.numel()
            self._count += uid.numel()
        return ranking + penalty

    def full_sort_predict(self, interaction):
        users, items = self.forward()
        return users[interaction[0]] @ items.T

    def pre_epoch_processing(self):
        self._sums, self._count = {}, 0

    def post_epoch_processing(self):
        return 'LGMRecOpt: ' + ', '.join(
            f'{key}={value / max(1, self._count):.6f}' for key, value in self._sums.items())

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return bpr_loss


    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss



MSCA_STRUCT_TOPK = 10
MSCA_MIN_COOCCURRENCE = 2  # equivalent to the authors' data > 1
UPSTREAM_SOURCE = 'https://github.com/recomall/MSCA/blob/main/src/models/msca.py'


def _canonical_interactions(interaction_matrix):
    matrix = interaction_matrix.astype(np.float32).tocsr(copy=True)
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError('TRAIN interactions must be finite and non-negative')
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.data[:] = 1  # Count distinct users, not duplicate event frequency.
    matrix.sort_indices()
    return matrix


def behavior_graph_fingerprint(interaction_matrix, *, topk, minimum,
                               graph_mode='cooccurrence', graph_seed=0,
                               self_loop=True, edge_weight='binary', neighbor_score='count'):
    """Identify the exact TRAIN matrix and structural graph parameters."""
    if neighbor_score != 'count':
        raise ValueError('neighbor_score=cosine is retired; only count is supported')
    matrix = _canonical_interactions(interaction_matrix)
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    digest.update(matrix.indptr.astype(np.int64, copy=False).tobytes())
    digest.update(matrix.indices.astype(np.int64, copy=False).tobytes())
    digest.update(matrix.data.astype(np.float32, copy=False).tobytes())
    digest.update(np.asarray([topk, minimum], dtype=np.int64).tobytes())
    digest.update(json.dumps(dict(version=2, graph_mode=graph_mode,
        graph_seed=graph_seed if graph_mode == 'random_relabel' else None,
        self_loop=self_loop, edge_weight=edge_weight), sort_keys=True).encode())
    return digest.hexdigest()


def _scipy_to_torch(matrix, device):
    matrix = matrix.tocoo()
    indices = torch.from_numpy(np.vstack((matrix.row, matrix.col))).long()
    values = torch.from_numpy(matrix.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(
        indices, values, matrix.shape, device=device
    ).coalesce()


def build_normalized_interactions(interaction_matrix, device):
    """Build MSCA's D_u^-1/2 R D_i^-1/2 from TRAIN interactions."""
    matrix = _canonical_interactions(interaction_matrix)
    user_degree = np.asarray(matrix.sum(axis=1)).reshape(-1)
    item_degree = np.asarray(matrix.sum(axis=0)).reshape(-1)
    user_scale = np.zeros_like(user_degree, dtype=np.float32)
    item_scale = np.zeros_like(item_degree, dtype=np.float32)
    np.power(user_degree, -0.5, out=user_scale, where=user_degree > 0)
    np.power(item_degree, -0.5, out=item_scale, where=item_degree > 0)
    normalized = sp.diags(user_scale).dot(matrix).dot(
        sp.diags(item_scale)
    )
    return _scipy_to_torch(normalized, device)


def build_structural_adjacency(interaction_matrix, *, topk, minimum,
                               device, graph_mode='cooccurrence', graph_seed=0,
                               self_loop=True, edge_weight='binary', block_size=1024,
                               neighbor_score='count'):
    """Directed row-TopK graph; blockwise product bounds temporary row count.

    The selected method uses binary cooccurrence edges and self membership.
    """
    for name, value in [('topk', topk), ('minimum', minimum), ('block_size', block_size)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if graph_mode != 'cooccurrence':
        raise ValueError('Only the selected cooccurrence graph is supported')
    if neighbor_score != 'count':
        raise ValueError('neighbor_score=cosine is retired; only count is supported')
    if edge_weight != 'binary' or self_loop is not True:
        raise ValueError('Invalid edge_weight or self_loop')
    if isinstance(graph_seed, bool) or not isinstance(graph_seed, Integral) or graph_seed < 0:
        raise ValueError('graph_seed must be a non-negative integer')
    interactions = _canonical_interactions(interaction_matrix)
    n_items = interactions.shape[1]
    rows, columns, weights = [], [], []
    for item in range(n_items):
        if item % block_size == 0:
            cooccurrence = (interactions[:, item:item + block_size].T @ interactions).tocsr()
        local = item % block_size
        start, end = cooccurrence.indptr[local:local + 2]
        candidates = cooccurrence.indices[start:end]
        counts = cooccurrence.data[start:end]
        eligible = (counts >= minimum) & (candidates != item)
        candidates, counts = candidates[eligible], counts[eligible]
        if candidates.size:
            order = np.lexsort((candidates, -counts))[:topk]
            selected = candidates[order]
            rows.extend([item] * len(selected))
            columns.extend(selected.tolist())
            weights.extend([1.] * len(selected))

    if self_loop:
        rows.extend(range(n_items))
        columns.extend(range(n_items))
        weights.extend([1.] * n_items)

    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32)
    degree = np.bincount(rows, weights=weights, minlength=n_items).astype(np.float32)
    scale = np.zeros_like(degree)
    np.power(degree, -.5, out=scale, where=degree > 0)
    values = weights * scale[rows] * scale[columns]
    adjacency = sp.coo_matrix(
        (values, (rows, columns)), shape=(n_items, n_items)
    )
    adjacency.eliminate_zeros()
    return _scipy_to_torch(adjacency, device)


class MSCABehaviorView(nn.Module):
    """Parameter-free MSCA structural encoder cached from one TRAIN graph."""

    def __init__(self, interaction_matrix, device, *,
                 topk=MSCA_STRUCT_TOPK,
                 minimum=MSCA_MIN_COOCCURRENCE, graph_mode='cooccurrence',
                 graph_seed=0, self_loop=True, edge_weight='binary', block_size=1024,
                 neighbor_score='count'):
        super().__init__()
        self.topk = topk
        self.minimum = minimum
        options = dict(graph_mode=graph_mode, graph_seed=graph_seed,
                       self_loop=self_loop, edge_weight=edge_weight, neighbor_score=neighbor_score)
        self.cache_fingerprint = behavior_graph_fingerprint(
            interaction_matrix, topk=self.topk, minimum=self.minimum, **options
        )
        # Rebuild from the current TRAIN matrix on every model construction.
        # Non-persistent buffers prevent a checkpoint from replacing this
        # graph with one produced by a different dataset or split.
        self.register_buffer(
            'structural_adjacency',
            build_structural_adjacency(
                interaction_matrix, topk=self.topk,
                minimum=self.minimum, device=device, block_size=block_size, **options
            ),
            persistent=False,
        )
        self.metadata = dict(fingerprint=self.cache_fingerprint, topk=topk,
                             minimum=minimum, **options,
                             normalized_nonzero_edges=self.structural_adjacency._nnz())
        self.register_buffer(
            'normalized_interactions',
            build_normalized_interactions(interaction_matrix, device),
            persistent=False,
        )

    def forward(self, item_id_embeddings):
        item_structural = torch.sparse.mm(
            self.structural_adjacency, item_id_embeddings
        )
        user_structural = torch.sparse.mm(
            self.normalized_interactions, item_structural
        )
        return torch.cat((user_structural, item_structural), dim=0)


class SparseGroupHypergraph(nn.Module):
    def __init__(self, structural_adjacency, *, layers=1):
        super().__init__()
        if isinstance(layers, bool) or not isinstance(layers, Integral) or layers < 1:
            raise ValueError('layers must be a positive integer')
        if structural_adjacency.layout != torch.sparse_coo or structural_adjacency.ndim != 2:
            raise ValueError('Expected sparse COO item adjacency')
        n_items = structural_adjacency.shape[0]
        if n_items < 1 or structural_adjacency.shape[1] != n_items:
            raise ValueError('Item adjacency must be nonempty and square')
        graph = structural_adjacency.detach().cpu().coalesce()
        values = graph.values()
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError('Graph weights must be finite and non-negative')
        rows, columns = graph.indices()[:, values > 0]
        if int((rows == columns).sum()) != n_items:
            raise ValueError('Each hyperedge must include its anchor self membership')
        sizes = torch.bincount(rows, minlength=n_items)
        width = int(sizes.max())
        # Padding is the row's anchor, never an unrelated item such as item 0.
        members = torch.arange(n_items)[:, None].expand(n_items, width).clone()
        mask = torch.zeros((n_items, width), dtype=torch.bool)
        offsets = torch.cumsum(sizes, dim=0) - sizes
        slots = torch.arange(rows.numel()) - offsets[rows]
        members[rows, slots] = columns
        mask[rows, slots] = True
        device = structural_adjacency.device
        self.register_buffer('members', members.to(device), persistent=False)
        self.register_buffer('mask', mask.to(device), persistent=False)
        self.register_buffer('_edge_ids', rows.to(device), persistent=False)
        self.register_buffer('_item_ids', columns.to(device), persistent=False)
        self.layers = int(layers)
        self.n_items = n_items
        support = {'shape': [n_items, n_items], 'rows': rows.tolist(),
                   'columns': columns.tolist()}
        fingerprint = hashlib.sha256(json.dumps(
            support, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.metadata = dict(
            version='sparse-group-hypergraph-v1', weighting='attention', layers=self.layers,
            n_hyperedges=n_items, nnz=int(rows.numel()), min_size=int(sizes.min()),
            max_size=width, singleton_count=int((sizes == 1).sum()),
            fingerprint=fingerprint,
            query_mode='anchor',
        )

    def weights(self, item_embeddings, projected_features):
        """Return [anchor, padded member] coefficients; padded entries are zero."""
        for name, value in (('item_embeddings', item_embeddings),
                            ('projected_features', projected_features)):
            if value.ndim != 2 or value.shape[0] != self.n_items or value.shape[1] < 1:
                raise ValueError(f'{name} must be a nonempty item feature matrix')
            if value.device != self.members.device or not value.is_floating_point():
                raise ValueError(f'{name} must be floating point on the graph device')
        if item_embeddings.dtype != projected_features.dtype:
            raise ValueError('Item embeddings and projected features must share dtype')
        features = F.normalize(projected_features, dim=1)
        gathered = features[self.members]
        query = features
        collaborative = F.normalize(item_embeddings, dim=1)
        scores = (gathered * query[:, None, :]).sum(dim=-1)
        scores = scores + (collaborative[self.members] * collaborative[:, None, :]).sum(dim=-1)
        return scores.masked_fill(~self.mask, -torch.inf).softmax(dim=1)

    def forward(self, item_embeddings, projected_features):
        # The memberships depend on initial representations. Repeated layers
        # propagate with these same weights rather than rebuilding a new graph.
        weights = self.weights(item_embeddings, projected_features)
        real_weights = weights[self.mask]
        mass = item_embeddings.new_zeros(self.n_items)
        mass.index_add_(0, self._item_ids, real_weights)
        current = item_embeddings
        for _ in range(self.layers):
            edges = (current[self.members] * weights[..., None]).sum(dim=1)
            messages = edges[self._edge_ids] * real_weights[:, None]
            current = torch.zeros_like(current).index_add_(0, self._item_ids, messages)
            current = current / mass.clamp_min(torch.finfo(current.dtype).tiny)[:, None]
        return current
