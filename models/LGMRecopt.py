"""TRAIN-only recommendation: local encoding, anchor hypergraph, behavior residual.

LGMRecOpt contains the model; the two nn.Module helpers below implement its
parameter-free graph operators. Behavior construction adapts MSCA; see
UPSTREAM_SOURCE below. The active objective is BPR plus representation regularization.
"""
import hashlib
import json
import math
import os
from logging import getLogger
from numbers import Integral

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from common.abstract_recommender import GeneralRecommender

IMPLEMENTATION_VERSION = 'lgmrec-opt-train-canonical-v3'
ARCHITECTURE_REVISION = 3


class LGMRecOpt(GeneralRecommender):
    """Anchor hypergraph with an identity-propagation diagnostic control.

    Algorithmically derived from LGMRec and MSCA-style behavior propagation;
    no dependency on the LGMRec model class and no alternate model branches.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        # Complete-input representations shared only within an evaluation pass.
        # Plain attributes: neither checkpoint state nor trainable structure.
        self._eval_embeddings = None
        if config['protocol'] != 'train_only':
            raise ValueError('Only train_only is supported')
        # Read once per model; explicit per-command settings keep paired runs clear.
        self.hyper_propagation = os.environ.get('LGMRECOPT_HYPER_PROPAGATION', 'group')
        if self.hyper_propagation not in ('group', 'identity'):
            raise ValueError('LGMRECOPT_HYPER_PROPAGATION must be group or identity')
        self.implementation_version = f'{IMPLEMENTATION_VERSION}-{self.hyper_propagation}'
        config['implementation_version'] = self.implementation_version
        config['hyper_propagation'] = self.hyper_propagation
        # Separate restored behavior propagation from the intermediate version
        # that omitted it; equal parameter shapes do not imply equal computation.
        self.register_buffer('_architecture_revision', torch.tensor(ARCHITECTURE_REVISION, dtype=torch.int64))
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
        for name, value in [('n_ui_layers', self.n_ui_layers), ('n_mm_layers', self.n_mm_layer)]:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f'{name} must be a non-negative integer')
        if self.n_hyper_layer < 1 or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError('Anchor hypergraph requires positive layers and alpha')
        if not math.isfinite(self.reg_weight) or self.reg_weight < 0:
            raise ValueError('reg_weight must be finite and non-negative')
        if not math.isfinite(self.behavior_eta) or self.behavior_eta < 0:
            raise ValueError('behavior_eta must be finite and non-negative')
        if self.v_feat is None or self.t_feat is None:
            raise ValueError('Both modality features are required')
        for features in (self.v_feat, self.t_feat):
            if features.ndim != 2 or features.shape[0] != self.n_items or features.shape[1] < 1:
                raise ValueError('Features must have one nonempty row per catalog item')
        if not torch.isfinite(self.v_feat).all() or not torch.isfinite(self.t_feat).all():
            raise ValueError('Features must be finite')
        self.n_nodes = self.n_users + self.n_items
        # All graph operators use the same binary TRAIN history. Otherwise
        # duplicate events inflate the readout while its degree is deduplicated.
        self.interaction_matrix = _canonical_interactions(dataset.inter_matrix(form='coo')).tocoo()
        # Rebuild TRAIN graphs at construction; follow .to() without storing
        # dataset-specific graphs in checkpoints.
        self.register_buffer('adj', _scipy_to_torch(
            self.interaction_matrix, self.device), persistent=False)
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
            user_aggregation='train_mean', routing='deterministic', incidence_dropout=False,
            behavior_output_branch=True, behavior_eta=self.behavior_eta)
        self.sparse_hypergraph.metadata['propagation'] = dict(
            mode=self.hyper_propagation,
            support_built=True, attention_computed=self.hyper_propagation == 'group',
            effective_layers=self.n_hyper_layer if self.hyper_propagation == 'group' else 0,
            item_input='cge_items',
            item_output='group_propagated' if self.hyper_propagation == 'group' else 'identity',
            user_readout='train_mean_of_hyper_items',
            fusion='normalize_visual_plus_text_then_alpha')
        getLogger().info('LGMRecOpt hypergraph propagation: %s',
                         json.dumps(self.sparse_hypergraph.metadata['propagation']))
        # Keep an explicit disabled marker for readers of experiment summaries.
        self.sparse_hypergraph.metadata['item_contrast'] = dict(
            enabled=False, computed=False, weight=0.0)
        self.pre_epoch_processing()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self._eval_embeddings = None
        revision = state_dict.get(prefix + '_architecture_revision')
        if revision is not None and (
                not isinstance(revision, torch.Tensor) or revision.numel() != 1
                or revision.item() != ARCHITECTURE_REVISION):
            error_msgs.append(f'{prefix}incompatible LGMRecOpt architecture revision; '
                              f'expected {ARCHITECTURE_REVISION}')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def train(self, mode=True):
        # Trainer calls eval() at the start of each evaluation, including when
        # already in eval mode. Never reuse representations across these calls.
        self._eval_embeddings = None
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        # Cached tensors must not survive device/dtype changes.
        self._eval_embeddings = None
        return super()._apply(fn, recurse=recurse)

    def get_norm_adj_mat(self):
        """Symmetric degree-normalized TRAIN user-item adjacency."""
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
        sumArr = A.sum(axis=1)
        # Preserve the selected implementation's epsilon and arithmetic order.
        diag = np.asarray(sumArr).reshape(-1) + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D.dot(A).dot(D)
        return np.asarray(sumArr), _scipy_to_torch(L, self.device)

    def cge(self):
        """Collaborative encoding, averaged over the input and all graph layers."""
        current = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        layers = [current]
        for _ in range(self.n_ui_layers):
            current = torch.sparse.mm(self.norm_adj, current)
            layers.append(current)
        return torch.stack(layers, dim=1).mean(dim=1)

    def mge(self, projected_items):
        """TRAIN-history user initialization followed by modality propagation."""
        users = torch.sparse.mm(self.adj, projected_items) * self.num_inters[:self.n_users]
        features = torch.cat((users, projected_items), dim=0)
        for _ in range(self.n_mm_layer):
            features = torch.sparse.mm(self.norm_adj, features)
        return features

    def forward(self, missing_modality=None, missing_items=None):
        # Explicit inference-only intervention; ordinary training/VALID is unchanged.
        if missing_modality is not None:
            if self.training:
                raise ValueError('Input-missing diagnosis requires eval mode')
            if missing_modality not in ('image', 'text', 'both'):
                raise ValueError('Expected image, text or both')
            if not isinstance(missing_items, torch.Tensor) or missing_items.dtype != torch.bool or missing_items.shape != (self.n_items,):
                raise ValueError('Expected one boolean missing flag per catalog item')
        elif missing_items is not None:
            raise ValueError('Missing flags require a modality')
        # 1. Local collaborative and modality representations.
        collaborative = self.cge()
        image_input = self.image_embedding.weight
        text_input = self.text_embedding.weight
        if missing_modality is not None:
            observed = (~missing_items).to(device=image_input.device).unsqueeze(1)
            if missing_modality in ('image', 'both'):
                image_input = image_input * observed
            if missing_modality in ('text', 'both'):
                text_input = text_input * observed
        visual = torch.mm(image_input, self.item_image_trs)
        visual_local = self.mge(visual)
        textual = torch.mm(text_input, self.item_text_trs)
        textual_local = self.mge(textual)
        local = collaborative + (F.normalize(visual_local) + F.normalize(textual_local))

        # 2. Modality-weighted anchor hypergraph; values are collaborative items.
        global_views = []
        collaborative_items = collaborative[self.n_users:]
        for features in (visual, textual):
            # Identity keeps the same local/global fusion and history readout;
            # only the learned group mixing (and its gradient path) is removed.
            items = (collaborative_items if self.hyper_propagation == 'identity'
                     else self.sparse_hypergraph(collaborative_items, features))
            users = torch.sparse.mm(self.adj, items) * self.num_inters[:self.n_users]
            global_views.extend((users, items))
        uv, iv, ut, it = global_views
        visual_global = torch.cat((uv, iv), dim=0)
        textual_global = torch.cat((ut, it), dim=0)
        combined = local + self.alpha * F.normalize(visual_global + textual_global)

        # Restore the original behavior branch from raw item IDs, including
        # its original graph normalization and both user/item contributions.
        users, items = torch.split(combined, [self.n_users, self.n_items], dim=0)
        residual = self.behavior_view(self.item_id_embedding.weight)
        ru, ri = torch.split(residual, [self.n_users, self.n_items], dim=0)
        final_users = users + self.behavior_eta * ru
        final_items = items + self.behavior_eta * ri
        return final_users, final_items

    @torch.no_grad()
    def diagnose_missing(self, valid_data, expected_valid, mask_seed=20261021):
        """Fixed-checkpoint VALID audit. Never use during training or model selection.

        Uses catalog-wide nested masks, shared across modalities/checkpoints.
        Subgroup metrics average users with subgroup targets, retaining full ranks.
        """
        from utils.topk_evaluator import TopKEvaluator
        from utils.metrics import recall_, ndcg_
        if self.training:
            raise ValueError('Call eval() before diagnosis')
        if not expected_valid or not all(math.isfinite(float(value)) for value in expected_valid.values()):
            raise ValueError('Expected finite reference VALID metrics from this checkpoint')
        config = valid_data.config
        evaluator = TopKEvaluator(config)
        permutation = np.random.default_rng(mask_seed).permutation(self.n_items)
        rows = []
        positive = valid_data.get_eval_items()

        def group_metrics(top, mask):
            groups = {}
            for name, flag in [('missing', True), ('observed', False)]:
                targets = [set(int(j) for j in items if bool(mask[int(j)]) == flag) for items in positive]
                selected = [i for i, items in enumerate(targets) if items]
                values = dict(users=len(selected), targets=sum(len(targets[i]) for i in selected))
                if selected:
                    hit = np.asarray([[int(j) in targets[i] for j in top[i]] for i in selected])
                    lengths = np.asarray([len(targets[i]) for i in selected])
                    for metric, function in [('recall', recall_), ('ndcg', ndcg_)]:
                        scores = function(hit, lengths)
                        values.update({f'{metric}@{k}': float(scores[k-1]) for k in config['topk']})
                groups[name] = values
            return groups

        scenarios = [(None, 0.)] + [(modality, ratio)
                                   for modality in ('image', 'text', 'both') for ratio in (.2, .5)]
        for modality, ratio in scenarios:
            mask = np.zeros(self.n_items, dtype=bool)
            mask[permutation[:int(self.n_items * ratio)]] = True
            flags = torch.as_tensor(mask, device=self.item_id_embedding.weight.device)
            users, items = self.forward(modality, flags if modality else None)
            if not torch.isfinite(users).all() or not torch.isfinite(items).all():
                raise FloatingPointError('Non-finite diagnostic representations')
            matrices = []
            for batch in valid_data:
                scores = users[batch[0]] @ items.T
                if not torch.isfinite(scores).all():
                    raise FloatingPointError('Non-finite diagnostic scores')
                scores[batch[1][0], batch[1][1]] = -1e10
                matrices.append(torch.topk(scores, max(config['topk']), dim=-1).indices.cpu())
            metrics = evaluator.evaluate(matrices, valid_data)
            if not all(math.isfinite(float(value)) for value in metrics.values()):
                raise FloatingPointError('Non-finite diagnostic metrics')
            top = torch.cat(matrices).numpy()
            if modality is None:
                if set(metrics) != set(expected_valid) or any(abs(metrics[k]-expected_valid[k]) > 1e-7 for k in metrics):
                    raise ValueError(f'Complete VALID does not match checkpoint: expected={expected_valid}, actual={metrics}')
                baseline, complete_top = metrics, top.copy()
            rows.append(dict(modality=modality or 'complete', ratio=ratio, missing_count=int(mask.sum()),
                             mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(), metrics=metrics,
                             retention={k: metrics[k]/baseline[k] if baseline[k] else None for k in metrics},
                             groups=group_metrics(top, mask), complete_groups=group_metrics(complete_top, mask),
                             mean_user_norm=float(users.norm(dim=1).mean()),
                             mean_item_norm=float(items.norm(dim=1).mean())))
            getLogger().info('Missing input diagnosis: %s', json.dumps(rows[-1]))
        return dict(mask_seed=mask_seed, item_permutation=permutation.tolist(), results=rows,
                    split='VALID', test_evaluated=False, complete_tolerance=1e-7,
                    task='full-training-missing-inference',
                    mask_rule='floor(ratio * catalog_size); same nested permutation for image/text/both',
                    subgroup_rule='macro over users with subgroup targets; full-catalog ranks; subgroup IDCG')

    def calculate_loss(self, interaction):
        """One TRAIN negative per positive; BPR plus the existing norm penalty."""
        self._eval_embeddings = None
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
        # Like RecBole LightGCN, compute the full graph once for all user batches.
        # Direct forward (including missing-input diagnosis) always recomputes.
        # Call eval() again after manually changing weights or graph settings.
        cacheable = (not self.training and not torch.is_grad_enabled()
                     and not torch.is_autocast_enabled(self.item_id_embedding.weight.device.type))
        if cacheable:
            if self._eval_embeddings is None:
                self._eval_embeddings = self.forward()
            users, items = self._eval_embeddings
        else:
            self._eval_embeddings = None
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
        """Preserve the original sum of unsquared L2 norms divided by batch size."""
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss
MSCA_STRUCT_TOPK = 10
MSCA_MIN_COOCCURRENCE = 2  # equivalent to the authors' data > 1
UPSTREAM_SOURCE = 'https://github.com/recomall/MSCA/blob/main/src/models/msca.py'


def _canonical_interactions(interaction_matrix):
    matrix = interaction_matrix.astype(np.float32).copy()
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError('TRAIN interactions must be finite and non-negative')
    matrix = matrix.tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.data[:] = 1  # Count distinct users, not duplicate event frequency.
    matrix.sort_indices()
    return matrix


def behavior_graph_fingerprint(interaction_matrix, *, topk, minimum):
    """Identify the exact TRAIN matrix and structural graph parameters."""
    matrix = _canonical_interactions(interaction_matrix)
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    digest.update(matrix.indptr.astype(np.int64, copy=False).tobytes())
    digest.update(matrix.indices.astype(np.int64, copy=False).tobytes())
    digest.update(matrix.data.astype(np.float32, copy=False).tobytes())
    digest.update(np.asarray([topk, minimum], dtype=np.int64).tobytes())
    # Restore the original encoder graph schema; construction uses no seed.
    digest.update(json.dumps(dict(version=2, graph_mode='cooccurrence',
        graph_seed=None, self_loop=True, edge_weight='binary'),
        sort_keys=True).encode())
    return digest.hexdigest()


def _scipy_to_torch(matrix, device):
    matrix = matrix.tocoo()
    indices = torch.from_numpy(np.vstack((matrix.row, matrix.col))).long()
    values = torch.from_numpy(matrix.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(
        indices, values, matrix.shape, device=device
    ).coalesce()


def build_normalized_interactions(interaction_matrix, device):
    """Build MSCA's D_u^-1/2 R D_i^-1/2 from binary TRAIN interactions."""
    matrix = _canonical_interactions(interaction_matrix)
    user_degree = np.asarray(matrix.sum(axis=1)).reshape(-1)
    item_degree = np.asarray(matrix.sum(axis=0)).reshape(-1)
    user_scale = np.zeros_like(user_degree, dtype=np.float32)
    item_scale = np.zeros_like(item_degree, dtype=np.float32)
    np.power(user_degree, -0.5, out=user_scale, where=user_degree > 0)
    np.power(item_degree, -0.5, out=item_scale, where=item_degree > 0)
    normalized = sp.diags(user_scale).dot(matrix).dot(sp.diags(item_scale))
    return _scipy_to_torch(normalized, device)


def build_structural_adjacency(interaction_matrix, *, topk, minimum,
                               device, block_size=1024):
    """Directed row-TopK graph; blockwise product bounds temporary row count.

    The selected method uses binary cooccurrence edges and self membership.
    """
    for name, value in [('topk', topk), ('minimum', minimum), ('block_size', block_size)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
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
    """Original parameter-free behavior encoder and anchor membership support."""

    def __init__(self, interaction_matrix, device, *,
                 topk=MSCA_STRUCT_TOPK,
                 minimum=MSCA_MIN_COOCCURRENCE, block_size=1024):
        super().__init__()
        self.topk = topk
        self.minimum = minimum
        self.cache_fingerprint = behavior_graph_fingerprint(
            interaction_matrix, topk=self.topk, minimum=self.minimum
        )
        # Rebuild from the current TRAIN matrix on every model construction.
        # Non-persistent buffers prevent a checkpoint from replacing this
        # graph with one produced by a different dataset or split.
        self.register_buffer(
            'structural_adjacency',
            build_structural_adjacency(
                interaction_matrix, topk=self.topk,
                minimum=self.minimum, device=device, block_size=block_size
            ),
            persistent=False,
        )
        self.metadata = dict(fingerprint=self.cache_fingerprint, topk=topk,
                             minimum=minimum, graph_mode='cooccurrence', graph_seed=None,
                             self_loop=True, edge_weight='binary', neighbor_score='count',
                             role='behavior_encoder_and_hypergraph_support', output_branch=True,
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
    """Shared fixed members, modality-conditioned weights, and two-way propagation."""
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


def _run_missing_diagnosis():
    """Model-local command: python -m models.LGMRecopt --checkpoint PATH."""
    import argparse
    import csv
    from datetime import datetime
    from pathlib import Path
    from uuid import uuid4
    from utils.configurator import Config
    from utils.dataset import RecDataset
    from utils.dataloader import TrainDataLoader, EvalDataLoader
    from utils.experiment_state import fingerprint_inputs, runtime_metadata
    from utils.utils import init_seed

    parser = argparse.ArgumentParser(description='Research-1 checkpoint: VALID input-missing diagnosis only')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mask-seed', type=int, default=20261021)
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    source = saved['config']
    expected_version = f'{IMPLEMENTATION_VERSION}-group'
    if saved.get('model') != 'LGMRecOpt' or saved.get('implementation_version') != expected_version:
        raise ValueError(f'Requires {expected_version} weights; historical checkpoints need their historical source')
    if saved.get('protocol') != 'train_only' or source.get('report_test') is not False:
        raise ValueError('Requires a TRAIN-only VALID-selected checkpoint')
    if os.environ.get('LGMRECOPT_HYPER_PROPAGATION', 'group') != 'group':
        raise ValueError('Unset LGMRECOPT_HYPER_PROPAGATION or set it to group')
    config = Config('LGMRecOpt', saved['dataset'])
    for key in list(config.final_config_dict):
        if key in source:
            config[key] = source[key]
    config['data_path'] = str(args.data_dir.resolve()) + os.sep
    config['device'] = torch.device(f'cuda:{args.gpu}' if source.get('use_gpu', True) and torch.cuda.is_available() else 'cpu')
    config['gpu_id'] = args.gpu
    config['save_recommended_topk'] = False
    hashes, signature = fingerprint_inputs(config)
    if signature != source.get('data_signature'):
        raise ValueError('Checkpoint and input data fingerprints differ')
    init_seed(int(config['seed']))
    train, valid, _ = RecDataset(config).split()
    train_data = TrainDataLoader(config, train, batch_size=config['train_batch_size'], shuffle=False)
    valid_data = EvalDataLoader(config, valid, additional_dataset=train, batch_size=config['eval_batch_size'])
    model = LGMRecOpt(config, train_data).to(config['device'])
    for key, features in [('image_embedding.weight', model.image_embedding.weight),
                          ('text_embedding.weight', model.text_embedding.weight)]:
        if not torch.equal(saved['model_state_dict'][key].cpu(), features.detach().cpu()):
            raise ValueError('Checkpoint contains modified input features: ' + key)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.eval()
    output = (args.output_dir or Path('log') / (
        f'missing-LGMRecOpt-seed{config["seed"]}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(checkpoint=str(args.checkpoint.resolve()),
                    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    checkpoint_version=saved['implementation_version'], epoch=saved['epoch'],
                    seed=int(config['seed']), data_signature=signature, input_hashes=hashes,
                    runtime=runtime_metadata(config['device']), finished=False)
    try:
        report = model.diagnose_missing(valid_data, saved['best_valid_result'], args.mask_seed)
        report.update(metadata)
        np.save(output / 'item-permutation.npy', np.asarray(report.pop('item_permutation'), dtype=np.int64))
        with (output / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            keys = list(report['results'][0]['metrics'])
            writer = csv.DictWriter(handle, fieldnames=['modality', 'ratio', 'missing_count'] + keys + ['retention_'+k for k in keys])
            writer.writeheader()
            for row in report['results']:
                writer.writerow({**{k: row[k] for k in ['modality', 'ratio', 'missing_count']}, **row['metrics'],
                                 **{'retention_'+k: v for k, v in row['retention'].items()}})
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for axis, metric in zip(axes, ['recall@20', 'ndcg@20']):
            for modality in ['image', 'text', 'both']:
                rows = [report['results'][0]] + [r for r in report['results'] if r['modality'] == modality]
                axis.plot([r['ratio'] for r in rows], [r['metrics'][metric] for r in rows], marker='o', label=modality)
            axis.set(xlabel='Missing item fraction', ylabel=metric, title=f'Seed {config["seed"]}')
            axis.legend()
            axis.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(output / 'sensitivity.png', dpi=180)
        fig.savefig(output / 'sensitivity.svg')
        plt.close(fig)
        report['finished'] = True
        (output / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        for row in report['results']:
            print(row['modality'], row['ratio'], row['metrics'])
        print('VALID diagnosis saved:', output)
    except Exception as exc:
        metadata['error'] = str(exc)
        (output / 'failure.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    _run_missing_diagnosis()
