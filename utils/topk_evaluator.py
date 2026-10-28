# coding: utf-8

import hashlib
import math
import os
from numbers import Integral

import numpy as np
import pandas as pd
import torch
from utils.metrics import metrics_dict, recall_, ndcg_
from utils.utils import get_local_time


# These metrics are typical in topk recommendations
topk_metrics = {metric.lower(): metric for metric in ['Recall', 'Recall2', 'Precision', 'NDCG', 'MAP']}


class TopKEvaluator(object):
    """Full-catalog ranking, user-averaged metrics and fixed missing-input evaluation."""

    def __init__(self, config):
        self.config = config
        self.metrics = config['metrics']
        self.topk = config['topk']
        self.save_recom_result = config['save_recommended_topk']
        self._check_args()

    @torch.no_grad()
    def rank(self, model, eval_data, embeddings=None):
        """Rank each user against the full catalog, excluding supplied history."""
        matrices = []
        for batch in eval_data:
            scores = (model.full_sort_predict(batch) if embeddings is None
                      else embeddings[0][batch[0]] @ embeddings[1].T)
            if not torch.isfinite(scores).all():
                raise FloatingPointError('Non-finite evaluation scores')
            scores[batch[1][0], batch[1][1]] = -1e10
            matrices.append(torch.topk(scores, max(self.topk), dim=-1).indices.cpu())
        return matrices

    def evaluate(self, batch_matrix_list, eval_data, is_test=False, idx=0):
        """Calculate full-precision metrics; optionally export final TEST ranks."""
        pos_items = eval_data.get_eval_items()
        pos_len_list = eval_data.get_eval_len_list()
        topk_index = torch.cat(batch_matrix_list, dim=0).cpu().numpy()
        # if save recommendation result?
        if self.save_recom_result and is_test:
            dataset_name = self.config['dataset']
            model_name = self.config['model']
            max_k = max(self.topk)
            dir_name = os.path.abspath(self.config['recommend_topk'])
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            file_path = os.path.join(dir_name, '{}-{}-idx{}-top{}-{}.csv'.format(
                model_name, dataset_name, idx, max_k, get_local_time()))
            x_df = pd.DataFrame(topk_index)
            x_df.insert(0, 'id', eval_data.get_eval_users())
            x_df.columns = ['id']+['top_'+str(i) for i in range(max_k)]
            x_df = x_df.astype(int)
            x_df.to_csv(file_path, sep='\t', index=False)
        assert len(pos_len_list) == len(topk_index)
        # if recom right?
        bool_rec_matrix = []
        for m, n in zip(pos_items, topk_index):
            bool_rec_matrix.append([True if i in m else False for i in n])
        bool_rec_matrix = np.asarray(bool_rec_matrix)

        # get metrics
        metric_dict = {}
        result_list = self._calculate_metrics(pos_len_list, bool_rec_matrix)
        for metric, value in zip(self.metrics, result_list):
            for k in self.topk:
                key = '{}@{}'.format(metric, k)
                # Keep precision for checkpoint selection; dict2str formats logs.
                metric_dict[key] = float(value[k - 1])
        return metric_dict

    def _group_metrics(self, top, positive, group_masks):
        groups = {}
        for name, mask in group_masks.items():
            targets = [set(int(j) for j in items if mask[int(j)]) for items in positive]
            selected = [i for i, items in enumerate(targets) if items]
            values = dict(users=len(selected), targets=sum(len(targets[i]) for i in selected))
            values.update({f'{metric}@{k}': None for metric in ('recall', 'ndcg') for k in self.topk})
            if selected:
                hit = np.asarray([[int(j) in targets[i] for j in top[i]] for i in selected])
                lengths = np.asarray([len(targets[i]) for i in selected])
                for metric, function in [('recall', recall_), ('ndcg', ndcg_)]:
                    scores = function(hit, lengths)
                    values.update({f'{metric}@{k}': float(scores[k-1]) for k in self.topk})
            groups[name] = values
        return groups

    @torch.no_grad()
    def diagnose_missing(self, model, valid_data, expected_valid, mask_seed=20261021,
                         imputation_modes=('zero', 'neighbor'), include_groups=True):
        """Fixed-weight VALID diagnosis; does not select weights or advance TRAIN masks.

        Zero/neighbor comparisons use this same model and identical nested masks.
        They are not comparisons between separately trained experiment groups.
        """
        if model.training:
            raise ValueError('Call eval() before diagnosis')
        if isinstance(mask_seed, bool) or not isinstance(mask_seed, Integral) or mask_seed < 0:
            raise ValueError('mask_seed must be a non-negative integer')
        if not expected_valid or not all(math.isfinite(float(value)) for value in expected_valid.values()):
            raise ValueError('Expected finite reference VALID metrics from this checkpoint')
        modes = tuple(imputation_modes)
        if not modes or len(set(modes)) != len(modes) or any(mode not in ('zero', 'neighbor') for mode in modes):
            raise ValueError('imputation_modes must contain unique zero/neighbor entries')
        permutation = np.random.default_rng(mask_seed).permutation(model.n_items)
        rows = []
        positive = valid_data.get_eval_items()

        scenarios = [(None, 0., 'none')] + [
            (modality, ratio, mode) for modality in ('image', 'text', 'both')
            for ratio in (.2, .5) for mode in modes]
        baseline, complete_top = None, None
        for modality, ratio, mode in scenarios:
            mask = np.zeros(model.n_items, dtype=bool)
            mask[permutation[:int(model.n_items * ratio)]] = True
            flags = torch.as_tensor(mask, device=model.item_id_embedding.weight.device)
            users, items = (model.forward() if modality is None
                            else model.forward(modality, flags, imputation=mode))
            if not torch.isfinite(users).all() or not torch.isfinite(items).all():
                raise FloatingPointError('Non-finite diagnostic representations')
            matrices = self.rank(model, valid_data, embeddings=(users, items))
            metrics = self.evaluate(matrices, valid_data)
            if not all(math.isfinite(float(value)) for value in metrics.values()):
                raise FloatingPointError('Non-finite diagnostic metrics')
            top = torch.cat(matrices).numpy() if include_groups else None
            if modality is None:
                if set(metrics) != set(expected_valid) or any(abs(metrics[k]-expected_valid[k]) > 1e-7 for k in metrics):
                    raise ValueError(f'Complete VALID does not match checkpoint: expected={expected_valid}, actual={metrics}')
                baseline, complete_top = metrics, top
            coverage = {}
            group_masks = {'missing': mask, 'observed': ~mask}
            for name in ('image', 'text'):
                active = modality in (name, 'both')
                missing = mask if active else np.zeros(model.n_items, dtype=bool)
                counts = model._observed_neighbor_counts(
                    torch.as_tensor(missing, device=flags.device)).cpu().numpy()
                with_neighbors = missing & (counts > 0)
                without_neighbors = missing & (counts == 0)
                total = int(missing.sum())
                covered = int(with_neighbors.sum())
                coverage[name] = dict(
                    missing_count=total, with_observed_neighbors=covered,
                    without_observed_neighbors=int(without_neighbors.sum()),
                    covered_fraction=covered / total if total else None,
                    mean_observed_neighbors=float(counts[missing].mean()) if total else None,
                    denominator='missing_items_in_this_modality')
                group_masks[name + '_missing_with_neighbors'] = with_neighbors
                group_masks[name + '_missing_without_neighbors'] = without_neighbors
            rows.append(dict(
                modality=modality or 'complete', ratio=ratio, imputation=mode,
                missing_count=int(mask.sum()), mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                metrics=metrics,
                retention={k: metrics[k]/baseline[k] if baseline[k] else None for k in metrics},
                coverage=coverage,
                groups=self._group_metrics(top, positive, group_masks) if include_groups else {},
                complete_groups=self._group_metrics(complete_top, positive, group_masks) if include_groups else {},
                mean_user_norm=float(users.norm(dim=1).mean()),
                mean_item_norm=float(items.norm(dim=1).mean())))
        zero_rows = {(row['modality'], row['ratio']): row for row in rows if row['imputation'] == 'zero'}
        for row in rows:
            reference = zero_rows.get((row['modality'], row['ratio']))
            if row['imputation'] == 'neighbor' and reference is not None:
                difference = {key: value - reference['metrics'][key] for key, value in row['metrics'].items()}
                relative = {key: difference[key] / reference['metrics'][key]
                            if reference['metrics'][key] else None for key in difference}
                row['comparison_to_zero'] = dict(absolute=difference, relative=relative)
        return dict(mask_seed=mask_seed, item_permutation=permutation.tolist(), results=rows,
                    split='VALID', test_evaluated=False, complete_tolerance=1e-7,
                    task='fixed-weight-missing-inference', imputation_modes=list(modes),
                    groups_included=include_groups, training_input_protocol=dict(model.missing_metadata),
                    comparison_scope='same frozen model and mask; not a cross-training-group comparison',
                    mask_rule='floor(ratio * catalog_size); same nested permutation for image/text/both',
                    subgroup_rule='macro over users with at least one subgroup target; full-catalog ranks; subgroup IDCG; empty groups have null metrics')

    def _check_args(self):
        settings = getattr(self.config, 'final_config_dict', self.config)
        diagnostics = settings.get('missing_diagnostics')
        if diagnostics is not None and not isinstance(diagnostics, bool):
            raise ValueError('missing_diagnostics must be boolean')
        for name in ('missing_eval_interval', 'missing_eval_seed'):
            value = settings.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 0):
                raise ValueError(f'{name} must be a non-negative integer')
        # Check metrics
        if isinstance(self.metrics, (str, list)):
            if isinstance(self.metrics, str):
                self.metrics = [self.metrics]
        else:
            raise TypeError('metrics must be str or list')

        # Convert metric to lowercase
        for m in self.metrics:
            if m.lower() not in topk_metrics:
                raise ValueError("There is no user grouped topk metric named {}!".format(m))
        self.metrics = [metric.lower() for metric in self.metrics]

        # Check topk:
        if isinstance(self.topk, (int, list)):
            if isinstance(self.topk, int):
                self.topk = [self.topk]
            for topk in self.topk:
                if topk <= 0:
                    raise ValueError(
                        'topk must be a positive integer or a list of positive integers, but get `{}`'.format(topk))
        else:
            raise TypeError('The topk must be a integer, list')

    def _calculate_metrics(self, pos_len_list, topk_index):
        """Average configured metrics over users."""
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_index, pos_len_list)
            result_list.append(result)
        return np.stack(result_list, axis=0)

    def __str__(self):
        mesg = 'The TopK Evaluator Info:\n' + '\tMetrics:[' + ', '.join(
            [topk_metrics[metric.lower()] for metric in self.metrics]) \
               + '], TopK:[' + ', '.join(map(str, self.topk)) + ']'
        return mesg
