"""Packaged defaults plus validated explicit overrides; no archived branches."""
from pathlib import Path
import torch
import yaml


class Config:
    def __init__(self, model='LGMRecOpt', dataset='baby', config_dict=None):
        if model not in ('LGMRec', 'LGMRecOpt') or dataset not in ('baby', 'sports', 'clothing'):
            raise ValueError('Unsupported model or dataset')
        root = Path(__file__).resolve().parents[1] / 'configs'
        values = {}
        model_file = 'LGMREC.yaml' if model == 'LGMRec' else 'LGMRecOpt.yaml'
        paths = [root / 'overall.yaml', root / 'dataset' / f'{dataset}.yaml',
                 root / 'model' / model_file]
        for path in paths:
            values.update(yaml.safe_load(path.read_text(encoding='utf-8')) or {})
        values.update(values.pop('dataset_overrides', {}).get(dataset, {}))
        values.update(protocol='train_only', report_test=False, ablation=None,
                      checkpoint_dir='data/runs/checkpoints', log_dir='data/runs/log',
                      recommend_topk='data/runs/recommendations/', gpu_id=0)
        extra = dict(config_dict or {})
        unknown = set(extra) - set(values) - {'model', 'dataset'}
        if unknown:
            raise ValueError('Unknown or retired configuration keys: ' + ', '.join(sorted(unknown)))
        if extra.get('model', model) != model or extra.get('dataset', dataset) != dataset:
            raise ValueError('Use model/dataset arguments, not conflicting overrides')
        values.update(extra)
        values.update(model=model, dataset=dataset)
        if values['protocol'] != 'train_only':
            raise ValueError('Only train_only is supported')
        if values['ablation'] not in (None, 'no_enhancement'):
            raise ValueError('Retired ablations live in references/legacy, not this runtime')
        values['evaluation_version'] = 'full-ranking-train-only-full-precision-v2'
        values['search_version'] = 'validation-only-grid-v1'
        values['valid_metric_bigger'] = values['valid_metric'].split('@')[0].lower() not in ('rmse', 'mae', 'logloss')
        seed = values['seed']
        if isinstance(seed, (list, tuple)) and len(seed) != 1:
            raise ValueError('Run each seed separately; never select the best seed')
        values['device'] = torch.device(f"cuda:{values['gpu_id']}" if torch.cuda.is_available() and values['use_gpu'] else 'cpu')
        for key in ('cl_weight', 'reg_weight', 'lambda_hcl'):
            value = values.get(key)
            if isinstance(value, list):
                values[key] = [float(x) for x in value]
            elif value is not None:
                values[key] = float(value)
        self.final_config_dict = values

    def __getitem__(self, key):
        return self.final_config_dict.get(key)

    def __setitem__(self, key, value):
        self.final_config_dict[key] = value

    def __contains__(self, key):
        return key in self.final_config_dict

    def __str__(self):
        return '\n' + '\n'.join(f'{k}={v}' for k, v in self.final_config_dict.items()) + '\n'

    __repr__ = __str__
