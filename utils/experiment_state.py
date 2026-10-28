"""Content identity for repeatable checkpoint evaluation and log comparisons."""
import hashlib
import json
from pathlib import Path
import platform


def fingerprint_inputs(config):
    directory = Path(config['data_path']) / config['dataset']
    hashes = {}
    for key in ('inter_file_name', 'vision_feature_file', 'text_feature_file'):
        digest = hashlib.sha256()
        with (directory / config[key]).open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        hashes[key] = digest.hexdigest()
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return hashes, signature


def runtime_metadata(device):
    """Record executable source identity and runtime, including uncommitted code."""
    import numpy as np
    import scipy
    import pandas as pd
    import yaml
    import torch
    device = torch.device(device)
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    sources = [package / 'main.py']
    for folder in ('common', 'configs', 'models', 'utils'):
        sources.extend(path for path in (package / folder).rglob('*')
                       if path.suffix in ('.py', '.yaml'))
    for path in sorted(sources):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
        digest.update(b'\0')
    return dict(
        source_sha256=digest.hexdigest(), source_scope='main.py and common/configs/models/utils .py/.yaml raw bytes',
        python=platform.python_version(),
        platform=platform.platform(), torch=torch.__version__, numpy=np.__version__,
        scipy=scipy.__version__, pandas=pd.__version__, pyyaml=yaml.__version__, cuda=torch.version.cuda,
        used_device=str(device),
        used_gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
    )
