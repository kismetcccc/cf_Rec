"""One training entry point with isolated artifacts and machine-readable status."""
import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
import traceback
from uuid import uuid4
import yaml

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('NUMEXPR_MAX_THREADS', '48')


def parse_overrides(entries):
    result = {}
    for entry in entries:
        key, sep, value = entry.partition('=')
        if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key) or not value.strip():
            raise ValueError(f'Expected KEY=VALUE: {entry}')
        parsed = yaml.safe_load(value)
        if isinstance(parsed, str) and re.fullmatch(r'[-+]?[\d.]+[eE][-+]?\d+', parsed):
            parsed = float(parsed)
        result[key] = parsed
    return result


def main():
    parser = argparse.ArgumentParser(description='LGMRec: TRAIN-only learning, VALID selection')
    parser.add_argument('-m', '--model', choices=['LGMRec', 'LGMRecOpt'], default='LGMRecOpt')
    parser.add_argument('-d', '--dataset', choices=['baby', 'sports', 'clothing'], default='baby')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--run-dir', type=Path, help='Optional new directory for status.json; training logs always go to log/')
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument('--save-checkpoint', action='store_true',
                            help='Opt in to saving the best VALID checkpoint')
    checkpoint.add_argument('--no-checkpoint', action='store_true',
                            help='Do not save checkpoints (the default)')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    args = parser.parse_args()
    try:
        options = parse_overrides(args.set)
    except (ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    reserved = {'model', 'dataset', 'seed', 'gpu_id', 'data_path', 'protocol',
                'report_test', 'checkpoint_dir', 'log_dir', 'recommend_topk',
                'feat_mod_mode', 'ablation', 'implementation_version'}
    if reserved.intersection(options):
        parser.error('Use named CLI options; protected overrides: ' + ', '.join(sorted(reserved.intersection(options))))
    run = args.run_dir.resolve() if args.run_dir else None
    artifacts = run or (Path('log/checkpoints') / (
        f'{datetime.now():%Y%m%d-%H%M%S}-{args.model}-seed{args.seed}-{uuid4().hex[:8]}')).resolve()
    options.update(seed=args.seed, gpu_id=args.gpu, protocol='train_only', report_test=False,
                   data_path=str(args.data_dir.resolve()) + os.sep,
                   checkpoint_dir=str(artifacts / 'checkpoints'), log_dir=str(Path('log').resolve()),
                   recommend_topk=str(artifacts / 'recommendations') + os.sep)
    from utils.configurator import Config
    from utils.quick_start import quick_start
    from utils.experiment_state import runtime_metadata
    # Reject misspelled/retired options before making an output directory.
    try:
        checked_config = Config(args.model, args.dataset, options)
    except ValueError as exc:
        parser.error(str(exc))
    if run is not None:
        try:
            run.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            parser.error(f'Run directory already exists; preserved unchanged: {run}')
    from utils.logger import init_logger
    logfile = init_logger(checked_config)
    logger = logging.getLogger()
    status = dict(schema_version=1, model=args.model, dataset=args.dataset,
                  seed=args.seed, variant='sparse_anchor_nohcl' if args.model == 'LGMRecOpt' else 'baseline', options=options,
                  checkpoint_enabled=args.save_checkpoint,
                  started_at=datetime.now().isoformat(), finished=False)
    status['log_file'] = logfile
    def save_status():
        if run is None:
            return
        temp = run / 'status.json.tmp'
        temp.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding='utf-8')
        temp.replace(run / 'status.json')
    save_status()
    print(f'Log: {logfile}', flush=True)
    try:
        status['runtime'] = runtime_metadata(checked_config['device'])
        logger.info('Runtime: %s', json.dumps(status['runtime'], ensure_ascii=False))
        logger.info('Checkpoint enabled: %s', args.save_checkpoint)
        result = quick_start(
            args.model, args.dataset, options,
            save_model=args.save_checkpoint, initialize_logger=False,
        )
        status.update(result, finished=True, returncode=0)
    except Exception:
        status.update(returncode=1, error=traceback.format_exc())
        logger.exception('Experiment failed')
        raise
    finally:
        status['ended_at'] = datetime.now().isoformat()
        logger.info('Experiment summary: %s', json.dumps(status, ensure_ascii=False))
        save_status()


if __name__ == '__main__':
    main()
