#!/usr/bin/env python3
"""
Compare config files against wandb finished runs and report what's still pending.

Usage:
    python check_pending.py
    python check_pending.py --config_dir config --prefixes phaseA_ phaseB_
"""

import argparse
import os
import glob
import wandb

ENTITY  = 'just-an-arbitrary-team-name'
PROJECT = 'action-spotting-final'

EXCLUDE_CONFIGS = {'baseline', 'store_splits'}


def fetch_finished_names(prefixes):
    api  = wandb.Api()
    runs = api.runs(f'{ENTITY}/{PROJECT}', filters={'state': 'finished'})
    names = set()
    for run in runs:
        if not prefixes or any(run.name.startswith(p) for p in prefixes):
            names.add(run.name)
    return names


def get_config_names(config_dir, prefixes):
    names = set()
    for path in glob.glob(os.path.join(config_dir, '*.json')):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in EXCLUDE_CONFIGS:
            continue
        if not prefixes or any(stem.startswith(p) for p in prefixes):
            names.add(stem)
    return names


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config_dir', default='config')
    p.add_argument('--prefixes', nargs='+', default=[],
                   help='Filter by name prefix (e.g. phaseA_ phaseB_). Empty = all.')
    args = p.parse_args()

    print(f'Fetching finished runs from {ENTITY}/{PROJECT} ...')
    finished = fetch_finished_names(args.prefixes)
    configs  = get_config_names(args.config_dir, args.prefixes)

    pending  = sorted(configs - finished)
    done     = sorted(configs & finished)
    unknown  = sorted(finished - configs)  # finished in wandb but no config file

    print(f'\nConfig files found : {len(configs)}')
    print(f'Finished in wandb  : {len(finished)}')

    print(f'\n{"="*50}')
    print(f'PENDING ({len(pending)}) — not yet finished in wandb:')
    print(f'{"="*50}')
    for name in pending:
        print(f'  {name}')

    print(f'\n{"="*50}')
    print(f'DONE ({len(done)}) — finished in wandb:')
    print(f'{"="*50}')
    for name in done:
        print(f'  {name}')

    if unknown:
        print(f'\n{"="*50}')
        print(f'WANDB ONLY ({len(unknown)}) — finished runs with no matching config:')
        print(f'{"="*50}')
        for name in unknown:
            print(f'  {name}')

    if pending:
        print(f'\nTo run all pending experiments:')
        print(f"  python run_experiments.py --models {' '.join(pending)}")


if __name__ == '__main__':
    main()
