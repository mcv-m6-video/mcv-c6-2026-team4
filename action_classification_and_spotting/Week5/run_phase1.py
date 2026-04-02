#!/usr/bin/env python3
"""
Dynamic scheduler that distributes training runs across multiple GPUs.

Each GPU runs one experiment at a time. As soon as a GPU finishes,
the next queued experiment is dispatched to it. A short delay between
consecutive starts prevents wandb from hitting its request rate limit.

Usage (from Week5/):
    # Run all phase1 experiments on GPUs 1,2,3 (defaults)
    python3 run_phase1.py

    # Override GPUs and/or models
    python3 run_phase1.py --gpus 1 2 --models phase1_tcn phase1_gru

    # Skip wandb + local file writes
    python3 run_phase1.py --dry-run
"""

import argparse
import os
import queue
import subprocess
import threading
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_GPUS = [1, 2, 3]

DEFAULT_MODELS = [
    'phase1_gru',
    'phase1_gru_deep',
    'phase1_tcn',
    'phase1_tcn_deep',
    'phase1_transformer',
    'phase1_transformer_deep',
    'phase1_tcn_unet_2l',
    'phase1_tcn_unet_3l',
]

DEFAULT_LOG_DIR = 'logs/phase1'
START_DELAY_S = 10   # seconds to wait between consecutive experiment starts

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _now():
    return datetime.now().strftime('%H:%M:%S')


def _run(model_name, gpu_id, gpu_queue, results, dry_run, log_dir):
    """Run one experiment on gpu_id, then return the GPU to the pool."""
    log_path = os.path.join(log_dir, f'{model_name}.log')
    print(f'[{_now()}] GPU {gpu_id} | START  {model_name:<35s}  log → {log_path}',
          flush=True)

    cmd = ['python3', 'main_classification.py', '--model', model_name]
    if dry_run:
        cmd.append('--dry-run')

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        with open(log_path, 'w') as log_file:
            proc = subprocess.run(
                cmd, env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        rc = proc.returncode
    except Exception as exc:
        print(f'[{_now()}] GPU {gpu_id} | ERROR  {model_name}: {exc}', flush=True)
        rc = -1

    status = 'OK' if rc == 0 else f'FAILED (exit {rc})'
    print(f'[{_now()}] GPU {gpu_id} | {status:<20s} {model_name}', flush=True)

    results[model_name] = rc
    gpu_queue.put(gpu_id)   # return GPU to pool


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Distribute training runs across GPUs with a dynamic queue.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--gpus', nargs='+', type=int, default=DEFAULT_GPUS,
        metavar='ID',
        help=f'CUDA device IDs to use (default: {DEFAULT_GPUS})',
    )
    parser.add_argument(
        '--models', nargs='+', type=str, default=DEFAULT_MODELS,
        metavar='NAME',
        help='Config names to run, i.e. the stem of config/<name>.json '
             '(default: all phase1 experiments)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Pass --dry-run to each experiment (disables wandb and local file writes)',
    )
    parser.add_argument(
        '--log-dir', type=str, default=DEFAULT_LOG_DIR,
        metavar='DIR',
        help=f'Directory for per-experiment log files (default: {DEFAULT_LOG_DIR})',
    )
    parser.add_argument(
        '--start-delay', type=int, default=START_DELAY_S,
        metavar='SECONDS',
        help=f'Seconds to wait between consecutive experiment starts to avoid '
             f'wandb rate limits (default: {START_DELAY_S})',
    )
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    gpu_queue = queue.Queue()
    for gpu in args.gpus:
        gpu_queue.put(gpu)

    results = {}
    threads = []

    print(f'[{_now()}] Scheduling {len(args.models)} experiments '
          f'across GPUs {args.gpus} '
          f'(start delay: {args.start_delay}s)')

    for i, model_name in enumerate(args.models):
        # Blocks until a GPU is free
        gpu_id = gpu_queue.get()
        t = threading.Thread(
            target=_run,
            args=(model_name, gpu_id, gpu_queue, results, args.dry_run, args.log_dir),
            daemon=True,
        )
        t.start()
        threads.append(t)

        # Stagger starts to avoid wandb rate-limit errors.
        # Skip the delay after the last experiment.
        if i < len(args.models) - 1:
            time.sleep(args.start_delay)

    for t in threads:
        t.join()

    # Summary
    ok     = [m for m, rc in results.items() if rc == 0]
    failed = [m for m, rc in results.items() if rc != 0]

    print(f'\n[{_now()}] ── Done ──────────────────────────────')
    print(f'  Passed  ({len(ok)}):  {", ".join(ok)  or "-"}')
    print(f'  Failed  ({len(failed)}):  {", ".join(failed) or "-"}')


if __name__ == '__main__':
    main()
