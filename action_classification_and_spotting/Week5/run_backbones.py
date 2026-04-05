#!/usr/bin/env python3
"""
Run backbone ablation experiments across GPUs.

Usage (from Week5/):
    python3 run_backbones.py
    python3 run_backbones.py --gpus 0 1 --models backbone_resnet18 backbone_resnet50
    python3 run_backbones.py --dry-run
"""

import argparse
import os
import queue
import subprocess
import threading
import time
from datetime import datetime

DEFAULT_GPUS = [1, 2, 3]

DEFAULT_MODELS = [
    'backbone_resnet18',
    'backbone_resnet50',
    'backbone_efficientnet_b0',
    'backbone_efficientnet_b3',
    'backbone_clip_vitb32',
    'backbone_convnext_tiny',
]

DEFAULT_LOG_DIR = 'logs/backbones'
START_DELAY_S = 10


def _now():
    return datetime.now().strftime('%H:%M:%S')


def _run(model_name, gpu_id, gpu_queue, results, dry_run, log_dir):
    log_path = os.path.join(log_dir, f'{model_name}.log')
    print(f'[{_now()}] GPU {gpu_id} | START  {model_name:<35s}  log -> {log_path}',
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
    gpu_queue.put(gpu_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int, default=DEFAULT_GPUS)
    parser.add_argument('--models', nargs='+', type=str, default=DEFAULT_MODELS)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--log-dir', type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument('--start-delay', type=int, default=START_DELAY_S)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    gpu_queue = queue.Queue()
    for gpu in args.gpus:
        gpu_queue.put(gpu)

    results = {}
    threads = []

    print(f'[{_now()}] Scheduling {len(args.models)} experiments '
          f'across GPUs {args.gpus}')

    for i, model_name in enumerate(args.models):
        gpu_id = gpu_queue.get()
        t = threading.Thread(
            target=_run,
            args=(model_name, gpu_id, gpu_queue, results, args.dry_run, args.log_dir),
            daemon=True,
        )
        t.start()
        threads.append(t)

        if i < len(args.models) - 1:
            time.sleep(args.start_delay)

    for t in threads:
        t.join()

    ok     = [m for m, rc in results.items() if rc == 0]
    failed = [m for m, rc in results.items() if rc != 0]

    print(f'\n[{_now()}] -- Done --')
    print(f'  Passed  ({len(ok)}):  {", ".join(ok) or "-"}')
    print(f'  Failed  ({len(failed)}):  {", ".join(failed) or "-"}')


if __name__ == '__main__':
    main()
