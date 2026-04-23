#!/usr/bin/env python3
"""
Check interrupted wandb runs and report the best resume point.

For each run, reconstructs which checkpoint epochs were saved
(best_loss, best_map12, best_map10) by replaying the logged history,
then reports the latest saved checkpoint and the epoch to resume from.

Usage:
    python3 check_resume.py
"""

import wandb
import pandas as pd

ENTITY  = 'just-an-arbitrary-team-name'
PROJECT = 'action-spotting-final'

RUNS = {
    '9sfd21r2': 'phaseD_rny008gsf_unet',
    '90u7hgea': 'phaseD_rny008gsf_unet_mixup',
}


def analyse_run(api, run_id, run_name):
    print(f'\n{"="*60}')
    print(f'Run: {run_name}  (id: {run_id})')
    print(f'{"="*60}')

    run = api.run(f'{ENTITY}/{PROJECT}/{run_id}')
    print(f'  State : {run.state}')
    print(f'  Config: num_epochs={run.config.get("num_epochs")}  '
          f'map_eval_freq={run.config.get("map_eval_freq")}')

    # Pull full history; pandas makes column access easy.
    history = run.history(samples=10_000, pandas=True)

    if history.empty:
        print('  ERROR: no history rows found.')
        return

    # Normalise: wandb sometimes stores epoch in the step counter only.
    if 'epoch' not in history.columns:
        history['epoch'] = history['_step']

    # Drop rows that are summary / final-log rows (no epoch value).
    history = history.dropna(subset=['epoch'])
    history['epoch'] = history['epoch'].astype(int)

    last_logged_epoch = int(history['epoch'].max())
    print(f'\n  Last epoch logged : {last_logged_epoch}')

    # ── Reconstruct best-checkpoint epochs ──────────────────────────────────
    # best_loss  : epoch where val_loss was at its all-time minimum so far
    # best_map12 : epoch where val_map12 was at its all-time maximum so far
    # best_map10 : epoch where val_map10 was at its all-time maximum so far

    best_loss_epoch  = None
    best_map12_epoch = None
    best_map10_epoch = None

    best_loss_val  =  float('inf')
    best_map12_val = -float('inf')
    best_map10_val = -float('inf')

    # Sort chronologically.
    history_sorted = history.sort_values('epoch')

    for _, row in history_sorted.iterrows():
        epoch = int(row['epoch'])

        if pd.notna(row.get('val_loss')):
            v = float(row['val_loss'])
            if v < best_loss_val:
                best_loss_val  = v
                best_loss_epoch = epoch

        if pd.notna(row.get('val_map12')):
            v = float(row['val_map12'])
            if v > best_map12_val:
                best_map12_val  = v
                best_map12_epoch = epoch

        if pd.notna(row.get('val_map10')):
            v = float(row['val_map10'])
            if v > best_map10_val:
                best_map10_val  = v
                best_map10_epoch = epoch

    print('\n  Saved checkpoints:')
    print(f'    checkpoint_best_loss.pt   → epoch {best_loss_epoch}'
          f'  (val_loss={best_loss_val:.5f})')
    print(f'    checkpoint_best_map12.pt  → epoch {best_map12_epoch}'
          f'  (val_map12={best_map12_val:.2f})')
    print(f'    checkpoint_best_map10.pt  → epoch {best_map10_epoch}'
          f'  (val_map10={best_map10_val:.2f})')

    # Latest checkpoint = max of the three saved epochs.
    candidates = {
        'checkpoint_best_loss.pt':  best_loss_epoch,
        'checkpoint_best_map12.pt': best_map12_epoch,
        'checkpoint_best_map10.pt': best_map10_epoch,
    }
    candidates = {k: v for k, v in candidates.items() if v is not None}

    if not candidates:
        print('\n  ERROR: could not determine any checkpoint epoch.')
        return

    latest_ckpt_file  = max(candidates, key=candidates.get)
    latest_ckpt_epoch = candidates[latest_ckpt_file]
    resume_from_epoch = latest_ckpt_epoch + 1

    num_epochs = int(run.config.get('num_epochs', 35))
    epochs_remaining = num_epochs - resume_from_epoch

    print(f'\n  ► Latest checkpoint : {latest_ckpt_file}  (epoch {latest_ckpt_epoch})')
    print(f'  ► Resume from epoch : {resume_from_epoch}')
    print(f'  ► Epochs remaining  : {epochs_remaining} / {num_epochs}')
    print(f'  ► Approx. time lost : ~{last_logged_epoch - latest_ckpt_epoch} epoch(s) '
          f'of training that must be repeated')


def main():
    api = wandb.Api()
    for run_id, run_name in RUNS.items():
        try:
            analyse_run(api, run_id, run_name)
        except Exception as exc:
            print(f'\nFailed to fetch run {run_id} ({run_name}): {exc}')

    print('\nDone.')


if __name__ == '__main__':
    main()
