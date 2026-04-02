#!/usr/bin/env python3
"""
Analysis script for Phase 1 experiment results.
Loads results from save_dir, compares models, and produces charts + printed analysis.

Usage:
    python analyze_phase1.py --save_dir /path/to/SN-BAS-2025_savedata
                             [--config_dir config/]
                             [--prefix phase1_]
                             [--out_dir analysis/]
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tabulate import tabulate


CLASS_NAMES = [
    'BALL PLAYER BLOCK', 'CROSS', 'DRIVE', 'FREE KICK', 'GOAL',
    'HEADER', 'HIGH PASS', 'OUT', 'PASS', 'PLAYER SUCCESSFUL TACKLE',
    'SHOT', 'THROW IN',
]
EXCLUDE_FROM_MAP10 = {'FREE KICK', 'GOAL'}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_results(save_dir, prefix):
    """Return dict {model_name: results_dict} for all subdirs matching prefix."""
    models = {}
    for name in sorted(os.listdir(save_dir)):
        if not name.startswith(prefix):
            continue
        results_path = os.path.join(save_dir, name, 'results.json')
        loss_path    = os.path.join(save_dir, name, 'loss.json')
        if not os.path.exists(results_path):
            continue

        with open(results_path) as f:
            results = json.load(f)
        with open(loss_path) as f:
            loss = json.load(f)

        # Support both old flat format and new nested-by-checkpoint format
        if 'best_loss' in results:
            # New format — use the best_loss entry as canonical
            entry = results['best_loss']
            entry['total_train_time_s'] = results.get('total_train_time_s')
        else:
            entry = results

        entry['loss_curve'] = loss
        models[name] = entry
    return models


def load_configs(config_dir, model_names):
    """Return dict {model_name: config_dict}."""
    configs = {}
    for name in model_names:
        path = os.path.join(config_dir, name + '.json')
        if os.path.exists(path):
            with open(path) as f:
                configs[name] = json.load(f)
    return configs


def short_name(model_name, prefix):
    return model_name[len(prefix):]


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_map_comparison(models, prefix, out_dir):
    names  = [short_name(k, prefix) for k in models]
    map12s = [models[k]['mAP12'] for k in models]
    map10s = [models[k]['mAP10'] for k in models]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    bars12 = ax.bar(x - w/2, map12s, w, label='mAP@12', color='steelblue')
    bars10 = ax.bar(x + w/2, map10s, w, label='mAP@10', color='coral')

    ax.set_ylabel('mAP (%)')
    ax.set_title('Phase 1 — mAP Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.legend()
    ax.bar_label(bars12, fmt='%.1f', padding=2, fontsize=8)
    ax.bar_label(bars10, fmt='%.1f', padding=2, fontsize=8)
    ax.set_ylim(0, max(map10s) * 1.2)
    fig.tight_layout()
    path = os.path.join(out_dir, 'map_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


def plot_per_class_ap(models, prefix, out_dir):
    names = list(models.keys())
    short = [short_name(k, prefix) for k in names]
    n_models = len(names)
    n_classes = len(CLASS_NAMES)

    data = np.zeros((n_models, n_classes))
    for i, name in enumerate(names):
        per_class = models[name].get('per_class_AP', {})
        for j, cls in enumerate(CLASS_NAMES):
            data[i, j] = per_class.get(cls, 0.0)

    fig, ax = plt.subplots(figsize=(14, max(4, n_models * 0.6 + 2)))
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=80)

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(short, fontsize=9)
    ax.set_title('Per-class AP (%) — all Phase 1 models')

    for i in range(n_models):
        for j in range(n_classes):
            val = data[i, j]
            color = 'white' if val < 20 or val > 60 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=7, color=color)

    fig.colorbar(im, ax=ax, label='AP (%)')
    fig.tight_layout()
    path = os.path.join(out_dir, 'per_class_ap.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


def plot_loss_curves(models, prefix, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = cm.tab10(np.linspace(0, 1, len(models)))

    for (name, entry), color in zip(models.items(), colors):
        short = short_name(name, prefix)
        curve = entry['loss_curve']
        epochs     = [e['epoch']    for e in curve]
        train_loss = [e['train']    for e in curve]
        val_loss   = [e['val']      for e in curve]
        best_ep    = entry.get('best_epoch', None)

        axes[0].plot(epochs, train_loss, color=color, label=short)
        axes[1].plot(epochs, val_loss,   color=color, label=short)
        if best_ep is not None:
            axes[1].axvline(best_ep, color=color, linestyle=':', alpha=0.5)

    for ax, title in zip(axes, ['Train Loss', 'Val Loss (dotted = best epoch)']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, 'loss_curves.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


def plot_time_vs_map(models, prefix, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = cm.tab10(np.linspace(0, 1, len(models)))

    for (name, entry), color in zip(models.items(), colors):
        short = short_name(name, prefix)
        t = (entry.get('total_train_time_s') or 0) / 3600
        ax.scatter(t, entry['mAP12'], color=color, s=80, zorder=3)
        ax.scatter(t, entry['mAP10'], color=color, s=80, marker='^', zorder=3)
        ax.annotate(short, (t, entry['mAP12']), textcoords='offset points',
                    xytext=(5, 3), fontsize=7)

    ax.set_xlabel('Train time (hours)')
    ax.set_ylabel('mAP (%)')
    ax.set_title('Train time vs mAP  (circle=mAP12, triangle=mAP10)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, 'time_vs_map.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


# ── printed analysis ──────────────────────────────────────────────────────────

def print_summary_table(models, prefix, configs):
    rows = []
    for name, entry in models.items():
        short = short_name(name, prefix)
        cfg = configs.get(name, {})
        neck = cfg.get('neck_architecture', 'max_pool')
        neck_params = cfg.get('neck_parameters', {})
        # Summarise neck params briefly
        param_str = ', '.join(f'{k}={v}' for k, v in neck_params.items()
                              if k not in ('dropout',))
        hours = (entry.get('total_train_time_s') or 0) / 3600
        rows.append([
            short,
            neck,
            param_str or '-',
            f"{entry['mAP12']:.2f}",
            f"{entry['mAP10']:.2f}",
            entry.get('best_epoch', '?'),
            f"{hours:.1f}h",
        ])

    # Sort by mAP12 desc
    rows.sort(key=lambda r: float(r[3]), reverse=True)
    headers = ['Model', 'Neck', 'Neck params', 'mAP12', 'mAP10', 'Best ep.', 'Train time']
    print('\n' + tabulate(rows, headers=headers, tablefmt='grid'))


def print_class_analysis(models, prefix):
    """Print per-class mean AP and variance across models, flag hard classes."""
    names = list(models.keys())
    data = {}
    for cls in CLASS_NAMES:
        vals = [models[n].get('per_class_AP', {}).get(cls, 0.0) for n in names]
        data[cls] = vals

    rows = []
    for cls in CLASS_NAMES:
        vals = data[cls]
        rows.append([
            cls,
            f"{np.mean(vals):.1f}",
            f"{np.std(vals):.1f}",
            f"{np.min(vals):.1f}",
            f"{np.max(vals):.1f}",
            '(excl. mAP10)' if cls in EXCLUDE_FROM_MAP10 else '',
        ])
    rows.sort(key=lambda r: float(r[1]))  # sort by mean AP asc
    headers = ['Class', 'Mean AP', 'Std', 'Min', 'Max', 'Note']
    print('\n' + tabulate(rows, headers=headers, tablefmt='grid'))


def print_convergence_analysis(models, prefix):
    """Flag models that look overfit or underfit."""
    rows = []
    for name, entry in models.items():
        short = short_name(name, prefix)
        curve = entry['loss_curve']
        best_ep = entry.get('best_epoch', len(curve) - 1)
        num_ep = len(curve)

        final_train = curve[-1]['train']
        final_val   = curve[-1]['val']
        best_val    = min(e['val'] for e in curve)
        overfit_gap = final_val - best_val   # how much val degraded after best epoch

        if best_ep <= 2:
            note = 'SUSPICIOUS: best at epoch 0-2 (possible bug or config issue)'
        elif best_ep >= num_ep - 3:
            note = 'May need more epochs (best near end)'
        elif overfit_gap > 0.02:
            note = f'Overfit after epoch {best_ep} (val degraded {overfit_gap:.4f})'
        else:
            note = 'OK'

        rows.append([short, best_ep, num_ep, f'{final_train:.4f}',
                     f'{final_val:.4f}', f'{overfit_gap:.4f}', note])

    headers = ['Model', 'Best ep', 'Num ep', 'Final train', 'Final val',
               'Val drift', 'Convergence note']
    print('\n' + tabulate(rows, headers=headers, tablefmt='grid'))


def print_next_steps_analysis(models, prefix, configs):
    """Data-driven suggestions for what to try next."""
    names = list(models.keys())

    best_name = max(names, key=lambda n: models[n]['mAP12'])
    best      = models[best_name]
    best_cfg  = configs.get(best_name, {})

    print('\n' + '='*70)
    print('ANALYSIS & NEXT STEPS')
    print('='*70)

    print(f'\nBest model so far: {short_name(best_name, prefix)}')
    print(f'  mAP12={best["mAP12"]:.2f}  mAP10={best["mAP10"]:.2f}  '
          f'best_epoch={best.get("best_epoch")}  '
          f'neck={best_cfg.get("neck_architecture")}')

    # Class-level hardness
    hard_classes = []
    easy_classes = []
    for cls in CLASS_NAMES:
        mean_ap = np.mean([models[n].get('per_class_AP', {}).get(cls, 0.0) for n in names])
        if mean_ap < 10:
            hard_classes.append((cls, mean_ap))
        elif mean_ap > 50:
            easy_classes.append((cls, mean_ap))
    hard_classes.sort(key=lambda x: x[1])
    easy_classes.sort(key=lambda x: x[1], reverse=True)

    if hard_classes:
        print(f'\nHard classes (mean AP < 10% across all models):')
        for cls, ap in hard_classes:
            note = ' [excluded from mAP10]' if cls in EXCLUDE_FROM_MAP10 else ' [counted in mAP10 — most impactful to fix]'
            print(f'  {cls}: {ap:.1f}%{note}')
        print('  → These likely suffer from class imbalance.')
        print('    Suggestions: weighted BCE loss, focal loss, oversampling.')

    if easy_classes:
        print(f'\nEasy classes (mean AP > 50% across all models):')
        for cls, ap in easy_classes:
            print(f'  {cls}: {ap:.1f}%')
        print('  → These converge well regardless of neck — focus effort elsewhere.')

    # Check for suspicious convergence
    suspicious = [n for n in names if models[n].get('best_epoch', 99) <= 2]
    if suspicious:
        print(f'\nSuspicious early convergence (best_epoch ≤ 2):')
        for n in suspicious:
            print(f'  {short_name(n, prefix)} — best_epoch={models[n]["best_epoch"]}')
        print('  → These likely have a bug or misconfiguration (check FIXME in TCN blocks).')

    # Neck architecture ranking
    print('\nNeck architecture ranking (by mAP12):')
    ranked = sorted(names, key=lambda n: models[n]['mAP12'], reverse=True)
    for i, n in enumerate(ranked):
        cfg = configs.get(n, {})
        print(f'  {i+1}. {short_name(n, prefix):25s}  mAP12={models[n]["mAP12"]:.2f}  '
              f'mAP10={models[n]["mAP10"]:.2f}  neck={cfg.get("neck_architecture", "?")}')

    # Depth: does it help?
    gru_base  = next((models[n]['mAP12'] for n in names if 'gru' in n and 'deep' not in n), None)
    gru_deep  = next((models[n]['mAP12'] for n in names if 'gru_deep' in n), None)
    tcn_base  = next((models[n]['mAP12'] for n in names if n.endswith('tcn')), None)
    tcn_deep  = next((models[n]['mAP12'] for n in names if 'tcn_deep' in n), None)
    if gru_base and gru_deep:
        print(f'\nDepth effect (GRU):   base={gru_base:.2f}  deep={gru_deep:.2f}'
              f'  (Δ={gru_deep - gru_base:+.2f})')
    if tcn_base and tcn_deep:
        print(f'Depth effect (TCN):   base={tcn_base:.2f}  deep={tcn_deep:.2f}'
              f'  (Δ={tcn_deep - tcn_base:+.2f})')

    print('\nSuggested next experiments:')
    suggestions = [
        ('Weighted BCE / Focal loss',
         f'Hard classes ({", ".join(c for c,_ in hard_classes[:3])}) '
         'drag mAP10 down significantly. Try pos_weight or focal loss.'),
        ('Better backbone',
         'All models use rny002. Try rny008, efficientnet, or a larger RegNet '
         'to get richer per-frame features before the neck.'),
        ('Fix TCN blocks (FIXME)',
         'TCN unet variants converge at epoch 0 — fix the bug before drawing '
         'conclusions about that architecture.'),
        ('More epochs for best model',
         f'{short_name(best_name, prefix)} best_epoch={best.get("best_epoch")} / {len(best["loss_curve"])} — '
         'check if the val curve is still declining at the end.'),
        ('Clip length / stride ablation',
         'All models use clip_len=50, stride=2. Longer clips or smaller '
         'stride may help temporal context for the neck.'),
    ]
    for i, (title, detail) in enumerate(suggestions):
        print(f'  {i+1}. {title}')
        print(f'     {detail}')

    print()


# ── main ──────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',  type=str,
                        default='/home/msiau/data/tmp/amarcos/SoccerNet/SN-BAS-2025_savedata')
    parser.add_argument('--config_dir', type=str, default='config')
    parser.add_argument('--prefix',    type=str, default='phase1_')
    parser.add_argument('--out_dir',   type=str, default='analysis')
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading results from: {args.save_dir}  (prefix={args.prefix!r})')
    models  = load_results(args.save_dir, args.prefix)
    configs = load_configs(args.config_dir, list(models.keys()))

    if not models:
        print('No results found.')
        return

    print(f'Found {len(models)} models: {", ".join(short_name(k, args.prefix) for k in models)}')

    print_summary_table(models, args.prefix, configs)

    print('\n--- Per-class AP statistics across models ---')
    print_class_analysis(models, args.prefix)

    print('\n--- Convergence analysis ---')
    print_convergence_analysis(models, args.prefix)

    print_next_steps_analysis(models, args.prefix, configs)

    print('Generating plots...')
    plot_map_comparison(models, args.prefix, args.out_dir)
    plot_per_class_ap(models, args.prefix, args.out_dir)
    plot_loss_curves(models, args.prefix, args.out_dir)
    plot_time_vs_map(models, args.prefix, args.out_dir)

    print(f'\nAll plots saved to: {args.out_dir}/')


if __name__ == '__main__':
    main()
