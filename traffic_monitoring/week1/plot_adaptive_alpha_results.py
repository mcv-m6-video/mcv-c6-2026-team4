"""
Plot results from adaptive alpha parameter study

Usage:
    python plot_adaptive_alpha_results.py --input adaptive_ro_results/adaptive_alpha_*.json --output_dir adaptive_alpha_plots
"""

import argparse
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(input_files):
    """Load and merge results from one or more JSON files"""
    all_results = []

    for filepath in input_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
            all_results.extend(data)

    # Sort by alpha value
    all_results.sort(key=lambda x: x['alpha'])

    return all_results


def extract_metric_arrays(results, metric_name):
    """Extract arrays of alpha values and corresponding metric values"""
    alphas = [r['alpha'] for r in results]
    total_metrics = [r['total_metrics'][metric_name] for r in results]

    return np.array(alphas), np.array(total_metrics)


def extract_per_frame_data(results, metric_name):
    """Extract per-frame data for heatmap"""
    all_frame_ids = set()
    for result in results:
        all_frame_ids.update(map(int, result['per_frame_metrics'].keys()))

    frame_ids = sorted(all_frame_ids)
    alphas = [r['alpha'] for r in results]

    data_matrix = np.full((len(alphas), len(frame_ids)), np.nan)

    for i, result in enumerate(results):
        for j, frame_id in enumerate(frame_ids):
            frame_key = str(frame_id)
            if frame_key in result['per_frame_metrics']:
                data_matrix[i, j] = result['per_frame_metrics'][frame_key][metric_name]

    return alphas, frame_ids, data_matrix


def plot_alpha_vs_total_metrics(results, output_dir):
    """Plot 1: Alpha vs Total Metrics (AP, AP50, AP75, etc.)"""
    metrics_to_plot = ['AP', 'AP50', 'AP75', 'AR_max100']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        alphas, values = extract_metric_arrays(results, metric)

        axes[idx].plot(alphas, values, 'o-', linewidth=2, markersize=8)
        axes[idx].set_xlabel('Alpha', fontsize=18)
        axes[idx].set_ylabel(metric, fontsize=18)
        axes[idx].set_title(f'{metric} vs Alpha (Adaptive Model)', fontsize=22, fontweight='bold')
        axes[idx].tick_params(axis='both', which='major', labelsize=16)
        axes[idx].grid(True, alpha=0.3)

        best_idx = np.argmax(values)
        axes[idx].axvline(alphas[best_idx], color='r', linestyle='--', alpha=0.5, label=f'Best: α={alphas[best_idx]:.2f}')
        axes[idx].legend(fontsize=14)

    plt.suptitle('Adaptive Model (mean_rho=0.01, variance_rho=0.06)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'adaptive_alpha_vs_total_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: adaptive_alpha_vs_total_metrics.png")


def plot_heatmap(results, output_dir, metric_name='AP50'):
    """Plot 2: Heatmap of Alpha vs Frame Number with metric as color"""
    alphas, frame_ids, data_matrix = extract_per_frame_data(results, metric_name)

    fig, ax = plt.subplots(figsize=(16, 8))

    im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)

    ax.set_xticks(np.arange(0, len(frame_ids), max(1, len(frame_ids)//20)))
    ax.set_xticklabels([frame_ids[i] for i in range(0, len(frame_ids), max(1, len(frame_ids)//20))], rotation=45)
    ax.set_yticks(np.arange(len(alphas)))
    ax.set_yticklabels([f'{a:.2f}' for a in alphas])

    ax.set_xlabel('Frame ID', fontsize=12)
    ax.set_ylabel('Alpha', fontsize=12)
    ax.set_title(f'{metric_name} Heatmap: Alpha vs Frame (Adaptive Model)', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_name, rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / f'heatmap_adaptive_alpha_vs_frame_{metric_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: heatmap_adaptive_alpha_vs_frame_{metric_name}.png")


def print_summary_table(results):
    """Print a summary table of results"""
    print("\n" + "="*80)
    print("SUMMARY TABLE (Adaptive Model: mean_rho=0.01, variance_rho=0.06)")
    print("="*80)
    print(f"{'Alpha':<8} {'AP':<8} {'AP50':<8} {'AP75':<8} {'AR@100':<8}")
    print("-"*80)

    for result in results:
        tm = result['total_metrics']
        print(f"{result['alpha']:<8.2f} {tm['AP']:<8.4f} {tm['AP50']:<8.4f} {tm['AP75']:<8.4f} {tm['AR_max100']:<8.4f}")

    print("="*80)

    best_ap = max(results, key=lambda x: x['total_metrics']['AP'])
    best_ap50 = max(results, key=lambda x: x['total_metrics']['AP50'])

    print(f"\nBest AP:    alpha = {best_ap['alpha']:.2f} (AP = {best_ap['total_metrics']['AP']:.4f})")
    print(f"Best AP50:  alpha = {best_ap50['alpha']:.2f} (AP50 = {best_ap50['total_metrics']['AP50']:.4f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Plot adaptive alpha parameter study results")
    parser.add_argument("--input", nargs='+', required=True, help="Input JSON file(s) with results")
    parser.add_argument("--output_dir", type=str, default="adaptive_alpha_plots", help="Output directory for plots")
    parser.add_argument("--metrics", nargs='+', default=['AP50', 'AP'], help="Metrics to plot in heatmaps")

    args = parser.parse_args()

    input_files = []
    for pattern in args.input:
        expanded = glob.glob(pattern)
        if expanded:
            input_files.extend(expanded)
        else:
            input_files.append(pattern)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Loading results from {len(input_files)} file(s)...")
    results = load_results(input_files)
    print(f"Loaded {len(results)} experiments with alpha values from {results[0]['alpha']:.2f} to {results[-1]['alpha']:.2f}")

    print_summary_table(results)

    print("\nGenerating plots...")
    plot_alpha_vs_total_metrics(results, output_dir)

    for metric in args.metrics:
        plot_heatmap(results, output_dir, metric_name=metric)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == '__main__':
    main()
