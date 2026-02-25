"""
    To plot results from std_bias study:

    python plot_std_bias_results.py --input std_bias_study_results.json --output_dir plots

    Or from multiple files:
    python plot_std_bias_results.py --input results_1.json results_2.json results_3.json results_4.json --output_dir plots
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(input_files):
    all_results = []

    for filepath in input_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
            all_results.extend(data)

    all_results.sort(key=lambda x: x['std_bias'])

    return all_results


def extract_metric_arrays(results, metric_name):
    std_bias_values = [r['std_bias'] for r in results]
    total_metrics = [r['total_metrics'][metric_name] for r in results]

    return np.array(std_bias_values), np.array(total_metrics)


def extract_per_frame_data(results, metric_name):
    all_frame_ids = set()
    for result in results:
        all_frame_ids.update(map(int, result['per_frame_metrics'].keys()))

    frame_ids = sorted(all_frame_ids)
    std_bias_values = [r['std_bias'] for r in results]

    data_matrix = np.full((len(std_bias_values), len(frame_ids)), np.nan)

    for i, result in enumerate(results):
        for j, frame_id in enumerate(frame_ids):
            frame_key = str(frame_id)
            if frame_key in result['per_frame_metrics']:
                data_matrix[i, j] = result['per_frame_metrics'][frame_key][metric_name]

    return std_bias_values, frame_ids, data_matrix


def plot_std_bias_vs_total_metrics(results, output_dir):
    metrics_to_plot = ['AP', 'AP50', 'AP75', 'AR_max100']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        std_bias_values, values = extract_metric_arrays(results, metric)

        axes[idx].plot(std_bias_values, values, 'o-', linewidth=2, markersize=8)
        axes[idx].set_xlabel('std_bias', fontsize=18)
        axes[idx].set_ylabel(metric, fontsize=18)
        axes[idx].set_title(f'{metric} vs std_bias', fontsize=22, fontweight='bold')
        axes[idx].tick_params(axis='both', which='major', labelsize=16)
        axes[idx].grid(True, alpha=0.3)

        best_idx = np.argmax(values)
        axes[idx].axvline(std_bias_values[best_idx], color='r', linestyle='--', alpha=0.5, label=f'Best: std_bias={std_bias_values[best_idx]:.2f}')
        axes[idx].legend(fontsize=16)

    plt.tight_layout()
    plt.savefig(output_dir / 'std_bias_vs_total_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: std_bias_vs_total_metrics.png")


def plot_heatmap(results, output_dir, metric_name='AP50'):
    std_bias_values, frame_ids, data_matrix = extract_per_frame_data(results, metric_name)

    fig, ax = plt.subplots(figsize=(16, 8))

    im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)

    ax.set_xticks(np.arange(0, len(frame_ids), max(1, len(frame_ids)//20)))
    ax.set_xticklabels([frame_ids[i] for i in range(0, len(frame_ids), max(1, len(frame_ids)//20))], rotation=45)
    ax.set_yticks(np.arange(len(std_bias_values)))
    ax.set_yticklabels([f'{s:.2f}' for s in std_bias_values])

    ax.set_xlabel('Frame ID', fontsize=12)
    ax.set_ylabel('std_bias', fontsize=12)
    ax.set_title(f'{metric_name} Heatmap: std_bias vs Frame', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_name, rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / f'heatmap_std_bias_vs_frame_{metric_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: heatmap_std_bias_vs_frame_{metric_name}.png")


def plot_per_frame_lines(results, output_dir, metric_name='AP50', max_std_bias_to_plot=10):
    std_bias_values, frame_ids, data_matrix = extract_per_frame_data(results, metric_name)

    if len(std_bias_values) > max_std_bias_to_plot:
        indices = np.linspace(0, len(std_bias_values)-1, max_std_bias_to_plot, dtype=int)
        std_bias_to_plot = [std_bias_values[i] for i in indices]
        data_to_plot = data_matrix[indices, :]
    else:
        std_bias_to_plot = std_bias_values
        data_to_plot = data_matrix

    fig, ax = plt.subplots(figsize=(16, 8))

    for i, std_bias in enumerate(std_bias_to_plot):
        valid_mask = ~np.isnan(data_to_plot[i, :])
        valid_frames = np.array(frame_ids)[valid_mask]
        valid_values = data_to_plot[i, :][valid_mask]

        ax.plot(valid_frames, valid_values, 'o-', label=f'std_bias={std_bias:.2f}', alpha=0.7, markersize=3)

    ax.set_xlabel('Frame ID', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} per Frame for Different std_bias Values', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'per_frame_lines_{metric_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: per_frame_lines_{metric_name}.png")


def print_summary_table(results):
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'std_bias':<10} {'AP':<8} {'AP50':<8} {'AP75':<8} {'AR@100':<8}")
    print("-"*80)

    for result in results:
        tm = result['total_metrics']
        print(f"{result['std_bias']:<10.2f} {tm['AP']:<8.4f} {tm['AP50']:<8.4f} {tm['AP75']:<8.4f} {tm['AR_max100']:<8.4f}")

    print("="*80)

    best_ap = max(results, key=lambda x: x['total_metrics']['AP'])
    best_ap50 = max(results, key=lambda x: x['total_metrics']['AP50'])

    print(f"\nBest AP:    std_bias = {best_ap['std_bias']:.2f} (AP = {best_ap['total_metrics']['AP']:.4f})")
    print(f"Best AP50:  std_bias = {best_ap50['std_bias']:.2f} (AP50 = {best_ap50['total_metrics']['AP50']:.4f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Plot std_bias parameter study results")
    parser.add_argument("--input", nargs='+', required=True, help="Input JSON file(s) with results")
    parser.add_argument("--output_dir", type=str, default="std_bias_plots", help="Output directory for plots")
    parser.add_argument("--metrics", nargs='+', default=['AP50', 'AP'], help="Metrics to plot in heatmaps and line plots")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Loading results from {len(args.input)} file(s)...")
    results = load_results(args.input)
    print(f"✓ Loaded {len(results)} experiments with std_bias values from {results[0]['std_bias']:.2f} to {results[-1]['std_bias']:.2f}")

    print_summary_table(results)

    print("\nGenerating plots...")

    plot_std_bias_vs_total_metrics(results, output_dir)

    for metric in args.metrics:
        plot_heatmap(results, output_dir, metric_name=metric)
        plot_per_frame_lines(results, output_dir, metric_name=metric)

    print(f"\n✓ All plots saved to {output_dir}/")


if __name__ == '__main__':
    main()
