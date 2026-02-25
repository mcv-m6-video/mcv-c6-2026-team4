"""
Plot results from alpha parameter study

Usage:
    # Plot from a single results file
    python plot_alpha_results.py --input alpha_study_results.json --output_dir plots

    # Plot from multiple results files (if you ran in parallel)
    python plot_alpha_results.py --input results_1.json results_2.json results_3.json results_4.json --output_dir plots
"""

import argparse
import json
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
    # Get all unique frame IDs across all experiments
    all_frame_ids = set()
    for result in results:
        all_frame_ids.update(map(int, result['per_frame_metrics'].keys()))

    frame_ids = sorted(all_frame_ids)
    alphas = [r['alpha'] for r in results]

    # Create matrix: rows = alphas, cols = frames
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
        axes[idx].set_title(f'{metric} vs Alpha', fontsize=22, fontweight='bold')
        axes[idx].tick_params(axis='both', which='major', labelsize=16)
        axes[idx].grid(True, alpha=0.3)

        # Mark best value
        best_idx = np.argmax(values)
        axes[idx].axvline(alphas[best_idx], color='r', linestyle='--', alpha=0.5, label=f'Best: α={alphas[best_idx]:.2f}')
        axes[idx].legend(fontsize=16)

    plt.tight_layout()
    plt.savefig(output_dir / 'alpha_vs_total_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: alpha_vs_total_metrics.png")


def plot_heatmap(results, output_dir, metric_name='AP50'):
    """Plot 2: Heatmap of Alpha vs Frame Number with metric as color"""
    alphas, frame_ids, data_matrix = extract_per_frame_data(results, metric_name)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot heatmap
    im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(0, len(frame_ids), max(1, len(frame_ids)//20)))
    ax.set_xticklabels([frame_ids[i] for i in range(0, len(frame_ids), max(1, len(frame_ids)//20))], rotation=45)
    ax.set_yticks(np.arange(len(alphas)))
    ax.set_yticklabels([f'{a:.2f}' for a in alphas])

    ax.set_xlabel('Frame ID', fontsize=12)
    ax.set_ylabel('Alpha', fontsize=12)
    ax.set_title(f'{metric_name} Heatmap: Alpha vs Frame', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_name, rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / f'heatmap_alpha_vs_frame_{metric_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: heatmap_alpha_vs_frame_{metric_name}.png")


def plot_per_frame_lines(results, output_dir, metric_name='AP50', max_alphas_to_plot=10):
    """Plot 3: Line plots showing metric per frame for different alpha values"""
    alphas, frame_ids, data_matrix = extract_per_frame_data(results, metric_name)

    # If too many alphas, subsample for readability
    if len(alphas) > max_alphas_to_plot:
        indices = np.linspace(0, len(alphas)-1, max_alphas_to_plot, dtype=int)
        alphas_to_plot = [alphas[i] for i in indices]
        data_to_plot = data_matrix[indices, :]
    else:
        alphas_to_plot = alphas
        data_to_plot = data_matrix

    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot each alpha as a line
    for i, alpha in enumerate(alphas_to_plot):
        # Remove NaN values for plotting
        valid_mask = ~np.isnan(data_to_plot[i, :])
        valid_frames = np.array(frame_ids)[valid_mask]
        valid_values = data_to_plot[i, :][valid_mask]

        ax.plot(valid_frames, valid_values, 'o-', label=f'α={alpha:.2f}', alpha=0.7, markersize=3)

    ax.set_xlabel('Frame ID', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} per Frame for Different Alpha Values', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'per_frame_lines_{metric_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: per_frame_lines_{metric_name}.png")


def print_summary_table(results):
    """Print a summary table of results"""
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Alpha':<8} {'AP':<8} {'AP50':<8} {'AP75':<8} {'AR@100':<8}")
    print("-"*80)

    for result in results:
        tm = result['total_metrics']
        print(f"{result['alpha']:<8.2f} {tm['AP']:<8.4f} {tm['AP50']:<8.4f} {tm['AP75']:<8.4f} {tm['AR_max100']:<8.4f}")

    print("="*80)

    # Find best alpha for each metric
    best_ap = max(results, key=lambda x: x['total_metrics']['AP'])
    best_ap50 = max(results, key=lambda x: x['total_metrics']['AP50'])

    print(f"\nBest AP:    α = {best_ap['alpha']:.2f} (AP = {best_ap['total_metrics']['AP']:.4f})")
    print(f"Best AP50:  α = {best_ap50['alpha']:.2f} (AP50 = {best_ap50['total_metrics']['AP50']:.4f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Plot alpha parameter study results")
    parser.add_argument("--input", nargs='+', required=True, help="Input JSON file(s) with results")
    parser.add_argument("--output_dir", type=str, default="alpha_plots", help="Output directory for plots")
    parser.add_argument("--metrics", nargs='+', default=['AP50', 'AP'], help="Metrics to plot in heatmaps and line plots")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load results
    print(f"Loading results from {len(args.input)} file(s)...")
    results = load_results(args.input)
    print(f"✓ Loaded {len(results)} experiments with alpha values from {results[0]['alpha']:.2f} to {results[-1]['alpha']:.2f}")

    # Print summary table
    print_summary_table(results)

    # Generate plots
    print("\nGenerating plots...")

    # Plot 1: Alpha vs Total Metrics
    plot_alpha_vs_total_metrics(results, output_dir)

    # Plots 2 & 3: For each specified metric
    for metric in args.metrics:
        plot_heatmap(results, output_dir, metric_name=metric)
        plot_per_frame_lines(results, output_dir, metric_name=metric)

    print(f"\n✓ All plots saved to {output_dir}/")


if __name__ == '__main__':
    main()
