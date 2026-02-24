"""
Plot results from variance_rho parameter study

Usage:
    # Plot from a single results file
    python plot_variance_rho_results.py --input adaptive_ro_results/variance_rho_A.json --output_dir variance_rho_plots

    # Plot from multiple results files (if you ran in parallel)
    python plot_variance_rho_results.py --input adaptive_ro_results/variance_rho_*.json --output_dir variance_rho_plots
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

    # Sort by variance_rho value
    all_results.sort(key=lambda x: x['variance_rho'])

    return all_results


def extract_metric_arrays(results, metric_name):
    """Extract arrays of variance_rho values and corresponding metric values"""
    variance_rhos = [r['variance_rho'] for r in results]
    total_metrics = [r['total_metrics'][metric_name] for r in results]

    return np.array(variance_rhos), np.array(total_metrics)


def extract_per_frame_data(results, metric_name):
    """Extract per-frame data for heatmap"""
    # Get all unique frame IDs across all experiments
    all_frame_ids = set()
    for result in results:
        all_frame_ids.update(map(int, result['per_frame_metrics'].keys()))

    frame_ids = sorted(all_frame_ids)
    variance_rhos = [r['variance_rho'] for r in results]

    # Create matrix: rows = variance_rhos, cols = frames
    data_matrix = np.full((len(variance_rhos), len(frame_ids)), np.nan)

    for i, result in enumerate(results):
        for j, frame_id in enumerate(frame_ids):
            frame_key = str(frame_id)
            if frame_key in result['per_frame_metrics']:
                data_matrix[i, j] = result['per_frame_metrics'][frame_key][metric_name]

    return variance_rhos, frame_ids, data_matrix


def plot_variance_rho_vs_total_metrics(results, output_dir):
    """Plot 1: variance_rho vs Total Metrics (AP, AP50, AP75, etc.)"""
    metrics_to_plot = ['AP', 'AP50', 'AP75', 'AR_max100']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        variance_rhos, values = extract_metric_arrays(results, metric)

        axes[idx].plot(variance_rhos, values, 'o-', linewidth=2, markersize=8)
        axes[idx].set_xlabel('variance_rho', fontsize=18)
        axes[idx].set_ylabel(metric, fontsize=18)
        axes[idx].set_title(f'{metric} vs variance_rho', fontsize=22, fontweight='bold')
        axes[idx].tick_params(axis='both', which='major', labelsize=16)
        axes[idx].grid(True, alpha=0.3)

        # Mark best value
        best_idx = np.argmax(values)
        axes[idx].axvline(variance_rhos[best_idx], color='r', linestyle='--', alpha=0.5, label=f'Best: ρ={variance_rhos[best_idx]:.4f}')
        axes[idx].legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / 'variance_rho_vs_total_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: variance_rho_vs_total_metrics.png")


def plot_heatmap(results, output_dir, metric_name='AP50'):
    """Plot 2: Heatmap of variance_rho vs Frame Number with metric as color"""
    variance_rhos, frame_ids, data_matrix = extract_per_frame_data(results, metric_name)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot heatmap
    im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(0, len(frame_ids), max(1, len(frame_ids)//20)))
    ax.set_xticklabels([frame_ids[i] for i in range(0, len(frame_ids), max(1, len(frame_ids)//20))], rotation=45)
    ax.set_yticks(np.arange(len(variance_rhos)))
    ax.set_yticklabels([f'{r:.4f}' for r in variance_rhos])

    ax.set_xlabel('Frame ID', fontsize=12)
    ax.set_ylabel('variance_rho', fontsize=12)
    ax.set_title(f'{metric_name} Heatmap: variance_rho vs Frame', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_name, rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / f'heatmap_variance_rho_vs_frame_{metric_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: heatmap_variance_rho_vs_frame_{metric_name}.png")


def plot_per_frame_lines(results, output_dir, metric_name='AP50', max_rhos_to_plot=10):
    """Plot 3: Line plots showing metric per frame for different variance_rho values"""
    variance_rhos, frame_ids, data_matrix = extract_per_frame_data(results, metric_name)

    # If too many rhos, subsample for readability
    if len(variance_rhos) > max_rhos_to_plot:
        indices = np.linspace(0, len(variance_rhos)-1, max_rhos_to_plot, dtype=int)
        rhos_to_plot = [variance_rhos[i] for i in indices]
        data_to_plot = data_matrix[indices, :]
    else:
        rhos_to_plot = variance_rhos
        data_to_plot = data_matrix

    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot each rho as a line
    for i, rho in enumerate(rhos_to_plot):
        # Remove NaN values for plotting
        valid_mask = ~np.isnan(data_to_plot[i, :])
        valid_frames = np.array(frame_ids)[valid_mask]
        valid_values = data_to_plot[i, :][valid_mask]

        ax.plot(valid_frames, valid_values, 'o-', label=f'ρ={rho:.4f}', alpha=0.7, markersize=3)

    ax.set_xlabel('Frame ID', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} per Frame for Different variance_rho Values', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'per_frame_lines_{metric_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: per_frame_lines_{metric_name}.png")


def print_summary_table(results):
    """Print a summary table of results"""
    print("\n" + "="*90)
    print("SUMMARY TABLE")
    print("="*90)
    print(f"{'mean_rho':<12} {'variance_rho':<14} {'AP':<8} {'AP50':<8} {'AP75':<8} {'AR@100':<8}")
    print("-"*90)

    for result in results:
        tm = result['total_metrics']
        print(f"{result['mean_rho']:<12.4f} {result['variance_rho']:<14.4f} {tm['AP']:<8.4f} {tm['AP50']:<8.4f} {tm['AP75']:<8.4f} {tm['AR_max100']:<8.4f}")

    print("="*90)

    # Find best variance_rho for each metric
    best_ap = max(results, key=lambda x: x['total_metrics']['AP'])
    best_ap50 = max(results, key=lambda x: x['total_metrics']['AP50'])

    print(f"\nBest AP:    variance_rho = {best_ap['variance_rho']:.4f} (AP = {best_ap['total_metrics']['AP']:.4f})")
    print(f"Best AP50:  variance_rho = {best_ap50['variance_rho']:.4f} (AP50 = {best_ap50['total_metrics']['AP50']:.4f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Plot variance_rho parameter study results")
    parser.add_argument("--input", nargs='+', required=True, help="Input JSON file(s) with results (supports glob patterns)")
    parser.add_argument("--output_dir", type=str, default="variance_rho_plots", help="Output directory for plots")
    parser.add_argument("--metrics", nargs='+', default=['AP50', 'AP'], help="Metrics to plot in heatmaps and line plots")

    args = parser.parse_args()

    # Expand glob patterns in input files
    input_files = []
    for pattern in args.input:
        expanded = glob.glob(pattern)
        if expanded:
            input_files.extend(expanded)
        else:
            input_files.append(pattern)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load results
    print(f"Loading results from {len(input_files)} file(s)...")
    results = load_results(input_files)
    print(f"Loaded {len(results)} experiments with variance_rho values from {results[0]['variance_rho']:.4f} to {results[-1]['variance_rho']:.4f}")

    # Print summary table
    print_summary_table(results)

    # Generate plots
    print("\nGenerating plots...")

    # Plot 1: variance_rho vs Total Metrics
    plot_variance_rho_vs_total_metrics(results, output_dir)

    # Plots 2 & 3: For each specified metric
    for metric in args.metrics:
        plot_heatmap(results, output_dir, metric_name=metric)
        plot_per_frame_lines(results, output_dir, metric_name=metric)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == '__main__':
    main()
