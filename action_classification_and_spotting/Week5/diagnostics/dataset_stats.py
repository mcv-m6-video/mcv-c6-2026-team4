#!/usr/bin/env python3
"""
Dataset statistics and visualization for SoccerNet Ball Action Spotting.

Usage:
    python diagnostics/dataset_stats.py --model baseline_quick
"""

import argparse
import os
import sys
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from util.io import load_json

FPS = 25  # SoccerNet frames are extracted at 25 fps
SPLITS = ['train', 'val', 'test']
EXCLUDE_CLASSES = {'FREE KICK', 'GOAL'}  # for mAP10 reference


# ── helpers ──────────────────────────────────────────────────────────────────

def load_config(model_name):
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', f'{model_name}.json')
    return load_json(config_path)


def load_classes(dataset):
    class_file = os.path.join(os.path.dirname(__file__), '..', 'data', dataset, 'class.txt')
    with open(class_file) as f:
        return [l.strip() for l in f if l.strip()]


def load_split_games(dataset, split):
    split_file = os.path.join(os.path.dirname(__file__), '..', 'data', dataset, f'{split}.json')
    return load_json(split_file)


def load_annotations(labels_dir, video_name):
    label_path = os.path.join(labels_dir, video_name, 'Labels-ball.json')
    return load_json(label_path)['annotations']


def parse_game_time(game_time_str):
    """Parse 'H - MM:SS' into total seconds within the half."""
    _, time_part = game_time_str.split(' - ')
    minutes, seconds = time_part.split(':')
    return int(minutes) * 60 + int(seconds)


# ── per-split stats ───────────────────────────────────────────────────────────

def compute_split_stats(games, labels_dir, classes):
    """Returns a dict of stats for one split."""
    total_frames = sum(g['num_frames'] for g in games)
    total_duration_s = total_frames / FPS

    class_counts = defaultdict(int)
    events_per_game = []
    temporal_positions = []   # seconds from start of half
    team_counts = defaultdict(int)
    co_occurrence = np.zeros((len(classes), len(classes)), dtype=int)

    for game in games:
        annotations = load_annotations(labels_dir, game['video'])
        events_per_game.append(len(annotations))

        clip_labels_seen = defaultdict(set)  # position_bucket -> set of class indices

        for ann in annotations:
            label = ann['label']
            class_counts[label] += 1
            team_counts[ann.get('team', 'unknown')] += 1

            t_s = parse_game_time(ann['gameTime'])
            temporal_positions.append(t_s)

            # bucket by 1-second windows for co-occurrence (same second = same clip context)
            bucket = t_s
            clip_labels_seen[bucket].add(classes.index(label))

        # Build co-occurrence from clips that contain multiple classes
        for bucket_labels in clip_labels_seen.values():
            for i in bucket_labels:
                for j in bucket_labels:
                    co_occurrence[i][j] += 1

    return {
        'num_games': len(games),
        'total_frames': total_frames,
        'total_duration_s': total_duration_s,
        'class_counts': dict(class_counts),
        'events_per_game': events_per_game,
        'temporal_positions': temporal_positions,
        'team_counts': dict(team_counts),
        'co_occurrence': co_occurrence,
        'total_events': sum(class_counts.values()),
    }


# ── printing ──────────────────────────────────────────────────────────────────

def print_split_summary(split, stats, classes):
    h = stats['total_duration_s'] / 3600
    print(f"\n{'='*60}")
    print(f"  SPLIT: {split.upper()}")
    print(f"{'='*60}")

    summary = [
        ['Games', stats['num_games']],
        ['Total frames', f"{stats['total_frames']:,}"],
        ['Total duration', f"{h:.2f} h  ({stats['total_duration_s']:.0f} s)"],
        ['Total annotated events', stats['total_events']],
        ['Avg events/game', f"{np.mean(stats['events_per_game']):.1f}"],
        ['Min events/game', min(stats['events_per_game'])],
        ['Max events/game', max(stats['events_per_game'])],
    ]
    print(tabulate(summary, tablefmt='simple'))

    print(f"\n  Class distribution ({split}):")
    class_table = []
    total = stats['total_events']
    for cls in classes:
        count = stats['class_counts'].get(cls, 0)
        pct = 100 * count / total if total > 0 else 0
        marker = '  *' if cls in EXCLUDE_CLASSES else ''
        class_table.append([cls + marker, count, f"{pct:.1f}%"])
    print(tabulate(class_table, headers=['Class', 'Count', '%'], tablefmt='simple'))
    print("  (* excluded from mAP10)")

    print(f"\n  Team breakdown ({split}):")
    team_table = [[team, count] for team, count in sorted(stats['team_counts'].items())]
    print(tabulate(team_table, headers=['Team', 'Events'], tablefmt='simple'))


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_class_counts(all_stats, classes):
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(6 * len(SPLITS), 5), sharey=False)
    fig.suptitle('Class distribution per split', fontsize=14)

    for ax, split in zip(axes, SPLITS):
        stats = all_stats[split]
        counts = [stats['class_counts'].get(c, 0) for c in classes]
        colors = ['#d62728' if c in EXCLUDE_CLASSES else '#1f77b4' for c in classes]
        bars = ax.barh(classes, counts, color=colors)
        ax.set_title(split)
        ax.set_xlabel('Event count')
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                    str(cnt), va='center', fontsize=8)
        ax.invert_yaxis()

    plt.tight_layout()


def plot_temporal_distribution(all_stats):
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(6 * len(SPLITS), 4), sharey=False)
    fig.suptitle('Temporal distribution of events within a half', fontsize=14)

    for ax, split in zip(axes, SPLITS):
        positions = all_stats[split]['temporal_positions']
        ax.hist(positions, bins=30, color='#1f77b4', edgecolor='white')
        ax.set_title(split)
        ax.set_xlabel('Time in half (seconds)')
        ax.set_ylabel('Event count')

    plt.tight_layout()


def plot_events_per_game(all_stats):
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(5 * len(SPLITS), 4))
    fig.suptitle('Events per game', fontsize=14)

    for ax, split in zip(axes, SPLITS):
        epg = all_stats[split]['events_per_game']
        if len(epg) == 1:
            ax.bar([0], epg, color='#1f77b4')
            ax.set_xticks([0])
            ax.set_xticklabels([f'game 1'])
        else:
            ax.bar(range(len(epg)), epg, color='#1f77b4')
            ax.set_xticks(range(len(epg)))
            ax.set_xticklabels([f'g{i+1}' for i in range(len(epg))])
        ax.set_title(split)
        ax.set_ylabel('Event count')
        ax.axhline(np.mean(epg), color='red', linestyle='--', label=f'mean={np.mean(epg):.0f}')
        ax.legend(fontsize=8)

    plt.tight_layout()


def plot_team_breakdown(all_stats):
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(4 * len(SPLITS), 4))
    fig.suptitle('Team breakdown (left vs right)', fontsize=14)

    for ax, split in zip(axes, SPLITS):
        team_counts = all_stats[split]['team_counts']
        teams = sorted(team_counts.keys())
        counts = [team_counts[t] for t in teams]
        ax.pie(counts, labels=teams, autopct='%1.1f%%', startangle=90)
        ax.set_title(split)

    plt.tight_layout()


def plot_co_occurrence(all_stats, classes):
    for split in SPLITS:
        co = all_stats[split]['co_occurrence'].copy().astype(float)
        # Normalize by diagonal (per-class total) to get conditional probability
        diag = np.diag(co).copy()
        diag[diag == 0] = 1
        co_norm = co / diag[:, None]
        np.fill_diagonal(co_norm, 0)  # zero out self-co-occurrence

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(co_norm, cmap='YlOrRd', vmin=0, vmax=co_norm.max())
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_title(f'Class co-occurrence (row-normalized) — {split}')
        plt.colorbar(im, ax=ax, label='P(col | row)')

        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j and co_norm[i, j] > 0:
                    ax.text(j, i, f'{co_norm[i,j]:.2f}', ha='center', va='center', fontsize=6)

        plt.tight_layout()


# ── main ──────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description='Dataset statistics for SoccerNet Ball Action Spotting')
    parser.add_argument('--model', type=str, required=True, help='Model config name (without .json)')
    return parser.parse_args()


def main():
    args = get_args()
    config = load_config(args.model)
    labels_dir = config['labels_dir']
    dataset = config['dataset']

    classes = load_classes(dataset)
    print(f"Dataset: {dataset}  |  {len(classes)} classes")
    print(f"Labels dir: {labels_dir}")

    all_stats = {}
    for split in SPLITS:
        games = load_split_games(dataset, split)
        all_stats[split] = compute_split_stats(games, labels_dir, classes)
        print_split_summary(split, all_stats[split], classes)

    # ── cross-split class count table ──
    print(f"\n{'='*60}")
    print("  CLASS COUNTS ACROSS SPLITS")
    print(f"{'='*60}")
    header = ['Class'] + SPLITS + ['Total']
    rows = []
    for cls in classes:
        counts = [all_stats[s]['class_counts'].get(cls, 0) for s in SPLITS]
        rows.append([cls] + counts + [sum(counts)])
    totals = ['TOTAL'] + [all_stats[s]['total_events'] for s in SPLITS] + [sum(all_stats[s]['total_events'] for s in SPLITS)]
    rows.append(totals)
    print(tabulate(rows, headers=header, tablefmt='grid'))

    # ── plots ──
    plot_class_counts(all_stats, classes)
    plot_temporal_distribution(all_stats)
    plot_events_per_game(all_stats)
    plot_team_breakdown(all_stats)
    plot_co_occurrence(all_stats, classes)

    plt.show()


if __name__ == '__main__':
    main()
