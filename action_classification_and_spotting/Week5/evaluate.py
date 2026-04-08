#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from tabulate import tabulate

from util.io import load_json
from util.dataset import load_classes
from dataset.frame import ActionSpotDataset
from model.model_classification import Model


EXCLUDE_CLASSES = {'FREE KICK', 'GOAL'}
N_GRID_FRAMES   = 8
N_CLIPS         = 2
BORDER_PX       = 4
LAST_FRAME_MS   = 2000


# ---- setup ----------------------------------------------------------------

class _Cfg(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


CHECKPOINTS = ['best_loss', 'best_map12', 'best_map10']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',       type=str, required=True)
    parser.add_argument('--checkpoint',  type=str, default='all',
                        choices=['all', 'best_loss', 'best_map12', 'best_map10'])
    parser.add_argument('--qualitative', action='store_true')
    parser.add_argument('--output-dir',  type=str, default='qualitative')
    return parser.parse_args()


def build_cfg(config, model_name):
    return _Cfg(
        frame_dir         = config['frame_dir'],
        save_dir          = config['save_dir'] + '/' + model_name,
        store_dir         = config['save_dir'] + '/splits',
        labels_dir        = config['labels_dir'],
        store_mode        = 'load',
        task              = config['task'],
        clip_len          = config['clip_len'],
        stride            = config.get('stride', 2),
        dataset           = config['dataset'],
        feature_arch      = config['feature_arch'],
        neck_architecture = config.get('neck_architecture', 'max_pool'),
        neck_parameters   = config.get('neck_parameters', {}),
        freeze_backbone   = config.get('freeze_backbone', False),
        num_classes       = config['num_classes'],
        device            = 'cuda' if torch.cuda.is_available() else 'cpu',
        num_workers       = config['num_workers'],
    )


# ---- inference ------------------------------------------------------------

class _IndexedDataset(Dataset):
    """Wraps ActionSpotDataset with deterministic index-based access."""

    def __init__(self, ds):
        self._ds = ds

    def __len__(self):
        return self._ds._total_len

    def __getitem__(self, idx):
        ds = self._ds
        frames = ds._frame_reader.load_frames(
            ds._frame_paths[idx], pad=True, stride=ds._stride)
        labels = np.zeros(len(ds._class_dict), np.int64)
        for lbl in ds._labels_store[idx]:
            labels[lbl['label'] - 1] = 1
        return frames, torch.tensor(labels, dtype=torch.int64), idx


def collect_scores(model, dataset, num_workers):
    n        = dataset._total_len
    n_cls    = len(dataset._class_dict)
    scores   = np.zeros((n, n_cls), np.float32)
    labels   = np.zeros((n, n_cls), np.int64)

    loader = DataLoader(
        _IndexedDataset(dataset), batch_size=8, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    for frames, lbl, idxs in tqdm(loader, desc='Inference'):
        preds = model.predict(frames)
        for j, i in enumerate(idxs.tolist()):
            scores[i] = preds[j]
            labels[i] = lbl[j].numpy()

    return scores, labels


# ---- quantitative ---------------------------------------------------------

def print_results(scores, labels, class_names, ckpt_name=None):
    ap   = average_precision_score(labels, scores, average=None)
    mask = np.array([n not in EXCLUDE_CLASSES for n in class_names])

    if ckpt_name:
        print(f'\n--- {ckpt_name} ---')
    print(tabulate(
        [[name, f'{ap[i]*100:.2f}'] for i, name in enumerate(class_names)],
        headers=['Class', 'AP'], tablefmt='grid',
    ))
    print(tabulate([
        ['mAP@12 (all classes)',            f'{np.mean(ap)*100:.2f}'],
        ['mAP@10 (excl. FREE KICK & GOAL)', f'{np.mean(ap[mask])*100:.2f}'],
    ], tablefmt='grid'))
    return ap


# ---- qualitative ----------------------------------------------------------

def _load_frames(dataset, idx):
    return dataset._frame_reader.load_frames(
        dataset._frame_paths[idx], pad=True, stride=dataset._stride)


def _try_font(size=11):
    for path in [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def find_clips(scores, labels, class_idx, n):
    s   = scores[:, class_idx]
    l   = labels[:, class_idx]
    pos = np.where(l == 1)[0]
    neg = np.where(l == 0)[0]
    tp  = pos[np.argsort(s[pos])[::-1]][:n]
    fp  = neg[np.argsort(s[neg])[::-1]][:n]
    fn  = pos[np.argsort(s[pos])][:n]
    return tp, fp, fn


def save_frame_grid(frames, gt_names, pred_scores, class_names, out_path):
    idxs    = np.linspace(0, frames.shape[0] - 1, N_GRID_FRAMES, dtype=int)
    sampled = frames[idxs].permute(0, 2, 3, 1).numpy().clip(0, 255).astype(np.uint8)
    top3    = sorted(enumerate(pred_scores), key=lambda x: x[1], reverse=True)[:3]

    gt_str   = 'GT: '   + (', '.join(gt_names) if gt_names else 'none')
    pred_str = 'Pred: ' + '   '.join(f'{class_names[j]} {s:.2f}' for j, s in top3)

    fig, axes = plt.subplots(1, N_GRID_FRAMES, figsize=(N_GRID_FRAMES * 3, 3.5))
    for ax, frame in zip(axes, sampled):
        ax.imshow(frame)
        ax.axis('off')
    fig.suptitle(f'{gt_str}\n{pred_str}', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def _overlay_frame(frame_np, gt_names, pred_scores, class_names, border_color, font):
    img  = Image.fromarray(frame_np)
    W, H = img.size
    draw = ImageDraw.Draw(img)

    for t in range(BORDER_PX):
        draw.rectangle([t, t, W - 1 - t, H - 1 - t], outline=border_color)

    top3     = sorted(enumerate(pred_scores), key=lambda x: x[1], reverse=True)[:3]
    gt_str   = 'GT: ' + (', '.join(gt_names) if gt_names else 'none')
    pred_str = '  '.join(f'{class_names[j]}:{s:.2f}' for j, s in top3)

    bar_h   = 36
    overlay = Image.new('RGBA', (W, bar_h), (0, 0, 0, 160))
    img     = img.convert('RGBA')
    img.paste(overlay, (0, H - bar_h), overlay)
    img  = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    draw.text((5, H - bar_h + 2),  gt_str,   fill='white',  font=font)
    draw.text((5, H - bar_h + 18), pred_str, fill='yellow', font=font)

    return img


def save_gif(frames, gt_names, pred_scores, class_names, border_color, fps, out_path):
    font      = _try_font()
    frame_ms  = max(int(1000 / fps), 20)
    frames_np = frames.permute(0, 2, 3, 1).numpy().clip(0, 255).astype(np.uint8)

    pil_frames = [
        _overlay_frame(f, gt_names, pred_scores, class_names, border_color, font)
        for f in frames_np
    ]
    durations         = [frame_ms] * len(pil_frames)
    durations[-1]     = LAST_FRAME_MS

    pil_frames[0].save(
        out_path, save_all=True, append_images=pil_frames[1:],
        loop=0, duration=durations, optimize=False,
    )


def plot_score_distributions(scores, labels, class_names, out_dir):
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for i, (name, ax) in enumerate(zip(class_names, axes.flat)):
        ax.hist(scores[labels[:, i] == 0, i], bins=30, alpha=0.6, label='Neg', color='tomato')
        ax.hist(scores[labels[:, i] == 1, i], bins=30, alpha=0.6, label='Pos', color='seagreen')
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'score_distributions.png'), dpi=150)
    plt.close()


def generate_qualitative(scores, labels, dataset, class_names, out_dir):
    fps       = 25 / dataset._stride
    grid_dir  = os.path.join(out_dir, 'grids')
    gif_dir   = os.path.join(out_dir, 'gifs')
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(gif_dir,  exist_ok=True)

    categories = [
        ('high_conf_pos', (0,   180, 0  )),
        ('high_conf_neg', (200, 0,   0  )),
        ('low_conf_pos',  (200, 130, 0  )),
    ]

    plot_score_distributions(scores, labels, class_names, out_dir)

    for class_idx, class_name in enumerate(tqdm(class_names, desc='Qualitative')):
        slug                    = class_name.replace(' ', '_')
        tp_idxs, fp_idxs, fn_idxs = find_clips(scores, labels, class_idx, N_CLIPS)

        for (cat, color), clip_idxs in zip(categories, [tp_idxs, fp_idxs, fn_idxs]):
            for i, ds_idx in enumerate(clip_idxs):
                frames   = _load_frames(dataset, ds_idx)
                gt_names = [class_names[j] for j in range(len(class_names))
                            if labels[ds_idx, j] == 1]

                save_frame_grid(
                    frames, gt_names, scores[ds_idx], class_names,
                    os.path.join(grid_dir, f'{slug}_{cat}_{i}.png'),
                )
                save_gif(
                    frames, gt_names, scores[ds_idx], class_names,
                    color, fps,
                    os.path.join(gif_dir, f'{slug}_{cat}_{i}.gif'),
                )


# ---- main -----------------------------------------------------------------

def main():
    args   = get_args()
    config = load_json(f'config/{args.model}.json')
    cfg    = build_cfg(config, args.model)

    classes     = load_classes(os.path.join('data', cfg.dataset, 'class.txt'))
    class_names = list(classes.keys())

    test_data = ActionSpotDataset(
        classes,
        os.path.join('data', cfg.dataset, 'test.json'),
        cfg.frame_dir, cfg.store_dir, cfg.store_mode,
        cfg.clip_len, None, pad_len=0,
        stride=cfg.stride, overlap=0,
        dataset=cfg.dataset, labels_dir=cfg.labels_dir, task=cfg.task,
    )

    ckpt_names = CHECKPOINTS if args.checkpoint == 'all' else [args.checkpoint]
    model      = Model(args=cfg)

    scores_cache = {}

    for ckpt_name in ckpt_names:
        ckpt_path = os.path.join(cfg.save_dir, 'checkpoints', f'checkpoint_{ckpt_name}.pt')
        if not os.path.exists(ckpt_path):
            print(f'Checkpoint {ckpt_name} not found, skipping.')
            continue

        model.load(torch.load(ckpt_path, map_location=cfg.device))
        print(f'Loaded: {ckpt_path}')

        scores, labels = collect_scores(model, test_data, cfg.num_workers)
        print_results(scores, labels, class_names, ckpt_name=ckpt_name)
        scores_cache[ckpt_name] = scores

    if args.qualitative:
        # Use the requested checkpoint for qualitative, or the last one evaluated
        qual_ckpt   = args.checkpoint if args.checkpoint != 'all' else ckpt_names[-1]
        qual_scores = scores_cache.get(qual_ckpt)
        if qual_scores is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            generate_qualitative(qual_scores, labels, test_data, class_names, args.output_dir)
            print(f'Saved qualitative output ({qual_ckpt}) to {args.output_dir}/')


if __name__ == '__main__':
    main()
