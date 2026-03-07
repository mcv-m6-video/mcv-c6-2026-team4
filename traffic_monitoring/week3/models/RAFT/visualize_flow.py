import sys
sys.path.append('core')

from PIL import Image
from glob import glob
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import math
import os.path as osp

from raft import RAFT
from utils.utils import InputPadder, forward_interpolate
import datasets
from utils import flow_viz
from utils import frame_utils

TRAIN_SIZE = [432, 960]

def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
    if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
        raise ValueError(f"Overlap should be less than patch size")
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap)) if image_shape[0] != TRAIN_SIZE[0] else [0]
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap)) if image_shape[1] != TRAIN_SIZE[1] else [0]
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]

def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]), indexing='ij')
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    h, w = h - 0.5, w - 0.5
    weights_hw = torch.exp(-0.5 * ((h**2 + w**2)**0.5 / sigma)**2) / (sigma * math.sqrt(2 * math.pi))

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (hi, wi) in enumerate(hws):
        weights[:, idx, hi:hi+patch_size[0], wi:wi+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (hi, wi) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, hi:hi+patch_size[0], wi:wi+patch_size[1]])
    return patch_weights

def compute_flow(model, image1, image2, weights=None, iters=24):
    image_size = image1.shape[1:]
    image1, image2 = image1[None].cuda(), image2[None].cuda()
    hws = compute_grid_indices(image_size)

    if weights is None:
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        _, flow_pr = model.module(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1,2,0).cpu().numpy()
    else:
        flows = 0
        flow_count = 0
        for idx, (h, w) in enumerate(hws):
            img1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            img2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            _, flow_pre = model.module(img1_tile, img2_tile, iters=iters, test_mode=True)
            padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)
        flow = (flows / flow_count)[0].permute(1,2,0).cpu().numpy()
    return flow

def compute_adaptive_image_size(image_size):
    scale = max(TRAIN_SIZE[0]/image_size[0], TRAIN_SIZE[1]/image_size[1])
    return (int(image_size[1]*scale), int(image_size[0]*scale))

def prepare_image(root_dir, viz_root_dir, fn1, fn2, keep_size):
    img1 = frame_utils.read_gen(osp.join(root_dir, fn1))
    img2 = frame_utils.read_gen(osp.join(root_dir, fn2))

    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)

    # Convert grayscale to 3 channels
    if img1.ndim == 2:
        img1 = np.stack([img1]*3, axis=-1)
    else:
        img1 = img1[..., :3]

    if img2.ndim == 2:
        img2 = np.stack([img2]*3, axis=-1)
    else:
        img2 = img2[..., :3]

    if not keep_size:
        dsize = compute_adaptive_image_size(img1.shape[:2])
        img1 = cv2.resize(img1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        img2 = cv2.resize(img2, dsize=dsize, interpolation=cv2.INTER_CUBIC)

    img1 = torch.from_numpy(img1).permute(2,0,1).float()
    img2 = torch.from_numpy(img2).permute(2,0,1).float()

    os.makedirs(viz_root_dir, exist_ok=True)
    filename = osp.splitext(osp.basename(fn1))[0]
    viz_fn = osp.join(viz_root_dir, filename+'.png')

    return img1, img2, viz_fn

def build_model(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model.cuda()
    model.eval()
    return model

def visualize_flow(root_dir, viz_root_dir, model, img_pairs, keep_size, iters=24):
    weights = None
    times = []
    for fn1, fn2 in img_pairs:
        print(f"Processing {fn1}, {fn2}...")
        image1, image2, viz_fn = prepare_image(root_dir, viz_root_dir, fn1, fn2, keep_size)

        # Measure inference time
        start_time = time.time()
        flow = compute_flow(model, image1, image2, weights, iters)
        end_time = time.time()
        inference_time = end_time - start_time
        times.append(inference_time)
        print(f"Inference time for this pair: {inference_time:.4f} s")

        flow_img = flow_viz.flow_to_image(flow)
        cv2.imwrite(viz_fn, flow_img[:, :, [2,1,0]])
        print(f"Saved flow visualization to: {osp.abspath(viz_fn)}")
    if times:
        mean_time = sum(times) / len(times)
        print(f"\nMean inference time per image pair: {mean_time:.4f} s")

def process_kitti(kitti_dir):
    img_pairs = []
    image_dir = osp.join(kitti_dir, "image_0")
    image_list = sorted(glob(osp.join(image_dir, "*_10.png")))
    for img1 in image_list:
        img2 = img1.replace("_10.png","_11.png")
        if osp.exists(img2):
            img_pairs.append((img1, img2))
    print(f"Found {len(img_pairs)} KITTI pairs")
    return img_pairs

def generate_pairs(dirname, start_idx, end_idx):
    img_pairs = []
    for idx in range(start_idx, end_idx):
        img_pairs.append((osp.join(dirname,f'{idx:06}.png'), osp.join(dirname,f'{idx+1:06}.png')))
    return img_pairs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoints/raft-kitti.pth')
    parser.add_argument('--root_dir', default='.')
    parser.add_argument('--data_dir', default='/data/113-2/users/gasbert/master/C6/KITTI/training')
    parser.add_argument('--seq_dir', default='demo_data')
    parser.add_argument('--viz_root_dir', default='viz_results')
    parser.add_argument('--start_idx', type=int, default=1)
    parser.add_argument('--end_idx', type=int, default=1200)
    parser.add_argument('--eval_type', default='kitti')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true')
    parser.add_argument('--keep_size', action='store_true')
    parser.add_argument('--iters', type=int, default=24)
    args = parser.parse_args()

    model = build_model(args)

    if args.eval_type == 'seq':
        img_pairs = generate_pairs(args.seq_dir, args.start_idx, args.end_idx)
    elif args.eval_type == 'kitti':
        img_pairs = process_kitti(args.data_dir)

    with torch.no_grad():
        visualize_flow(args.root_dir, args.viz_root_dir, model, img_pairs, args.keep_size, args.iters)