#!/usr/bin/env python3
"""
File containing the main training script for T-DEED.
"""

#Standard imports
import argparse
import torch
import os
import numpy as np
import random
import time
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
import sys
from torch.utils.data import DataLoader
from tabulate import tabulate
import wandb

#Local imports
from util.io import load_json, store_json
from util.eval_classification import evaluate
from dataset.datasets import get_datasets
from model.model_classification import Model


def get_args():
    #Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true',
                        help='Disable all logging (wandb and local files)')
    return parser.parse_args()

def update_args(args, config):
    #Update arguments with config file
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model # + '-' + str(args.seed) -> in case multiple seeds
    args.store_dir = config['save_dir'] + '/' + "splits"
    args.labels_dir = config['labels_dir']
    args.store_mode = config['store_mode']
    args.task = config['task']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.dataset = config['dataset']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch = config['feature_arch']
    args.learning_rate = config['learning_rate']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.only_test = config['only_test']
    args.device = config['device']
    args.num_workers = config['num_workers']
    args.neck_architecture = config.get('neck_architecture', 'max_pool')
    args.neck_parameters = config.get('neck_parameters', {})
    args.map_eval_freq = config.get('map_eval_freq', 2)
    args.loss = config.get('loss', 'bce')
    args.loss_parameters = config.get('loss_parameters', {})

    return args

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs)])


def main(args):
    # Set seed
    print('Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = 'config/' + args.model + '.json'
    config = load_json(config_path)
    args = update_args(args, config)

    wandb.init(
        project='action-classification',
        name=args.model,
        config={**config, 'seed': args.seed},
        mode='disabled' if args.dry_run else 'online',
    )

    # Directory for storing / reading model checkpoints
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Get datasets train, validation (and validation for map -> Video dataset)
    classes, train_data, val_data, test_data = get_datasets(args)

    class_names = list(classes.keys())
    exclude_classes = {'FREE KICK', 'GOAL'}
    mask_10 = np.array([name not in exclude_classes for name in class_names])

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Re-run changing "mode" to "load" in the config JSON.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    def worker_init_fn(id):
        random.seed(id + epoch * 100)

    # Dataloaders
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )
        
    val_loader = DataLoader(
        val_data, shuffle=True, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )

    # Model
    model = Model(args=args)

    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    # Compute pos_weight from training label frequencies for weighted BCE
    pos_weight = None
    if args.loss == 'weighted_bce':
        counts = np.zeros(args.num_classes, dtype=np.float32)
        for clip_labels in train_data._labels_store:
            for lbl in clip_labels:
                counts[lbl['label'] - 1] += 1  # labels are 1-indexed
        total = len(train_data._labels_store)
        pos_weight = (total - counts) / np.maximum(counts, 1)
        max_pos_weight = args.loss_parameters.get('max_pos_weight', 50.0)
        pos_weight = np.minimum(pos_weight, max_pos_weight)
        print('Weighted BCE pos_weight (capped at {}):'.format(max_pos_weight),
              np.round(pos_weight, 1))

    model.configure_loss(args.loss, pos_weight, args.loss_parameters)

    best_epoch_loss = None
    best_epoch_map12 = None
    best_epoch_map10 = None
    total_train_time = None

    if not args.only_test:
        # Warmup schedule
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(
            args, optimizer, num_steps_per_epoch)

        losses = []
        best_loss = float('inf')
        best_map12_val = -float('inf')
        best_map10_val = -float('inf')
        best_epoch_loss = 0
        best_epoch_map12 = 0
        best_epoch_map10 = 0
        epoch = 0
        train_start_time = time.time()

        print('START TRAINING EPOCHS')
        for epoch in range(epoch, num_epochs):

            epoch_start_time = time.time()

            if args.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            train_loss = model.epoch(
                train_loader, optimizer, scaler,
                lr_scheduler=lr_scheduler)

            val_loss = model.epoch(val_loader)

            epoch_time = time.time() - epoch_start_time
            current_lr = lr_scheduler.get_last_lr()[0]

            peak_vram_mb = (
                torch.cuda.max_memory_allocated() / 1024 ** 2
                if args.device == 'cuda' else 0.0
            )

            better_loss = val_loss < best_loss
            if better_loss:
                best_loss = val_loss
                best_epoch_loss = epoch

            #Printing info epoch
            print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f} LR: {:.2e} '
                  'VRAM: {:.0f}MB Time: {:.1f}s'.format(
                epoch, train_loss, val_loss, current_lr, peak_vram_mb, epoch_time))
            if better_loss:
                print('New best epoch (loss)!')

            wandb_log = {
                'train_loss': train_loss, 'val_loss': val_loss,
                'lr': current_lr, 'epoch_time': epoch_time,
                'peak_vram_mb': peak_vram_mb,
                'epoch': epoch,
            }

            # Evaluate mAP on validation set every map_eval_freq epochs
            if epoch % args.map_eval_freq == 0:
                val_ap = evaluate(model, val_data)
                val_map12 = float(np.mean(val_ap) * 100)
                val_map10 = float(np.mean(val_ap[mask_10]) * 100)

                better_map12 = val_map12 > best_map12_val
                better_map10 = val_map10 > best_map10_val

                if better_map12:
                    best_map12_val = val_map12
                    best_epoch_map12 = epoch
                if better_map10:
                    best_map10_val = val_map10
                    best_epoch_map10 = epoch

                print('  Val mAP12: {:0.2f} Val mAP10: {:0.2f}{}{}'.format(
                    val_map12, val_map10,
                    ' | New best mAP12!' if better_map12 else '',
                    ' | New best mAP10!' if better_map10 else '',
                ))

                wandb_log['val_map12'] = val_map12
                wandb_log['val_map10'] = val_map10

                if args.save_dir is not None and not args.dry_run:
                    if better_map12:
                        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_best_map12.pt'))
                    if better_map10:
                        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_best_map10.pt'))

            wandb.log(wandb_log)

            losses.append({
                'epoch': epoch, 'train': train_loss, 'val': val_loss,
                'lr': current_lr, 'epoch_time_s': epoch_time,
                'peak_vram_mb': peak_vram_mb,
            })

            if args.save_dir is not None and not args.dry_run:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)

                if better_loss:
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_best_loss.pt'))

        total_train_time = time.time() - train_start_time

    print('START INFERENCE')

    # Evaluate each checkpoint on the test set and report separately
    checkpoints = [
        ('best_loss',  'checkpoint_best_loss.pt'),
        ('best_map12', 'checkpoint_best_map12.pt'),
        ('best_map10', 'checkpoint_best_map10.pt'),
    ]
    best_epochs = {
        'best_loss':  best_epoch_loss,
        'best_map12': best_epoch_map12,
        'best_map10': best_epoch_map10,
    }

    all_results = {'total_train_time_s': total_train_time}
    wandb_final = {'total_train_time_s': total_train_time}

    for ckpt_name, ckpt_file in checkpoints:
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        if not os.path.exists(ckpt_path):
            print(f'Checkpoint {ckpt_file} not found, skipping.')
            continue

        print(f'\n--- Evaluating {ckpt_name} (epoch {best_epochs[ckpt_name]}) ---')
        model.load(torch.load(ckpt_path))

        ap_score = evaluate(model, test_data)

        # Report results per-class in table
        table = []
        for i, class_name in enumerate(class_names):
            table.append([class_name, f"{ap_score[i]*100:.2f}"])
        print(tabulate(table, ["Class", "Average Precision"], tablefmt="grid"))

        map12 = float(np.mean(ap_score) * 100)
        map10 = float(np.mean(ap_score[mask_10]) * 100)

        avg_table = [
            ["mAP@12 (all)", f"{map12:.2f}"],
            ["mAP@10 (excl. FREE KICK & GOAL)", f"{map10:.2f}"],
        ]
        print(tabulate(avg_table, ["", "Average Precision"], tablefmt="grid"))

        all_results[ckpt_name] = {
            'mAP12': map12,
            'mAP10': map10,
            'per_class_AP': {name: float(ap_score[i] * 100) for i, name in enumerate(class_names)},
            'best_epoch': best_epochs[ckpt_name],
        }

        wandb_final[f'{ckpt_name}/mAP12'] = map12
        wandb_final[f'{ckpt_name}/mAP10'] = map10
        wandb_final[f'{ckpt_name}/best_epoch'] = best_epochs[ckpt_name]
        for i, name in enumerate(class_names):
            wandb_final[f'{ckpt_name}/AP/{name}'] = float(ap_score[i] * 100)

    if not args.dry_run:
        store_json(os.path.join(args.save_dir, 'results.json'), all_results, pretty=True)

    wandb.log(wandb_final)
    wandb.finish()

    print('CORRECTLY FINISHED TRAINING AND INFERENCE')


if __name__ == '__main__':
    main(get_args())