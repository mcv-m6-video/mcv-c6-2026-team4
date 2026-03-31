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
    )

    # Directory for storing / reading model checkpoints
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Get datasets train, validation (and validation for map -> Video dataset)
    classes, train_data, val_data, test_data = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Re-run changing "mode" to "load" in the config JSON.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    def worker_init_fn(id):
        random.seed(id + epoch * 100)

    # Dataloaders
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )
        
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )

    # Model
    model = Model(args=args)

    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    best_epoch = None
    total_train_time = None

    if not args.only_test:
        # Warmup schedule
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(
            args, optimizer, num_steps_per_epoch)
        
        losses = []
        best_criterion = float('inf')
        best_epoch = 0
        epoch = 0
        train_start_time = time.time()

        print('START TRAINING EPOCHS')
        for epoch in range(epoch, num_epochs):

            epoch_start_time = time.time()

            train_loss = model.epoch(
                train_loader, optimizer, scaler,
                lr_scheduler=lr_scheduler)

            val_loss = model.epoch(val_loader)

            epoch_time = time.time() - epoch_start_time
            current_lr = lr_scheduler.get_last_lr()[0]

            better = False
            if val_loss < best_criterion:
                best_criterion = val_loss
                best_epoch = epoch
                better = True

            #Printing info epoch
            print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f} LR: {:.2e} Time: {:.1f}s'.format(
                epoch, train_loss, val_loss, current_lr, epoch_time))
            if better:
                print('New best epoch!')

            wandb.log({
                'train_loss': train_loss, 'val_loss': val_loss,
                'lr': current_lr, 'epoch_time': epoch_time,
                'epoch': epoch,
            })

            losses.append({
                'epoch': epoch, 'train': train_loss, 'val': val_loss,
                'lr': current_lr, 'epoch_time_s': epoch_time,
            })

            if args.save_dir is not None:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)

                if better:
                    torch.save( model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_best.pt') )

        total_train_time = time.time() - train_start_time

    print('START INFERENCE')
    model.load(torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pt')))

    # Evaluation on test split
    ap_score = evaluate(model, test_data)

    # Report results per-class in table
    table = []
    for i, class_name in enumerate(classes.keys()):
        table.append([class_name, f"{ap_score[i]*100:.2f}"])

    headers = ["Class", "Average Precision"]
    print(tabulate(table, headers, tablefmt="grid"))

    # Compute mAP12 (all classes) and mAP10 (excluding FREE KICK and GOAL)
    class_names = list(classes.keys())
    exclude_classes = {'FREE KICK', 'GOAL'}
    mask_10 = np.array([name not in exclude_classes for name in class_names])

    map12 = np.mean(ap_score) * 100
    map10 = np.mean(ap_score[mask_10]) * 100

    # Report average results in table
    avg_table = [
        ["mAP@12 (all)", f"{map12:.2f}"],
        ["mAP@10 (excl. FREE KICK & GOAL)", f"{map10:.2f}"],
    ]
    headers = ["", "Average Precision"]

    print(tabulate(avg_table, headers, tablefmt="grid"))

    results = {
        'mAP12': map12,
        'mAP10': map10,
        'per_class_AP': {name: float(ap_score[i] * 100) for i, name in enumerate(class_names)},
        'best_epoch': best_epoch,
        'total_train_time_s': total_train_time,
    }
    store_json(os.path.join(args.save_dir, 'results.json'), results, pretty=True)

    wandb.log({
        **{f'AP/{name}': ap_score[i] * 100 for i, name in enumerate(class_names)},
        'AP/mAP12': map12,
        'AP/mAP10': map10,
        'best_epoch': best_epoch,
        'total_train_time_s': total_train_time,
    })
    wandb.finish()

    print('CORRECTLY FINISHED TRAINING AND INFERENCE')


if __name__ == '__main__':
    main(get_args())