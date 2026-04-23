#!/usr/bin/env python3
"""
File containing the main training script.
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
from util.eval_spotting import evaluate
from dataset.datasets import get_datasets
from model.model_spotting import Model


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
    args.map_eval_freq = config.get('map_eval_freq', 2)
    args.neck_architecture = config.get('neck_architecture', 'identity')
    args.neck_parameters = config.get('neck_parameters', {})
    args.focal_gamma = config.get('focal_gamma', 0.0)
    args.focal_alpha = config.get('focal_alpha', 5.0)
    args.label_mode = config.get('label_mode', 'onehot')
    args.label_sigma = config.get('label_sigma', 1.0)
    args.early_stop_patience = config.get('early_stop_patience', None)

    # Knowledge distillation
    args.kd_alpha = config.get('kd_alpha', 0.0)
    args.kd_temperature = config.get('kd_temperature', 4.0)
    args.teacher_checkpoint = config.get('teacher_checkpoint', None)
    args.teacher_feature_arch = config.get('teacher_feature_arch', 'rny002_gsf')
    args.teacher_clip_len = config.get('teacher_clip_len', 100)

    # Mixup
    args.mixup = config.get('mixup', False)

    # Resume from checkpoint
    args.resume_checkpoint = config.get('resume_checkpoint', None)
    args.resume_epoch = config.get('resume_epoch', 0)
    args.resume_best_loss = config.get('resume_best_loss', float('inf'))
    args.resume_best_map12 = config.get('resume_best_map12', -float('inf'))
    args.resume_best_map10 = config.get('resume_best_map10', -float('inf'))

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
        project='action-spotting-final',
        name=args.model,
        config={**config, 'seed': args.seed},
        mode='disabled' if args.dry_run else 'online',
        entity='just-an-arbitrary-team-name'
    )

    # Directory for storing / reading model checkpoints
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Get datasets train, validation (and validation for map -> Video dataset)
    classes, train_data, val_data, val_video_data, test_data = get_datasets(args)

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

    # Resume from a previous checkpoint
    if args.resume_checkpoint:
        print(f'Resuming from {args.resume_checkpoint} (starting at epoch {args.resume_epoch})')
        model.load(torch.load(args.resume_checkpoint, map_location=args.device))

    # Load teacher for knowledge distillation
    if args.teacher_checkpoint and args.kd_alpha > 0:
        from model.teacher import load_teacher
        teacher = load_teacher(
            checkpoint_path=args.teacher_checkpoint,
            num_classes=args.num_classes,
            clip_len=args.teacher_clip_len,
            feature_arch=args.teacher_feature_arch,
            device=args.device,
        )
        model.set_teacher(teacher)

    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

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
        best_loss = args.resume_best_loss
        best_map12_val = args.resume_best_map12
        best_map10_val = args.resume_best_map10
        best_map12_05_val = -float('inf')
        best_map10_05_val = -float('inf')
        best_epoch_loss = args.resume_epoch
        best_epoch_map12 = args.resume_epoch
        best_epoch_map10 = args.resume_epoch
        last_improved = args.resume_epoch
        train_start_time = time.time()

        # Fast-forward LR scheduler to match the resume epoch so the cosine
        # annealing curve continues from where it was interrupted.
        if args.resume_epoch > 0:
            steps_to_skip = args.resume_epoch * num_steps_per_epoch
            print(f'Fast-forwarding LR scheduler by {steps_to_skip} steps '
                  f'({args.resume_epoch} epochs × {num_steps_per_epoch} steps/epoch)')
            for _ in range(steps_to_skip):
                lr_scheduler.step()

        print('START TRAINING EPOCHS')
        for epoch in range(args.resume_epoch, num_epochs):

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
                last_improved = epoch

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
                'kd_alpha': getattr(args, 'kd_alpha', 0.0),
            }

            # Evaluate mAP on validation set every map_eval_freq epochs
            if epoch % args.map_eval_freq == 0:
                val_results = evaluate(model, val_video_data)
                # Primary metric for checkpoint selection: delta = 1s
                val_map12 = float(val_results[1.0]['mAP'] * 100)
                val_map10 = float(np.mean(val_results[1.0]['AP_per_class'][mask_10]) * 100)
                val_map12_05 = float(val_results[0.5]['mAP'] * 100)
                val_map10_05 = float(np.mean(val_results[0.5]['AP_per_class'][mask_10]) * 100)

                better_map12    = val_map12    > best_map12_val
                better_map10    = val_map10    > best_map10_val
                better_map12_05 = val_map12_05 > best_map12_05_val
                better_map10_05 = val_map10_05 > best_map10_05_val

                if better_map12:
                    best_map12_val = val_map12
                    best_epoch_map12 = epoch
                if better_map10:
                    best_map10_val = val_map10
                    best_epoch_map10 = epoch
                if better_map12_05:
                    best_map12_05_val = val_map12_05
                if better_map10_05:
                    best_map10_05_val = val_map10_05

                if better_map12 or better_map10 or better_map12_05 or better_map10_05:
                    last_improved = epoch

                print('  Val mAP12@1: {:0.2f} Val mAP10@1: {:0.2f} | '
                      'mAP12@0.5: {:0.2f} mAP10@0.5: {:0.2f}{}{}'.format(
                    val_map12, val_map10, val_map12_05, val_map10_05,
                    ' | New best mAP12!' if better_map12 else '',
                    ' | New best mAP10!' if better_map10 else '',
                ))

                wandb_log['val_map12'] = val_map12
                wandb_log['val_map10'] = val_map10
                wandb_log['val_map12@0.5'] = val_map12_05
                wandb_log['val_map10@0.5'] = val_map10_05

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

            if (args.early_stop_patience is not None
                    and epoch - last_improved >= args.early_stop_patience):
                print(f'Early stopping at epoch {epoch}: no improvement in '
                      f'{args.early_stop_patience} epochs (last improved: {last_improved}).')
                break

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

        test_results = evaluate(model, test_data, nms_window=5)

        ap_score_1 = test_results[1.0]['AP_per_class']
        map12_1 = float(test_results[1.0]['mAP'] * 100)
        map10_1 = float(np.mean(ap_score_1[mask_10]) * 100)

        ap_score_05 = test_results[0.5]['AP_per_class']
        map12_05 = float(test_results[0.5]['mAP'] * 100)
        map10_05 = float(np.mean(ap_score_05[mask_10]) * 100)

        # Per-class AP at delta = 1s
        table = [[name, f"{ap_score_1[i]*100:.2f}", f"{ap_score_05[i]*100:.2f}"]
                 for i, name in enumerate(class_names)]
        print(tabulate(table, ["Class", "AP @1s", "AP @0.5s"], tablefmt="grid"))

        avg_table = [
            ["mAP@12 (all) @1s",                  f"{map12_1:.2f}"],
            ["mAP@10 (excl. FREE KICK & GOAL) @1s", f"{map10_1:.2f}"],
            ["mAP@12 (all) @0.5s",                f"{map12_05:.2f}"],
            ["mAP@10 (excl. FREE KICK & GOAL) @0.5s", f"{map10_05:.2f}"],
        ]
        print(tabulate(avg_table, ["", "Average Precision"], tablefmt="grid"))

        all_results[ckpt_name] = {
            'mAP12': map12_1,
            'mAP10': map10_1,
            'mAP12@0.5': map12_05,
            'mAP10@0.5': map10_05,
            'per_class_AP': {name: float(ap_score_1[i] * 100) for i, name in enumerate(class_names)},
            'per_class_AP@0.5': {name: float(ap_score_05[i] * 100) for i, name in enumerate(class_names)},
            'best_epoch': best_epochs[ckpt_name],
        }

        wandb_final[f'{ckpt_name}/mAP12'] = map12_1
        wandb_final[f'{ckpt_name}/mAP10'] = map10_1
        wandb_final[f'{ckpt_name}/mAP12@0.5'] = map12_05
        wandb_final[f'{ckpt_name}/mAP10@0.5'] = map10_05
        wandb_final[f'{ckpt_name}/best_epoch'] = best_epochs[ckpt_name]
        for i, name in enumerate(class_names):
            wandb_final[f'{ckpt_name}/AP/{name}'] = float(ap_score_1[i] * 100)
            wandb_final[f'{ckpt_name}/AP@0.5/{name}'] = float(ap_score_05[i] * 100)

    if not args.dry_run:
        store_json(os.path.join(args.save_dir, 'results.json'), all_results, pretty=True)

    wandb.log(wandb_final)
    wandb.finish()

    print('CORRECTLY FINISHED TRAINING AND INFERENCE')


if __name__ == '__main__':
    main(get_args())
