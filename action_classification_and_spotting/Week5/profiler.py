#!/usr/bin/env python3
import argparse
import torch
from thop import profile

from util.io import load_json
from model.model_classification import Model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width',  type=int, default=398)
    return parser.parse_args()


class _Cfg(dict):
    """Dict that also supports attribute access and 'in' — compatible with Model.__init__."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def build_cfg(config):
    return _Cfg(
        feature_arch      = config['feature_arch'],
        num_classes       = config['num_classes'],
        clip_len          = config['clip_len'],
        neck_architecture = config.get('neck_architecture', 'max_pool'),
        neck_parameters   = config.get('neck_parameters', {}),
        freeze_backbone   = config.get('freeze_backbone', False),
        device            = 'cpu',
    )


def main():
    args = get_args()
    config = load_json(f'config/{args.model}.json')
    cfg = build_cfg(config)

    model = Model(args=cfg)
    model._model.eval()

    dummy = torch.randn(1, cfg.clip_len, 3, args.height, args.width)
    macs, params = profile(model._model, inputs=(dummy,), verbose=False)

    print(f'\nModel:   {args.model}')
    print(f'GMACs:   {macs / 1e9:.2f}')
    print(f'Params:  {params / 1e6:.2f} M')


if __name__ == '__main__':
    main()
