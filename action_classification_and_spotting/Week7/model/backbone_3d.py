"""
3-D video backbone wrappers for action spotting.

Every wrapper:
  - Accepts  x : (B, C, T, H, W)  [float, already normalized/standardized]
  - Returns  out: (B, T', D)       [T' == T for all current models]
  - Exposes  out_dim property -> int

Supported architectures
-----------------------
pytorchvideo via torch.hub (internet needed on first call per model):
  x3d_s, x3d_m, x3d_l    – X3D family (Kinetics-400 pretrained)
  r3d_50                  – slow_r50 / 3-D ResNet-50 (Kinetics-400)
  slowfast_r50            – SlowFast-R50 (Kinetics-400)
  slowfast_r101           – SlowFast-R101 (Kinetics-400)

torchvision (no extra deps):
  r3d_18                  – 3-D ResNet-18 (Kinetics-400 pretrained)

Temporal dimension notes
------------------------
All X3D variants:  no temporal downsampling (all temporal strides = 1). T' = T.
R3D-18 / R3D-50:   no temporal downsampling (all temporal strides = 1). T' = T.
SlowFast:          fast pathway T' = T, slow pathway T' = ceil(T / alpha).
                   Only the fast pathway is used for per-frame output.
                   LIMITATION: the slow pathway's richer semantic features are
                   still injected into the fast pathway via lateral connections
                   during the forward pass, but the final per-frame descriptor
                   only reflects the fast pathway channel width (~256 for R50),
                   which is 1/8 of the slow pathway's 2048 channels. Using the
                   fused (slow+fast) representation would require upsampling the
                   slow features to T, which we deliberately avoid.

X3D feature channels
--------------------
We keep only the trunk (blocks 0..4), dropping the classification head.
This gives the raw ResStage output (~432 ch for X3D-M, smaller for X3D-S,
larger for X3D-L). The pretrained head projection (432->2048) is NOT included
because accessing it robustly across pytorchvideo versions would require
fragile attribute introspection. The neck that follows can adapt to any D.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared spatial pooling helper
# ─────────────────────────────────────────────────────────────────────────────

class _SpatialAvgPool(nn.Module):
    """(B, D, T, H, W) → (B, T, D) via mean over spatial dims H and W."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=(-2, -1))   # (B, D, T)
        return x.permute(0, 2, 1)  # (B, T, D)


# ─────────────────────────────────────────────────────────────────────────────
# X3D  (pytorchvideo via torch.hub)
# ─────────────────────────────────────────────────────────────────────────────

_PYTORCHVIDEO_HUB = 'facebookresearch/pytorchvideo:main'

_X3D_HUB_NAMES = {
    'x3d_s': 'x3d_s',
    'x3d_m': 'x3d_m',
    'x3d_l': 'x3d_l',
}


class X3DBackbone(nn.Module):
    """
    X3D-S / -M / -L feature extractor (pytorchvideo, Kinetics-400).

    Loads the pytorchvideo hub model and keeps blocks 0..4 (stem +
    4 ResStages), matching the strategy used by the reference group.
    The classification head (block 5) is dropped entirely.

    Temporal dimension is fully preserved – X3D uses temporal stride = 1
    in all layers, so T' = T regardless of input length.

    Requires internet access on the first call per arch to download
    weights via torch.hub.
    """

    def __init__(self, arch: str, pretrained: bool = True):
        super().__init__()
        hub_name = _X3D_HUB_NAMES[arch]
        hub_model = torch.hub.load(
            _PYTORCHVIDEO_HUB, hub_name,
            pretrained=pretrained, verbose=False,
        )
        # pytorchvideo Net: hub_model.blocks = ModuleList
        # [0] Stem  [1-4] ResStages  [5] X3DHead  — drop the head
        self._blocks = nn.ModuleList(list(hub_model.blocks)[:5])
        self._pool = _SpatialAvgPool()

        # Determine output channel count via a small dummy forward.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 4, 64, 64)
            for blk in self._blocks:
                dummy = blk(dummy)
            self._out_dim: int = dummy.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T, H, W)
        for blk in self._blocks:
            x = blk(x)                # (B, D, T, H', W')
        return self._pool(x)          # (B, T, D)

    @property
    def out_dim(self) -> int:
        return self._out_dim


# ─────────────────────────────────────────────────────────────────────────────
# R3D-18  (torchvision)
# ─────────────────────────────────────────────────────────────────────────────

class R3D18Backbone(nn.Module):
    """
    3-D ResNet-18 feature extractor (torchvision, Kinetics-400 pretrained).

    Removes avgpool and fc; temporal dimension is preserved (stride = 1
    in all temporal convolutions). feat_dim = 512.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models.video as V
        model = V.r3d_18(weights=V.R3D_18_Weights.DEFAULT if pretrained else None)

        # Collect named children up to (but not including) 'avgpool'.
        trunk_modules = []
        for name, module in model.named_children():
            if name == 'avgpool':
                break
            trunk_modules.append(module)
        self._trunk = nn.Sequential(*trunk_modules)
        self._pool = _SpatialAvgPool()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 4, 64, 64)
            self._out_dim: int = self._trunk(dummy).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._pool(self._trunk(x))  # (B, T, D)

    @property
    def out_dim(self) -> int:
        return self._out_dim


# ─────────────────────────────────────────────────────────────────────────────
# R3D-50 / slow_r50  (pytorchvideo)
# ─────────────────────────────────────────────────────────────────────────────

class R3D50Backbone(nn.Module):
    """
    3-D ResNet-50 feature extractor (pytorchvideo slow_r50, Kinetics-400).

    pytorchvideo's slow_r50 is the slow pathway of SlowFast — a standard
    3-D ResNet-50 with temporal stride = 1 throughout. feat_dim = 2048.

    In pytorchvideo's Net, each block processes a plain Tensor (not a
    list), so nn.Sequential works directly as the trunk.

    Requires internet access on the first call to download via torch.hub.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        hub_model = torch.hub.load(
            _PYTORCHVIDEO_HUB, 'slow_r50',
            pretrained=pretrained, verbose=False,
        )
        # hub_model.blocks: [Stem, S1, S2, S3, S4, ResNetBasicHead]
        # Drop the head (last block).
        self._trunk = nn.Sequential(*list(hub_model.blocks)[:-1])
        self._pool = _SpatialAvgPool()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 4, 64, 64)
            self._out_dim: int = self._trunk(dummy).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._pool(self._trunk(x))  # (B, T, D)

    @property
    def out_dim(self) -> int:
        return self._out_dim


# ─────────────────────────────────────────────────────────────────────────────
# SlowFast  (pytorchvideo)
# ─────────────────────────────────────────────────────────────────────────────

class _SlowFastTrunk(nn.Module):
    """
    Runs a sequence of pytorchvideo blocks that each map
    List[Tensor] -> List[Tensor]  (SlowFast multi-pathway design).
    """

    def __init__(self, blocks):
        super().__init__()
        self._blocks = nn.ModuleList(blocks)

    def forward(self, x_list):
        for blk in self._blocks:
            x_list = blk(x_list)
        return x_list


class SlowFastBackbone(nn.Module):
    """
    SlowFast-R50 / -R101 feature extractor (pytorchvideo, Kinetics-400).

    Two-pathway model.  The slow pathway is derived on-the-fly by
    subsampling the input clip by `alpha` (default 8, matching the
    pretrained model).

    LIMITATION — fast pathway only:
        Only the fast pathway output (index 1 in the trunk's output list)
        is used for per-frame prediction, preserving the full T temporal
        dimension.  The slow pathway (T // alpha temporal positions, ~2048
        channels) is NOT directly used in the final descriptor even though
        it enriches the fast pathway via lateral connections during the
        forward pass.  The fast pathway has ~256 channels for R50, 1/8 of
        the slow pathway width.  Fusing both pathways for per-frame output
        would require temporally upsampling the slow features to T, which
        we deliberately avoid as it is considered a hacky workaround.

    Requires internet access on the first call to download via torch.hub.
    """

    _HUB_NAMES = {
        'slowfast_r50':  'slowfast_r50',
        'slowfast_r101': 'slowfast_r101',
    }

    def __init__(self, arch: str, pretrained: bool = True, alpha: int = 8):
        super().__init__()
        self._alpha = alpha
        hub_model = torch.hub.load(
            _PYTORCHVIDEO_HUB, self._HUB_NAMES[arch],
            pretrained=pretrained, verbose=False,
        )
        # SlowFast blocks process List[Tensor] → List[Tensor].
        trunk_blocks = list(hub_model.blocks)[:-1]  # drop head
        self._trunk = _SlowFastTrunk(trunk_blocks)
        self._pool = _SpatialAvgPool()

        # Determine fast-pathway feature channels via a dummy forward.
        with torch.no_grad():
            T = 16
            dummy_fast = torch.zeros(1, 3, T, 64, 64)
            dummy_slow = torch.zeros(1, 3, max(1, T // alpha), 64, 64)
            out = self._trunk([dummy_slow, dummy_fast])
            # out[1] is the fast pathway: (1, D_fast, T, H', W')
            self._out_dim: int = out[1].shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T, H, W)
        slow = x[:, :, :: self._alpha, :, :]  # (B, C, ceil(T/alpha), H, W)
        out = self._trunk([slow, x])           # [slow_feat, fast_feat]
        # Use fast pathway: full T temporal resolution.
        return self._pool(out[1])              # (B, T, D_fast)

    @property
    def out_dim(self) -> int:
        return self._out_dim


# ─────────────────────────────────────────────────────────────────────────────
# Registry and public factory
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_3D_ARCHS: frozenset = frozenset({
    'x3d_s', 'x3d_m', 'x3d_l',
    'r3d_18',
    'r3d_50',
    'slowfast_r50', 'slowfast_r101',
})


def build_video_backbone(arch: str, pretrained: bool = True):
    """
    Instantiate a 3-D video backbone and return ``(module, feat_dim)``.

    The returned module maps  ``(B, C, T, H, W) → (B, T', D)``
    where ``T' == T`` for all current implementations.

    Args:
        arch:       one of the strings in VIDEO_3D_ARCHS
        pretrained: load Kinetics-400 pretrained weights (default True)

    Returns:
        (backbone_module, feat_dim)
    """
    if arch in ('x3d_s', 'x3d_m', 'x3d_l'):
        backbone = X3DBackbone(arch, pretrained=pretrained)
    elif arch == 'r3d_18':
        backbone = R3D18Backbone(pretrained=pretrained)
    elif arch == 'r3d_50':
        backbone = R3D50Backbone(pretrained=pretrained)
    elif arch in ('slowfast_r50', 'slowfast_r101'):
        backbone = SlowFastBackbone(arch, pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown 3-D backbone '{arch}'. "
            f"Available: {sorted(VIDEO_3D_ARCHS)}"
        )
    return backbone, backbone.out_dim
