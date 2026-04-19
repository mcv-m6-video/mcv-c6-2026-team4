"""
T-DEED teacher model for knowledge distillation.

Loads a pretrained T-DEED checkpoint and produces soft targets (logits)
for the student model. The teacher is always frozen (no gradients).

Usage:
    teacher = load_teacher(checkpoint_path, args)
    # In training loop:
    with torch.no_grad():
        teacher_logits = teacher.get_logits(frames)  # (B, T_teacher, C)
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# We need the T-DEED source to build the model architecture.
# Either install it or point to the cloned repo.
_TDEED_ROOT = os.environ.get(
    'TDEED_ROOT',
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'T-DEED'),
)


def _import_tdeed():
    """Import T-DEED model classes by temporarily adding its root to sys.path."""
    if _TDEED_ROOT not in sys.path:
        sys.path.insert(0, _TDEED_ROOT)
    from model.model import TDEEDModel
    return TDEEDModel


class TeacherWrapper(nn.Module):
    """
    Wraps a T-DEED model for distillation.

    Key responsibilities:
    - Freeze all parameters
    - Handle clip_len mismatch between teacher and student
    - Expose raw logits (pre-softmax) for KD loss
    """

    def __init__(self, tdeed_model, teacher_clip_len, num_classes):
        super().__init__()
        self._model = tdeed_model._model
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False
        self._teacher_clip_len = teacher_clip_len
        self._num_classes = num_classes

    @torch.no_grad()
    def get_logits(self, frames):
        """
        Args:
            frames: (B, T_student, C, H, W) uint8 or float

        Returns:
            logits: (B, T_student, num_classes+1) raw logits (pre-softmax)

        If the student clip_len differs from teacher clip_len, we interpolate
        the teacher output temporally to match.
        """
        B, T_student, C, H, W = frames.shape
        T_teacher = self._teacher_clip_len

        if T_student == T_teacher:
            x = frames
        else:
            # Temporally resample frames to match teacher's expected clip_len
            # Use nearest-neighbor to avoid blending frames
            indices = torch.linspace(0, T_student - 1, T_teacher).long()
            x = frames[:, indices]

        pred, _ = self._model(x, inference=True)

        # T-DEED may return dict (with displacement head) or tensor
        if isinstance(pred, dict):
            logits = pred['im_feat']  # (B, T_teacher, C+1)
        elif isinstance(pred, tuple):
            logits = pred[0]
        else:
            logits = pred

        if len(logits.shape) > 3:
            logits = logits[-1]

        # Only keep the first num_classes+1 columns (drop second head if double-head)
        logits = logits[:, :, :self._num_classes + 1]

        # Interpolate back to student temporal resolution if needed
        if logits.shape[1] != T_student:
            # (B, T, C) -> (B, C, T) for interpolate -> (B, C, T') -> (B, T', C)
            logits = logits.permute(0, 2, 1)
            logits = F.interpolate(logits, size=T_student, mode='linear', align_corners=False)
            logits = logits.permute(0, 2, 1)

        return logits


def load_teacher(checkpoint_path, num_classes=12, clip_len=100,
                 feature_arch='rny002_gsf', device='cuda'):
    """
    Load a pretrained T-DEED checkpoint as a frozen teacher.

    Args:
        checkpoint_path: path to the .pt checkpoint file
        num_classes:     number of foreground classes (12 for SN-BAS)
        clip_len:        clip_len used when training T-DEED (100 by default)
        feature_arch:    backbone used in T-DEED ('rny002_gsf' or 'rny008_gsf')
        device:          'cuda' or 'cpu'

    Returns:
        TeacherWrapper instance
    """
    from argparse import Namespace

    TDEEDModel = _import_tdeed()

    # Build T-DEED args matching the checkpoint config
    teacher_args = Namespace(
        modality='rgb',
        feature_arch=feature_arch,
        temporal_arch='ed_sgp_mixer',
        clip_len=clip_len,
        num_classes=num_classes,
        n_layers=2,
        sgp_ks=9,
        sgp_r=4,
        radi_displacement=4,
        crop_dim=None,
    )

    teacher_model = TDEEDModel(device=device, args=teacher_args)
    state_dict = torch.load(checkpoint_path, map_location=device)
    teacher_model.load(state_dict)

    wrapper = TeacherWrapper(teacher_model, clip_len, num_classes)
    wrapper = wrapper.to(device)

    print(f'[Teacher] Loaded T-DEED ({feature_arch}) from {checkpoint_path}')
    print(f'[Teacher] clip_len={clip_len}, num_classes={num_classes}, '
          f'params={sum(p.numel() for p in wrapper.parameters()):,} (all frozen)')

    return wrapper
