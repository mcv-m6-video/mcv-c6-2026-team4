"""
Temporal neck modules for action spotting.

Unlike the classification necks (Week5), these are sequence-to-sequence:
input (B, T, D) -> output (B, T, D'), preserving the temporal dimension so
that the FC head can produce one prediction per frame.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Identity (baseline — no temporal modeling)
# ---------------------------------------------------------------------------

class IdentityNeck(nn.Module):
    """Pass-through: no temporal modeling, each frame is classified independently."""

    def __init__(self, feat_dim):
        super().__init__()
        self._out_dim = feat_dim

    def forward(self, x):      # (B, T, D)
        return x               # (B, T, D)

    @property
    def out_dim(self):
        return self._out_dim


# ---------------------------------------------------------------------------
# BiGRU
# ---------------------------------------------------------------------------

class GRUNeck(nn.Module):
    """
    Bidirectional GRU over the temporal dimension.
    Returns all hidden states so each frame gets context from both directions.

    Parameters (neck_parameters):
        hidden_dim    int   GRU hidden size per direction (default: feat_dim)
        num_layers    int   stacked GRU layers (default: 1)
        bidirectional bool  (default: True)
        dropout       float dropout between layers; ignored when num_layers=1
                            (default: 0.0)
    """

    def __init__(self, feat_dim, hidden_dim=None, num_layers=1,
                 bidirectional=True, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or feat_dim
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self._gru = nn.GRU(
            feat_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self._out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):          # (B, T, D)
        out, _ = self._gru(x)      # (B, T, out_dim)
        return out                 # keep all T timesteps

    @property
    def out_dim(self):
        return self._out_dim


# ---------------------------------------------------------------------------
# TCN
# ---------------------------------------------------------------------------

class _TCNBlock(nn.Module):
    """Single dilated residual conv block. Input/output shape: (B, C, T)."""

    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        # Symmetric padding -> output length == input length
        padding = (kernel_size - 1) * dilation // 2
        self._conv = nn.Conv1d(
            channels, channels, kernel_size,
            dilation=dilation, padding=padding,
        )
        self._norm = nn.BatchNorm1d(channels)
        self._dropout = nn.Dropout(dropout)

    def forward(self, x):   # (B, C, T)
        return x + self._dropout(F.relu(self._norm(self._conv(x))))


class TCNNeck(nn.Module):
    """
    Stack of dilated residual 1-D conv blocks with exponentially growing
    dilation (1, 2, 4, ...). Preserves sequence length -> output (B, T, D).

    With kernel_size=3 and num_layers=5 the receptive field spans
    1 + 2*(3-1)*(1+2+4+8+16) = 63 frames, enough to cover T=50.

    Parameters (neck_parameters):
        num_layers  int   number of blocks (default: 5)
        kernel_size int   conv kernel size (default: 3)
        dropout     float per-block dropout (default: 0.1)
    """

    def __init__(self, feat_dim, num_layers=5, kernel_size=3, dropout=0.1):
        super().__init__()
        self._blocks = nn.ModuleList([
            _TCNBlock(feat_dim, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_layers)
        ])
        self._out_dim = feat_dim

    def forward(self, x):               # (B, T, D)
        x = x.permute(0, 2, 1)         # (B, D, T) — Conv1d expects channels-first
        for block in self._blocks:
            x = block(x)
        return x.permute(0, 2, 1)      # (B, T, D) — keep all timesteps

    @property
    def out_dim(self):
        return self._out_dim


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self._dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):   # (B, T, D)
        x = x + self.pe[:, :x.size(1)]
        return self._dropout(x)


class TransformerNeck(nn.Module):
    """
    Transformer encoder over the temporal dimension (pre-norm).
    Returns all encoder outputs -> (B, T, D), so each frame gets full
    bidirectional context via self-attention.

    Parameters (neck_parameters):
        num_layers       int   encoder layers (default: 2)
        num_heads        int   attention heads; feat_dim must be divisible
                               (default: 4)
        dim_feedforward  int   FFN hidden width (default: feat_dim * 4)
        dropout          float attention + FFN dropout (default: 0.1)
    """

    def __init__(self, feat_dim, num_layers=2, num_heads=4,
                 dim_feedforward=None, dropout=0.1):
        super().__init__()
        dim_feedforward = dim_feedforward or feat_dim * 4
        self._pos_enc = _PositionalEncoding(feat_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # pre-norm: more stable training
        )
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._out_dim = feat_dim

    def forward(self, x):           # (B, T, D)
        x = self._pos_enc(x)
        x = self._encoder(x)        # (B, T, D)
        return x                    # keep all timesteps

    @property
    def out_dim(self):
        return self._out_dim


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_NECK_REGISTRY = {
    'identity':    IdentityNeck,
    'gru':         GRUNeck,
    'tcn':         TCNNeck,
    'transformer': TransformerNeck,
}


def create_neck(arch, feat_dim, neck_params):
    """
    Instantiate a temporal neck and return (neck_module, out_dim).

    Args:
        arch        str   one of 'identity', 'gru', 'tcn', 'transformer'
        feat_dim    int   backbone output dimension
        neck_params dict  architecture-specific keyword arguments
    """
    arch = arch or 'identity'
    neck_params = neck_params or {}

    if arch not in _NECK_REGISTRY:
        raise NotImplementedError(
            f"Unknown neck architecture '{arch}'. "
            f"Available: {list(_NECK_REGISTRY.keys())}"
        )

    neck = _NECK_REGISTRY[arch](feat_dim, **neck_params)
    return neck, neck.out_dim
