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
# U-Net (1-D temporal)
# ---------------------------------------------------------------------------

class _DoubleConv1d(nn.Module):
    """Two conv + BN + ReLU, preserves length."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        self._conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self._norm1 = nn.BatchNorm1d(out_ch)
        self._conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding)
        self._norm2 = nn.BatchNorm1d(out_ch)
        self._dropout = nn.Dropout(dropout)

    def forward(self, x):                             # (B, C, T)
        x = F.relu(self._norm1(self._conv1(x)))
        x = self._dropout(x)
        x = F.relu(self._norm2(self._conv2(x)))
        return x


class UNetNeck(nn.Module):
    """
    1-D U-Net over the temporal dimension with constant channel width.

    An input 1x1 conv projects feat_dim -> hidden_dim (skipped when equal).
    Encoder/decoder run at constant hidden_dim; decoder concat doubles the
    channels momentarily (2*hidden_dim -> hidden_dim via DoubleConv).
    Output: (B, T, hidden_dim).

    Parameters (neck_parameters):
        hidden_dim   int   internal + output channel width (default: feat_dim)
        num_levels   int   encoder/decoder depth (default: 2)
        kernel_size  int   conv kernel size (default: 3)
        dropout      float per-block dropout (default: 0.0)
    """

    def __init__(self, feat_dim, hidden_dim=None, num_levels=2,
                 kernel_size=3, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or feat_dim

        self._proj_in = (nn.Conv1d(feat_dim, hidden_dim, 1)
                         if hidden_dim != feat_dim else nn.Identity())

        self._enc = nn.ModuleList([
            _DoubleConv1d(hidden_dim, hidden_dim, kernel_size, dropout)
            for _ in range(num_levels)
        ])
        self._bottleneck = _DoubleConv1d(
            hidden_dim, hidden_dim, kernel_size, dropout)
        self._dec = nn.ModuleList([
            _DoubleConv1d(hidden_dim * 2, hidden_dim, kernel_size, dropout)
            for _ in range(num_levels)
        ])

        self._out_dim = hidden_dim

    def forward(self, x):                             # (B, T, D)
        x = x.permute(0, 2, 1)                        # (B, D, T)
        x = self._proj_in(x)                           # (B, hidden_dim, T)

        skips = []
        for enc in self._enc:
            x = enc(x)
            skips.append(x)
            x = F.max_pool1d(x, 2)

        x = self._bottleneck(x)

        for dec, skip in zip(self._dec, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-1],
                              mode='linear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return x.permute(0, 2, 1)                    # (B, T, hidden_dim)

    @property
    def out_dim(self):
        return self._out_dim


# ---------------------------------------------------------------------------
# Attention-Bottleneck U-Net (1-D temporal)
# ---------------------------------------------------------------------------

class AttentionBottleneckUNetNeck(nn.Module):
    """
    1-D U-Net where the bottleneck DoubleConv is replaced with multi-head
    self-attention. At num_levels=2 the bottleneck T = T_input/4 (≈12 frames
    for clip_len=50), at num_levels=3 T = T_input/8 (≈6 frames). Attention
    is cheap at these lengths while capturing global temporal context that
    local convolutions miss.

    Parameters (neck_parameters):
        hidden_dim   int   internal + output channel width (default: feat_dim)
        num_levels   int   encoder/decoder depth (default: 2)
        kernel_size  int   conv kernel size in encoder/decoder (default: 3)
        dropout      float per-block dropout (default: 0.0)
        num_heads    int   attention heads at bottleneck (default: 4)
        attn_layers  int   transformer layers at bottleneck (default: 1)
    """

    def __init__(self, feat_dim, hidden_dim=None, num_levels=2,
                 kernel_size=3, dropout=0.0, num_heads=4, attn_layers=1):
        super().__init__()
        hidden_dim = hidden_dim or feat_dim

        self._proj_in = (nn.Conv1d(feat_dim, hidden_dim, 1)
                         if hidden_dim != feat_dim else nn.Identity())

        self._enc = nn.ModuleList([
            _DoubleConv1d(hidden_dim, hidden_dim, kernel_size, dropout)
            for _ in range(num_levels)
        ])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self._bottleneck = nn.TransformerEncoder(encoder_layer, num_layers=attn_layers)

        self._dec = nn.ModuleList([
            _DoubleConv1d(hidden_dim * 2, hidden_dim, kernel_size, dropout)
            for _ in range(num_levels)
        ])

        self._out_dim = hidden_dim

    def forward(self, x):                             # (B, T, D)
        x = x.permute(0, 2, 1)                        # (B, D, T)
        x = self._proj_in(x)                           # (B, hidden_dim, T)

        skips = []
        for enc in self._enc:
            x = enc(x)
            skips.append(x)
            x = F.max_pool1d(x, 2)

        x = x.permute(0, 2, 1)                        # (B, T_bot, hidden_dim)
        x = self._bottleneck(x)
        x = x.permute(0, 2, 1)                        # (B, hidden_dim, T_bot)

        for dec, skip in zip(self._dec, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-1],
                              mode='linear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return x.permute(0, 2, 1)                    # (B, T, hidden_dim)

    @property
    def out_dim(self):
        return self._out_dim


# ---------------------------------------------------------------------------
# TCN-block U-Net (1-D temporal)
# ---------------------------------------------------------------------------

class _TCNLevelBlock(nn.Module):
    """Stack of dilated residual TCN blocks at a single U-Net level."""

    def __init__(self, channels, kernel_size, dropout, num_dilations=3):
        super().__init__()
        self._blocks = nn.ModuleList([
            _TCNBlock(channels, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_dilations)
        ])

    def forward(self, x):   # (B, C, T)
        for block in self._blocks:
            x = block(x)
        return x


class TCNUNetNeck(nn.Module):
    """
    1-D U-Net where each encoder/decoder level uses a dilated-TCN stack
    instead of plain DoubleConv1d. With num_dilations=3 each level applies
    dilations (1, 2, 4), giving a per-level RF of 1+2*(k-1)*7 frames before
    the next max-pool halving. Combines the multi-scale hierarchy of U-Net
    with the wide per-level receptive field of TCN.

    Parameters (neck_parameters):
        hidden_dim     int   internal + output channel width (default: feat_dim)
        num_levels     int   encoder/decoder depth (default: 2)
        kernel_size    int   conv kernel size (default: 3)
        dropout        float per-block dropout (default: 0.0)
        num_dilations  int   dilated blocks per level (default: 3, → dilations 1,2,4)
    """

    def __init__(self, feat_dim, hidden_dim=None, num_levels=2,
                 kernel_size=3, dropout=0.0, num_dilations=3):
        super().__init__()
        hidden_dim = hidden_dim or feat_dim

        self._proj_in = (nn.Conv1d(feat_dim, hidden_dim, 1)
                         if hidden_dim != feat_dim else nn.Identity())

        self._enc = nn.ModuleList([
            _TCNLevelBlock(hidden_dim, kernel_size, dropout, num_dilations)
            for _ in range(num_levels)
        ])
        self._bottleneck = _TCNLevelBlock(
            hidden_dim, kernel_size, dropout, num_dilations)

        # After skip-concat channels double; a 1x1 conv projects back before TCN.
        self._dec = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
                _TCNLevelBlock(hidden_dim, kernel_size, dropout, num_dilations),
            )
            for _ in range(num_levels)
        ])

        self._out_dim = hidden_dim

    def forward(self, x):                             # (B, T, D)
        x = x.permute(0, 2, 1)                        # (B, D, T)
        x = self._proj_in(x)

        skips = []
        for enc in self._enc:
            x = enc(x)
            skips.append(x)
            x = F.max_pool1d(x, 2)

        x = self._bottleneck(x)

        for dec, skip in zip(self._dec, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-1],
                              mode='linear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return x.permute(0, 2, 1)                    # (B, T, hidden_dim)

    @property
    def out_dim(self):
        return self._out_dim


# ---------------------------------------------------------------------------
# TCN-block U-Net with additive skip connections
# ---------------------------------------------------------------------------

class TCNUNetAddNeck(nn.Module):
    """
    Identical to TCNUNetNeck except skip connections are **additive**
    (upsample + skip) instead of concatenative (cat([upsample, skip])).

    Because channels never double, no 1x1 projection conv is needed in the
    decoder. This gives exactly the same parameter count as FlatTCNNeck
    (both have num_levels*2+1 _TCNLevelBlock groups), making the two models
    a controlled pair for the temporal-downsampling ablation.

    Parameters: same as TCNUNetNeck.
    """

    def __init__(self, feat_dim, hidden_dim=None, num_levels=2,
                 kernel_size=3, dropout=0.0, num_dilations=3):
        super().__init__()
        hidden_dim = hidden_dim or feat_dim

        self._proj_in = (nn.Conv1d(feat_dim, hidden_dim, 1)
                         if hidden_dim != feat_dim else nn.Identity())

        self._enc = nn.ModuleList([
            _TCNLevelBlock(hidden_dim, kernel_size, dropout, num_dilations)
            for _ in range(num_levels)
        ])
        self._bottleneck = _TCNLevelBlock(
            hidden_dim, kernel_size, dropout, num_dilations)
        self._dec = nn.ModuleList([
            _TCNLevelBlock(hidden_dim, kernel_size, dropout, num_dilations)
            for _ in range(num_levels)
        ])

        self._out_dim = hidden_dim

    def forward(self, x):                             # (B, T, D)
        x = x.permute(0, 2, 1)                        # (B, D, T)
        x = self._proj_in(x)

        skips = []
        for enc in self._enc:
            x = enc(x)
            skips.append(x)
            x = F.max_pool1d(x, 2)

        x = self._bottleneck(x)

        for dec, skip in zip(self._dec, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-1],
                              mode='linear', align_corners=False)
            x = x + skip                              # additive — no channel doubling
            x = dec(x)

        return x.permute(0, 2, 1)                    # (B, T, hidden_dim)

    @property
    def out_dim(self):
        return self._out_dim


# ---------------------------------------------------------------------------
# Flat TCN (same blocks as TCNUNet but no temporal downsampling)
# ---------------------------------------------------------------------------

class FlatTCNNeck(nn.Module):
    """
    TCNUNetNeck with pooling and upsampling removed — the only difference.

    Keeps the full U-Net shape: encoder levels write skip connections,
    decoder levels read them via concatenation + 1x1 projection. Because
    temporal resolution never changes, skip tensors have the same size as
    the decoder input so no interpolation is needed. Parameter count is
    therefore identical to TCNUNetNeck with the same hyperparameters,
    making the two a controlled pair for the temporal-downsampling ablation.

    Parameters: same as TCNUNetNeck.
    """

    def __init__(self, feat_dim, hidden_dim=None, num_levels=2,
                 kernel_size=3, dropout=0.0, num_dilations=3):
        super().__init__()
        hidden_dim = hidden_dim or feat_dim

        self._proj_in = (nn.Conv1d(feat_dim, hidden_dim, 1)
                         if hidden_dim != feat_dim else nn.Identity())

        self._enc = nn.ModuleList([
            _TCNLevelBlock(hidden_dim, kernel_size, dropout, num_dilations)
            for _ in range(num_levels)
        ])
        self._bottleneck = _TCNLevelBlock(
            hidden_dim, kernel_size, dropout, num_dilations)
        self._dec = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
                _TCNLevelBlock(hidden_dim, kernel_size, dropout, num_dilations),
            )
            for _ in range(num_levels)
        ])

        self._out_dim = hidden_dim

    def forward(self, x):                             # (B, T, D)
        x = x.permute(0, 2, 1)                        # (B, D, T)
        x = self._proj_in(x)

        skips = []
        for enc in self._enc:
            x = enc(x)
            skips.append(x)
            # no max_pool — resolution stays at T throughout

        x = self._bottleneck(x)

        for dec, skip in zip(self._dec, reversed(skips)):
            # no interpolate — skip is already the same size as x
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return x.permute(0, 2, 1)                    # (B, T, hidden_dim)

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
    'unet':        UNetNeck,
    'unet_attn':   AttentionBottleneckUNetNeck,
    'unet_tcn':     TCNUNetNeck,
    'unet_tcn_add': TCNUNetAddNeck,
    'flat_tcn':     FlatTCNNeck,
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
