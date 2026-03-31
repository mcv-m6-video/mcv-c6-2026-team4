"""
Temporal neck modules that aggregate frame-level features (B, T, D) -> (B, D).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Max pool (baseline)
# ---------------------------------------------------------------------------

class MaxPoolNeck(nn.Module):

    def __init__(self, feat_dim):
        super().__init__()
        self._out_dim = feat_dim

    def forward(self, x):           # (B, T, D)
        return torch.max(x, dim=1)[0]   # (B, D)

    @property
    def out_dim(self):
        return self._out_dim


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------

class GRUNeck(nn.Module):
    """
    Bidirectional GRU over the temporal dimension.
    Output: mean-pooled hidden states -> (B, hidden_dim * directions).

    Parameters (neck_parameters):
        hidden_dim   int   internal and output width (default: feat_dim)
        num_layers   int   stacked GRU layers (default: 1)
        bidirectional bool (default: True)
        dropout      float dropout between layers; ignored when num_layers=1
                          (default: 0.0)
    """

    def __init__(self, feat_dim, hidden_dim=None, num_layers=1,
                 bidirectional=True, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or feat_dim
        # PyTorch warns (and ignores) dropout when num_layers == 1
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self._gru = nn.GRU(
            feat_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self._out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):           # (B, T, D)
        out, _ = self._gru(x)       # (B, T, out_dim)
        return out.mean(dim=1)      # (B, out_dim)

    @property
    def out_dim(self):
        return self._out_dim


# ---------------------------------------------------------------------------
# TCN
# ---------------------------------------------------------------------------

class _TCNBlock(nn.Module):
    """Single dilated residual conv block. Keeps (B, C, T) shape."""

    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        # Symmetric (non-causal) padding for "same" output length
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
    dilation (1, 2, 4, ...) so the receptive field covers the full clip.

    With kernel_size=3 and num_layers=5 the receptive field is
    sum_i(1 + 2*(3-1)*2^i, i=0..4) = 63 frames — enough for T=50.

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
        x = x.permute(0, 2, 1)         # (B, D, T) — Conv1d expects channels first
        for block in self._blocks:
            x = block(x)
        return x.mean(dim=-1)           # (B, D)

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
    Transformer encoder over the temporal dimension (pre-norm variant).
    Output: mean-pooled encoder outputs -> (B, D).

    Parameters (neck_parameters):
        num_layers       int   transformer encoder layers (default: 2)
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

    def forward(self, x):               # (B, T, D)
        x = self._pos_enc(x)
        x = self._encoder(x)            # (B, T, D)
        return x.mean(dim=1)            # (B, D)

    @property
    def out_dim(self):
        return self._out_dim


# ---------------------------------------------------------------------------
# TCN UNet
# ---------------------------------------------------------------------------

class TCNUNetNeck(nn.Module):
    """
    UNet-style neck built from TCN blocks.

    Encoder:   TCN blocks at each level, strided Conv1d halves T and doubles C.
    Bottleneck: TCN blocks at lowest resolution.
    Decoder:   Bilinear upsample → concat skip → 1×1 projection → TCN blocks.
               Projection merges 3×C channels (2×C from below + C from skip) → C.
    Pooling:   Temporal attention (learned per-frame weights) or mean pool.

    Channel widths double at each encoder level: D, 2D, 4D, ...
    Output dimension is always feat_dim (D).

    Parameters (neck_parameters):
        num_levels       int   depth of the U (default: 3 → T, T/2, T/4)
        blocks_per_level int   TCN blocks at each level/bottleneck (default: 2)
        kernel_size      int   TCN conv kernel size (default: 3)
        dropout          float per-block dropout (default: 0.1)
        pooling          str   'attention' or 'mean' (default: 'attention')
    """

    def __init__(self, feat_dim, num_levels=3, blocks_per_level=2,
                 kernel_size=3, dropout=0.1, pooling='attention'):
        super().__init__()

        # Channel width at each level: D, 2D, 4D, ...
        channels = [feat_dim * (2 ** i) for i in range(num_levels)]

        # ---- Encoder -------------------------------------------------------
        # blocks_per_level TCN blocks at each level except the bottleneck
        self._encoder_blocks = nn.ModuleList([
            nn.ModuleList([
                _TCNBlock(channels[i], kernel_size, dilation=2 ** j, dropout=dropout)
                for j in range(blocks_per_level)
            ])
            for i in range(num_levels - 1)
        ])
        # Stride-2 conv to halve T and double C between encoder levels
        self._downsamplers = nn.ModuleList([
            nn.Conv1d(channels[i], channels[i + 1], kernel_size=2, stride=2)
            for i in range(num_levels - 1)
        ])

        # ---- Bottleneck ----------------------------------------------------
        self._bottleneck = nn.ModuleList([
            _TCNBlock(channels[-1], kernel_size, dilation=2 ** j, dropout=dropout)
            for j in range(blocks_per_level)
        ])

        # ---- Decoder -------------------------------------------------------
        # After upsampling from level i+1 we have 2C channels; skip has C channels;
        # concat gives 3C → project back to C with a 1×1 conv.
        self._decoder_projections = nn.ModuleList([
            nn.Conv1d(channels[i + 1] + channels[i], channels[i], kernel_size=1)
            for i in range(num_levels - 1)
        ])
        self._decoder_blocks = nn.ModuleList([
            nn.ModuleList([
                _TCNBlock(channels[i], kernel_size, dilation=2 ** j, dropout=dropout)
                for j in range(blocks_per_level)
            ])
            for i in range(num_levels - 1)
        ])

        # ---- Pooling -------------------------------------------------------
        self._pooling = pooling
        if pooling == 'attention':
            self._attention_fc = nn.Linear(feat_dim, 1)

        self._out_dim = feat_dim

    def forward(self, x):                          # (B, T, D)
        x = x.permute(0, 2, 1)                     # (B, D, T) — Conv1d format

        # Encoder
        skips = []
        for enc_blocks, downsampler in zip(self._encoder_blocks, self._downsamplers):
            for block in enc_blocks:
                x = block(x)
            skips.append(x)
            x = downsampler(x)

        # Bottleneck
        for block in self._bottleneck:
            x = block(x)

        # Decoder (deepest level first, back to level 0)
        for i in range(len(self._encoder_blocks) - 1, -1, -1):
            skip = skips[i]
            x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
            x = torch.cat([x, skip], dim=1)        # (B, 3C, T_i)
            x = self._decoder_projections[i](x)    # (B, C, T_i)
            for block in self._decoder_blocks[i]:
                x = block(x)

        x = x.permute(0, 2, 1)                     # (B, T, D)

        # Pooling
        if self._pooling == 'attention':
            w = torch.softmax(self._attention_fc(x), dim=1)    # (B, T, 1)
            return (w * x).sum(dim=1)                           # (B, D)
        return x.mean(dim=1)                                    # (B, D)

    @property
    def out_dim(self):
        return self._out_dim


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_NECK_REGISTRY = {
    'max_pool':    MaxPoolNeck,
    'gru':         GRUNeck,
    'tcn':         TCNNeck,
    'tcn_unet':    TCNUNetNeck,
    'transformer': TransformerNeck,
}


def create_neck(arch, feat_dim, neck_params):
    """
    Instantiate a temporal neck and return (neck_module, out_dim).

    Args:
        arch        str   one of 'max_pool', 'gru', 'tcn', 'transformer'
        feat_dim    int   feature dimension coming from the backbone
        neck_params dict  architecture-specific keyword arguments
    """
    arch = arch or 'max_pool'
    neck_params = neck_params or {}

    if arch not in _NECK_REGISTRY:
        raise NotImplementedError(
            f"Unknown neck architecture '{arch}'. "
            f"Available: {list(_NECK_REGISTRY.keys())}"
        )

    neck = _NECK_REGISTRY[arch](feat_dim, **neck_params)
    return neck, neck.out_dim
