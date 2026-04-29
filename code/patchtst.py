"""
PatchTST Model Architecture
----------------------------
Implements the PatchTST model from "A Time Series is Worth 64 Words:
Long-Term Forecasting with Transformers" (Nie et al., ICLR 2023).

Contains:
- Patching: splits univariate time series into subseries-level patches
- Transformer Encoder: maps patches to latent representations
- Prediction Head: linear layer to produce forecast output
"""

import torch
import torch.nn as nn
from config import config


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., ICLR 2022)."""
    def __init__(self, n_channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n_channels))
        self.beta  = nn.Parameter(torch.zeros(n_channels))

    def forward(self, x, mode):
        # x: (batch, seq_len, n_channels)
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)
            self.std  = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
            x = (x - self.mean) / self.std
            x = x * self.gamma + self.beta
        elif mode == 'denorm':
            x = (x - self.beta) / self.gamma
            x = x * self.std + self.mean
        return x


class _TSTEncoderLayer(nn.Module):
    """Transformer encoder layer with BatchNorm, matching the paper."""
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, D)
        attn_out, _ = self.attn(x, x, x)
        x = x + self.drop(attn_out)
        B, L, D = x.shape
        x = self.norm1(x.reshape(B * L, D)).reshape(B, L, D)
        x = x + self.drop(self.ff(x))
        x = self.norm2(x.reshape(B * L, D)).reshape(B, L, D)
        return x


class PatchTST(nn.Module):
    def __init__(self, config=config):
        super().__init__()

        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.d_ff = config.d_ff
        self.dropout = config.dropout

        # number of patches
        self.n_patches = (self.seq_len - self.patch_len) // self.stride + 1

        # number of input channels (features)
        self.n_channels = config.n_channels

        # reversible instance normalization
        self.revin = RevIN(self.n_channels)

        # patch projection: project each patch to d_model dimensions
        self.patch_projection = nn.Linear(self.patch_len, self.d_model)

        # positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(self.n_patches, self.d_model))

        # transformer encoder with BatchNorm (per paper)
        self.transformer_encoder = nn.ModuleList([
            _TSTEncoderLayer(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.n_layers)
        ])

        # prediction head: flatten patches and project to pred_len
        self.prediction_head = nn.Linear(self.n_patches * self.d_model, self.pred_len)

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, n_channels)
        batch_size = x.shape[0]

        # RevIN normalization
        x = self.revin(x, 'norm')

        # channel independence
        # (batch_size, seq_len, n_channels) -> (batch_size * n_channels, seq_len)
        x = x.permute(0, 2, 1)                                      # (batch_size, n_channels, seq_len)
        x = x.reshape(batch_size * self.n_channels, self.seq_len)   # (batch_size * n_channels, seq_len)

        # patching
        # unfold splits seq_len into overlapping patches
        # (batch_size * n_channels, seq_len) -> (batch_size * n_channels, n_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)   # (batch_size * n_channels, n_patches, patch_len)

        # patch projection + positional encoding
        # (batch_size * n_channels, n_patches, patch_len) -> (batch_size * n_channels, n_patches, d_model)
        x = self.patch_projection(x)                    # linear projection
        x = self.dropout_layer(x + self.pos_encoding)   # add positional encoding

        # transformer encoder
        # (batch_size * n_channels, n_patches, d_model) -> (batch_size * n_channels, n_patches, d_model)
        for layer in self.transformer_encoder:
            x = layer(x)

        # flatten + prediction head
        # (batch_size * n_channels, n_patches, d_model) -> (batch_size * n_channels, n_patches * d_model)
        x = x.reshape(batch_size * self.n_channels, -1) # flatten
        x = self.dropout_layer(x)
        x = self.prediction_head(x)                     # (batch_size * n_channels, pred_len)

        # reshape back
        # (batch_size * n_channels, pred_len) -> (batch_size, pred_len, n_channels)
        x = x.reshape(batch_size, self.n_channels, self.pred_len)   # (batch_size, n_channels, pred_len)
        x = x.permute(0, 2, 1)                                      # (batch_size, pred_len, n_channels)

        # RevIN denormalization
        x = self.revin(x, 'denorm')

        return x


if __name__ == "__main__":
    model = PatchTST(config)
    x = torch.randn(128, config.seq_len, config.n_channels)
    out = model(x)
    print(f"input shape:  {x.shape}")    # (128, 336, 7)
    print(f"output shape: {out.shape}")  # (128, 96, 7)