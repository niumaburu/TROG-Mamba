"""
Unified baseline model zoo for long-term time-series forecasting.

All neural models use the same forward signature as Informer/Autoformer-style
forecasting code:
    forward(x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None) -> [B, pred_len, c_out]

Supported names in get_model(args):
    Informer, Autoformer, TGCN, TimeMachine, SMamba, iTransformer,
    DLinear, PatchTST, Gateformer, PeriodNet

ARIMA is implemented as ARIMAForecaster for the special non-neural branch in
training_baselines.py.

These implementations are designed for fair, same-loader comparisons in one
project. They are lightweight/reproducible versions of the corresponding model
families, not line-by-line copies of each official repository.
"""

import math
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _getattr(args, name, default):
    return getattr(args, name, default)


def _select_output_channels(pred: torch.Tensor, c_out: int) -> torch.Tensor:
    """pred: [B, pred_len, C]. For MS setting, keep the last c_out channels."""
    if pred.shape[-1] == c_out:
        return pred
    return pred[:, :, -c_out:]


class SeriesDecomp(nn.Module):
    """Moving-average decomposition used by Autoformer/DLinear-style models."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, C]
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        trend = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1), :].to(dtype=x.dtype, device=x.device)


class DataEmbedding(nn.Module):
    """Value + positional + optional time-feature embedding."""

    def __init__(self, value_dim: int, d_model: int, time_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(value_dim, d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.time_embedding = nn.Linear(time_dim, d_model) if time_dim > 0 else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.value_embedding(x)
        h = h + self.position_embedding(h)
        if self.time_embedding is not None and x_mark is not None:
            if x_mark.shape[-1] == self.time_embedding.in_features:
                h = h + self.time_embedding(x_mark.float())
        return self.dropout(h)


def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


class ForecastHead(nn.Module):
    """Flatten historical hidden states to pred_len for every variable."""

    def __init__(self, seq_len: int, d_model: int, pred_len: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(seq_len * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, L, N, D] -> [B, pred_len, N]
        B, L, N, D = h.shape
        y = h.permute(0, 2, 1, 3).contiguous().view(B, N, L * D)
        y = self.head(y).transpose(1, 2).contiguous()
        return y


# -----------------------------------------------------------------------------
# Informer / Autoformer-style encoder-decoder baselines
# -----------------------------------------------------------------------------


class Informer(nn.Module):
    """
    Practical Informer-family baseline.

    It keeps the encoder-decoder Transformer forecasting interface. For a strict
    official reproduction, replace this class with zhouhaoyi/Informer2020's
    ProbSparse encoder/decoder blocks; the training interface remains the same.
    """

    def __init__(self, args):
        super().__init__()
        self.pred_len = int(args.pred_len)
        self.enc_in = int(args.enc_in)
        self.c_out = int(args.c_out)
        d_model = int(_getattr(args, "d_model", 64))
        n_heads = int(_getattr(args, "n_heads", 4))
        e_layers = int(_getattr(args, "e_layers", 2))
        d_layers = int(_getattr(args, "d_layers", 1))
        d_ff = int(_getattr(args, "d_ff", 4 * d_model))
        dropout = float(_getattr(args, "dropout", 0.1))
        time_dim = int(_getattr(args, "time_dim", 4))

        self.enc_embedding = DataEmbedding(self.enc_in, d_model, time_dim, dropout)
        self.dec_embedding = DataEmbedding(self.enc_in, d_model, time_dim, dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=e_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=d_layers)
        self.projection = nn.Linear(d_model, self.c_out if self.c_out < self.enc_in else self.enc_in)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_dec is None:
            x_dec = torch.zeros(x_enc.size(0), self.pred_len, x_enc.size(-1), device=x_enc.device, dtype=x_enc.dtype)
        enc_out = self.encoder(self.enc_embedding(x_enc, x_mark_enc))
        dec_in = self.dec_embedding(x_dec, x_mark_dec)
        mask = _causal_mask(dec_in.size(1), dec_in.device)
        dec_out = self.decoder(dec_in, enc_out, tgt_mask=mask)
        out = self.projection(dec_out[:, -self.pred_len :, :])
        return _select_output_channels(out, self.c_out)


class Autoformer(nn.Module):
    """
    Autoformer-family baseline with moving-average decomposition plus
    encoder-decoder attention. It follows the same interface as the official
    Autoformer long-term forecasting code.
    """

    def __init__(self, args):
        super().__init__()
        self.pred_len = int(args.pred_len)
        self.enc_in = int(args.enc_in)
        self.c_out = int(args.c_out)
        d_model = int(_getattr(args, "d_model", 64))
        n_heads = int(_getattr(args, "n_heads", 4))
        e_layers = int(_getattr(args, "e_layers", 2))
        d_layers = int(_getattr(args, "d_layers", 1))
        d_ff = int(_getattr(args, "d_ff", 4 * d_model))
        dropout = float(_getattr(args, "dropout", 0.1))
        time_dim = int(_getattr(args, "time_dim", 4))
        kernel = int(_getattr(args, "moving_avg", 25))

        self.decomp = SeriesDecomp(kernel)
        self.enc_embedding = DataEmbedding(self.enc_in, d_model, time_dim, dropout)
        self.dec_embedding = DataEmbedding(self.enc_in, d_model, time_dim, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=e_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=d_layers)
        self.seasonal_projection = nn.Linear(d_model, self.enc_in)
        self.trend_projection = nn.Linear(self.enc_in, self.enc_in)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_dec is None:
            zeros = torch.zeros(x_enc.size(0), self.pred_len, x_enc.size(-1), device=x_enc.device, dtype=x_enc.dtype)
            x_dec = torch.cat([x_enc[:, -min(x_enc.size(1), self.pred_len) :, :], zeros], dim=1)

        seasonal_enc, trend_enc = self.decomp(x_enc)
        seasonal_dec, trend_dec = self.decomp(x_dec)
        enc_out = self.encoder(self.enc_embedding(seasonal_enc, x_mark_enc))
        dec_in = self.dec_embedding(seasonal_dec, x_mark_dec)
        mask = _causal_mask(dec_in.size(1), dec_in.device)
        dec_out = self.decoder(dec_in, enc_out, tgt_mask=mask)
        seasonal_out = self.seasonal_projection(dec_out[:, -self.pred_len :, :])

        trend_base = trend_dec[:, -self.pred_len :, :]
        if trend_base.size(1) < self.pred_len:
            trend_base = trend_enc[:, -1:, :].repeat(1, self.pred_len, 1)
        trend_out = self.trend_projection(trend_base)
        out = seasonal_out + trend_out
        return _select_output_channels(out, self.c_out)


# -----------------------------------------------------------------------------
# DLinear
# -----------------------------------------------------------------------------


class DLinear(nn.Module):
    """Decomposition-Linear baseline."""

    def __init__(self, args):
        super().__init__()
        self.seq_len = int(args.seq_len)
        self.pred_len = int(args.pred_len)
        self.enc_in = int(args.enc_in)
        self.c_out = int(args.c_out)
        self.individual = bool(_getattr(args, "individual", False))
        kernel = int(_getattr(args, "moving_avg", 25))
        self.decomp = SeriesDecomp(kernel)

        if self.individual:
            self.linear_seasonal = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)])
            self.linear_trend = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)])
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

        self.reset_parameters()

    def reset_parameters(self):
        if self.individual:
            for i in range(self.enc_in):
                nn.init.constant_(self.linear_seasonal[i].weight, 1.0 / self.seq_len)
                nn.init.constant_(self.linear_trend[i].weight, 1.0 / self.seq_len)
        else:
            nn.init.constant_(self.linear_seasonal.weight, 1.0 / self.seq_len)
            nn.init.constant_(self.linear_trend.weight, 1.0 / self.seq_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        seasonal, trend = self.decomp(x_enc)
        seasonal = seasonal.permute(0, 2, 1)  # [B, C, L]
        trend = trend.permute(0, 2, 1)

        if self.individual:
            seasonal_out = torch.zeros(seasonal.size(0), self.enc_in, self.pred_len, device=x_enc.device, dtype=x_enc.dtype)
            trend_out = torch.zeros_like(seasonal_out)
            for i in range(self.enc_in):
                seasonal_out[:, i, :] = self.linear_seasonal[i](seasonal[:, i, :])
                trend_out[:, i, :] = self.linear_trend[i](trend[:, i, :])
        else:
            seasonal_out = self.linear_seasonal(seasonal)
            trend_out = self.linear_trend(trend)
        out = (seasonal_out + trend_out).permute(0, 2, 1).contiguous()
        return _select_output_channels(out, self.c_out)


# -----------------------------------------------------------------------------
# PatchTST
# -----------------------------------------------------------------------------


class PatchTST(nn.Module):
    """Patch-based channel-independent Transformer baseline."""

    def __init__(self, args):
        super().__init__()
        self.seq_len = int(args.seq_len)
        self.pred_len = int(args.pred_len)
        self.enc_in = int(args.enc_in)
        self.c_out = int(args.c_out)
        self.patch_len = int(_getattr(args, "patch_len", 16))
        self.stride = int(_getattr(args, "stride", 8))
        d_model = int(_getattr(args, "d_model", 64))
        n_heads = int(_getattr(args, "n_heads", 4))
        e_layers = int(_getattr(args, "e_layers", 2))
        d_ff = int(_getattr(args, "d_ff", 4 * d_model))
        dropout = float(_getattr(args, "dropout", 0.1))

        if self.seq_len < self.patch_len:
            raise ValueError("seq_len must be >= patch_len for PatchTST.")
        self.patch_num = (self.seq_len - self.patch_len) // self.stride + 1
        self.patch_embedding = nn.Linear(self.patch_len, d_model)
        self.position_embedding = PositionalEncoding(d_model, max_len=max(10000, self.patch_num + 1))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=e_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.patch_num * d_model, self.pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x_enc: [B, L, C] -> [B*C, patch_num, patch_len]
        B, L, C = x_enc.shape
        x = x_enc.permute(0, 2, 1).contiguous()
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().view(B * C, self.patch_num, self.patch_len)
        h = self.patch_embedding(patches)
        h = self.dropout(h + self.position_embedding(h))
        h = self.encoder(h)
        y = self.head(h.reshape(B * C, self.patch_num * h.size(-1)))
        y = y.view(B, C, self.pred_len).permute(0, 2, 1).contiguous()
        return _select_output_channels(y, self.c_out)


# -----------------------------------------------------------------------------
# iTransformer
# -----------------------------------------------------------------------------


class iTransformer(nn.Module):
    """Inverted Transformer: variables are tokens, temporal points are features."""

    def __init__(self, args):
        super().__init__()
        self.seq_len = int(args.seq_len)
        self.pred_len = int(args.pred_len)
        self.enc_in = int(args.enc_in)
        self.c_out = int(args.c_out)
        d_model = int(_getattr(args, "d_model", 64))
        n_heads = int(_getattr(args, "n_heads", 4))
        e_layers = int(_getattr(args, "e_layers", 2))
        d_ff = int(_getattr(args, "d_ff", 4 * d_model))
        dropout = float(_getattr(args, "dropout", 0.1))

        self.value_embedding = nn.Linear(self.seq_len, d_model)
        self.variable_embedding = nn.Parameter(torch.randn(1, self.enc_in, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=e_layers)
        self.head = nn.Linear(d_model, self.pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        means = x_enc.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = (x_enc - means) / stdev

        # [B, L, C] -> [B, C, L] -> [B, C, D]
        h = self.value_embedding(x.permute(0, 2, 1)) + self.variable_embedding
        h = self.encoder(h)
        y = self.head(h).permute(0, 2, 1).contiguous()
        y = y * stdev[:, 0:1, :] + means[:, 0:1, :]
        return _select_output_channels(y, self.c_out)


# -----------------------------------------------------------------------------
# T-GCN
# -----------------------------------------------------------------------------


class TGCNCell(nn.Module):
    def __init__(self, num_nodes: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.gate = nn.Linear(input_dim + hidden_dim, 2 * hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def graph_conv(self, x: torch.Tensor, adj: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        # x: [B, N, F], adj: [N, N]
        support = torch.einsum("ij,bjf->bif", adj, x)
        return layer(support)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x_t: [B, N], h: [B, N, H]
        x_t = x_t.unsqueeze(-1)
        combined = torch.cat([x_t, h], dim=-1)
        gates = torch.sigmoid(self.graph_conv(combined, adj, self.gate))
        z, r = torch.chunk(gates, 2, dim=-1)
        candidate = torch.cat([x_t, r * h], dim=-1)
        h_tilde = torch.tanh(self.graph_conv(candidate, adj, self.update))
        h = z * h + (1.0 - z) * h_tilde
        return h


class TGCN(nn.Module):
    """Temporal Graph Convolutional Network baseline."""

    def __init__(self, args):
        super().__init__()
        self.seq_len = int(args.seq_len)
        self.pred_len = int(args.pred_len)
        self.enc_in = int(args.enc_in)
        self.c_out = int(args.c_out)
        hidden_dim = int(_getattr(args, "hidden_dim", _getattr(args, "d_model", 64)))
        self.cell = TGCNCell(self.enc_in, input_dim=1, hidden_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, self.pred_len)
        self.register_buffer("static_adj", torch.eye(self.enc_in), persistent=False)

    @staticmethod
    def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        adj = torch.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
        adj = torch.clamp(adj, min=0.0)
        adj = 0.5 * (adj + adj.t())
        adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        deg = adj.sum(dim=-1).clamp(min=1e-6)
        deg_inv_sqrt = deg.pow(-0.5)
        return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)

    def set_static_adj(self, adj: torch.Tensor):
        self.static_adj = self.normalize_adj(adj.detach().float())

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        B, L, N = x_enc.shape
        adj = self.static_adj.to(x_enc.device)
        h = torch.zeros(B, N, self.cell.hidden_dim, device=x_enc.device, dtype=x_enc.dtype)
        for t in range(L):
            h = self.cell(x_enc[:, t, :], h, adj)
        y = self.head(h).transpose(1, 2).contiguous()
        return _select_output_channels(y, self.c_out)


# -----------------------------------------------------------------------------
# Lightweight Mamba-family blocks for TimeMachine / S-Mamba
# -----------------------------------------------------------------------------


class MambaLiteBlock(nn.Module):
    """
    Dependency-free Mamba-like sequence block.

    It uses input projection, depthwise causal-ish convolution, gated update and
    residual FFN. This keeps the same sequence-modeling role without requiring
    the external mamba-ssm CUDA package.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, kernel_size: int = 5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_ff)
        self.dwconv = nn.Conv1d(d_ff, d_ff, kernel_size=kernel_size, padding=kernel_size - 1, groups=d_ff)
        self.out_proj = nn.Linear(d_ff, d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        residual = x
        u, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        u = u.transpose(1, 2)
        u = self.dwconv(u)[..., : x.size(1)].transpose(1, 2)
        u = F.silu(u) * torch.sigmoid(gate)
        x = residual + self.dropout(self.out_proj(u))
        x = x + self.dropout(self.ffn(x))
        return x


class TimeMachine(nn.Module):
    """
    TimeMachine-style baseline: alternates temporal Mamba-like scanning and
    variable/channel scanning on [time, variable] representations.
    """

    def __init__(self, args):
        super().__init__()
        self.seq_len = int(args.seq_len)
        self.pred_len = int(args.pred_len)
        self.enc_in = int(args.enc_in)
        self.c_out = int(args.c_out)
        d_model = int(_getattr(args, "d_model", 64))
        d_ff = int(_getattr(args, "d_ff", 2 * d_model))
        layers = int(_getattr(args, "e_layers", 2))
        dropout = float(_getattr(args, "dropout", 0.1))
        time_dim = int(_getattr(args, "time_dim", 4))

        self.value_embedding = nn.Linear(1, d_model)
        self.variable_embedding = nn.Parameter(torch.randn(1, 1, self.enc_in, d_model) * 0.02)
        self.time_embedding = nn.Linear(time_dim, d_model) if time_dim > 0 else None
        self.temporal_blocks = nn.ModuleList([MambaLiteBlock(d_model, d_ff, dropout) for _ in range(layers)])
        self.variable_blocks = nn.ModuleList([MambaLiteBlock(d_model, d_ff, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = ForecastHead(self.seq_len, d_model, self.pred_len, dropout)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        B, L, N = x_enc.shape
        h = self.value_embedding(x_enc.unsqueeze(-1)) + self.variable_embedding
        if self.time_embedding is not None and x_mark_enc is not None and x_mark_enc.shape[-1] == self.time_embedding.in_features:
            h = h + self.time_embedding(x_mark_enc.float()).unsqueeze(2)

        for tb, vb in zip(self.temporal_blocks, self.variable_blocks):
            ht = h.permute(0, 2, 1, 3).contiguous().view(B * N, L, -1)
            ht = tb(ht).view(B, N, L, -1).permute(0, 2, 1, 3).contiguous()
            hv = ht.contiguous().view(B * L, N, -1)
            hv = vb(hv).view(B, L, N, -1)
            h = h + ht + hv
        h = self.norm(h)
        y = self.head(h)
        return _select_output_channels(y, self.c_out)


class SMamba(nn.Module):
    """S-Mamba-style baseline with bidirectional temporal scan and channel mixing."""

    def __init__(self, args):
        super().__init__()
        self.seq_len = int(args.seq_len)
        self.pred_len = int(args.pred_len)
        self.enc_in = int(args.enc_in)
        self.c_out = int(args.c_out)
        d_model = int(_getattr(args, "d_model", 64))
        d_ff = int(_getattr(args, "d_ff", 2 * d_model))
        layers = int(_getattr(args, "e_layers", 2))
        dropout = float(_getattr(args, "dropout", 0.1))
        time_dim = int(_getattr(args, "time_dim", 4))

        self.value_embedding = nn.Linear(1, d_model)
        self.variable_embedding = nn.Parameter(torch.randn(1, 1, self.enc_in, d_model) * 0.02)
        self.time_embedding = nn.Linear(time_dim, d_model) if time_dim > 0 else None
        self.forward_blocks = nn.ModuleList([MambaLiteBlock(d_model, d_ff, dropout) for _ in range(layers)])
        self.backward_blocks = nn.ModuleList([MambaLiteBlock(d_model, d_ff, dropout) for _ in range(layers)])
        self.channel_gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = ForecastHead(self.seq_len, d_model, self.pred_len, dropout)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        B, L, N = x_enc.shape
        h = self.value_embedding(x_enc.unsqueeze(-1)) + self.variable_embedding
        if self.time_embedding is not None and x_mark_enc is not None and x_mark_enc.shape[-1] == self.time_embedding.in_features:
            h = h + self.time_embedding(x_mark_enc.float()).unsqueeze(2)

        seq = h.permute(0, 2, 1, 3).contiguous().view(B * N, L, -1)
        f = seq
        b = torch.flip(seq, dims=[1])
        for fb, bb in zip(self.forward_blocks, self.backward_blocks):
            f = fb(f)
            b = bb(b)
        b = torch.flip(b, dims=[1])
        gate = self.channel_gate(torch.cat([f, b], dim=-1))
        seq = gate * f + (1.0 - gate) * b
        h = seq.view(B, N, L, -1).permute(0, 2, 1, 3).contiguous()

        # Channel/variable dependency mixing.
        ch = h.view(B * L, N, -1)
        ch_mean = ch.mean(dim=1, keepdim=True)
        ch = ch + self.channel_mixer(ch - ch_mean)
        h = ch.view(B, L, N, -1)
        h = self.norm(h)
        y = self.head(h)
        return _select_output_channels(y, self.c_out)


# -----------------------------------------------------------------------------
# Gateformer
# -----------------------------------------------------------------------------


class Gateformer(nn.Module):
    """Gateformer-style baseline with gated fusion of time and variable branches."""

    def __init__(self, args):
        super().__init__()
        self.seq_len = int(args.seq_len)
        self.pred_len = int(args.pred_len)
        self.enc_in = int(args.enc_in)
        self.c_out = int(args.c_out)
        d_model = int(_getattr(args, "d_model", 64))
        n_heads = int(_getattr(args, "n_heads", 4))
        e_layers = int(_getattr(args, "e_layers", 2))
        d_ff = int(_getattr(args, "d_ff", 4 * d_model))
        dropout = float(_getattr(args, "dropout", 0.1))
        time_dim = int(_getattr(args, "time_dim", 4))

        self.time_embedding = DataEmbedding(self.enc_in, d_model, time_dim, dropout)
        time_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.time_encoder = nn.TransformerEncoder(time_layer, num_layers=e_layers)
        self.time_head = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(self.seq_len * d_model, self.pred_len * self.enc_in))

        self.var_embedding = nn.Linear(self.seq_len, d_model)
        self.var_id = nn.Parameter(torch.randn(1, self.enc_in, d_model) * 0.02)
        var_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.var_encoder = nn.TransformerEncoder(var_layer, num_layers=e_layers)
        self.var_head = nn.Linear(d_model, self.pred_len)

        self.gate = nn.Sequential(
            nn.Linear(2 * self.enc_in, self.enc_in),
            nn.Sigmoid(),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        B, L, C = x_enc.shape
        ht = self.time_encoder(self.time_embedding(x_enc, x_mark_enc))
        yt = self.time_head(ht).view(B, self.pred_len, C)

        hv = self.var_embedding(x_enc.permute(0, 2, 1)) + self.var_id
        hv = self.var_encoder(hv)
        yv = self.var_head(hv).permute(0, 2, 1).contiguous()

        g = self.gate(torch.cat([yt, yv], dim=-1))
        y = g * yt + (1.0 - g) * yv
        return _select_output_channels(y, self.c_out)


# -----------------------------------------------------------------------------
# PeriodNet
# -----------------------------------------------------------------------------


class PeriodBlock(nn.Module):
    """Period-aware block: finds dominant FFT periods and processes period matrices."""

    def __init__(self, enc_in: int, top_k: int = 3, dropout: float = 0.1):
        super().__init__()
        self.enc_in = enc_in
        self.top_k = int(top_k)
        self.conv = nn.Sequential(
            nn.Conv2d(enc_in, enc_in, kernel_size=3, padding=1, groups=enc_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(enc_in, enc_in, kernel_size=1),
        )
        self.norm = nn.LayerNorm(enc_in)

    def _periods(self, x: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
        # x: [B, L, C]
        B, L, C = x.shape
        xf = torch.fft.rfft(x, dim=1)
        amp = xf.abs().mean(dim=(0, 2))
        amp[0] = 0.0
        k = min(self.top_k, amp.numel() - 1)
        if k <= 0:
            return [L], torch.ones(1, device=x.device, dtype=x.dtype)
        vals, idx = torch.topk(amp, k=k)
        periods = []
        for f in idx.detach().cpu().tolist():
            f = max(1, int(f))
            periods.append(max(1, int(round(L / f))))
        weights = torch.softmax(vals, dim=0).to(dtype=x.dtype)
        return periods, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        periods, weights = self._periods(x)
        outs = []
        for p in periods:
            if L % p != 0:
                pad_len = p - (L % p)
                pad = x[:, -1:, :].repeat(1, pad_len, 1)
                xp = torch.cat([x, pad], dim=1)
            else:
                xp = x
            total_len = xp.size(1)
            # [B, total_len, C] -> [B, C, blocks, period]
            xp = xp.reshape(B, total_len // p, p, C).permute(0, 3, 1, 2).contiguous()
            yp = self.conv(xp).permute(0, 2, 3, 1).reshape(B, total_len, C)
            outs.append(yp[:, :L, :])
        stacked = torch.stack(outs, dim=0)  # [K, B, L, C]
        y = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
        return self.norm(x + y)


class PeriodNet(nn.Module):
    """PeriodNet-style FFT-period modeling baseline."""

    def __init__(self, args):
        super().__init__()
        self.seq_len = int(args.seq_len)
        self.pred_len = int(args.pred_len)
        self.enc_in = int(args.enc_in)
        self.c_out = int(args.c_out)
        layers = int(_getattr(args, "e_layers", 2))
        top_k = int(_getattr(args, "period_top_k", 3))
        dropout = float(_getattr(args, "dropout", 0.1))
        self.blocks = nn.ModuleList([PeriodBlock(self.enc_in, top_k=top_k, dropout=dropout) for _ in range(layers)])
        self.decomp = SeriesDecomp(int(_getattr(args, "moving_avg", 25)))
        self.seasonal_head = nn.Linear(self.seq_len, self.pred_len)
        self.trend_head = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        seasonal, trend = self.decomp(x_enc)
        h = seasonal
        for block in self.blocks:
            h = block(h)
        y_seasonal = self.seasonal_head(h.permute(0, 2, 1))
        y_trend = self.trend_head(trend.permute(0, 2, 1))
        y = (y_seasonal + y_trend).permute(0, 2, 1).contiguous()
        return _select_output_channels(y, self.c_out)


# -----------------------------------------------------------------------------
# ARIMA wrapper for non-neural branch
# -----------------------------------------------------------------------------


class ARIMAForecaster:
    """Statsmodels ARIMA wrapper. Used by training_baselines.py, not nn.Module."""

    def __init__(self, order=(2, 0, 2), fallback="last"):
        self.order = tuple(order)
        self.fallback = fallback

    def forecast_one_series(self, history: np.ndarray, pred_len: int) -> np.ndarray:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(history.astype(np.float64), order=self.order)
            fitted = model.fit()
            pred = fitted.forecast(steps=pred_len)
            pred = np.asarray(pred, dtype=np.float32)
            if not np.all(np.isfinite(pred)):
                raise ValueError("non-finite ARIMA prediction")
            return pred
        except Exception:
            if self.fallback == "mean":
                value = float(np.mean(history))
            else:
                value = float(history[-1])
            return np.full(pred_len, value, dtype=np.float32)

    def forecast_batch(self, x: np.ndarray, pred_len: int, c_out: int) -> np.ndarray:
        # x: [B, L, C]
        B, L, C = x.shape
        start_c = C - c_out
        out = np.zeros((B, pred_len, c_out), dtype=np.float32)
        for b in range(B):
            for j, c in enumerate(range(start_c, C)):
                out[b, :, j] = self.forecast_one_series(x[b, :, c], pred_len)
        return out


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------


_MODEL_TABLE = {
    "informer": Informer,
    "autoformer": Autoformer,
    "tgcn": TGCN,
    "t-gcn": TGCN,
    "timemachine": TimeMachine,
    "time_machine": TimeMachine,
    "s-mamba": SMamba,
    "smamba": SMamba,
    "s_mamba": SMamba,
    "itransformer": iTransformer,
    "iTransformer".lower(): iTransformer,
    "dlinear": DLinear,
    "d-linear": DLinear,
    "patchtst": PatchTST,
    "patch_tst": PatchTST,
    "gateformer": Gateformer,
    "periodnet": PeriodNet,
}


def get_model(args) -> nn.Module:
    name = str(args.model).lower()
    if name not in _MODEL_TABLE:
        supported = ", ".join(sorted(_MODEL_TABLE.keys()))
        raise ValueError(f"Unsupported model={args.model}. Supported: {supported}, ARIMA")
    return _MODEL_TABLE[name](args)
