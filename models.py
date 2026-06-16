"""
Ablation-ready sparse graph residual bidirectional Mamba forecaster.

Class exported for training.py:
    GCN_mamba_TSForecast

Input:
    x_enc: [B, seq_len, N]
Output:
    pred : [B, pred_len, N] for features=M

Main modules and switches:
    --use_bimamba          false -> forward-only Mamba, ablates bidirectionality
    --use_graph_adapter    false -> removes graph residual adapter
    --use_graph_diffusion  false -> replaces graph diffusion with a self projection
    --use_sna              false -> removes explicit selective neighbor aggregation
    --use_tanh_gate        false -> replaces tanh signed gate with sigmoid gate
    --use_orth_res         false -> removes orthogonal residual
    --disable_ffn          true  -> disables FFN for large-node datasets
"""

from typing import Optional, List
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    HAS_MAMBA_SSM = True
except Exception:
    Mamba = None
    HAS_MAMBA_SSM = False

try:
    from revin import RevIN
except Exception:
    RevIN = None


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class ChunkedGatedMLP(nn.Module):
    def __init__(self, d_model: int, hidden_ratio: float = 1.0, dropout: float = 0.1, chunk_nodes: int = 64, enabled: bool = True):
        super().__init__()
        self.enabled = bool(enabled)
        self.chunk_nodes = int(chunk_nodes)
        hidden = max(d_model, int(d_model * hidden_ratio))
        self.fc1 = nn.Linear(d_model, hidden * 2)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc1(x).chunk(2, dim=-1)
        y = F.silu(a) * b
        y = self.dropout(y)
        y = self.fc2(y)
        return self.dropout(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros_like(x)
        if self.chunk_nodes <= 0 or x.size(2) <= self.chunk_nodes:
            return self._forward_chunk(x)
        outs = []
        for start in range(0, x.size(2), self.chunk_nodes):
            end = min(start + self.chunk_nodes, x.size(2))
            outs.append(self._forward_chunk(x[:, :, start:end, :]))
        return torch.cat(outs, dim=2)


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    if adj.dim() != 2 or adj.size(0) != adj.size(1):
        raise ValueError(f"Expected square adjacency [N,N], got {tuple(adj.shape)}")
    adj = torch.nan_to_num(adj.float(), nan=0.0, posinf=0.0, neginf=0.0)
    adj = torch.clamp(adj, min=0.0)
    adj = 0.5 * (adj + adj.t())
    adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    deg = adj.sum(dim=-1).clamp(min=1e-6)
    deg_inv_sqrt = deg.pow(-0.5)
    return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)


def build_topk_sparse_adj(adj: torch.Tensor, top_k: int) -> torch.Tensor:
    adj = normalize_adj(adj).float()
    N = adj.size(0)
    k = max(1, min(int(top_k) + 1, N))
    values, indices = torch.topk(adj, k=k, dim=-1)
    values = torch.clamp(values, min=0.0)
    values = values / (values.sum(dim=-1, keepdim=True) + 1e-6)
    row = torch.arange(N, device=adj.device).unsqueeze(1).expand(N, k).reshape(-1)
    col = indices.reshape(-1)
    val = values.reshape(-1).float()
    return torch.sparse_coo_tensor(
        indices=torch.stack([row, col], dim=0),
        values=val,
        size=(N, N),
        device=adj.device,
        dtype=torch.float32,
    ).coalesce()


def _ensure_float32_sparse_adj(sparse_adj: torch.Tensor, device: torch.device) -> torch.Tensor:
    sparse_adj = sparse_adj.coalesce().to(device=device)
    if sparse_adj.dtype == torch.float32:
        return sparse_adj
    return torch.sparse_coo_tensor(
        indices=sparse_adj.indices(),
        values=sparse_adj.values().float(),
        size=sparse_adj.size(),
        device=device,
        dtype=torch.float32,
    ).coalesce()


def sparse_graph_aggregate(x: torch.Tensor, sparse_adj: torch.Tensor, chunk_size: int = 32768) -> torch.Tensor:
    """Sparse aggregation for [B,L,N,D]. Computes sparse.mm in float32.

    This avoids the CUDA error:
        addmm_sparse_cuda not implemented for Half
    when AMP is enabled.
    """
    if x.dim() != 4:
        raise ValueError(f"Expected x [B,L,N,D], got {tuple(x.shape)}")
    original_dtype = x.dtype
    B, L, N, D = x.shape
    sparse_adj = _ensure_float32_sparse_adj(sparse_adj, device=x.device)
    if sparse_adj.size(0) != N or sparse_adj.size(1) != N:
        raise ValueError(f"sparse_adj {tuple(sparse_adj.size())} does not match N={N}")
    x_flat = x.permute(2, 0, 1, 3).contiguous().view(N, B * L * D)
    total_cols = x_flat.size(1)
    chunk_size = max(1, int(chunk_size))
    ctx = torch.cuda.amp.autocast(enabled=False) if x.is_cuda else nullcontext()
    outs = []
    with ctx:
        for start in range(0, total_cols, chunk_size):
            end = min(start + chunk_size, total_cols)
            out_chunk = torch.sparse.mm(sparse_adj, x_flat[:, start:end].float())
            outs.append(out_chunk.to(dtype=original_dtype))
    out = torch.cat(outs, dim=1)
    return out.view(N, B, L, D).permute(1, 2, 0, 3).contiguous()


class FallbackMambaLikeBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(u)
        conv = self.dwconv(u.transpose(1, 2))[..., : x.size(1)].transpose(1, 2)
        y, _ = self.gru(conv)
        y = y * torch.sigmoid(gate)
        return self.dropout(self.out_proj(y))


class FastBiMambaBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        d_model = int(args.d_model)
        d_state = int(getattr(args, "d_state", 16))
        d_conv = int(getattr(args, "d_conv", 4))
        expand = int(getattr(args, "mamba_expand", 2))
        dropout = float(getattr(args, "mamba_dropout", getattr(args, "dropout", 0.1)))
        force_fallback = bool(getattr(args, "force_fallback_mamba", False))
        self.use_bimamba = bool(getattr(args, "use_bimamba", True))
        self.use_fused_mamba = HAS_MAMBA_SSM and (not force_fallback)
        if self.use_fused_mamba:
            self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.forward_mamba = FallbackMambaLikeBlock(d_model, dropout=dropout)
            self.backward_mamba = FallbackMambaLikeBlock(d_model, dropout=dropout)
        self.fusion_gate = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N, D = x.shape
        z = x.permute(0, 2, 1, 3).contiguous().view(B * N, L, D)
        y_f = self.forward_mamba(z)
        if not self.use_bimamba:
            y = y_f
        else:
            y_b = self.backward_mamba(torch.flip(z, dims=[1]))
            y_b = torch.flip(y_b, dims=[1])
            gate = torch.sigmoid(self.fusion_gate(torch.cat([y_f, y_b], dim=-1)))
            y = gate * y_f + (1.0 - gate) * y_b
        y = self.dropout(y)
        return y.view(B, N, L, D).permute(0, 2, 1, 3).contiguous()


class SparseGraphResidualAdapter(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = int(args.d_model)
        self.graph_hops = max(1, int(getattr(args, "graph_hops", 1)))
        self.hop_decay = float(getattr(args, "hop_decay", 0.7))
        self.use_graph_diffusion = bool(getattr(args, "use_graph_diffusion", True))
        self.use_sna = bool(getattr(args, "use_sna", True))
        self.use_tanh_gate = bool(getattr(args, "use_tanh_gate", True))
        self.use_orth_res = bool(getattr(args, "use_orth_res", True))
        self.graph_chunk_size = int(getattr(args, "graph_chunk_size", 32768))
        dropout = float(getattr(args, "dropout", 0.1))

        self.hop_projs = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for _ in range(self.graph_hops)])
        self.self_proj = nn.Linear(self.d_model, self.d_model)
        self.hop_logits = nn.Parameter(torch.zeros(self.graph_hops))
        prior = torch.tensor([self.hop_decay ** k for k in range(self.graph_hops)], dtype=torch.float32)
        self.register_buffer("hop_decay_prior", prior)

        self.neighbor_proj = nn.Linear(self.d_model, self.d_model)
        self.selector = nn.Linear(self.d_model * 2, self.d_model)
        self.gate = nn.Linear(self.d_model * 2, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)

        self.graph_scale = nn.Parameter(torch.tensor(float(getattr(args, "graph_scale_init", 0.1)), dtype=torch.float32))
        self.alpha_res = nn.Parameter(torch.tensor(float(getattr(args, "alpha_res", 0.03)), dtype=torch.float32))

    @staticmethod
    def _orthogonal_residual(h0: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        dot = (h0 * h).sum(dim=-1, keepdim=True)
        denom = (h * h).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        redundant = dot / denom * h
        return torch.tanh(h0 - redundant)

    def _diffuse(self, h: torch.Tensor, sparse_adj: torch.Tensor) -> torch.Tensor:
        if not self.use_graph_diffusion:
            return self.self_proj(h)
        current = h
        hop_outs: List[torch.Tensor] = []
        for k in range(self.graph_hops):
            current = sparse_graph_aggregate(current, sparse_adj, chunk_size=self.graph_chunk_size)
            hop_outs.append(self.hop_projs[k](current))
        if self.graph_hops == 1:
            return hop_outs[0]
        weights = torch.softmax(self.hop_logits, dim=0)
        prior = self.hop_decay_prior.to(device=weights.device, dtype=weights.dtype)
        weights = weights * prior
        weights = weights / (weights.sum() + 1e-12)
        return sum(weights[k] * hop_outs[k] for k in range(self.graph_hops))

    def forward(self, h: torch.Tensor, h0: torch.Tensor, sparse_adj: torch.Tensor) -> torch.Tensor:
        graph_msg = self._diffuse(h, sparse_adj)

        if self.use_sna:
            selector = torch.sigmoid(self.selector(torch.cat([h, graph_msg], dim=-1)))
            graph_msg = selector * self.neighbor_proj(graph_msg)
        else:
            graph_msg = self.neighbor_proj(graph_msg)

        gate_input = torch.cat([h, graph_msg], dim=-1)
        if self.use_tanh_gate:
            gate = torch.tanh(self.gate(gate_input))
        else:
            gate = torch.sigmoid(self.gate(gate_input))
        delta = gate * self.out_proj(graph_msg)

        if self.use_orth_res:
            delta = delta + self.alpha_res * self._orthogonal_residual(h0, h)
        delta = self.dropout(delta)
        return torch.tanh(self.graph_scale) * delta


class FastGraphBiMambaLayer(nn.Module):
    def __init__(self, args, layer_idx: int):
        super().__init__()
        d_model = int(args.d_model)
        dropout = float(getattr(args, "dropout", 0.1))
        ffn_ratio = float(getattr(args, "ffn_ratio", 1.0))
        self.use_graph_adapter = bool(getattr(args, "use_graph_adapter", True))
        disable_ffn = bool(getattr(args, "disable_ffn", False))
        self.layer_idx = int(layer_idx)
        self.norm_mamba = RMSNorm(d_model)
        self.temporal = FastBiMambaBlock(args)
        self.norm_graph = RMSNorm(d_model)
        self.graph_adapter = SparseGraphResidualAdapter(args)
        self.norm_ffn = RMSNorm(d_model)
        self.ffn = ChunkedGatedMLP(
            d_model=d_model,
            hidden_ratio=ffn_ratio,
            dropout=dropout,
            chunk_nodes=int(getattr(args, "ffn_chunk_nodes", 64)),
            enabled=not disable_ffn,
        )

    def forward(self, x: torch.Tensor, h0: torch.Tensor, sparse_adj: torch.Tensor, enable_graph: bool) -> torch.Tensor:
        x = x + self.temporal(self.norm_mamba(x))
        if self.use_graph_adapter and enable_graph:
            x = x + self.graph_adapter(self.norm_graph(x), h0, sparse_adj)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class FastGraphBiMambaStack(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_layers = max(1, int(getattr(args, "layer_num", 2)))
        self.layers = nn.ModuleList([FastGraphBiMambaLayer(args, i) for i in range(self.num_layers)])
        self.graph_layers = sorted({0, self.num_layers // 2, self.num_layers - 1})

    def forward(self, x: torch.Tensor, sparse_adj: torch.Tensor) -> torch.Tensor:
        h0 = x
        for i, layer in enumerate(self.layers):
            x = layer(x, h0, sparse_adj, enable_graph=(i in self.graph_layers))
        return x


class GCN_mamba_TSForecast(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seq_len = int(args.seq_len)
        self.pred_len = int(args.pred_len)
        self.enc_in = int(args.enc_in)
        self.c_out = int(args.c_out)
        self.d_model = int(args.d_model)
        self.top_k = int(getattr(args, "top_k", 5))
        self.time_dim = int(getattr(args, "time_dim", 0))
        dropout = float(getattr(args, "dropout", 0.1))

        self.value_embedding = nn.Linear(1, self.d_model)
        self.variable_embedding = nn.Parameter(torch.zeros(1, 1, self.enc_in, self.d_model))
        nn.init.trunc_normal_(self.variable_embedding, std=0.02)
        self.time_embedding = None
        if self.time_dim > 0 and bool(getattr(args, "use_time_features", True)):
            self.time_embedding = nn.Linear(self.time_dim, self.d_model)
        self.input_norm = RMSNorm(self.d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.backbone = FastGraphBiMambaStack(args)
        self.output_norm = RMSNorm(self.d_model)
        self.head_dropout = nn.Dropout(float(getattr(args, "head_dropout", dropout)))
        self.forecast_head = nn.Linear(self.seq_len * self.d_model, self.pred_len)

        self.use_revin = bool(getattr(args, "use_revin", False))
        if self.use_revin:
            if RevIN is None:
                raise ImportError("use_revin=True but revin.py cannot be imported.")
            if self.c_out != self.enc_in:
                raise ValueError("RevIN denorm requires c_out == enc_in. Use features=M.")
            self.revin = RevIN(
                num_features=self.enc_in,
                affine=True,
                subtract_last=bool(getattr(args, "revin_subtract_last", False)),
            )
        else:
            self.revin = None

        eye = torch.eye(self.enc_in, dtype=torch.float32)
        self.register_buffer("static_adj", eye.clone())
        self.register_buffer("topk_sparse_adj", build_topk_sparse_adj(eye, self.top_k))

    @torch.no_grad()
    def set_static_adj(self, adj: Optional[torch.Tensor]):
        if adj is None:
            adj = torch.eye(self.enc_in, device=self.static_adj.device, dtype=torch.float32)
        if adj.dim() != 2 or adj.size(0) != self.enc_in or adj.size(1) != self.enc_in:
            raise ValueError(f"adj must be [{self.enc_in},{self.enc_in}], got {tuple(adj.shape)}")
        adj = adj.to(device=self.static_adj.device, dtype=torch.float32)
        self.static_adj = adj
        self.topk_sparse_adj = build_topk_sparse_adj(adj, self.top_k).to(self.static_adj.device)

    def _embed(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.value_embedding(x_enc.unsqueeze(-1))
        h = h + self.variable_embedding
        if self.time_embedding is not None and x_mark_enc is not None:
            h = h + self.time_embedding(x_mark_enc.float()).unsqueeze(2)
        h = self.input_norm(h)
        return self.embedding_dropout(h)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        x_dec: Optional[torch.Tensor] = None,
        x_mark_dec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_revin:
            x_enc = self.revin(x_enc, mode="norm")
        B, L, N = x_enc.shape
        if L != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {L}")
        if N != self.enc_in:
            raise ValueError(f"Expected enc_in={self.enc_in}, got {N}")
        h = self._embed(x_enc, x_mark_enc)
        h = self.backbone(h, self.topk_sparse_adj)
        h = self.output_norm(h)
        h_node = h.permute(0, 2, 1, 3).contiguous().view(B, N, L * self.d_model)
        h_node = self.head_dropout(h_node)
        pred = self.forecast_head(h_node).transpose(1, 2).contiguous()
        if self.use_revin:
            pred = self.revin(pred, mode="denorm")
        if self.c_out < self.enc_in:
            pred = pred[:, :, -self.c_out:]
        return pred
