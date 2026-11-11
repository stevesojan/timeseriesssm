"""Series branch: one token per channel, mixed across channels.

Each channel's whole (RevIN-normalized) lookback is embedded as one token (iTransformer, Liu et al., ICLR 2024), and
pre-norm Transformer layers attend across the channel axis. No channel positional encoding is used, so the module is
permutation-equivariant over channels, like the rest of TimeSeriesSSM.

Outputs:
  * a forecast branch  [B, H, C] (Linear(dim -> horizon) per token), added by the caller through a gate;
  * one context vector per N-HiTS stack [B, C, hidden_dim], added to that stack's patch tokens (TimeXer-style
    series-level context for the patch-level model). The context projections are zero-initialized, so the cascade
    starts exactly as without series tokens.

Attention uses F.scaled_dot_product_attention (flash / memory-efficient kernels), so C x C scores are not
materialized for large channel counts (Traffic: 862).

``mixer`` selects the channel-interaction layer: "attention", "star" (SOFTS' STar
Aggregate-Redistribute, Han et al., NeurIPS 2024: every series is mapped by an MLP, stochastic pooling over channels
forms one global core, the core is concatenated back to every series and fused by a second MLP with a residual;
O(C) and no C x C interaction), or "both" (STAR then attention in every layer).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ChannelAttentionLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"series_dim {dim} must be divisible by series_heads {heads}.")
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.drop = nn.Dropout(dropout)
        self.p_attn = dropout

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t [B, C, D]
        b, c, d = t.shape
        q, k, v = self.qkv(self.norm1(t)).view(b, c, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p_attn if self.training else 0.0)
        t = t + self.drop(self.out(a.transpose(1, 2).reshape(b, c, d)))
        return t + self.drop(self.ffn(self.norm2(t)))


class _STARLayer(nn.Module):
    """SOFTS STAR (official form): core = StochPool_c(MLP1(S)); S <- S + MLP2([S, core]); then a pre-norm FFN.
    Stochastic pooling: per feature, a channel is sampled with probability softmax over channels (training);
    the softmax-weighted average is used in evaluation. Permutation-invariant pooling keeps the layer
    channel-permutation equivariant (in expectation during training, exactly in evaluation)."""

    def __init__(self, dim: int, core_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.gen1 = nn.Linear(dim, dim)
        self.gen2 = nn.Linear(dim, core_dim)
        self.gen3 = nn.Linear(dim + core_dim, dim)
        self.gen4 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.drop = nn.Dropout(dropout)

    def core(self, h: torch.Tensor) -> torch.Tensor:  # h [B, C, K] -> [B, 1, K]
        w = torch.softmax(h, dim=1)
        if self.training:
            b, c, k = h.shape
            idx = torch.multinomial(w.permute(0, 2, 1).reshape(b * k, c), 1).view(b, k)  # one channel per feature
            return torch.gather(h, 1, idx.unsqueeze(1))
        return (w * h).sum(1, keepdim=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t [B, C, D]
        u = self.norm1(t)
        core = self.core(self.gen2(F.gelu(self.gen1(u))))
        fused = torch.cat([u, core.expand(-1, u.shape[1], -1)], dim=-1)
        t = t + self.drop(self.gen4(F.gelu(self.gen3(fused))))
        return t + self.drop(self.ffn(self.norm2(t)))


class SeriesTokenMixer(nn.Module):
    def __init__(self, context_len: int, horizon: int, dim: int, layers: int, heads: int, dropout: float,
                 n_stacks: int, hidden_dim: int, mixer: str = "attention", core_dim: int = 0) -> None:
        super().__init__()
        if mixer not in ("attention", "star", "both"):
            raise ValueError(f"series mixer must be attention, star or both, got {mixer!r}")
        self.embed = nn.Linear(context_len, dim)
        n = max(1, layers)
        cd = core_dim or dim
        if mixer == "attention":
            mods = [_ChannelAttentionLayer(dim, heads, dropout) for _ in range(n)]
        elif mixer == "star":
            mods = [_STARLayer(dim, cd, dropout) for _ in range(n)]
        else:
            mods = [m for _ in range(n) for m in (_STARLayer(dim, cd, dropout), _ChannelAttentionLayer(dim, heads, dropout))]
        self.layers = nn.ModuleList(mods)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, horizon)
        self.ctx = nn.ModuleList([nn.Linear(dim, hidden_dim) for _ in range(n_stacks)])
        for p in self.ctx:
            nn.init.zeros_(p.weight)
            nn.init.zeros_(p.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """x [B, L, C] normalized lookback -> (forecast [B, H, C], [context [B, C, hidden_dim]] per stack)."""
        t = self.embed(x.transpose(1, 2))
        for layer in self.layers:
            t = layer(t)
        t = self.norm(t)
        return self.head(t).transpose(1, 2), [p(t) for p in self.ctx]
