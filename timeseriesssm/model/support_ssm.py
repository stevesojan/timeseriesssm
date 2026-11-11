"""Sample-axis SSM over a labeled support bank of training windows (optional; not used in the paper).

Mixes the windows of a set with a rank-R factorized global kernel ``q_n * sum_m k_m (x) v_m``; a *sample*
is a window and a *feature* is a channel, applied per token position:

    out[b, p, c] = q[b, p] * sum_k w[b, k] * k[k, p] * v[k, p, c]

Queries read from the support set only; supports never read from queries or from each other, so a
support masked out for a query (e.g. overlapping its future during training) cannot reach it indirectly.
The pooled sum is a masked mean (not an unnormalized sum) so it does not depend on how many supports
survive the per-query overlap mask. Permutation-equivariant over channels and over the support set.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SupportSetSSM(nn.Module):
    def __init__(self, hidden_dim: int, rank_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, rank_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, rank_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, rank_dim, bias=False)
        self.out_proj = nn.Linear(rank_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_q: torch.Tensor, h_s: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """h_q [B, P, C, H] queries, h_s [K, P, C, H] supports, mask [B, K] (True = usable) -> [B, P, C, H]."""
        q = self.q_proj(h_q.mean(dim=2))  # [B, P, R]
        k = self.k_proj(h_s.mean(dim=2))  # [K, P, R]
        kv = k.unsqueeze(2) * self.v_proj(h_s)  # [K, P, C, R]
        if mask is None:
            pooled = kv.mean(dim=0).unsqueeze(0)  # [1, P, C, R]
        else:
            w = mask.to(kv.dtype)
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = torch.einsum("bk,kpcr->bpcr", w, kv)
        return self.dropout(self.out_proj(q.unsqueeze(2) * pooled))
