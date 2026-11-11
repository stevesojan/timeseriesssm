"""Global permutation-equivariant feature SSM with factorized kernels."""

from __future__ import annotations

import torch
import torch.nn as nn


class GlobalFeatureSSM(nn.Module):
    """
    Feature-axis global SSM convolution:
        h_f = <q_f, sum_{f'} k_{f'} * v_{f'}>
    where q_f, k_f are from positional encodings and v_f is from content.

    This implementation avoids explicit F x F kernel construction.
    """

    def __init__(
        self,
        hidden_dim: int,
        z_dim: int,
        rank_dim: int,
        symmetric_kernel: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.symmetric_kernel = symmetric_kernel

        self.q_proj = nn.Linear(z_dim, rank_dim, bias=False)
        self.k_proj = self.q_proj if symmetric_kernel else nn.Linear(z_dim, rank_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, rank_dim, bias=False)
        self.out_proj = nn.Linear(rank_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, F, H]
            z: [F, Z] (static) or [B, N, F, Z] (content-updated)
        Returns:
            [B, N, F, H]
        """
        if x.dim() != 4:
            raise ValueError("x must have shape [B, N, F, H].")

        v = self.v_proj(x)  # [B, N, F, R]
        if z.dim() == 2:
            q = self.q_proj(z)  # [F, R]
            k = self.k_proj(z)  # [F, R]
            pooled = torch.einsum("fr,bnfr->bnr", k, v)  # [B, N, R]
            h_rank = torch.einsum("fr,bnr->bnfr", q, pooled)  # [B, N, F, R]
        elif z.dim() == 4:
            q = self.q_proj(z)  # [B, N, F, R]
            k = self.k_proj(z)  # [B, N, F, R]
            pooled = torch.einsum("bnfr,bnfr->bnr", k, v)  # [B, N, R]
            h_rank = q * pooled.unsqueeze(2)  # [B, N, F, R]
        else:
            raise ValueError("z must have shape [F, Z] or [B, N, F, Z].")

        h = self.out_proj(h_rank)
        return self.dropout(h)


class SpectralGSSC(nn.Module):
    """
    GSSC as defined by Huang, Miao & Li (2024, arXiv:2406.05815, Eq. 4), along the feature axis:
        h_f = sum_{f'} <Z_f U_q, Z_{f'} U_k> * U_o x_{f'} + <Z_f U_sq, Z_f U_sk> * U_s x_f
    with codes Z_f in R^{d x M} (entries phi_m(lambda_j) e_{j,f}, at most divided by one global scalar). The U act on the filter axis and
    <.,.> sums over the eigenvector index j only, so every hidden feature is mixed by a spectral filter
    sum_j psi(lambda_j) e_j e_j^T, which does not change when an eigenvector changes sign. Evaluated through a
    [d, H] state per position; no F x F kernel is formed. Takes static codes only.
    """

    def __init__(self, hidden_dim: int, num_eig: int, num_filters: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_eig, self.num_filters = int(num_eig), int(num_filters)
        self.u_q = nn.Linear(num_filters, hidden_dim, bias=False)
        self.u_k = nn.Linear(num_filters, hidden_dim, bias=False)
        self.u_sq = nn.Linear(num_filters, hidden_dim, bias=False)
        self.u_sk = nn.Linear(num_filters, hidden_dim, bias=False)
        self.u_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.u_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, F, H]
            z: [F, d * M] static codes (flattened Z_f)
        Returns:
            [B, N, F, H]
        """
        if x.dim() != 4 or z.dim() != 2:
            raise ValueError("SpectralGSSC takes x [B, N, F, H] and static codes z [F, d*M].")
        zz = z.view(z.shape[0], self.num_eig, self.num_filters)            # [F, d, M]
        state = torch.einsum("fjh,bnfh->bnjh", self.u_k(zz), self.u_o(x))  # [B, N, d, H], formed once
        h = torch.einsum("fjh,bnjh->bnfh", self.u_q(zz), state)
        h = h + (self.u_sq(zz) * self.u_sk(zz)).sum(1) * self.u_s(x)        # self term
        return self.dropout(h)


class ChannelAttention(nn.Module):
    """
    Content-only alternative to GlobalFeatureSSM (ablation): multi-head self-attention across the F channels at every
    patch position, without positional codes (as across the variate tokens of iTransformer). O(F^2) per position;
    the codes ``z`` are ignored.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        b, n, f, h = x.shape
        t = x.reshape(b * n, f, h)
        y, _ = self.attn(t, t, t, need_weights=False)
        return self.dropout(y.reshape(b, n, f, h))


class ChannelSTAR(nn.Module):
    """
    Content-only alternative to GlobalFeatureSSM (ablation): the STAR interaction of SOFTS (Han et al., 2024) across
    the F channels at every patch position. An MLP maps every token, a core is pooled over the channels (stochastic
    pooling in training, softmax-weighted average in evaluation), and a second MLP fuses the core with each token.
    O(F) per position; the codes ``z`` are ignored.
    """

    def __init__(self, hidden_dim: int, core_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gen1 = nn.Linear(hidden_dim, hidden_dim)
        self.gen2 = nn.Linear(hidden_dim, core_dim)
        self.gen3 = nn.Linear(hidden_dim + core_dim, hidden_dim)
        self.gen4 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        b, n, f, h = x.shape
        t = x.reshape(b * n, f, h)
        c = self.gen2(torch.nn.functional.gelu(self.gen1(t)))  # [BN, F, K]
        w = torch.softmax(c, dim=1)
        if self.training:
            k = c.shape[-1]
            idx = torch.multinomial(w.permute(0, 2, 1).reshape(-1, f), 1).view(b * n, k)  # one channel per feature
            core = torch.gather(c, 1, idx.unsqueeze(1))
        else:
            core = (w * c).sum(1, keepdim=True)
        y = self.gen4(torch.nn.functional.gelu(self.gen3(torch.cat([t, core.expand(-1, f, -1)], dim=-1))))
        return self.dropout(y.reshape(b, n, f, h))
