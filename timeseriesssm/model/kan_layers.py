"""Kolmogorov-Arnold feed-forward layers for TimeSeriesSSMBlock (alternative to the GLU FFN).

- ``ChebyKANLayer``: every edge (i -> o) is a learnable univariate function
  phi_io(x) = sum_k c[i, o, k] T_k(tanh x) with Chebyshev polynomials T_k (TimeKAN, ICLR 2025, M-KAN form).
- ``BSplineKANLayer``: phi_io(x) = w_io * SiLU(x) + sum_j c[i, o, j] B_j(x) with cubic B-splines on a fixed
  uniform grid (original KAN / efficient-KAN form, as in DecompKAN). Outside the grid only the SiLU path acts.
- ``KANFFN``: LN -> KAN(H -> w) -> LN -> KAN(w -> H); a residual-branch replacement for the GLU FFN
  (inner and outer Kolmogorov-Arnold functions). Acts per token on the last dim only, so the block keeps
  its channel-permutation equivariance and time causality.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def chebyshev_basis(x: torch.Tensor, order: int) -> torch.Tensor:
    """x in [-1, 1], [..., D] -> [..., D, order + 1] with T_0..T_order via T_{k+1} = 2x T_k - T_{k-1}."""
    ts = [torch.ones_like(x)]
    if order >= 1:
        ts.append(x)
    for _ in range(2, order + 1):
        ts.append(2.0 * x * ts[-1] - ts[-2])
    return torch.stack(ts, dim=-1)


def bspline_basis(x: torch.Tensor, knots: torch.Tensor, order: int) -> torch.Tensor:
    """Cox-de Boor on an extended uniform knot vector. x [..., D] -> [..., D, n_knots - 1 - order]."""
    x = x.unsqueeze(-1)
    b = ((x >= knots[:-1]) & (x < knots[1:])).to(x.dtype)
    for k in range(1, order + 1):
        left = (x - knots[: -(k + 1)]) / (knots[k:-1] - knots[: -(k + 1)])
        right = (knots[k + 1:] - x) / (knots[k + 1:] - knots[1:-k])
        b = left * b[..., :-1] + right * b[..., 1:]
    return b


def cubic_bspline_basis_uniform(x: torch.Tensor, t0: float, h: float, n_basis: int) -> torch.Tensor:
    """Closed-form uniform cubic B-splines (same values as Cox-de Boor, ~20% less time/memory).

    Basis j is supported on [t0 + j h, t0 + (j + 4) h); s is the position inside that support in units of h.
    """
    s = ((x - t0) / h).unsqueeze(-1) - torch.arange(n_basis, device=x.device, dtype=x.dtype)
    s = s.clamp(0.0, 4.0)
    return torch.where(s < 1, s ** 3 / 6,
           torch.where(s < 2, (-3 * s ** 3 + 12 * s ** 2 - 12 * s + 4) / 6,
           torch.where(s < 3, (3 * s ** 3 - 24 * s ** 2 + 60 * s - 44) / 6, (4 - s) ** 3 / 6)))


class _CubicBSplineUniform(torch.autograd.Function):
    """Same values/gradients as ``cubic_bspline_basis_uniform``, ~3x faster and ~40% less memory.

    On a uniform grid each value activates only 4 cubic bases (indices j-3..j for u in [j, j+1)) with
    weights that are cubics in t = u - j. Forward computes those 4 weights on [..., D] and scatter-adds them
    into the dense basis (out-of-range bases get weight 0 at a clamped index, so no padding/copy is needed);
    backward gathers the 4 incoming gradients and applies the analytic derivatives, so autograd keeps only
    (t, idx, mask) instead of every intermediate of the piecewise formula.
    """

    @staticmethod
    def forward(ctx, x, t0: float, h: float, n_basis: int):
        u = (x - t0) / h
        j = torch.floor(u).clamp(-4, n_basis + 4)  # beyond this all 4 bases are out of range anyway
        t = u - j
        t2, t3 = t * t, t * t * t
        w = torch.stack([(1 - t) ** 3, 3 * t3 - 6 * t2 + 4, -3 * t3 + 3 * t2 + 3 * t + 1, t3], dim=-1) / 6
        k = j.long().unsqueeze(-1) - 3 + torch.arange(4, device=x.device)  # basis indices j-3..j
        mask = (k >= 0) & (k < n_basis)
        idx = k.clamp(0, n_basis - 1)
        out = torch.zeros(*x.shape, n_basis, device=x.device, dtype=x.dtype)
        out.scatter_add_(-1, idx, w * mask)
        ctx.save_for_backward(t, idx, mask)
        ctx.h = h
        return out

    @staticmethod
    def backward(ctx, g):
        t, idx, mask = ctx.saved_tensors
        t2 = t * t
        dw = torch.stack([-(1 - t) ** 2 / 2, 1.5 * t2 - 2 * t, -1.5 * t2 + t + 0.5, t2 / 2], dim=-1)
        return (g.gather(-1, idx) * dw * mask).sum(-1) / ctx.h, None, None, None


class ChebyKANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, order: int = 3) -> None:
        super().__init__()
        self.order = int(order)
        self.coef = nn.Parameter(torch.randn(in_dim, out_dim, order + 1) / math.sqrt(in_dim * (order + 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = chebyshev_basis(torch.tanh(x), self.order)  # [..., in, K+1]
        w = self.coef.permute(1, 0, 2).reshape(self.coef.shape[1], -1)  # [out, in * (K+1)]
        return F.linear(basis.reshape(*basis.shape[:-2], -1), w)


class BSplineKANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid: int = 5, order: int = 3,
                 grid_range: tuple[float, float] = (-2.0, 2.0)) -> None:
        super().__init__()
        self.order = int(order)
        h = (grid_range[1] - grid_range[0]) / grid
        knots = grid_range[0] + h * torch.arange(-order, grid + order + 1, dtype=torch.float32)
        self.register_buffer("knots", knots, persistent=False)
        self._t0, self._h = float(knots[0]), float(h)
        n_basis = grid + order
        self.n_basis = n_basis
        self.base = nn.Linear(in_dim, out_dim, bias=False)
        self.coef = nn.Parameter(torch.randn(in_dim, out_dim, n_basis) * (0.1 / math.sqrt(in_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.order == 3:
            basis = _CubicBSplineUniform.apply(x, self._t0, self._h, self.n_basis)  # [..., in, G+3]
        else:
            basis = bspline_basis(x, self.knots.to(x.dtype), self.order)
        w = self.coef.permute(1, 0, 2).reshape(self.coef.shape[1], -1)  # [out, in * n_basis]
        return self.base(F.silu(x)) + F.linear(basis.reshape(*basis.shape[:-2], -1), w)


class KANFFN(nn.Module):
    def __init__(self, hidden_dim: int, kind: str, width: int, order: int = 3, grid: int = 5,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if kind == "cheby_kan":
            make = lambda i, o: ChebyKANLayer(i, o, order=order)  # noqa: E731
        elif kind == "bspline_kan":
            make = lambda i, o: BSplineKANLayer(i, o, grid=grid, order=order)  # noqa: E731
        else:
            raise ValueError(f"unknown KAN kind {kind!r}")
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.inner = make(hidden_dim, width)
        self.norm_mid = nn.LayerNorm(width)
        self.outer = make(width, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.outer(self.dropout(self.norm_mid(self.inner(self.norm_in(x)))))
