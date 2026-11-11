"""Causal factorized time-axis SSM (prefix-sum of the GSSC kernel)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalFactorizedSSM(nn.Module):
    """
    Rank-R factorized mix along the patch/time axis with a causal prefix-sum.

    Same kernel class as GlobalFeatureSSM, but position t only aggregates ``<= t``.
    Input layout is TimeSeriesSSM's ``[B, P, C, H]`` (patches, channels).
    """

    def __init__(
        self,
        hidden_dim: int,
        z_dim: int,
        rank_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(z_dim, rank_dim, bias=False)
        self.k_proj = nn.Linear(z_dim, rank_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, rank_dim, bias=False)
        self.out_proj = nn.Linear(rank_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        visible: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, P, C, H]
            z: [P, Z] or [B, P, C, Z]
            visible: optional [B, P, C] or [P] / [B, P] mask; False positions
                do not contribute to the causal pool (horizon target tokens).
        Returns:
            [B, P, C, H]
        """
        if x.dim() != 4:
            raise ValueError("x must have shape [B, P, C, H].")
        x_t = x.transpose(1, 2)  # [B, C, P, H]
        v = self.v_proj(x_t)  # [B, C, P, R]

        if z.dim() == 2:
            if z.shape[0] != x.shape[1]:
                raise ValueError("z [P, Z] must match x.shape[1].")
            q = self.q_proj(z)  # [P, R]
            k = self.k_proj(z)
            kv = v * k.view(1, 1, z.shape[0], -1)
        elif z.dim() == 4:
            z_t = z.transpose(1, 2) if z.shape[1] == x.shape[1] else z
            per_window = z_t.shape[1] == 1 and (z_t.shape[0], z_t.shape[2]) == (x_t.shape[0], x_t.shape[2])
            if z_t.shape[:3] != x_t.shape[:3] and not per_window:  # [B, 1, P, Z]: one code per window
                raise ValueError("z must broadcast to [B, P, C, Z].")
            q = self.q_proj(z_t)
            k = self.k_proj(z_t)
            kv = v * k
        else:
            raise ValueError("z must have shape [P, Z] or [B, P, C, Z].")

        if visible is not None:
            vis = visible
            if vis.dim() == 1:
                vis = vis.view(1, vis.shape[0], 1)
            elif vis.dim() == 2:
                vis = vis.unsqueeze(-1)
            if vis.shape != x.shape[:3]:
                if vis.shape[0] == 1:
                    vis = vis.expand(x.shape[0], -1, x.shape[2] if vis.shape[2] == 1 else vis.shape[2])
                if vis.shape != x.shape[:3]:
                    raise ValueError("visible must broadcast to [B, P, C].")
            kv = kv * vis.transpose(1, 2).unsqueeze(-1).to(dtype=kv.dtype)

        pooled = torch.cumsum(kv, dim=2)
        if z.dim() == 2:
            h_rank = pooled * q.view(1, 1, q.shape[0], -1)
        else:
            h_rank = pooled * q
        h = self.out_proj(h_rank)
        return self.dropout(h).transpose(1, 2)


def _fp32_or_wider(t: torch.Tensor) -> torch.Tensor:
    """Upcast half/bfloat16 to float32 for the scan; float32 and float64 pass through unchanged."""
    return t if t.dtype in (torch.float32, torch.float64) else t.float()


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


class CausalDecaySSM(nn.Module):
    """
    Selective, decaying state space mix along the patch/time axis (S4D/Mamba-style).

    Per rank coordinate r the state follows

        s_t = a_t * s_{t-1} + k_t * v_t,     a_t = exp(-softplus(W_dt x_t + b_dt) * exp(A_log)),

    and the output is ``q_t * s_t`` projected back to ``hidden_dim``. ``a_t`` in (0, 1) gives an
    input-dependent recency bias; with ``exp(A_log) -> 0`` (a_t -> 1) this reduces exactly to
    :class:`CausalFactorizedSSM` (uniform prefix sum).

    ``bidirectional_context``: the first ``n_context`` positions (the fully observed lookback) are
    additionally mixed by a backward scan restricted to the context, so context tokens see the whole
    lookback. Horizon positions (``>= n_context``) read only the forward state, so no information
    flows from the horizon (or from later horizon steps) into any position.
    """

    def __init__(
        self,
        hidden_dim: int,
        z_dim: int,
        rank_dim: int,
        dropout: float = 0.0,
        bidirectional_context: bool = False,
        dt_min: float = 0.01,
        dt_max: float = 0.3,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(z_dim, rank_dim, bias=False)
        self.k_proj = nn.Linear(z_dim, rank_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, rank_dim, bias=False)
        self.out_proj = nn.Linear(rank_dim, hidden_dim, bias=False)
        self.dt_proj = nn.Linear(hidden_dim, rank_dim)
        self.A_log = nn.Parameter(torch.log(torch.linspace(0.5, 4.0, rank_dim)))
        with torch.no_grad():
            dt = torch.exp(torch.linspace(torch.log(torch.tensor(dt_min)), torch.log(torch.tensor(dt_max)), rank_dim))
            self.dt_proj.bias.copy_(_inv_softplus(dt))
            self.dt_proj.weight.mul_(0.1)
        self.bidirectional_context = bool(bidirectional_context)
        self.dropout = nn.Dropout(dropout)

    # forward scan implementation: "parallel" (closed form, one batched op) or "loop" (sequential reference);
    # both compute s_t = sum_{u<=t} (prod_{u<j<=t} a_j) kv_u, they differ only in floating-point summation order
    SCAN = "parallel"
    PARALLEL_MAX_P = 128  # P x P weights per (batch, channel, rank): beyond this fall back to the loop

    def log_decay(self, x_t: torch.Tensor) -> torch.Tensor:
        """x_t [B, C, P, H] -> log a [B, C, P, R] (< 0), taken before the exponential (no log(exp()) round trip)."""
        return -F.softplus(self.dt_proj(x_t)) * torch.exp(self.A_log)

    def decay(self, x_t: torch.Tensor) -> torch.Tensor:
        """x_t [B, C, P, H] -> a [B, C, P, R] in (0, 1)."""
        return torch.exp(self.log_decay(x_t))

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        visible: torch.Tensor | None = None,
        n_context: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, P, C, H]; z: [P, Z] or [B, P, C, Z]; visible: optional [B, P, C] mask
            n_context: number of leading (observed lookback) positions for the backward scan.
        Returns:
            [B, P, C, H]
        """
        if x.dim() != 4:
            raise ValueError("x must have shape [B, P, C, H].")
        x_t = x.transpose(1, 2)  # [B, C, P, H]
        v = self.v_proj(x_t)
        if z.dim() == 2:
            q = self.q_proj(z).view(1, 1, z.shape[0], -1)
            k = self.k_proj(z).view(1, 1, z.shape[0], -1)
        elif z.dim() == 4:
            z_t = z.transpose(1, 2) if z.shape[1] == x.shape[1] else z
            q, k = self.q_proj(z_t), self.k_proj(z_t)
        else:
            raise ValueError("z must have shape [P, Z] or [B, P, C, Z].")
        kv = v * k
        if visible is not None:
            vis = visible
            if vis.dim() == 1:
                vis = vis.view(1, vis.shape[0], 1)
            elif vis.dim() == 2:
                vis = vis.unsqueeze(-1)
            vis = vis.expand(x.shape[0], x.shape[1], x.shape[2]) if vis.shape != x.shape[:3] else vis
            kv = kv * vis.transpose(1, 2).unsqueeze(-1).to(dtype=kv.dtype)
        # the decay and the recurrence run in fp32 even under bf16 autocast (products of many a_t < 1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            log_a = self.log_decay(_fp32_or_wider(x_t))  # [B, C, P, R]
            kv = _fp32_or_wider(kv)
            return self._scan(x, kv, log_a, q, n_context)

    @staticmethod
    def _parallel_fwd(kv: torch.Tensor, log_a: torch.Tensor) -> torch.Tensor:
        """s_t = sum_{u<=t} exp(L_t - L_u) kv_u with L = cumsum(log a): the unrolled recurrence in one batched op.
        The upper triangle (u > t) is masked to -inf BEFORE exp, so no positive exponent can overflow."""
        p = kv.shape[2]
        cum = torch.cumsum(log_a, dim=2)                                       # [B, C, P, R]
        diff = cum.unsqueeze(3) - cum.unsqueeze(2)                             # [B, C, P(t), P(u), R]
        causal = torch.ones(p, p, dtype=torch.bool, device=kv.device).tril().view(1, 1, p, p, 1)
        return (torch.exp(diff.masked_fill(~causal, float("-inf"))) * kv.unsqueeze(2)).sum(3)

    @classmethod
    def _parallel_fwd_chunked(cls, kv: torch.Tensor, log_a: torch.Tensor,
                              max_bytes: float = 256 * 2 ** 20) -> torch.Tensor:
        """``_parallel_fwd`` over chunks of the (B*C) rows, which are independent: identical maths, bounded memory."""
        if kv.is_cuda:
            torch.cuda.empty_cache()
        b, c, p, r = kv.shape
        rows = max(1, int(max_bytes // (p * p * r * kv.element_size())))
        kv2, la2 = kv.reshape(1, b * c, p, r), log_a.reshape(1, b * c, p, r)
        out = [cls._parallel_fwd(kv2[:, i:i + rows], la2[:, i:i + rows]) for i in range(0, b * c, rows)]
        return torch.cat(out, dim=1).view(b, c, p, r)

    def _scan(self, x, kv, log_a, q, n_context):
        p = x.shape[1]
        a = None
        if self.SCAN == "parallel" and p <= self.PARALLEL_MAX_P:
            try:
                s_fwd = self._parallel_fwd(kv, log_a)
            except torch.OutOfMemoryError:
                if torch.is_grad_enabled():
                    raise  # training: run_one's micro-batch backoff handles it (behaviour unchanged)
                # evaluation only (previously a crash): the same scan over independent row chunks
                s_fwd = self._parallel_fwd_chunked(kv, log_a)
        else:
            a = torch.exp(log_a)
            states = []
            s = torch.zeros_like(kv[:, :, 0])
            for t in range(p):
                s = a[:, :, t] * s + kv[:, :, t]
                states.append(s)
            s_fwd = torch.stack(states, dim=2)

        if self.bidirectional_context and n_context is not None and n_context > 1:
            if a is None:
                a = torch.exp(log_a)
            nc = min(int(n_context), p)
            back = [None] * nc
            s = torch.zeros_like(kv[:, :, 0])
            for t in range(nc - 1, -1, -1):
                s = a[:, :, t] * s + kv[:, :, t]
                back[t] = s
            s_bwd = torch.stack(back, dim=2) - kv[:, :, :nc]  # kv_t counted once (in s_fwd)
            s_fwd = torch.cat([s_fwd[:, :, :nc] + s_bwd, s_fwd[:, :, nc:]], dim=2)

        h = self.out_proj(s_fwd * _fp32_or_wider(q))
        return self.dropout(h).transpose(1, 2)
