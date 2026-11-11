"""N-HiTS-style hierarchical cascade of GSSC stacks (Challu et al., AAAI 2023, adapted).

Each stack k:
  1. multi-rate input sampling: average-pools the current *residual* lookback with kernel ``pool_k``;
  2. patches the pooled series with ``patch_len // pool_k`` so every stack sees the same number of
     context tokens (coarse stacks: each token summarizes a longer, smoothed span);
  3. mixes tokens with TimeSeriesSSMBlocks (time SSM + channel GSSC, same kernels as the main model);
  4. emits a **backcast** over the full lookback (subtracted from the residual passed to stack k+1) and
     ``ceil(H / ratio_k)`` forecast coefficients that are linearly **interpolated** to the horizon.
Forecasts of all stacks are summed. Heads are channel-shared, so the cascade stays permutation-equivariant
over channels. Point forecasts only (no quantiles / covariates).

Optional sample axis (``use_support=True``): a bank of K labeled support windows (lookback + true future)
runs through the same stacks. Each support token gets an embedding of its *residual* future (true future
minus the forecasts of earlier stacks for that support, i.e. what this stack still has to explain), and
after every block the query tokens read the support states through ``SupportSetSSM``.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .heads import FlattenForecastHead
from .support_ssm import SupportSetSSM
from .patch_embed import PatchEmbed
from .timeseries_block import TimeSeriesSSMBlock


class NHiTSStack(nn.Module):
    def __init__(self, *, context_len: int, horizon: int, patch_len: int, pool: int, ratio: int,
                 hidden_dim: int, channel_z_dim: int, time_z_dim: int, n_blocks: int,
                 block_kwargs: dict, dropout: float, need_backcast: bool = True,
                 grad_checkpoint: bool = False, use_support: bool = False, rank_dim: int = 0) -> None:
        super().__init__()
        self.grad_checkpoint = bool(grad_checkpoint)
        if patch_len % pool != 0:
            raise ValueError(f"patch_len {patch_len} must be divisible by pool kernel {pool}.")
        self.pool = int(pool)
        self.sub_patch = patch_len // pool
        self.n_tokens = -(-context_len // patch_len)
        self.context_len, self.horizon = int(context_len), int(horizon)
        self.n_coef = max(1, -(-horizon // max(1, ratio)))
        self.embed = PatchEmbed(patch_len=self.sub_patch, hidden_dim=hidden_dim, lookahead_dim=0)
        self.blocks = nn.ModuleList([
            TimeSeriesSSMBlock(hidden_dim=hidden_dim, channel_z_dim=channel_z_dim, time_z_dim=time_z_dim,
                               dropout=dropout, **block_kwargs)
            for _ in range(max(1, n_blocks))
        ])
        # the last stack's backcast would be discarded, so it gets no head
        self.backcast_head = (FlattenForecastHead(hidden_dim, self.n_tokens, self.n_tokens * patch_len,
                                                  n_quantiles=0, dropout=dropout) if need_backcast else None)
        self.forecast_head = FlattenForecastHead(hidden_dim, self.n_tokens, self.n_coef, n_quantiles=0,
                                                 dropout=dropout)
        self.use_support = bool(use_support)
        if self.use_support:
            self.label_proj = nn.Linear(self.horizon, hidden_dim)  # support residual future -> token context
            self.support_ssm = nn.ModuleList([SupportSetSSM(hidden_dim, rank_dim or hidden_dim, dropout)
                                              for _ in self.blocks])
            self.support_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.blocks])

    def forward(self, resid: torch.Tensor, z_ch: torch.Tensor, z_time: torch.Tensor, n_query: int | None = None,
                support_label: torch.Tensor | None = None, support_mask: torch.Tensor | None = None,
                ctx: torch.Tensor | None = None):
        """resid [B, L_pad, C] -> (backcast [B, L_pad, C], forecast [B, H, C]).

        With a support bank, ``resid`` holds the queries in rows ``[:n_query]`` and the supports after them;
        ``support_label`` [K, H, C] is the supports' residual future, ``support_mask`` [n_query, K] or None.
        """
        b, l, c = resid.shape
        x = resid.permute(0, 2, 1)  # [B, C, L]
        if self.pool > 1:
            x = F.avg_pool1d(x, kernel_size=self.pool, stride=self.pool)
        x = x.reshape(b, c, self.n_tokens, self.sub_patch).permute(0, 2, 1, 3)  # [B, P, C, sub_patch]
        h = self.embed(x)
        if ctx is not None:  # series-level context [B, C, hidden] (series tokens), broadcast over patches
            h = h + ctx.unsqueeze(1).to(h.dtype)
        sup = self.use_support and support_label is not None and n_query is not None and n_query < b
        if sup:  # label context on support tokens only
            lab = self.label_proj(support_label.permute(0, 2, 1))  # [K, C, H]
            h = torch.cat([h[:n_query], h[n_query:] + lab.unsqueeze(1)], dim=0)
        for i, blk in enumerate(self.blocks):
            if self.grad_checkpoint and torch.is_grad_enabled():
                h = checkpoint(blk, h, z_ch, z_time, n_context=self.n_tokens, use_reentrant=False)
            else:
                h = blk(h, z_ch, z_time, n_context=self.n_tokens)
            if sup:
                h_q, h_s = h[:n_query], h[n_query:]
                h_q = self.support_norm[i](h_q + self.support_ssm[i](h_q, h_s, support_mask))
                h = torch.cat([h_q, h_s], dim=0)
        backcast = self.backcast_head(h)[0] if self.backcast_head is not None else None
        coef, _ = self.forecast_head(h)  # [B, n_coef, C]
        if self.n_coef != self.horizon:
            coef = F.interpolate(coef.permute(0, 2, 1), size=self.horizon, mode="linear", align_corners=True)
            coef = coef.permute(0, 2, 1)
        return backcast, coef


class NHiTSCascade(nn.Module):
    def __init__(self, *, context_len: int, horizon: int, patch_len: int, pools: Sequence[int],
                 ratios: Sequence[int], blocks_per_stack: int, hidden_dim: int, channel_z_dim: int,
                 time_z_dim: int, block_kwargs: dict, dropout: float, grad_checkpoint: bool = False,
                 use_support: bool = False, stack_overrides: Sequence[dict] | None = None) -> None:
        super().__init__()
        if len(pools) != len(ratios):
            raise ValueError("nhits_pool_kernels and nhits_downsample must have the same length.")
        self.patch_len = int(patch_len)
        self.context_len, self.horizon = int(context_len), int(horizon)
        self.stacks = nn.ModuleList([
            NHiTSStack(context_len=context_len, horizon=horizon, patch_len=patch_len, pool=p, ratio=r,
                       hidden_dim=hidden_dim, channel_z_dim=channel_z_dim, time_z_dim=time_z_dim,
                       n_blocks=blocks_per_stack, dropout=dropout,
                       block_kwargs={**block_kwargs, **(stack_overrides[i] if stack_overrides else {})},
                       need_backcast=i < len(pools) - 1, grad_checkpoint=grad_checkpoint,
                       use_support=use_support, rank_dim=int(block_kwargs.get("rank_dim", 0)))
            for i, (p, r) in enumerate(zip(pools, ratios))
        ])

    @property
    def n_tokens(self) -> int:
        return self.stacks[0].n_tokens

    def forward(self, x: torch.Tensor, z_ch: torch.Tensor, z_time: torch.Tensor,
                support_x: torch.Tensor | None = None, support_y: torch.Tensor | None = None,
                support_mask: torch.Tensor | None = None, ctx: Sequence[torch.Tensor] | None = None) -> torch.Tensor:
        """x [B, L, C] normalized lookback -> forecast [B, H, C] (normalized).
        ``ctx``: optional per-stack series context [B, C, hidden] (series tokens; not with a support bank).

        Optional support bank: ``support_x`` [K, L, C], ``support_y`` [K, H, C] (normalized like the queries),
        ``support_mask`` [B, K] (True = support usable by that query).
        """
        n_q = x.shape[0]
        use_sup = support_x is not None and support_y is not None
        if use_sup:
            x = torch.cat([x, support_x], dim=0)
        pad = self.n_tokens * self.patch_len - x.shape[1]
        resid = F.pad(x, (0, 0, pad, 0)) if pad > 0 else x  # left-pad so the latest step stays last
        forecast = 0.0
        if ctx is not None and use_sup:
            raise NotImplementedError("series-token context is not combined with a support bank.")
        for i, stack in enumerate(self.stacks):
            label = (support_y - forecast[n_q:]) if use_sup and torch.is_tensor(forecast) else support_y
            backcast, f = stack(resid, z_ch, z_time, n_query=n_q if use_sup else None,
                                support_label=label, support_mask=support_mask,
                                ctx=None if ctx is None else ctx[i])
            if backcast is not None:
                resid = resid - backcast
            forecast = forecast + f
        return forecast[:n_q]
