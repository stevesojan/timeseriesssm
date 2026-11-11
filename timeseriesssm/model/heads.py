"""Point + quantile forecast heads and pinball / MAE losses."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class ForecastHead(nn.Module):
    """Map horizon patch tokens [B, P_h, C, H] to point and optional quantiles."""

    def __init__(
        self,
        hidden_dim: int,
        patch_len: int,
        n_quantiles: int = 9,
    ) -> None:
        super().__init__()
        self.patch_len = int(patch_len)
        self.n_quantiles = int(n_quantiles)
        out = self.patch_len * (1 + self.n_quantiles)
        self.proj = nn.Linear(hidden_dim, out)

    def forward(self, horizon_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            point: [B, T_h, C]
            quantiles: [B, T_h, C, Q] or None
        """
        raw = self.proj(horizon_tokens)  # [B, P, C, patch_len * (1+Q)]
        b, p, c, _ = raw.shape
        raw = raw.view(b, p, c, 1 + self.n_quantiles, self.patch_len)
        raw = raw.permute(0, 1, 4, 2, 3).reshape(b, p * self.patch_len, c, 1 + self.n_quantiles)
        point = raw[..., 0]
        if self.n_quantiles <= 0:
            return point, None
        quantiles = raw[..., 1:]
        return point, quantiles


class FlattenForecastHead(nn.Module):
    """
    PatchTST-style head: per channel, flatten the context tokens [P_ctx * H] and map them linearly
    to the whole (patch-padded) horizon. Channel-shared weights keep it permutation-equivariant.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_context_patches: int,
        horizon: int,
        n_quantiles: int = 9,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_context_patches = int(n_context_patches)
        self.horizon = int(horizon)
        self.n_quantiles = int(n_quantiles)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(self.n_context_patches * hidden_dim, self.horizon * (1 + self.n_quantiles))

    def forward(self, context_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """context_tokens [B, P_ctx, C, H] -> point [B, horizon, C], quantiles [B, horizon, C, Q] or None."""
        b, p, c, h = context_tokens.shape
        if p != self.n_context_patches:
            raise ValueError(f"FlattenForecastHead built for {self.n_context_patches} context patches, got {p}.")
        flat = context_tokens.permute(0, 2, 1, 3).reshape(b, c, p * h)
        raw = self.proj(self.dropout(flat)).view(b, c, 1 + self.n_quantiles, self.horizon)
        raw = raw.permute(0, 3, 1, 2)  # [B, horizon, C, 1+Q]
        point = raw[..., 0]
        if self.n_quantiles <= 0:
            return point, None
        return point, raw[..., 1:]


class ImputeHead(nn.Module):
    def __init__(self, hidden_dim: int, patch_len: int) -> None:
        super().__init__()
        self.patch_len = int(patch_len)
        self.proj = nn.Linear(hidden_dim, self.patch_len)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, P, C, H] -> [B, P, C, patch_len]."""
        return self.proj(tokens)


class ClassifyHead(nn.Module):
    def __init__(self, hidden_dim: int, n_classes: int) -> None:
        super().__init__()
        self.pool_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = tokens.mean(dim=(1, 2))
        return self.mlp(self.pool_norm(pooled))


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """
    pred: [B, T, C, Q], target: [B, T, C], quantiles: [Q]
    """
    q = quantiles.view(*([1] * (pred.dim() - 1)), -1).to(dtype=pred.dtype, device=pred.device)
    err = target.unsqueeze(-1) - pred
    return torch.maximum(q * err, (q - 1.0) * err).mean()


def forecast_loss(
    point: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor | None = None,
    quantile_levels: Sequence[float] | torch.Tensor | None = None,
    mae_weight: float = 1.0,
    mse_weight: float = 0.0,
    pinball_weight: float = 1.0,
) -> torch.Tensor:
    t = target.to(dtype=point.dtype)
    loss = mae_weight * (point - t).abs().mean()
    if mse_weight > 0:
        loss = loss + mse_weight * F.mse_loss(point, t)
    if (
        pinball_weight > 0
        and quantiles is not None
        and quantile_levels is not None
        and len(list(quantile_levels)) > 0
    ):
        levels = torch.as_tensor(list(quantile_levels), device=point.device, dtype=point.dtype)
        loss = loss + pinball_weight * pinball_loss(quantiles, t, levels)
    return loss
