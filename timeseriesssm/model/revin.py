"""Reversible instance normalization per series / channel."""

from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Normalize each channel over the lookback time axis; invert after the head."""

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def _stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return mean, std

    def normalize(self, x: torch.Tensor,
                  channel_index: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, C] (or the channel subset ``channel_index`` of the configured channels)
        Returns:
            normalized x, mean, std (mean/std kept for invert; affine is applied after).
        """
        mean, std = self._stats(x)
        y = (x - mean) / std
        if self.affine and self.weight is not None and self.bias is not None:
            w, b = self.weight, self.bias
            if channel_index is not None:
                w, b = w[channel_index], b[channel_index]
            y = y * w.view(1, 1, -1) + b.view(1, 1, -1)
        return y, mean, std

    def denormalize(
        self,
        y: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        channel_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Invert RevIN. ``y`` is [B, H, C] or [B, H, C, Q]; stats from lookback."""
        if self.affine and self.weight is not None and self.bias is not None:
            w = self.weight
            b = self.bias
            if channel_index is not None:
                w = w[channel_index]
                b = b[channel_index]
            view = (1, 1, -1, 1) if y.dim() == 4 else (1, 1, -1)
            y = (y - b.view(*view[: y.dim()])) / w.view(*view[: y.dim()]).clamp_min(self.eps)
        if channel_index is not None and mean.shape[-1] != y.shape[2]:  # stats of the full set -> subset them
            mean = mean.index_select(-1, channel_index)
            std = std.index_select(-1, channel_index)
        if y.dim() == 4:
            mean = mean.unsqueeze(-1)
            std = std.unsqueeze(-1)
        return y * std + mean
