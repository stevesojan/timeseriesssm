"""Patch embedding and contiguous-horizon mask tokens."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_time(x: torch.Tensor, patch_len: int) -> tuple[torch.Tensor, int]:
    """Right-pad time dim of [B, T, C] so T is a multiple of ``patch_len``."""
    t = int(x.shape[1])
    pad = (patch_len - (t % patch_len)) % patch_len
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    return x, pad


def time_to_patches(x: torch.Tensor, patch_len: int) -> torch.Tensor:
    """[B, T, C] -> [B, P, C, patch_len]. T must be divisible by patch_len."""
    if x.shape[1] % patch_len != 0:
        raise ValueError("time length must be divisible by patch_len (pad first).")
    b, t, c = x.shape
    p = t // patch_len
    return x.reshape(b, p, patch_len, c).permute(0, 1, 3, 2).contiguous()


def patches_to_time(patches: torch.Tensor) -> torch.Tensor:
    """[B, P, C, patch_len] -> [B, T, C]."""
    b, p, c, pl = patches.shape
    return patches.permute(0, 1, 3, 2).reshape(b, p * pl, c)


class PatchEmbed(nn.Module):
    """Project each channel's length-``patch_len`` slice to ``hidden_dim``."""

    def __init__(self, patch_len: int = 32, hidden_dim: int = 64, lookahead_dim: int = 0) -> None:
        super().__init__()
        self.patch_len = int(patch_len)
        self.lookahead_dim = int(lookahead_dim)
        in_dim = self.patch_len + self.lookahead_dim
        self.proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, patches: torch.Tensor, lookahead: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            patches: [B, P, C, patch_len]
            lookahead: optional [B, P, C, lookahead_dim] (future-cov tokens)
        """
        if lookahead is not None:
            if lookahead.shape[-1] != self.lookahead_dim:
                raise ValueError("lookahead last dim must match lookahead_dim.")
            inp = torch.cat([patches, lookahead], dim=-1)
        elif self.lookahead_dim > 0:
            zeros = patches.new_zeros(*patches.shape[:3], self.lookahead_dim)
            inp = torch.cat([patches, zeros], dim=-1)
        else:
            inp = patches
        return self.proj(inp)


class HorizonMaskTokens(nn.Module):
    """Learned placeholder tokens for masked target / past-only horizon patches."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.token = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def expand(self, batch: int, n_patches: int, n_channels: int) -> torch.Tensor:
        return self.token.view(1, 1, 1, -1).expand(batch, n_patches, n_channels, -1)


def n_patches_for(length: int, patch_len: int) -> int:
    return int(math.ceil(length / patch_len))
