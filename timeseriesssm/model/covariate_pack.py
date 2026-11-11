"""Pack targets / past-only / past-future covariates and lookahead tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .patch_embed import n_patches_for, pad_time, time_to_patches


@dataclass
class PackedWindow:
    """Channel-stacked lookback and horizon tensors plus role masks."""

    lookback: torch.Tensor  # [B, T_pad, C_all]
    horizon: torch.Tensor  # [B, H_pad, C_all]  (zeros where unknown)
    visible_horizon: torch.Tensor  # [B, H_pad, C_all] bool
    target_index: torch.Tensor  # [C_tgt] long
    past_index: torch.Tensor
    future_index: torch.Tensor
    n_target: int
    n_past: int
    n_future: int
    lookback_pad: int
    horizon_pad: int


def _as_3d(x: torch.Tensor | None, batch: int, time: int, name: str) -> torch.Tensor | None:
    if x is None:
        return None
    if x.dim() == 2:
        x = x.unsqueeze(-1)
    if x.dim() != 3:
        raise ValueError(f"{name} must have shape [B, T, C].")
    if x.shape[0] != batch or x.shape[1] != time:
        raise ValueError(f"{name} batch/time mismatch: {tuple(x.shape)} vs B={batch} T={time}.")
    return x


def pack_window(
    past_targets: torch.Tensor,
    horizon: int,
    *,
    past_covariates: torch.Tensor | None = None,
    future_covariates: torch.Tensor | None = None,
    patch_len: int = 32,
) -> PackedWindow:
    """
    Stack channels as [targets | past-only | past-future].

    Horizon slots for targets and past-only covariates are zeros and not visible.
    Past-future covariates must provide ``future_covariates`` of length ``horizon``.
    """
    if past_targets.dim() == 2:
        past_targets = past_targets.unsqueeze(-1)
    if past_targets.dim() != 3:
        raise ValueError("past_targets must have shape [B, T] or [B, T, C].")
    b, t, c_tgt = past_targets.shape
    past_covariates = _as_3d(past_covariates, b, t, "past_covariates")
    c_past = 0 if past_covariates is None else int(past_covariates.shape[2])
    c_fut = 0
    if future_covariates is not None:
        if future_covariates.dim() == 2:
            future_covariates = future_covariates.unsqueeze(-1)
        if future_covariates.shape[0] != b or future_covariates.shape[1] != int(horizon):
            raise ValueError("future_covariates must have shape [B, horizon, C_fut].")
        c_fut = int(future_covariates.shape[2])

    parts_lb = [past_targets]
    if past_covariates is not None:
        parts_lb.append(past_covariates)
    if c_fut > 0:
        # Past of future-known covariates is not always supplied; fill zeros
        # unless past_covariates already included them. Callers should pass
        # lookback of known dynamic series via ``future_lookback`` if needed.
        parts_lb.append(past_targets.new_zeros(b, t, c_fut))
    lookback = torch.cat(parts_lb, dim=-1)

    hor_tgt = past_targets.new_zeros(b, int(horizon), c_tgt)
    hor_past = past_targets.new_zeros(b, int(horizon), c_past) if c_past else None
    hor_parts = [hor_tgt]
    vis_parts = [torch.zeros(b, int(horizon), c_tgt, dtype=torch.bool, device=past_targets.device)]
    if hor_past is not None:
        hor_parts.append(hor_past)
        vis_parts.append(torch.zeros(b, int(horizon), c_past, dtype=torch.bool, device=past_targets.device))
    if c_fut > 0 and future_covariates is not None:
        hor_parts.append(future_covariates.to(dtype=past_targets.dtype, device=past_targets.device))
        vis_parts.append(torch.ones(b, int(horizon), c_fut, dtype=torch.bool, device=past_targets.device))
    horizon_raw = torch.cat(hor_parts, dim=-1)
    visible = torch.cat(vis_parts, dim=-1)

    lookback, lb_pad = pad_time(lookback, patch_len)
    horizon_raw, hz_pad = pad_time(horizon_raw, patch_len)
    if hz_pad:
        visible = F.pad(visible, (0, 0, 0, hz_pad), value=False)

    n_all = lookback.shape[-1]
    device = lookback.device
    target_index = torch.arange(c_tgt, device=device)
    past_index = torch.arange(c_tgt, c_tgt + c_past, device=device)
    future_index = torch.arange(c_tgt + c_past, n_all, device=device)
    return PackedWindow(
        lookback=lookback,
        horizon=horizon_raw,
        visible_horizon=visible,
        target_index=target_index,
        past_index=past_index,
        future_index=future_index,
        n_target=c_tgt,
        n_past=c_past,
        n_future=c_fut,
        lookback_pad=lb_pad,
        horizon_pad=hz_pad,
    )


def pack_window_with_future_lookback(
    past_targets: torch.Tensor,
    horizon: int,
    *,
    past_covariates: torch.Tensor | None = None,
    future_covariates: torch.Tensor | None = None,
    future_lookback: torch.Tensor | None = None,
    patch_len: int = 32,
) -> PackedWindow:
    """Like ``pack_window`` but fills past-future lookback from ``future_lookback``."""
    packed = pack_window(
        past_targets,
        horizon,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        patch_len=patch_len,
    )
    if future_lookback is None or packed.n_future == 0:
        return packed
    if future_lookback.dim() == 2:
        future_lookback = future_lookback.unsqueeze(-1)
    fl, _ = pad_time(future_lookback.to(dtype=packed.lookback.dtype, device=packed.lookback.device), patch_len)
    if fl.shape[1] != packed.lookback.shape[1] or fl.shape[2] != packed.n_future:
        raise ValueError("future_lookback must match lookback time and C_fut after padding.")
    packed.lookback[:, :, packed.future_index] = fl
    return packed


def lookahead_from_future(
    future_horizon_patches: torch.Tensor,
    n_ctx_patches: int,
    patch_len: int,
) -> torch.Tensor:
    """
    TimesFM-3-style lookahead: each context patch sees a summary of known future.

    Args:
        future_horizon_patches: [B, P_hor, C_fut, patch_len] visible future-cov patches
    Returns:
        [B, P_ctx, C_fut, patch_len] repeated future mean patch
    """
    summary = future_horizon_patches.mean(dim=1, keepdim=True)
    if summary.shape[-1] != patch_len:
        summary = F.pad(summary, (0, patch_len - summary.shape[-1]))[..., :patch_len]
    return summary.expand(-1, n_ctx_patches, -1, -1)


def split_target_horizon(horizon_values: torch.Tensor, n_target: int) -> torch.Tensor:
    return horizon_values[:, :, :n_target]


def patches_for_role(packed: PackedWindow, patch_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    lb = time_to_patches(packed.lookback, patch_len)
    hz = time_to_patches(packed.horizon, patch_len)
    return lb, hz


def visible_patch_mask(packed: PackedWindow, patch_len: int) -> torch.Tensor:
    """[B, P_ctx+P_hor, C] — context fully visible; horizon follows ``visible_horizon``."""
    b, t_lb, c = packed.lookback.shape
    p_ctx = t_lb // patch_len
    vis_ctx = torch.ones(b, p_ctx, c, dtype=torch.bool, device=packed.lookback.device)
    vis_h = packed.visible_horizon.reshape(b, packed.horizon.shape[1] // patch_len, patch_len, c).any(dim=2)
    return torch.cat([vis_ctx, vis_h], dim=1)


def n_horizon_patches(horizon: int, patch_len: int) -> int:
    return n_patches_for(horizon, patch_len)
