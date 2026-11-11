"""Patch-index, calendar, and frequency encodings for the time axis."""

from __future__ import annotations

import math
import torch
import torch.nn as nn

FREQ_TO_ID: dict[str, int] = {
    "S": 0,
    "T": 1,
    "min": 1,
    "H": 2,
    "h": 2,
    "D": 3,
    "W": 4,
    "M": 5,
    "ME": 5,
    "Q": 6,
    "QE": 6,
    "Y": 7,
    "YE": 7,
}


class TimeEncoding(nn.Module):
    """``z_time`` of width ``z_dim`` for each patch index, plus optional calendar/freq."""

    def __init__(
        self,
        z_dim: int,
        max_patches: int = 512,
        mode: str = "learned",
        use_calendar: bool = True,
        use_freq_embed: bool = True,
        n_freqs: int = 8,
        per_window: bool = False,
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.mode = str(mode)
        self.use_calendar = bool(use_calendar)
        self.use_freq_embed = bool(use_freq_embed)
        # per_window: every window keeps its own calendar code ([B, P, 1, Z] output), so a forecast does not depend
        # on the other windows of its batch; off = the batch-averaged code of earlier runs ([P, Z])
        self.per_window = bool(per_window)
        if self.mode == "learned":
            self.patch_embed = nn.Embedding(max_patches, z_dim)
        else:
            self.register_buffer(
                "sinusoid",
                _sinusoidal(max_patches, z_dim),
                persistent=False,
            )
            self.patch_embed = None
        cal_in = 8  # hour sin/cos, dow sin/cos, month sin/cos, plus 2 reserved
        self.cal_proj = nn.Linear(cal_in, z_dim) if use_calendar else None
        self.freq_embed = nn.Embedding(n_freqs, z_dim) if use_freq_embed else None

    def forward(
        self,
        n_patches: int,
        *,
        calendar_features: torch.Tensor | None = None,
        freq_id: int | torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Returns ``z_time`` of shape [P, Z].

        ``calendar_features`` if given is [B, P, cal_in] or [P, cal_in] and is
        averaged into a single [P, Z] addend (shared across the batch). With
        ``per_window`` and [B, P, cal_in] features, returns [B, P, 1, Z] instead
        (one code per window, broadcast over channels).
        """
        if self.mode == "learned" and self.patch_embed is not None:
            idx = torch.arange(n_patches, device=device or self.patch_embed.weight.device)
            z = self.patch_embed(idx)
        else:
            sinusoid = self.sinusoid
            z = sinusoid[:n_patches].to(device=device or sinusoid.device)

        per_window = False
        if self.cal_proj is not None and calendar_features is not None:
            cal = calendar_features
            per_window = self.per_window and cal.dim() == 3
            if cal.dim() == 3 and not per_window:
                cal = cal.mean(dim=0)
            z = z + self.cal_proj(cal.to(dtype=z.dtype, device=z.device))

        if self.freq_embed is not None and freq_id is not None:
            if isinstance(freq_id, int):
                fid = torch.tensor(freq_id, device=z.device, dtype=torch.long)
            else:
                fid = freq_id.to(device=z.device, dtype=torch.long).reshape(-1)[0]
            z = z + self.freq_embed(fid.clamp(0, self.freq_embed.num_embeddings - 1))
        return z.unsqueeze(2) if per_window else z


def _sinusoidal(n: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(n, dim)
    pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


def calendar_from_unix(ts: torch.Tensor, patch_len: int) -> torch.Tensor:
    """
    Approximate hour/dow/month features from unix seconds.

    Args:
        ts: [B, T] unix timestamps (or any seconds-like scale).
    Returns:
        [B, P, 8] patch-averaged cyclic features.
    """
    if ts.dim() != 2:
        raise ValueError("timestamps must have shape [B, T].")
    t = ts.to(dtype=torch.float32)
    hour = (t / 3600.0) % 24.0
    dow = (t / 86400.0) % 7.0
    month = ((t / 86400.0) / 30.437) % 12.0
    feats = torch.stack(
        [
            torch.sin(2 * math.pi * hour / 24.0),
            torch.cos(2 * math.pi * hour / 24.0),
            torch.sin(2 * math.pi * dow / 7.0),
            torch.cos(2 * math.pi * dow / 7.0),
            torch.sin(2 * math.pi * month / 12.0),
            torch.cos(2 * math.pi * month / 12.0),
            torch.sin(2 * math.pi * (t / 86400.0)),
            torch.cos(2 * math.pi * (t / 86400.0)),
        ],
        dim=-1,
    )
    b, length, f = feats.shape
    pad = (patch_len - (length % patch_len)) % patch_len
    if pad:
        feats = F_pad_time(feats, pad)
        length = length + pad
    p = length // patch_len
    return feats.reshape(b, p, patch_len, f).mean(dim=2)


def F_pad_time(x: torch.Tensor, pad: int) -> torch.Tensor:
    return torch.nn.functional.pad(x, (0, 0, 0, pad))


def freq_id_from_name(name: str | None) -> int:
    if not name:
        return 3
    return FREQ_TO_ID.get(str(name), FREQ_TO_ID.get(str(name).upper(), 3))


def default_calendar_dim() -> int:
    return 8
