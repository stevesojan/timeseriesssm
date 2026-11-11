"""Alternating causal-time + set-variate GSSC block."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..global_ssm import ChannelAttention, ChannelSTAR, GlobalFeatureSSM, SpectralGSSC
from ..selective_ssm import SelectivePositionUpdate
from .causal_ssm import CausalDecaySSM, CausalFactorizedSSM
from .kan_layers import KANFFN


class TimeSeriesSSMBlock(nn.Module):
    """Causal time mix, then permutation-equivariant channel GSSC, then GLU/cross."""

    def __init__(
        self,
        hidden_dim: int,
        channel_z_dim: int,
        time_z_dim: int,
        rank_dim: int,
        use_selective: bool = True,
        use_channel_ssm: bool = True,
        use_time_ssm: bool = True,
        feature_ssm_symmetric_kernel: bool = True,
        dropout: float = 0.0,
        use_glu_ffn: bool = True,
        ffn_mult: float = 2.0,
        use_channel_cross: bool = True,
        cross_rank: int = 16,
        time_kernel: str = "causal_factorized",
        time_context_bidirectional: bool = False,
        ffn_kind: str = "glu",
        kan_width: int = 0,
        kan_order: int = 3,
        kan_grid: int = 5,
        channel_mix_scale: float = 1.0,
        channel_mix_learn: bool = False,
        channel_kernel: str = "rank",
        channel_num_eig: int = 0,
        channel_num_filters: int = 0,
    ) -> None:
        super().__init__()
        self.use_selective = use_selective
        # channel blend: scale on the channel GSSC + cross updates; a fixed 0 skips them entirely
        if channel_mix_learn:
            self.mix_gate = nn.Parameter(torch.tensor(float(channel_mix_scale)))
        else:
            self.register_buffer("mix_gate", torch.tensor(float(channel_mix_scale)), persistent=False)
        self._mix_fixed = None if channel_mix_learn else float(channel_mix_scale)
        if self._mix_fixed == 0.0:
            use_channel_ssm, use_channel_cross = False, False
        self.use_channel_ssm = use_channel_ssm
        self.use_time_ssm = use_time_ssm

        self.selective = (
            SelectivePositionUpdate(z_dim=channel_z_dim, hidden_dim=hidden_dim, rank_dim=rank_dim)
            if use_selective
            else None
        )
        if time_kernel == "causal_decay":
            self.time_ssm = CausalDecaySSM(
                hidden_dim=hidden_dim,
                z_dim=time_z_dim,
                rank_dim=rank_dim,
                dropout=dropout,
                bidirectional_context=time_context_bidirectional,
            )
        else:
            self.time_ssm = CausalFactorizedSSM(
                hidden_dim=hidden_dim,
                z_dim=time_z_dim,
                rank_dim=rank_dim,
                dropout=dropout,
            )
        # channel_kernel: "rank" = rank-R state with addresses W_g z (the reported models); "spectral" = Eq. 4 of
        # Huang et al. (2024) on static codes with a global scale, sign-invariant (ablation; needs channel_num_eig/_filters)
        if channel_kernel == "spectral":
            if use_selective:
                raise ValueError("channel_kernel='spectral' takes static codes (use_selective=False).")
            if channel_num_eig * channel_num_filters != channel_z_dim:
                raise ValueError("channel_kernel='spectral' needs channel_num_eig * channel_num_filters == channel_z_dim.")
            self.channel_ssm = SpectralGSSC(hidden_dim=hidden_dim, num_eig=channel_num_eig,
                                            num_filters=channel_num_filters, dropout=dropout)
        elif channel_kernel in ("attention", "star"):  # content-only channel mixers in place of step 3 (ablation)
            if use_selective:
                raise ValueError(f"channel_kernel={channel_kernel!r} ignores the codes (use_selective=False).")
            self.channel_ssm = (ChannelAttention(hidden_dim, dropout=dropout) if channel_kernel == "attention"
                                else ChannelSTAR(hidden_dim, rank_dim, dropout=dropout))
        elif channel_kernel == "rank":
            self.channel_ssm = GlobalFeatureSSM(
                hidden_dim=hidden_dim,
                z_dim=channel_z_dim,
                rank_dim=rank_dim,
                symmetric_kernel=feature_ssm_symmetric_kernel,
                dropout=dropout,
            )
        else:
            raise ValueError(f"unknown channel_kernel {channel_kernel!r}")
        self.norm_time = nn.LayerNorm(hidden_dim)
        self.norm_channel = nn.LayerNorm(hidden_dim)
        self.resid_dropout = nn.Dropout(dropout)

        ffn_hidden = max(1, int(hidden_dim * ffn_mult))
        # KAN feed-forward (Chebyshev or B-spline edges) replaces the GLU when requested
        self.kan_ffn = (KANFFN(hidden_dim, ffn_kind, kan_width or hidden_dim, order=kan_order, grid=kan_grid,
                               dropout=dropout) if use_glu_ffn and ffn_kind in ("cheby_kan", "bspline_kan") else None)
        self.swiglu = ffn_kind == "swiglu"  # same layers as GLU, SiLU gate instead of sigmoid
        if use_glu_ffn and self.kan_ffn is None:
            self.norm_ffn = nn.LayerNorm(hidden_dim)
            self.ffn_in = nn.Linear(hidden_dim, 2 * ffn_hidden)
            self.ffn_out = nn.Linear(ffn_hidden, hidden_dim)
        else:
            self.norm_ffn = None
            self.ffn_in = None
            self.ffn_out = None

        if use_channel_cross:
            self.cross_u = nn.Linear(hidden_dim, cross_rank, bias=False)
            self.cross_v = nn.Linear(hidden_dim, cross_rank, bias=False)
            self.cross_out = nn.Linear(cross_rank, hidden_dim, bias=False)
            self.norm_cross = nn.LayerNorm(hidden_dim)
        else:
            self.cross_u = None
            self.cross_v = None
            self.cross_out = None
            self.norm_cross = None

    def _mix(self, h: torch.Tensor) -> torch.Tensor:
        return h if self._mix_fixed == 1.0 else h * self.mix_gate.to(h.dtype)

    def forward(
        self,
        x: torch.Tensor,
        z_channel: torch.Tensor,
        z_time: torch.Tensor,
        visible: torch.Tensor | None = None,
        n_context: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, P, C, H]
            z_channel: [C, Zc]
            z_time: [P, Zt] or [B, P, C, Zt]
            visible: optional causal-pool mask [B, P, C]
        """
        if x.dim() != 4:
            raise ValueError("x must have shape [B, P, C, H].")

        if self.use_time_ssm:
            if isinstance(self.time_ssm, CausalDecaySSM):
                h_t = self.time_ssm(x, z_time, visible=visible, n_context=n_context)
            else:
                h_t = self.time_ssm(x, z_time, visible=visible)
            x = self.norm_time(x + self.resid_dropout(h_t))
        else:
            x = self.norm_time(x)

        z_use = z_channel
        if self.use_channel_ssm and x.shape[2] > 1:
            if self.use_selective and self.selective is not None and z_channel.dim() == 2:
                z_use = self.selective(z_channel, x)
            h_c = self.channel_ssm(x, z_use)
            x = self.norm_channel(x + self.resid_dropout(self._mix(h_c)))
        else:
            x = self.norm_channel(x)

        if self.kan_ffn is not None:
            x = x + self.resid_dropout(self.kan_ffn(x))
        elif self.norm_ffn is not None and self.ffn_in is not None and self.ffn_out is not None:
            t = self.norm_ffn(x)
            if self.swiglu:
                a, g = self.ffn_in(t).chunk(2, dim=-1)
                t = a * F.silu(g)
            else:
                t = F.glu(self.ffn_in(t), dim=-1)
            t = self.ffn_out(t)
            x = x + self.resid_dropout(t)

        if self.cross_u is not None and self.cross_v is not None and self.cross_out is not None:
            u = self.cross_u(x)
            v = self.cross_v(x)
            g = u.mean(dim=2, keepdim=True)
            cross = self._mix(self.cross_out(v * g))
            if self.norm_cross is not None:
                x = self.norm_cross(x + self.resid_dropout(cross))
            else:
                x = x + self.resid_dropout(cross)
        return x
