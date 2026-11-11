"""TimeSeriesSSM orchestrator: CPM decode, point+quantile heads, unified predict."""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .channel_graph import ChannelGraphEncoder, ChannelGraphMethod
from .covariate_pack import (
    PackedWindow,
    lookahead_from_future,
    pack_window_with_future_lookback,
    patches_for_role,
    visible_patch_mask,
)
from .nhits_cascade import NHiTSCascade
from .heads import (
    DEFAULT_QUANTILES,
    ClassifyHead,
    FlattenForecastHead,
    ForecastHead,
    ImputeHead,
    forecast_loss,
)
from .patch_embed import HorizonMaskTokens, PatchEmbed, patches_to_time
from .revin import RevIN
from .series_tokens import SeriesTokenMixer
from .time_encoding import TimeEncoding, calendar_from_unix, freq_id_from_name
from .timeseries_block import TimeSeriesSSMBlock

TimeMix = Literal["causal_factorized", "causal_decay", "off"]
HeadKind = Literal["mask_tokens", "flatten"]
CascadeKind = Literal["single", "nhits"]
ChannelMix = Literal["gssc", "off"]


_ALPHA_SPAN = 0.999  # adaptive alpha = 1 + 0.999 * sigmoid(raw), i.e. in [1, 1.999]
_LOGZ_TABLE: dict = {}


def _barron_rho(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Barron's general robust loss rho(x, alpha, 1) for alpha in [1, 2): |a-2|/a * ((x^2/|a-2| + 1)^(a/2) - 1)."""
    b = (2.0 - alpha).clamp_min(1e-6)
    return b / alpha * ((x * x / b + 1.0).pow(alpha / 2.0) - 1.0)


def _log_partition(alpha: torch.Tensor) -> torch.Tensor:
    """log Z(alpha) = log of the integral of exp(-rho(x, alpha, 1)) over x, for alpha in [1, 2), by linear interpolation
    in a table computed once by numerical integration (trapezoid, |x| <= 80; tails at alpha >= 1 decay at least like
    exp(-|x|)). Differentiable in alpha."""
    key = (alpha.device, alpha.dtype)
    if key not in _LOGZ_TABLE:
        grid = torch.linspace(1.0, 1.0 + _ALPHA_SPAN, 2001, dtype=torch.float64)
        x = torch.linspace(0.0, 80.0, 80001, dtype=torch.float64)
        vals = []
        for g in grid.split(64):
            f = torch.exp(-_barron_rho(x.view(1, -1), g.view(-1, 1)))
            vals.append(torch.log(2.0 * torch.trapezoid(f, x, dim=1)))
        _LOGZ_TABLE[key] = (grid.to(alpha.device, alpha.dtype), torch.cat(vals).to(alpha.device, alpha.dtype))
    grid, logz = _LOGZ_TABLE[key]
    pos = ((alpha - grid[0]) / (grid[1] - grid[0])).clamp(0, len(grid) - 1.000001)
    i0 = pos.floor().long()
    w = pos - i0.to(pos.dtype)
    return logz[i0] * (1 - w) + logz[(i0 + 1).clamp_max(len(grid) - 1)] * w


def barron_nll(err: torch.Tensor, alpha: torch.Tensor, log_c: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of Barron's adaptive distribution: rho(err / c, alpha) + log c + log Z(alpha)."""
    return _barron_rho(err * torch.exp(-log_c), alpha) + log_c + _log_partition(alpha)


def topk_frequency_part(x: torch.Tensor, k: int) -> torch.Tensor:
    """x [B, T, C] -> the sum of its k largest-amplitude rFFT components per (sample, channel), in the time domain."""
    spec = torch.fft.rfft(x.float(), dim=1)
    k = min(k, spec.shape[1])
    idx = spec.abs().topk(k, dim=1).indices
    mask = torch.zeros_like(spec.real).scatter_(1, idx, 1.0)
    return torch.fft.irfft(spec * mask, n=x.shape[1], dim=1).to(x.dtype)


def bures_wasserstein(z: torch.Tensor, z_hat: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Squared 2-Wasserstein distance between the Gaussians fitted to the rows of z and z_hat, both [B, D, C] with
    rows = (sample, channel) and dimension D, divided by D. Closed form (Bures):
    |mu - mu_hat|^2 + Tr(S) + Tr(S_hat) - 2 Tr((S^1/2 S_hat S^1/2)^1/2); computed in float64."""
    a = z.permute(0, 2, 1).reshape(-1, z.shape[1]).double()
    b = z_hat.permute(0, 2, 1).reshape(-1, z_hat.shape[1]).double()
    n, d = a.shape
    eye = torch.eye(d, dtype=a.dtype, device=a.device)
    mu, mu_h = a.mean(0), b.mean(0)
    s = (a - mu).T @ (a - mu) / max(1, n - 1) + eps * eye
    s_h = (b - mu_h).T @ (b - mu_h) / max(1, n - 1) + eps * eye
    lam, vec = torch.linalg.eigh(s.detach())  # the label covariance carries no gradient
    s_half = vec @ torch.diag(lam.clamp_min(0).sqrt()) @ vec.T
    m = s_half @ s_h @ s_half
    cross = torch.linalg.eigvalsh((m + m.T) / 2).clamp_min(0).add(eps).sqrt().sum()
    bw = (mu - mu_h).pow(2).sum() + torch.trace(s) + torch.trace(s_h) - 2 * cross
    return (bw / d).to(z.dtype)


def patch_structural_loss(pred: torch.Tensor, target: torch.Tensor, patch: int, eps: float = 1e-5) -> torch.Tensor:
    """Per horizon patch (length ``patch``; the last one is left-padded by repeating its first step): mean of
    (1 - Pearson correlation) + |std(pred) - std(target)| + |mean(pred) - mean(target)| over patches and channels."""
    h = pred.shape[1]
    n = -(-h // patch)
    pad = n * patch - h
    if pad:
        pred = torch.cat([pred[:, :1].expand(-1, pad, -1), pred], 1)
        target = torch.cat([target[:, :1].expand(-1, pad, -1), target], 1)
    p = pred.reshape(pred.shape[0], n, patch, -1)
    t = target.reshape(target.shape[0], n, patch, -1)
    pm, tm = p.mean(2, keepdim=True), t.mean(2, keepdim=True)
    pc, tc = p - pm, t - tm
    ps, ts = pc.pow(2).mean(2).add(eps).sqrt(), tc.pow(2).mean(2).add(eps).sqrt()
    corr = (pc * tc).mean(2) / (ps * ts)
    return (1 - corr).mean() + (ps - ts).abs().mean() + (pm - tm).abs().mean()


def haar_coeffs(x: torch.Tensor, levels: int) -> torch.Tensor:
    """Multi-level orthonormal Haar transform along dim 1 of x [B, H, C] -> all coefficients concatenated on dim 1
    (details of every level + the final approximation). An odd length is padded by repeating the last step."""
    out = []
    a = x
    for _ in range(max(1, levels)):
        if a.shape[1] < 2:
            break
        if a.shape[1] % 2:
            a = torch.cat([a, a[:, -1:]], dim=1)
        ev, od = a[:, 0::2], a[:, 1::2]
        out.append((ev - od) / math.sqrt(2.0))
        a = (ev + od) / math.sqrt(2.0)
    out.append(a)
    return torch.cat(out, dim=1)


def _dct_lowpass(n: int, frac: float) -> torch.Tensor:
    """[n, n] projection onto the first ceil(frac * n) orthonormal DCT-II basis vectors (a smooth low-pass that,
    unlike rFFT truncation, does not assume the window wraps around)."""
    k = max(1, min(n, int(math.ceil(frac * n))))
    i = torch.arange(n, dtype=torch.float64)
    f = torch.arange(k, dtype=torch.float64)
    basis = torch.cos(math.pi / n * (i[None, :] + 0.5) * f[:, None])  # [k, n]
    basis = basis / basis.norm(dim=1, keepdim=True)
    return (basis.T @ basis).float()


@dataclass
class TimeSeriesSSMConfig:
    hidden_dim: int = 64
    rank_dim: int = 48
    pe_dim: int = 12
    pe_num_filters: int = 4
    num_blocks: int = 2
    dropout: float = 0.05
    patch_len: int = 32
    use_revin: bool = True
    use_contiguous_patch_mask: bool = True
    time_mix: TimeMix = "causal_factorized"
    # causal_decay only: backward scan over the observed lookback (horizon still reads forward only)
    time_context_bidirectional: bool = False
    # "flatten" = PatchTST-style linear head over context tokens; needs context_len/horizon_len
    head: HeadKind = "mask_tokens"
    context_len: int = 0
    horizon_len: int = 0
    # "nhits": hierarchical multi-rate stacks with backcast residuals + interpolated forecasts
    # (point forecasts only; needs context_len/horizon_len). "single": the original one-stack model.
    cascade: CascadeKind = "single"
    nhits_pool_kernels: tuple[int, ...] = (4, 2, 1)
    nhits_downsample: tuple[int, ...] = (8, 4, 1)
    nhits_blocks_per_stack: int = 1
    # recompute block activations in backward (exact gradients, ~1/3 slower; lowers peak VRAM)
    grad_checkpoint: bool = False
    # sample axis: 'off' = no cross-window mixing (NSSM-style); 'support' = queries read a labeled bank of
    # training windows via SupportSetSSM (cascade='nhits' only)
    sample_axis: Literal["off", "support"] = "off"
    channel_mix: ChannelMix = "gssc"
    channel_graph: ChannelGraphMethod = "lag_correlation"
    channel_graph_max_lag: int = 4
    # sign of each retained Laplacian eigenvector: as returned by the solver ("solver", earlier runs), or
    # "canonical" = largest-magnitude entry positive, so codes recomputed from permuted channels are permuted codes
    channel_graph_sign: Literal["solver", "canonical"] = "solver"
    # channel mixing kernel: "rank" = rank-R state with addresses W_g z (earlier runs, the reported models);
    # "spectral" = Eq. 4 of Huang, Miao & Li (2024) on static codes scaled by one global scalar (no LayerNorm),
    # invariant to the eigenvector signs (ablation; needs use_selective=False)
    channel_kernel: Literal["rank", "spectral", "attention", "star"] = "rank"
    time_pe: str = "learned"
    use_calendar: bool = True
    # per-window calendar codes (forecast independent of the other windows in the batch); off = batch-averaged
    calendar_per_window: bool = False
    use_freq_embed: bool = True
    use_channel_independent: bool = False
    use_selective: bool = True
    use_glu_ffn: bool = True
    ffn_mult: float = 2.0
    # feed-forward kind: 'glu' (default, sigmoid gate), 'swiglu' (SiLU gate, same params) or a
    # Kolmogorov-Arnold FFN ('cheby_kan' | 'bspline_kan')
    ffn_kind: Literal["glu", "swiglu", "cheby_kan", "bspline_kan"] = "glu"
    kan_width: int = 0  # 0 = hidden_dim
    kan_order: int = 3
    kan_grid: int = 5
    # N-HiTS only: 'all' stacks use ffn_kind, or only the 'last' (full-resolution) stack does (others GLU)
    kan_stacks: Literal["all", "last"] = "all"
    # Residual Cycle Forecasting (CycleNet, NeurIPS 2024): learnable per-channel periodic profile Q[W, C]
    # subtracted from the normalized lookback and added back on the horizon; 0 = off. N-HiTS and single-stack
    # (target-only, unpadded lookback) paths.
    # Channel-specific parameters: the model is then equivariant only if Q is permuted with the channels.
    rcf_cycle: int = 0
    use_channel_cross: bool = True
    cross_rank: int = 16
    feature_ssm_symmetric_kernel: bool = True
    num_residual_stages: int = 1
    use_window_icl: bool = False
    use_multi_period: bool = False
    period_kernel: int = 2
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES
    max_patches: int = 512
    static_dim: int = 0
    n_classes: int = 0
    mae_weight: float = 1.0
    mse_weight: float = 0.0
    pinball_weight: float = 1.0
    # FreDF (Wang et al., ICLR 2025): loss = (1 - a) * time-domain loss + a * mean |rFFT(pred) - rFFT(target)| over the
    # horizon axis. 0 = off (time-domain loss only).
    freq_loss_weight: float = 0.0
    # ---- linear path, channel blend and series branch (cascade="nhits" only); off by default ----
    # linear path: RLinear-style Linear(context_len -> horizon_len) on the RevIN-normalized lookback, shared across
    # channels; the deep (N-HiTS) forecast is added through a per-horizon-step gate initialized at deep_gate_init
    linear_path: bool = False
    deep_gate_init: float = 0.0
    # channel blend: scale on every block's channel GSSC + channel-cross updates (1 = full mixing, 0 = independent);
    # channel_mix_learn makes it a learned per-block scalar initialized at channel_mix_scale
    channel_mix_scale: float = 1.0
    channel_mix_learn: bool = False
    # series tokens (iTransformer/TimeXer-style): one whole-lookback token per channel, attention across channels;
    # feeds a gated forecast branch (gate init 0) and a zero-initialized context added to every stack's patch tokens
    series_tokens: bool = False
    series_dim: int = 128
    series_layers: int = 1
    series_heads: int = 4
    # channel interaction inside the series-token branch: "attention", "star" (SOFTS STAR series-core
    # with stochastic pooling, O(C)), or "both" (STAR then attention per layer); star_core_dim 0 = series_dim
    series_mixer: str = "attention"
    star_core_dim: int = 0
    # ---- further options (cascade="nhits" only); off by default ----
    # Time-o1 (Wang et al., NeurIPS 2025): L1 on the first round(label_pca_frac * H) principal components of the
    # horizon labels (projection from an SVD of the training label windows, set with set_label_basis); the loss
    # becomes (1 - freq_loss_weight - label_pca_weight) * time loss + ... * FreDF + label_pca_weight * Time-o1
    label_pca_weight: float = 0.0
    label_pca_frac: float = 0.5
    # low-pass deep head: the N-HiTS (deep) forecast is projected onto the first ceil(lowpass_frac * H) DCT-II basis
    # vectors of the horizon (smooth, no wrap-around ringing); the linear path / RCF / series branch are untouched
    lowpass_frac: float = 0.0
    # future-statistics head (SAN-style, Liu et al., NeurIPS 2023, simplified): from the lookback's per-slice
    # mean/std, an MLP predicts each horizon slice's mean shift and log-scale change relative to the RevIN
    # statistics; zero-initialized output, so it starts as plain RevIN. Requires use_revin.
    stat_head: bool = False
    stat_head_hidden: int = 64
    # auxiliary statistics-head loss (SAN-faithful): stat_head_aux > 0 supervises the predicted per-slice mean shift / log-scale with
    # the TRUE future slice statistics (SAN's statistics loss); stat_head_linear uses a zero-init linear map plus the
    # MLP behind a zero-init scalar gate (capacity only if it pays off) instead of the zero-init MLP alone
    stat_head_aux: float = 0.0
    stat_head_linear: bool = False
    # FAN (Ye et al., NeurIPS 2024): per window and channel, the top-fan_k rFFT components (by amplitude) of the
    # normalized lookback are removed from every forecaster's input; an MLP on [those components, the lookback]
    # forecasts their future, supervised (fan_aux) by the top-fan_k components of the normalized label window
    fan_k: int = 0
    fan_aux: float = 1.0
    fan_hidden: int = 64
    # wavelet-domain loss: L1 on the multi-level orthonormal Haar coefficients of the forecast error over the horizon
    # (time- and frequency-localized decorrelation, for non-stationary horizons); shares the loss budget with the
    # time loss: (1 - freq - pca - wavelet) * time + ... + wavelet_loss_weight * wavelet
    wavelet_loss_weight: float = 0.0
    wavelet_levels: int = 4
    # time-term horizon weighting: "arctan" (xPatch, Stitsyuk & Choi, AAAI 2025) weights step i by
    # -arctan(i) + pi/4 + 1, renormalized to mean 1 so the balance with the other loss terms is unchanged
    horizon_weight: str = "none"
    # time-term loss: "fixed" = mae_weight * L1 + mse_weight * MSE, "adaptive" = Barron's general adaptive
    # robust loss (CVPR 2019) as a negative log-likelihood, rho(x, alpha, c) + log c + log Z(alpha), with a learnable
    # shape alpha in [1, 2) (1 ~ smooth L1 / median, 2 ~ MSE / mean) and scale c per channel, alpha initialized at
    # adaptive_alpha_init. Horizon weights (arctan) multiply the per-step NLL.
    time_loss: str = "fixed"
    adaptive_alpha_init: float = 1.5
    # DistDF (Wang et al., ICLR 2026): squared Bures-Wasserstein distance between the batch Gaussians of the joint
    # [lookback, label] and [lookback, forecast] sequences (rows = sample x channel), divided by L + H
    distdf_weight: float = 0.0
    # patch-wise structural loss (Kudrat et al., ICML 2025; fixed equal sub-weights instead of the paper's
    # gradient-based weighting): per horizon patch of patch_len, (1 - Pearson corr) + |std diff| + |mean diff|
    ps_loss_weight: float = 0.0


class TimeSeriesSSM(nn.Module):
    """
    Unified forecasting model.

    ``predict(past_targets, ..., horizon, quantile_levels)`` returns
    point ``[B, H, C_tgt]`` and optional quantiles ``[B, H, C_tgt, Q]``.
    """

    def __init__(self, config: TimeSeriesSSMConfig | None = None) -> None:
        super().__init__()
        self.config = config or TimeSeriesSSMConfig()
        cfg = self.config
        self.time_z_dim = cfg.pe_dim * cfg.pe_num_filters
        self.channel_z_dim = cfg.pe_dim * cfg.pe_num_filters
        self.patch_embed = PatchEmbed(
            patch_len=cfg.patch_len,
            hidden_dim=cfg.hidden_dim,
            lookahead_dim=cfg.patch_len,
        )
        self.mask_tokens = HorizonMaskTokens(cfg.hidden_dim)
        self.time_encoding = TimeEncoding(
            z_dim=self.time_z_dim,
            max_patches=cfg.max_patches,
            mode=cfg.time_pe,
            use_calendar=cfg.use_calendar,
            use_freq_embed=cfg.use_freq_embed,
            per_window=cfg.calendar_per_window,
        )
        self.blocks = nn.ModuleList(
            [
                TimeSeriesSSMBlock(
                    hidden_dim=cfg.hidden_dim,
                    channel_z_dim=self.channel_z_dim,
                    time_z_dim=self.time_z_dim,
                    rank_dim=cfg.rank_dim,
                    use_selective=cfg.use_selective,
                    use_channel_ssm=cfg.channel_mix == "gssc" and not cfg.use_channel_independent,
                    use_time_ssm=cfg.time_mix in ("causal_factorized", "causal_decay"),
                    time_kernel=cfg.time_mix,
                    time_context_bidirectional=cfg.time_context_bidirectional,
                    feature_ssm_symmetric_kernel=cfg.feature_ssm_symmetric_kernel,
                    dropout=cfg.dropout,
                    use_glu_ffn=cfg.use_glu_ffn,
                    ffn_mult=cfg.ffn_mult,
                    ffn_kind=cfg.ffn_kind, kan_width=cfg.kan_width,
                    kan_order=cfg.kan_order, kan_grid=cfg.kan_grid,
                    use_channel_cross=cfg.use_channel_cross and not cfg.use_channel_independent,
                    cross_rank=cfg.cross_rank,
                    channel_kernel=cfg.channel_kernel, channel_num_eig=cfg.pe_dim,
                    channel_num_filters=cfg.pe_num_filters,
                )
                for _ in range(max(1, cfg.num_blocks))
            ]
        )
        n_q = len(cfg.quantile_levels)
        self.forecast_head = self._make_head(n_q)
        extra_stages = max(0, int(cfg.num_residual_stages) - 1)
        self.residual_blocks = nn.ModuleList(
            [
                TimeSeriesSSMBlock(
                    hidden_dim=cfg.hidden_dim,
                    channel_z_dim=self.channel_z_dim,
                    time_z_dim=self.time_z_dim,
                    rank_dim=cfg.rank_dim,
                    use_selective=cfg.use_selective,
                    use_channel_ssm=cfg.channel_mix == "gssc" and not cfg.use_channel_independent,
                    use_time_ssm=cfg.time_mix in ("causal_factorized", "causal_decay"),
                    time_kernel=cfg.time_mix,
                    time_context_bidirectional=cfg.time_context_bidirectional,
                    dropout=cfg.dropout,
                    use_glu_ffn=cfg.use_glu_ffn,
                    ffn_mult=cfg.ffn_mult,
                    ffn_kind=cfg.ffn_kind, kan_width=cfg.kan_width,
                    kan_order=cfg.kan_order, kan_grid=cfg.kan_grid,
                    use_channel_cross=cfg.use_channel_cross and not cfg.use_channel_independent,
                    cross_rank=cfg.cross_rank,
                    channel_kernel=cfg.channel_kernel, channel_num_eig=cfg.pe_dim,
                    channel_num_filters=cfg.pe_num_filters,
                )
                for _ in range(extra_stages)
            ]
        )
        self.residual_heads = nn.ModuleList([self._make_head(n_q) for _ in range(extra_stages)])
        self.impute_head = ImputeHead(cfg.hidden_dim, cfg.patch_len)
        self.classify_head = ClassifyHead(cfg.hidden_dim, cfg.n_classes) if cfg.n_classes > 0 else None
        self.static_film = nn.Linear(cfg.static_dim, 2 * cfg.hidden_dim) if cfg.static_dim > 0 else None
        self.period_down = nn.AvgPool1d(cfg.period_kernel, stride=cfg.period_kernel) if cfg.use_multi_period else None

        self.nhits: NHiTSCascade | None = None
        self.rcf_q = None
        self.linear_head = self.deep_gate = self.series = self.series_gate = None
        self.register_buffer("lowpass_proj", None, persistent=False)
        self.label_basis = None
        self.stat_mlp = self.stat_lin = self.stat_gate = None
        self.loss_alpha_raw = self.loss_log_c = None
        self.fan_in = self.fan_mix = self.fan_out = None
        self._stat_cache = self._fan_cache = None
        if (cfg.linear_path or cfg.series_tokens or cfg.channel_mix_scale != 1.0 or cfg.channel_mix_learn) \
                and cfg.cascade != "nhits":
            raise ValueError("linear_path, series_tokens and the channel blend need cascade='nhits'.")
        if cfg.sample_axis == "support" and cfg.cascade != "nhits":
            raise ValueError("sample_axis='support' is implemented for cascade='nhits' only.")
        if cfg.cascade == "nhits":
            if cfg.context_len <= 0 or cfg.horizon_len <= 0:
                raise ValueError("cascade='nhits' requires context_len and horizon_len in the config.")
            self.nhits = NHiTSCascade(
                context_len=cfg.context_len, horizon=cfg.horizon_len, patch_len=cfg.patch_len,
                pools=cfg.nhits_pool_kernels, ratios=cfg.nhits_downsample,
                blocks_per_stack=cfg.nhits_blocks_per_stack, hidden_dim=cfg.hidden_dim,
                channel_z_dim=self.channel_z_dim, time_z_dim=self.time_z_dim, dropout=cfg.dropout,
                block_kwargs=dict(
                    rank_dim=cfg.rank_dim, use_selective=cfg.use_selective,
                    use_channel_ssm=cfg.channel_mix == "gssc" and not cfg.use_channel_independent,
                    use_time_ssm=cfg.time_mix in ("causal_factorized", "causal_decay"),
                    feature_ssm_symmetric_kernel=cfg.feature_ssm_symmetric_kernel,
                    use_glu_ffn=cfg.use_glu_ffn, ffn_mult=cfg.ffn_mult, ffn_kind=cfg.ffn_kind,
                    kan_width=cfg.kan_width, kan_order=cfg.kan_order, kan_grid=cfg.kan_grid,
                    use_channel_cross=cfg.use_channel_cross and not cfg.use_channel_independent,
                    cross_rank=cfg.cross_rank, time_kernel=cfg.time_mix,
                    time_context_bidirectional=cfg.time_context_bidirectional,
                    channel_mix_scale=cfg.channel_mix_scale, channel_mix_learn=cfg.channel_mix_learn,
                    channel_kernel=cfg.channel_kernel, channel_num_eig=cfg.pe_dim,
                    channel_num_filters=cfg.pe_num_filters),
                grad_checkpoint=cfg.grad_checkpoint, use_support=cfg.sample_axis == "support",
                stack_overrides=([{"ffn_kind": "glu"}] * (len(cfg.nhits_pool_kernels) - 1) + [{}]
                                 if cfg.kan_stacks == "last" else None),
            )
            # linear path (deep forecast behind a per-step gate) and series tokens (gated forecast branch +
            # zero-initialized per-stack context); both off by default
            self.linear_head = nn.Linear(cfg.context_len, cfg.horizon_len) if cfg.linear_path else None
            self.deep_gate = (nn.Parameter(torch.full((cfg.horizon_len,), float(cfg.deep_gate_init)))
                              if cfg.linear_path else None)
            self.series = (SeriesTokenMixer(cfg.context_len, cfg.horizon_len, cfg.series_dim, cfg.series_layers,
                                            cfg.series_heads, cfg.dropout, len(cfg.nhits_pool_kernels),
                                            cfg.hidden_dim, mixer=cfg.series_mixer, core_dim=cfg.star_core_dim)
                           if cfg.series_tokens else None)
            self.series_gate = nn.Parameter(torch.zeros(cfg.horizon_len)) if cfg.series_tokens else None
            # further options
            if cfg.lowpass_frac > 0:
                self.register_buffer("lowpass_proj", _dct_lowpass(cfg.horizon_len, cfg.lowpass_frac), persistent=False)
            if cfg.stat_head:  # without RevIN it corrects the forecast about the same lookback statistics
                n_in = 2 * (-(-cfg.context_len // cfg.patch_len))
                self.stat_slices_out = -(-cfg.horizon_len // cfg.patch_len)
                self.stat_mlp = nn.Sequential(nn.Linear(n_in, cfg.stat_head_hidden), nn.GELU(),
                                              nn.Linear(cfg.stat_head_hidden, 2 * self.stat_slices_out))
                if cfg.stat_head_linear:  # zero-init linear map + MLP behind a zero-init gate
                    self.stat_lin = nn.Linear(n_in, 2 * self.stat_slices_out)
                    nn.init.zeros_(self.stat_lin.weight)
                    nn.init.zeros_(self.stat_lin.bias)
                    self.stat_gate = nn.Parameter(torch.zeros(()))
                else:
                    nn.init.zeros_(self.stat_mlp[-1].weight)
                    nn.init.zeros_(self.stat_mlp[-1].bias)
            if cfg.fan_k > 0:
                if not cfg.use_revin:
                    raise ValueError("fan_k > 0 is applied on the RevIN-normalized lookback; it needs use_revin=True.")
                L, Hh, hd = cfg.context_len, cfg.horizon_len, cfg.fan_hidden
                self.fan_in = nn.Linear(L, hd)
                self.fan_mix = nn.Linear(hd + L, 2 * hd)
                self.fan_out = nn.Linear(2 * hd, Hh)
            # the one-stack encoder/decoder is unused on this path: drop it so it holds no parameters
            self.blocks = nn.ModuleList()
            self.forecast_head = nn.Identity()
            self.patch_embed = nn.Identity()
            self.mask_tokens = nn.Identity()
            self.impute_head = nn.Identity()

        self.channel_encoder: ChannelGraphEncoder | None = None
        self.revin: RevIN | None = None
        self._n_channels: int | None = None
        self._n_target: int | None = None
        self._learned_z_channel: nn.Parameter | None = None

    def _make_head(self, n_q: int) -> nn.Module:
        cfg = self.config
        if cfg.head == "flatten":
            if cfg.context_len <= 0 or cfg.horizon_len <= 0:
                raise ValueError("head='flatten' requires context_len and horizon_len in the config.")
            pl = cfg.patch_len
            n_ctx = -(-cfg.context_len // pl)
            h_pad = -(-cfg.horizon_len // pl) * pl
            return FlattenForecastHead(cfg.hidden_dim, n_ctx, h_pad, n_quantiles=n_q, dropout=cfg.dropout)
        return ForecastHead(cfg.hidden_dim, cfg.patch_len, n_quantiles=n_q)

    def configure_channels(self, n_channels: int, n_target: int | None = None) -> None:
        """Build RevIN + channel graph for a known joint channel count."""
        n_channels = int(n_channels)
        # configured once per channel count; previously this also required RevIN, so with use_revin=False every
        # forward rebuilt the channel encoder (dropping the fitted graph and learned weights) and the RCF table
        if self._n_channels == n_channels:
            if n_target is not None:
                self._n_target = int(n_target)
            return
        self._n_channels = n_channels
        self._n_target = int(n_target) if n_target is not None else n_channels
        cfg = self.config
        self.revin = RevIN(n_channels) if cfg.use_revin else None
        # zero init: an RCF model starts as exactly the plain model
        self.rcf_q = nn.Parameter(torch.zeros(cfg.rcf_cycle, n_channels)) if cfg.rcf_cycle > 0 else None
        if cfg.time_loss == "adaptive":  # Barron adaptive loss: per-channel shape (alpha) and scale (c)
            u = (min(max(cfg.adaptive_alpha_init, 1.0 + 1e-3), 1.998) - 1.0) / _ALPHA_SPAN
            self.loss_alpha_raw = nn.Parameter(torch.full((n_channels,), math.log(u / (1.0 - u))))
            self.loss_log_c = nn.Parameter(torch.zeros(n_channels))
        elif cfg.time_loss != "fixed":
            raise ValueError(f"unknown time_loss {cfg.time_loss!r}")
        if hasattr(self, "_learned_z_channel"):
            delattr(self, "_learned_z_channel")
        if n_channels >= 2 and cfg.channel_mix == "gssc" and not cfg.use_channel_independent:
            self.channel_encoder = ChannelGraphEncoder(
                num_channels=n_channels,
                pe_dim=cfg.pe_dim,
                num_filters=cfg.pe_num_filters,
                method=cfg.channel_graph,
                max_lag=cfg.channel_graph_max_lag,
                dropout=cfg.dropout,
                sign=cfg.channel_graph_sign,
                code_norm="global_rms" if cfg.channel_kernel == "spectral" else "layer",
            )
            self._learned_z_channel = None
        else:
            self.channel_encoder = None
            self._learned_z_channel = nn.Parameter(torch.zeros(max(1, n_channels), self.channel_z_dim))

    def set_channel_adjacency(self, adj: torch.Tensor) -> None:
        """Known channel graph for channel_graph='provided' (applied at fit_channel_graph)."""
        self._provided_channel_adj = torch.as_tensor(adj, dtype=torch.float32)

    def fit_channel_graph(self, train_series: torch.Tensor) -> None:
        """``train_series`` is [T, C] or [B, T, C] on the joint packed channels."""
        if train_series.dim() == 2:
            c = int(train_series.shape[1])
        else:
            c = int(train_series.shape[-1])
        self.configure_channels(c, n_target=self._n_target or c)
        if self.channel_encoder is not None:
            adj = getattr(self, "_provided_channel_adj", None)
            if self.config.channel_graph == "provided" and adj is not None:
                self.channel_encoder.set_adjacency(adj)
            self.channel_encoder.fit(train_series)

    def _z_channel(self, n_channels: int, device: torch.device) -> torch.Tensor:
        if self.channel_encoder is not None and self.channel_encoder.is_fitted():
            return self.channel_encoder.encodings().to(device=device)
        if self._learned_z_channel is not None:
            z = self._learned_z_channel
            if z.shape[0] < n_channels:
                z = F.pad(z, (0, 0, 0, n_channels - z.shape[0]))
            return z[:n_channels].to(device=device)
        return torch.zeros(n_channels, self.channel_z_dim, device=device)

    def _apply_static(self, h: torch.Tensor, static: torch.Tensor | None) -> torch.Tensor:
        if static is None or self.static_film is None:
            return h
        gb = self.static_film(static.to(dtype=h.dtype, device=h.device))
        gamma, beta = gb.chunk(2, dim=-1)
        return h * (1.0 + gamma.view(h.shape[0], 1, 1, -1)) + beta.view(h.shape[0], 1, 1, -1)

    def _encode_tokens(
        self,
        packed: PackedWindow,
        *,
        timestamps: torch.Tensor | None,
        freq: str | None,
        static_covariates: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        cfg = self.config
        pl = cfg.patch_len
        lb_p, hz_p = patches_for_role(packed, pl)
        p_ctx = lb_p.shape[1]
        p_hor = hz_p.shape[1]
        b, _, c, _ = lb_p.shape

        lookahead = None
        if packed.n_future > 0:
            fut_hz = hz_p[:, :, packed.future_index, :]
            la_fut = lookahead_from_future(fut_hz, p_ctx, pl)
            lookahead = lb_p.new_zeros(b, p_ctx, c, pl)
            lookahead[:, :, packed.future_index, :] = la_fut

        ctx_h = self.patch_embed(lb_p, lookahead=lookahead)
        vis_hz = packed.visible_horizon.reshape(b, p_hor, pl, c).any(dim=2)
        hz_h = self.mask_tokens.expand(b, p_hor, c)
        if packed.n_future > 0:
            fut_tokens = self.patch_embed(hz_p[:, :, packed.future_index, :], lookahead=None)
            hz_h = hz_h.clone()
            fut_idx = packed.future_index
            sel = vis_hz[:, :, fut_idx]
            # write visible future-cov tokens
            full_fut = hz_h[:, :, fut_idx, :].clone()
            full_fut = torch.where(sel.unsqueeze(-1), fut_tokens, full_fut)
            hz_h = hz_h.clone()
            hz_h[:, :, fut_idx, :] = full_fut

        tokens = torch.cat([ctx_h, hz_h], dim=1)
        tokens = self._apply_static(tokens, static_covariates)

        cal = None
        if timestamps is not None and cfg.use_calendar:
            ts = timestamps
            if ts.dim() == 2:
                need = packed.lookback.shape[1] + packed.horizon.shape[1] - packed.lookback_pad - packed.horizon_pad
                if ts.shape[1] >= packed.lookback.shape[1] - packed.lookback_pad:
                    # use available timestamps; pad to lookback+horizon if short
                    if ts.shape[1] < need:
                        last = ts[:, -1:]
                        extra = last.expand(-1, need - ts.shape[1])
                        ts = torch.cat([ts, extra], dim=1)
                    cal_all = calendar_from_unix(ts[:, : need + packed.lookback_pad + packed.horizon_pad], pl)
                    cal = cal_all
            if cal is not None and cal.shape[1] != tokens.shape[1]:
                if cal.shape[1] > tokens.shape[1]:
                    cal = cal[:, : tokens.shape[1]]
                else:
                    cal = F.pad(cal, (0, 0, 0, tokens.shape[1] - cal.shape[1]))

        device = tokens.device
        z_time = self.time_encoding(
            tokens.shape[1],
            calendar_features=cal,
            freq_id=freq_id_from_name(freq),
            device=device,
        )
        z_ch = self._z_channel(c, device)
        vis = visible_patch_mask(packed, pl)
        return tokens, z_ch, z_time, p_ctx, p_hor

    def _run_stack(
        self,
        tokens: torch.Tensor,
        z_ch: torch.Tensor,
        z_time: torch.Tensor,
        visible: torch.Tensor,
        blocks: nn.ModuleList,
        n_context: int | None = None,
    ) -> torch.Tensor:
        h = tokens
        for block in blocks:
            if self.config.grad_checkpoint and torch.is_grad_enabled():
                h = checkpoint(block, h, z_ch, z_time, visible=visible, n_context=n_context,
                               use_reentrant=False)
            else:
                h = block(h, z_ch, z_time, visible=visible, n_context=n_context)
        if self.period_down is not None:
            # [B, P, C, H] -> pool over P
            b, p, c, d = h.shape
            x = h.permute(0, 2, 3, 1).reshape(b * c, d, p)
            down = self.period_down(x)
            up = F.interpolate(down, size=p, mode="linear", align_corners=False)
            h = h + up.reshape(b, c, d, p).permute(0, 3, 1, 2)
        return h

    def _decode_horizon(
        self,
        tokens: torch.Tensor,
        p_ctx: int,
        p_hor: int,
        packed: PackedWindow,
        mean: torch.Tensor | None,
        std: torch.Tensor | None,
        quantile_levels: Sequence[float],
        head: ForecastHead,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(head, FlattenForecastHead):
            point_all, q_all = head(tokens[:, :p_ctx])
        else:
            point_all, q_all = head(tokens[:, p_ctx : p_ctx + p_hor])
        h_pad = packed.horizon.shape[1]
        point_all = point_all[:, :h_pad]
        tgt = packed.target_index
        point = point_all[:, :, tgt]
        quantiles = None
        if q_all is not None and len(quantile_levels) > 0:
            q_all = q_all[:, :h_pad]
            quantiles = q_all[:, :, tgt, : len(quantile_levels)]
        # drop time padding on the horizon
        if packed.horizon_pad:
            cut = packed.horizon.shape[1] - packed.horizon_pad
            point = point[:, :cut]
            if quantiles is not None:
                quantiles = quantiles[:, :cut]
        if self.revin is not None and mean is not None and std is not None:
            point = self.revin.denormalize(point, mean, std, channel_index=tgt)
            if quantiles is not None:
                quantiles = self.revin.denormalize(quantiles, mean, std, channel_index=tgt)
        return point, quantiles

    def forward_window(
        self,
        past_targets: torch.Tensor,
        horizon: int,
        *,
        past_covariates: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_lookback: torch.Tensor | None = None,
        static_covariates: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
        freq: str | None = None,
        quantile_levels: Sequence[float] | None = None,
        support: dict | None = None,
        cycle_phase: torch.Tensor | None = None,
        channel_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cfg = self.config
        if self.nhits is not None:
            if past_covariates is not None or future_covariates is not None:
                raise NotImplementedError("cascade='nhits' supports target-only point forecasts.")
            return self._forward_nhits(past_targets, horizon, timestamps=timestamps, freq=freq, support=support,
                                       cycle_phase=cycle_phase, channel_index=channel_index)
        if channel_index is not None:
            raise NotImplementedError("channel subsets (channel_index) need cascade='nhits'.")
        if support is not None:
            raise NotImplementedError("support banks require cascade='nhits'.")
        levels = tuple(quantile_levels) if quantile_levels is not None else cfg.quantile_levels
        packed = pack_window_with_future_lookback(
            past_targets,
            horizon,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            future_lookback=future_lookback,
            patch_len=cfg.patch_len,
        )
        self.configure_channels(packed.lookback.shape[-1], n_target=packed.n_target)

        lookback = packed.lookback
        horizon_vals = packed.horizon
        mean = std = None
        if self.revin is not None:
            lookback, mean, std = self.revin.normalize(lookback)
            if packed.n_future > 0:
                fut = packed.future_index
                hz = horizon_vals.clone()
                m = mean[:, :, fut]
                s = std[:, :, fut]
                vis = packed.visible_horizon[:, :, fut].to(dtype=lookback.dtype)
                hz[:, :, fut] = torch.where(
                    packed.visible_horizon[:, :, fut],
                    (horizon_vals[:, :, fut] - m) / s,
                    horizon_vals[:, :, fut],
                )
                packed.horizon = hz
            packed.lookback = lookback
        n_lb = past_targets.shape[1]
        if self.rcf_q is not None:
            if cycle_phase is None or cycle_phase.shape[1] < n_lb + horizon:
                raise ValueError("rcf_cycle > 0 needs cycle_phase [B, L + H] (phase of every lookback/horizon step).")
            if packed.lookback_pad or packed.n_past or packed.n_future:
                raise NotImplementedError("RCF on the single-stack path needs a target-only, unpadded lookback.")
            packed.lookback = packed.lookback - self.rcf_q[cycle_phase[:, :n_lb]]

        tokens, z_ch, z_time, p_ctx, p_hor = self._encode_tokens(
            packed, timestamps=timestamps, freq=freq, static_covariates=static_covariates
        )
        vis = visible_patch_mask(packed, cfg.patch_len)  # context visible; horizon future-cov visible
        h = self._run_stack(tokens, z_ch, z_time, vis, self.blocks, n_context=p_ctx)
        point, quantiles = self._decode_horizon(
            h, p_ctx, p_hor, packed, mean, std, levels, self.forecast_head
        )
        for rblock, rhead in zip(self.residual_blocks, self.residual_heads):
            hr = rblock(h, z_ch, z_time, visible=vis, n_context=p_ctx)
            d_point, d_q = self._decode_horizon(hr, p_ctx, p_hor, packed, mean, std, levels, rhead)
            point = point + 0.1 * d_point
            if quantiles is not None and d_q is not None:
                quantiles = quantiles + 0.1 * d_q
            h = hr
        # match requested horizon length
        point = point[:, :horizon]
        if self.rcf_q is not None:
            # point is already denormalized; RevIN's inverse is affine in y, so adding Q in normalized space equals
            # adding Q * std / w here (same as the N-HiTS path, which adds before denormalizing)
            cyc = self.rcf_q[cycle_phase[:, n_lb:n_lb + horizon]]
            if self.revin is not None:
                if self.revin.affine:
                    cyc = cyc / self.revin.weight.view(1, 1, -1).clamp_min(self.revin.eps)
                cyc = cyc * std
            point = point + cyc
        if quantiles is not None:
            quantiles = quantiles[:, :horizon]
            if quantiles.shape[-1] > len(levels):
                quantiles = quantiles[..., : len(levels)]
        if len(levels) == 0:
            quantiles = None
        return point, quantiles

    def _forward_nhits(self, past_targets: torch.Tensor, horizon: int, *, timestamps=None, freq=None,
                       support: dict | None = None, cycle_phase: torch.Tensor | None = None,
                       channel_index: torch.Tensor | None = None):
        """``support`` (sample axis): {"x": [K, L, C], "y": [K, H, C], "mask": [B, K] bool or None}.
        ``cycle_phase`` (RCF): [B, L + H] long, phase in [0, rcf_cycle) of every lookback and horizon step.
        ``channel_index`` (training on a channel subset, memory saving): [c] long indices into the configured
        channels; ``past_targets`` then holds only those channels. Every per-channel parameter (RevIN affine, RCF
        table, channel-graph encodings) is indexed with it; all mixing modules are channel-set equivariant."""
        cfg = self.config
        x = past_targets if past_targets.dim() == 3 else past_targets.unsqueeze(-1)
        c = x.shape[-1]
        if channel_index is None:
            self.configure_channels(c, n_target=c)
        elif self._n_channels is None:
            raise ValueError("channel_index needs the model configured on the full channel set first.")
        elif support is not None:
            raise NotImplementedError("channel subsets are not combined with a support bank.")
        mean = std = None
        stat_feat = self._stat_features(x) if self.stat_mlp is not None else None
        if self.revin is not None:
            x, mean, std = self.revin.normalize(x, channel_index=channel_index)
        elif stat_feat is not None:  # statistics head without RevIN: the lookback statistics, input left as it is
            mean = x.mean(1, keepdim=True)
            std = x.std(1, keepdim=True, unbiased=False).clamp_min(1e-5)
        lookback = x.shape[1]
        rcf_q = self.rcf_q
        if rcf_q is not None:
            if cycle_phase is None or cycle_phase.shape[1] < lookback + horizon:
                raise ValueError("rcf_cycle > 0 needs cycle_phase [B, L + H] (phase of every lookback/horizon step).")
            if channel_index is not None:
                rcf_q = rcf_q[:, channel_index]
            x = x - rcf_q[cycle_phase[:, :lookback]]  # [B, L, C]
        fan_fc = None
        if self.fan_in is not None:  # FAN: remove the instance's dominant frequencies, forecast them separately
            x_non = topk_frequency_part(x, cfg.fan_k)
            h_fan = F.relu(self.fan_in(x_non.transpose(1, 2)))
            h_fan = F.relu(self.fan_mix(torch.cat([h_fan, x.transpose(1, 2)], dim=-1)))
            fan_fc = self.fan_out(h_fan).transpose(1, 2)  # [B, H, C]
            x = x - x_non
        n_tok = self.nhits.n_tokens
        cal = None
        if timestamps is not None and cfg.use_calendar:
            ts = timestamps[:, : x.shape[1]]
            pad = n_tok * cfg.patch_len - ts.shape[1]
            if pad > 0:
                ts = torch.cat([ts[:, :1].expand(-1, pad), ts], dim=1)
            cal = calendar_from_unix(ts, cfg.patch_len)
        z_time = self.time_encoding(n_tok, calendar_features=cal, freq_id=freq_id_from_name(freq),
                                    device=x.device)
        z_ch = (self._z_channel(c, x.device) if channel_index is None
                else self._z_channel(self._n_channels, x.device)[channel_index])
        sx = sy = smask = None
        if support is not None and cfg.sample_axis == "support":
            sx, sy, smask = support["x"], support["y"], support.get("mask")
            if self.revin is not None:  # each support normalized by its own lookback, future by the same stats
                sx, s_mean, s_std = self.revin.normalize(sx)
                sy = (sy - s_mean) / s_std
                if self.revin.affine:
                    sy = sy * self.revin.weight.view(1, 1, -1) + self.revin.bias.view(1, 1, -1)
        series_fc = ctx = None
        if self.series is not None:
            series_fc, ctx = self.series(x)
        point = self.nhits(x, z_ch, z_time, support_x=sx, support_y=sy, support_mask=smask, ctx=ctx)
        if self.lowpass_proj is not None:  # smooth deep forecast (DCT low-pass over the full horizon_len)
            point = torch.einsum("hk,bkc->bhc", self.lowpass_proj.to(point.dtype), point)
        point = point[:, :horizon]
        if self.linear_head is not None:
            lin = self.linear_head(x.transpose(1, 2)).transpose(1, 2)[:, :horizon]
            point = lin + self.deep_gate[:horizon].view(1, -1, 1) * point
        if series_fc is not None:
            point = point + self.series_gate[:horizon].view(1, -1, 1) * series_fc[:, :horizon]
        if fan_fc is not None:
            point = point + fan_fc[:, :horizon]
            self._fan_cache = (fan_fc[:, :horizon], mean, std, channel_index)
        if rcf_q is not None:
            point = point + rcf_q[cycle_phase[:, lookback:lookback + horizon]]
        if self.revin is not None:
            point = self.revin.denormalize(point, mean, std, channel_index=channel_index)
        if stat_feat is not None:  # future statistics: mean shift and scale change per horizon slice
            if self.stat_lin is not None:
                d = self.stat_lin(stat_feat) + self.stat_gate * self.stat_mlp(stat_feat)
            else:
                d = self.stat_mlp(stat_feat)  # [B, C, 2 * n_out]
            n_out = self.stat_slices_out
            self._stat_cache = (d[..., :n_out], d[..., n_out:], mean, std)
            d_mu = d[..., :n_out].repeat_interleave(cfg.patch_len, dim=-1)[..., :horizon].transpose(1, 2)
            d_ls = d[..., n_out:].repeat_interleave(cfg.patch_len, dim=-1)[..., :horizon].transpose(1, 2)
            point = mean + d_mu * std + (point - mean) * torch.exp(d_ls.clamp(-3.0, 3.0))
        return point, None

    def _stat_features(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, L, C] raw lookback -> [B, C, 2 * n_slices]: per patch-length slice, (slice mean - mean) / std and
        log(slice std / std), relative to the whole-lookback (RevIN) statistics."""
        b, l, c = x.shape
        pl = self.config.patch_len
        n = -(-l // pl)
        pad = n * pl - l
        xs = torch.cat([x[:, :1].expand(-1, pad, -1), x], dim=1) if pad > 0 else x
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True, unbiased=False).clamp_min(1e-5)
        sl = xs.view(b, n, pl, c)
        m = (sl.mean(2) - mean) / std
        sd = torch.log(sl.std(2, unbiased=False).clamp_min(1e-5) / std)
        return torch.cat([m, sd], dim=1).transpose(1, 2)

    def _stat_aux_loss(self, target: torch.Tensor) -> torch.Tensor:
        """SAN statistics loss: predicted vs true future slice mean shift ((slice mean - mu) / sigma) and log-scale
        (log(slice std / sigma)), relative to the lookback statistics; slices of patch_len from the horizon start."""
        d_mu, d_ls, mean, std = self._stat_cache  # [B, C, m], [B, C, m], [B, 1, C], [B, 1, C]
        pl, m = self.config.patch_len, d_mu.shape[-1]
        h = target.shape[1]
        pad = m * pl - h
        y = torch.cat([target, target[:, -1:].expand(-1, pad, -1)], dim=1) if pad > 0 else target[:, : m * pl]
        sl = y.reshape(y.shape[0], m, pl, -1)  # [B, m, pl, C]
        mu_t = ((sl.mean(2) - mean) / std).transpose(1, 2)
        ls_t = torch.log(sl.std(2, unbiased=False).clamp_min(1e-5) / std).clamp(-3.0, 3.0).transpose(1, 2)
        return F.mse_loss(d_mu, mu_t) + F.mse_loss(d_ls, ls_t)

    def _fan_aux_loss(self, target: torch.Tensor) -> torch.Tensor:
        """FAN prior loss: the frequency branch's forecast vs the top-fan_k frequency part of the label window,
        expressed in the same (RevIN-normalized) space."""
        fan_fc, mean, std, cidx = self._fan_cache
        yn = (target - mean) / std
        if self.revin is not None and self.revin.affine:
            w, b = self.revin.weight, self.revin.bias
            if cidx is not None:
                w, b = w[cidx], b[cidx]
            yn = yn * w.view(1, 1, -1) + b.view(1, 1, -1)
        return F.mse_loss(fan_fc, topk_frequency_part(yn, self.config.fan_k).detach())

    def adaptive_loss_params(self, n_channels: int, channel_index: torch.Tensor | None = None):
        """(alpha [C] in [1, 2), log c [C]) of the adaptive time loss, for the given channel subset."""
        a, lc = self.loss_alpha_raw, self.loss_log_c
        if channel_index is not None:
            a, lc = a[channel_index], lc[channel_index]
        return 1.0 + _ALPHA_SPAN * torch.sigmoid(a.float()), lc.float()

    def set_label_basis(self, basis: torch.Tensor) -> None:
        """Time-o1 projection [H, K] (first K right-singular vectors of the standardized training label windows)."""
        self.label_basis = basis.detach().float()

    def predict(
        self,
        past_targets: torch.Tensor,
        past_covariates: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        static_covariates: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
        horizon: int = 96,
        quantile_levels: Sequence[float] | None = (),
        future_lookback: torch.Tensor | None = None,
        freq: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        levels: Sequence[float]
        if quantile_levels is None:
            levels = self.config.quantile_levels
        else:
            levels = tuple(quantile_levels)
        self.eval()
        with torch.no_grad():
            return self.forward_window(
                past_targets,
                horizon,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                future_lookback=future_lookback,
                static_covariates=static_covariates,
                timestamps=timestamps,
                freq=freq,
                quantile_levels=levels,
            )

    def loss(
        self,
        past_targets: torch.Tensor,
        future_targets: torch.Tensor,
        *,
        past_covariates: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_lookback: torch.Tensor | None = None,
        static_covariates: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
        freq: str | None = None,
        support: dict | None = None,
        cycle_phase: torch.Tensor | None = None,
        channel_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        horizon = int(future_targets.shape[1])
        point, quantiles = self.forward_window(
            past_targets,
            horizon,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            future_lookback=future_lookback,
            static_covariates=static_covariates,
            timestamps=timestamps,
            freq=freq,
            quantile_levels=self.config.quantile_levels,
            support=support,
            cycle_phase=cycle_phase,
            channel_index=channel_index,
        )
        target = future_targets
        if target.dim() == 2:
            target = target.unsqueeze(-1)
        target = target[:, : point.shape[1], : point.shape[2]]
        cfg = self.config
        if cfg.time_loss == "adaptive":
            err = point - target.to(dtype=point.dtype)
            alpha, log_c = self.adaptive_loss_params(point.shape[2], channel_index)
            nll = barron_nll(err.float(), alpha.view(1, 1, -1), log_c.view(1, 1, -1))
            if cfg.horizon_weight == "arctan":
                i = torch.arange(1, point.shape[1] + 1, device=point.device, dtype=nll.dtype)
                rho = -torch.atan(i) + math.pi / 4 + 1.0
                nll = nll * (rho / rho.mean()).view(1, -1, 1)
            loss = nll.mean().to(point.dtype)
        elif cfg.horizon_weight == "arctan":
            err = point - target.to(dtype=point.dtype)
            i = torch.arange(1, point.shape[1] + 1, device=point.device, dtype=point.dtype)
            rho = -torch.atan(i) + math.pi / 4 + 1.0
            rho = (rho / rho.mean()).view(1, -1, 1)
            loss = (rho * (cfg.mae_weight * err.abs() + cfg.mse_weight * err.pow(2))).mean()
        elif cfg.horizon_weight != "none":
            raise ValueError(f"unknown horizon_weight {cfg.horizon_weight!r}")
        else:
            loss = forecast_loss(
                point,
                target,
                quantiles=quantiles,
                quantile_levels=cfg.quantile_levels,
                mae_weight=cfg.mae_weight,
                mse_weight=cfg.mse_weight,
                pinball_weight=cfg.pinball_weight,
            )
        a, b, w = cfg.freq_loss_weight, cfg.label_pca_weight, cfg.wavelet_loss_weight
        dw, pw = cfg.distdf_weight, cfg.ps_loss_weight
        if a > 0 or b > 0 or w > 0 or dw > 0 or pw > 0:
            if a + b + w + dw + pw > 1.0:
                raise ValueError("the auxiliary loss weights (freq, label_pca, wavelet, distdf, ps) must sum to <= 1.")
            t = target.to(dtype=point.dtype)
            loss = (1.0 - a - b - w - dw - pw) * loss
            if dw > 0:
                x = past_targets if past_targets.dim() == 3 else past_targets.unsqueeze(-1)
                x = x[..., : point.shape[2]].to(point.dtype)
                loss = loss + dw * bures_wasserstein(torch.cat([x, t], 1), torch.cat([x, point], 1))
            if pw > 0:
                loss = loss + pw * patch_structural_loss(point.float(), t.float(), cfg.patch_len)
            if w > 0:
                loss = loss + w * haar_coeffs(point.float() - t.float(), self.config.wavelet_levels).abs().mean()
            if a > 0:
                loss = loss + a * (torch.fft.rfft(point.float(), dim=1) - torch.fft.rfft(t.float(), dim=1)).abs().mean()
            if b > 0:
                if self.label_basis is None:
                    raise ValueError("label_pca_weight > 0 needs set_label_basis() (training label SVD).")
                pb = self.label_basis.to(point.device)[: point.shape[1]]
                loss = loss + b * torch.einsum("bhc,hk->bkc", point.float() - t.float(), pb).abs().mean()
        # auxiliary supervision of the normalization candidates (outside the forecast-loss budget)
        if cfg.stat_head and cfg.stat_head_aux > 0 and self._stat_cache is not None:
            loss = loss + cfg.stat_head_aux * self._stat_aux_loss(target.to(point.dtype))
        if cfg.fan_k > 0 and cfg.fan_aux > 0 and self._fan_cache is not None:
            loss = loss + cfg.fan_aux * self._fan_aux_loss(target.to(point.dtype))
        return loss

    def impute(
        self,
        window: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct masked positions in ``window`` [B, T, C] (True = missing).
        """
        if window.dim() == 2:
            window = window.unsqueeze(-1)
        filled = window.masked_fill(mask.bool(), 0.0)
        horizon = max(self.config.patch_len, 1)
        packed = pack_window_with_future_lookback(filled, horizon, patch_len=self.config.patch_len)
        self.configure_channels(packed.lookback.shape[-1], n_target=packed.n_target)
        tokens, z_ch, z_time, p_ctx, _p_hor = self._encode_tokens(
            packed, timestamps=None, freq=None, static_covariates=None
        )
        vis = visible_patch_mask(packed, self.config.patch_len)
        h = self._run_stack(tokens, z_ch, z_time, vis, self.blocks, n_context=p_ctx)
        ctx = h[:, :p_ctx]
        recon = patches_to_time(self.impute_head(ctx))
        recon = recon[:, : window.shape[1], : window.shape[2]]
        return torch.where(mask.bool(), recon, window)

    def classify(self, window: torch.Tensor) -> torch.Tensor:
        if self.classify_head is None:
            raise RuntimeError("classify_head is not configured (n_classes=0).")
        if window.dim() == 2:
            window = window.unsqueeze(-1)
        packed = pack_window_with_future_lookback(window, self.config.patch_len, patch_len=self.config.patch_len)
        self.configure_channels(packed.lookback.shape[-1], n_target=packed.n_target)
        tokens, z_ch, z_time, _p_ctx, _p_hor = self._encode_tokens(
            packed, timestamps=None, freq=None, static_covariates=None
        )
        vis = visible_patch_mask(packed, self.config.patch_len)
        h = self._run_stack(tokens, z_ch, z_time, vis, self.blocks, n_context=_p_ctx)
        return self.classify_head(h)
