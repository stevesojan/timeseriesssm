"""Correctness tests for the TimeSeriesSSM time-axis kernels, heads and channel graph (Stage 0)."""
import math
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timeseriesssm.model import TimeSeriesSSM, TimeSeriesSSMConfig  # noqa: E402
from timeseriesssm.model.causal_ssm import CausalDecaySSM, CausalFactorizedSSM  # noqa: E402
from timeseriesssm.model.timeseries_block import TimeSeriesSSMBlock  # noqa: E402

B, P, C, H, Z, R = 2, 7, 5, 16, 12, 8


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, P, C, H, generator=g), torch.randn(P, Z, generator=g)


def test_decay_reduces_to_prefix_sum_when_decay_is_one():
    x, z = _inputs()
    old = CausalFactorizedSSM(H, Z, R)
    new = CausalDecaySSM(H, Z, R)
    new.load_state_dict({**old.state_dict(), **{k: v for k, v in new.state_dict().items()
                                               if k.startswith(("dt_proj", "A_log"))}})
    with torch.no_grad():
        new.A_log.fill_(-50.0)  # exp(A_log) ~ 0 -> a_t = 1
    assert torch.allclose(old(x, z), new(x, z), atol=1e-5)


@pytest.mark.parametrize("bidirectional", [False, True])
def test_no_leakage_from_later_positions_or_horizon(bidirectional):
    torch.manual_seed(0)
    n_ctx = 4
    m = CausalDecaySSM(H, Z, R, bidirectional_context=bidirectional).eval()
    x, z = _inputs(1)
    vis = torch.ones(B, P, C, dtype=torch.bool)
    vis[:, n_ctx:] = False  # horizon tokens are not observed
    base = m(x, z, visible=vis, n_context=n_ctx)
    # perturbing horizon position t must not change any position < t
    for t in range(n_ctx, P):
        x2 = x.clone()
        x2[:, t] += 10.0 * torch.randn_like(x2[:, t])
        out = m(x2, z, visible=vis, n_context=n_ctx)
        assert torch.allclose(out[:, :t], base[:, :t], atol=1e-5), f"leak from position {t}"
    if not bidirectional:  # strictly causal inside the context as well
        x3 = x.clone()
        x3[:, 2] += 5.0
        assert torch.allclose(m(x3, z, visible=vis, n_context=n_ctx)[:, :2], base[:, :2], atol=1e-5)


def test_bidirectional_context_sees_whole_lookback():
    torch.manual_seed(0)
    m = CausalDecaySSM(H, Z, R, bidirectional_context=True).eval()
    x, z = _inputs(2)
    base = m(x, z, n_context=4)
    x2 = x.clone()
    x2[:, 3] += 5.0  # last context patch
    assert not torch.allclose(m(x2, z, n_context=4)[:, 0], base[:, 0])


@pytest.mark.parametrize("kernel", ["causal_factorized", "causal_decay"])
def test_block_is_channel_permutation_equivariant(kernel):
    torch.manual_seed(0)
    blk = TimeSeriesSSMBlock(H, Z, Z, R, time_kernel=kernel, time_context_bidirectional=True).eval()
    x, zt = _inputs(3)
    zc = torch.randn(C, Z)
    perm = torch.randperm(C)
    out = blk(x, zc, zt, n_context=4)
    out_p = blk(x[:, :, perm], zc[perm], zt, n_context=4)
    assert torch.allclose(out[:, :, perm], out_p, atol=1e-5)


@pytest.mark.parametrize("cfg_kwargs", [
    {},
    {"time_mix": "causal_decay"},
    {"time_mix": "causal_decay", "time_context_bidirectional": True},
    {"time_mix": "causal_decay", "time_context_bidirectional": True, "head": "flatten",
     "context_len": 96, "horizon_len": 48},
])
def test_forecast_shapes_and_backward(cfg_kwargs):
    torch.manual_seed(0)
    cfg = TimeSeriesSSMConfig(hidden_dim=16, rank_dim=8, patch_len=16, quantile_levels=(),
                              mse_weight=1.0, mae_weight=0.0, pinball_weight=0.0, **cfg_kwargs)
    m = TimeSeriesSSM(cfg)
    past, fut = torch.randn(3, 96, C), torch.randn(3, 48, C)
    m.fit_channel_graph(past.reshape(-1, C))
    point, q = m.predict(past, horizon=48)
    assert point.shape == (3, 48, C) and q is None
    m.train()
    loss = m.loss(past, fut)
    loss.backward()
    assert torch.isfinite(loss)


def test_flatten_head_requires_lengths():
    with pytest.raises(ValueError):
        TimeSeriesSSM(TimeSeriesSSMConfig(head="flatten"))


def test_provided_channel_graph():
    torch.manual_seed(0)
    cfg = TimeSeriesSSMConfig(hidden_dim=16, rank_dim=8, patch_len=16, quantile_levels=(),
                              channel_graph="provided")
    m = TimeSeriesSSM(cfg)
    ring = torch.zeros(C, C)
    for i in range(C):
        ring[i, (i + 1) % C] = ring[(i + 1) % C, i] = 1.0
    m.set_channel_adjacency(ring)
    m.fit_channel_graph(torch.randn(200, C))
    z = m._z_channel(C, torch.device("cpu"))
    assert z.shape[0] == C and torch.isfinite(z).all()
    point, _ = m.predict(torch.randn(2, 96, C), horizon=32)
    assert point.shape == (2, 32, C)


def test_nhits_cascade_forecast_and_backward():
    torch.manual_seed(0)
    cfg = TimeSeriesSSMConfig(hidden_dim=16, rank_dim=8, patch_len=16, quantile_levels=(), cascade="nhits",
                              context_len=96, horizon_len=48, time_mix="causal_decay",
                              time_context_bidirectional=True, mse_weight=1.0, mae_weight=0.0, pinball_weight=0.0)
    m = TimeSeriesSSM(cfg)
    assert len(m.blocks) == 0 and len(m.nhits.stacks) == 3
    past, fut = torch.randn(3, 96, C), torch.randn(3, 48, C)
    m.fit_channel_graph(past.reshape(-1, C))
    point, q = m.predict(past, horizon=48)
    assert point.shape == (3, 48, C) and q is None
    m.train()
    loss = m.loss(past, fut)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in m.nhits.parameters() if p.requires_grad)


def test_nhits_cascade_is_channel_permutation_equivariant():
    from timeseriesssm.model.nhits_cascade import NHiTSCascade
    torch.manual_seed(0)
    casc = NHiTSCascade(context_len=96, horizon=48, patch_len=16, pools=(4, 2, 1), ratios=(8, 4, 1),
                        blocks_per_stack=1, hidden_dim=H, channel_z_dim=Z, time_z_dim=Z, dropout=0.0,
                        block_kwargs=dict(rank_dim=R, time_kernel="causal_decay",
                                          time_context_bidirectional=True)).eval()
    x, zc, zt = torch.randn(2, 96, C), torch.randn(C, Z), torch.randn(6, Z)
    perm = torch.randperm(C)
    assert torch.allclose(casc(x, zc, zt)[..., perm], casc(x[..., perm], zc[perm], zt), atol=1e-5)


@pytest.mark.parametrize("cascade", ["single", "nhits"])
def test_grad_checkpoint_gives_identical_loss_and_grads(cascade):
    kw = dict(hidden_dim=16, rank_dim=8, patch_len=16, quantile_levels=(), mse_weight=1.0, mae_weight=0.0,
              pinball_weight=0.0, time_mix="causal_decay", time_context_bidirectional=True,
              context_len=96, horizon_len=48, dropout=0.1)
    kw.update(cascade="nhits") if cascade == "nhits" else kw.update(head="flatten")
    past, fut = torch.randn(3, 96, C), torch.randn(3, 48, C)
    out = []
    for ckpt in (False, True):
        torch.manual_seed(0)
        m = TimeSeriesSSM(TimeSeriesSSMConfig(grad_checkpoint=ckpt, **kw))
        m.fit_channel_graph(past.reshape(-1, C))
        m.train()
        torch.manual_seed(1)  # same dropout masks in both runs
        loss = m.loss(past, fut)
        loss.backward()
        out.append((loss.detach(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}))
    (l0, g0), (l1, g1) = out
    assert torch.allclose(l0, l1, atol=1e-6) and g0.keys() == g1.keys()
    assert all(torch.allclose(g0[k], g1[k], atol=1e-5) for k in g0)


def _support_model(**extra):
    cfg = TimeSeriesSSMConfig(hidden_dim=16, rank_dim=8, patch_len=16, quantile_levels=(), cascade="nhits",
                              context_len=96, horizon_len=48, time_mix="causal_decay", sample_axis="support",
                              time_context_bidirectional=True, mse_weight=1.0, mae_weight=0.0, pinball_weight=0.0,
                              use_calendar=False, **extra)
    m = TimeSeriesSSM(cfg)
    m.fit_channel_graph(torch.randn(400, C))
    return m


def test_support_bank_forecast_backward_and_all_grads():
    torch.manual_seed(0)
    m = _support_model()
    past, fut = torch.randn(3, 96, C), torch.randn(3, 48, C)
    sup = {"x": torch.randn(4, 96, C), "y": torch.randn(4, 48, C), "mask": torch.ones(3, 4, dtype=torch.bool)}
    m.train()
    loss = m.loss(past, fut, support=sup)
    loss.backward()
    assert torch.isfinite(loss)
    missing = [n for n, p in m.nhits.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing
    point, _ = m.predict(past, horizon=48)  # no bank -> plain N-HiTS path still works
    assert point.shape == (3, 48, C)


def test_masked_support_cannot_reach_query_and_queries_do_not_mix():
    torch.manual_seed(0)
    m = _support_model().eval()
    past = torch.randn(3, 96, C)
    sx, sy = torch.randn(4, 96, C), torch.randn(4, 48, C)
    mask = torch.ones(3, 4, dtype=torch.bool)
    mask[0, 2] = False  # support 2 overlaps query 0's future
    base, _ = m.forward_window(past, 48, support={"x": sx, "y": sy, "mask": mask})
    sx2, sy2 = sx.clone(), sy.clone()
    sx2[2] += 3.0 * torch.randn_like(sx2[2]); sy2[2] += 3.0 * torch.randn_like(sy2[2])  # (RevIN removes shifts)
    out, _ = m.forward_window(past, 48, support={"x": sx2, "y": sy2, "mask": mask})
    assert torch.allclose(out[0], base[0], atol=1e-5)  # masked for query 0 -> no influence
    assert not torch.allclose(out[1], base[1], atol=1e-5)  # usable by query 1 -> influences it
    past2 = past.clone(); past2[1] += 3.0 * torch.randn_like(past2[1])
    out2, _ = m.forward_window(past2, 48, support={"x": sx, "y": sy, "mask": mask})
    assert torch.allclose(out2[0], base[0], atol=1e-5) and torch.allclose(out2[2], base[2], atol=1e-5)


def test_support_bank_order_invariance_and_channel_equivariance():
    torch.manual_seed(0)
    m = _support_model().eval()
    past, sx, sy = torch.randn(2, 96, C), torch.randn(5, 96, C), torch.randn(5, 48, C)
    base, _ = m.forward_window(past, 48, support={"x": sx, "y": sy})
    k = torch.randperm(5)
    out, _ = m.forward_window(past, 48, support={"x": sx[k], "y": sy[k]})
    assert torch.allclose(out, base, atol=1e-5)
    casc = m.nhits
    zc, zt = torch.randn(C, m.channel_z_dim), torch.randn(casc.n_tokens, m.time_z_dim)
    perm = torch.randperm(C)
    a = casc(past, zc, zt, support_x=sx, support_y=sy)
    b = casc(past[..., perm], zc[perm], zt, support_x=sx[..., perm], support_y=sy[..., perm])
    assert torch.allclose(a[..., perm], b, atol=1e-5)


# ---------------------------------------------------------------- KAN feed-forward
from timeseriesssm.model.kan_layers import (  # noqa: E402
    BSplineKANLayer, ChebyKANLayer, bspline_basis, chebyshev_basis)


def test_chebyshev_recurrence_matches_closed_form():
    x = torch.linspace(-0.99, 0.99, 101)
    t = chebyshev_basis(x, 5)
    ref = torch.stack([torch.cos(k * torch.arccos(x)) for k in range(6)], dim=-1)
    assert torch.allclose(t, ref, atol=1e-5)


def test_bspline_partition_of_unity_inside_grid():
    layer = BSplineKANLayer(4, 3, grid=5, order=3, grid_range=(-2.0, 2.0))
    x = torch.linspace(-1.999, 1.999, 257)
    b = bspline_basis(x, layer.knots, 3)
    assert b.shape == (257, 8) and (b >= -1e-7).all()
    assert torch.allclose(b.sum(-1), torch.ones(257), atol=1e-5)


@pytest.mark.parametrize("kind,width,expected", [("cheby_kan", 96, 2 * 128 * 96 * 4),
                                                  ("bspline_kan", 48, 2 * 128 * 48 * 9)])
def test_kan_ffn_param_count_matches_plan(kind, width, expected):
    blk = TimeSeriesSSMBlock(128, Z, Z, 96, ffn_kind=kind, kan_width=width)
    n = sum(p.numel() for n_, p in blk.named_parameters() if n_.startswith("kan_ffn.") and "norm" not in n_)
    assert n == expected
    glu = TimeSeriesSSMBlock(128, Z, Z, 96)
    n_glu = sum(p.numel() for n_, p in glu.named_parameters() if n_.startswith(("ffn_in", "ffn_out", "norm_ffn")))
    assert abs(n - n_glu) / n_glu < 0.15


@pytest.mark.parametrize("kind", ["cheby_kan", "bspline_kan"])
@pytest.mark.parametrize("cascade", ["single", "nhits"])
def test_kan_ffn_forecast_backward_equivariance(kind, cascade):
    torch.manual_seed(0)
    kw = dict(hidden_dim=16, rank_dim=8, patch_len=16, quantile_levels=(), mse_weight=1.0, mae_weight=0.0,
              pinball_weight=0.0, time_mix="causal_decay", time_context_bidirectional=True, context_len=96,
              horizon_len=48, ffn_kind=kind, kan_width=12)
    kw.update(cascade="nhits") if cascade == "nhits" else kw.update(head="flatten")
    m = TimeSeriesSSM(TimeSeriesSSMConfig(**kw))
    past, fut = torch.randn(3, 96, C), torch.randn(3, 48, C)
    m.fit_channel_graph(past.reshape(-1, C))
    m.train()
    loss = m.loss(past, fut)
    loss.backward()
    assert torch.isfinite(loss)
    kan = [(n, p) for n, p in m.named_parameters() if ".kan_ffn." in n]
    assert kan and all(p.grad is not None for _, p in kan)
    blk = TimeSeriesSSMBlock(H, Z, Z, R, time_kernel="causal_decay", time_context_bidirectional=True,
                             ffn_kind=kind, kan_width=12).eval()
    x, zt = _inputs(4)
    zc, perm = torch.randn(C, Z), torch.randperm(C)
    assert torch.allclose(blk(x, zc, zt, n_context=4)[:, :, perm], blk(x[:, :, perm], zc[perm], zt, n_context=4),
                          atol=1e-5)


def test_closed_form_cubic_bspline_matches_cox_de_boor():
    from timeseriesssm.model.kan_layers import cubic_bspline_basis_uniform
    layer = BSplineKANLayer(4, 3, grid=5, order=3)
    x = torch.linspace(-3.0, 3.0, 301)  # includes points outside the grid
    ref = bspline_basis(x, layer.knots, 3)
    assert torch.allclose(cubic_bspline_basis_uniform(x, layer._t0, layer._h, 8), ref, atol=1e-5)


def test_swiglu_is_parameter_identical_and_uses_silu_gate():
    torch.manual_seed(0)
    glu = TimeSeriesSSMBlock(H, Z, Z, R, use_channel_cross=False, use_channel_ssm=False, use_time_ssm=False)
    swi = TimeSeriesSSMBlock(H, Z, Z, R, use_channel_cross=False, use_channel_ssm=False, use_time_ssm=False,
                             ffn_kind="swiglu")
    assert [(n, p.shape) for n, p in glu.named_parameters()] == [(n, p.shape) for n, p in swi.named_parameters()]
    swi.load_state_dict(glu.state_dict())
    x, zt = _inputs(5)
    zc = torch.randn(C, Z)
    t = glu.norm_ffn(glu.norm_channel(glu.norm_time(x)))
    a, g = glu.ffn_in(t).chunk(2, dim=-1)
    ref = glu.norm_channel(glu.norm_time(x)) + glu.ffn_out(a * torch.nn.functional.silu(g))
    assert torch.allclose(swi.eval()(x, zc, zt), ref, atol=1e-5)
    assert not torch.allclose(glu.eval()(x, zc, zt), ref, atol=1e-4)


def test_fast_cubic_bspline_matches_closed_form_and_gradcheck():
    from timeseriesssm.model.kan_layers import _CubicBSplineUniform, cubic_bspline_basis_uniform
    layer = BSplineKANLayer(4, 3, grid=5, order=3)
    t0, h = layer._t0, layer._h
    x = (torch.rand(500, 6, dtype=torch.float64) * 12 - 6).requires_grad_()  # well inside and outside the grid
    ref = cubic_bspline_basis_uniform(x, t0, h, 8)
    new = _CubicBSplineUniform.apply(x, t0, h, 8)
    assert torch.allclose(new, ref, atol=1e-12)
    g = torch.randn_like(ref)
    assert torch.allclose(torch.autograd.grad(new, x, g)[0], torch.autograd.grad(ref, x, g)[0], atol=1e-12)
    xs = (torch.rand(20, 3, dtype=torch.float64) * 10 - 5).requires_grad_()
    assert torch.autograd.gradcheck(lambda v: _CubicBSplineUniform.apply(v, t0, h, 8), (xs,))


def test_kan_last_stack_only():
    cfg = TimeSeriesSSMConfig(hidden_dim=16, rank_dim=8, patch_len=16, quantile_levels=(), cascade="nhits",
                              context_len=96, horizon_len=48, ffn_kind="bspline_kan", kan_width=8, kan_stacks="last")
    m = TimeSeriesSSM(cfg)
    kinds = [all(b.kan_ffn is not None for b in st.blocks) for st in m.nhits.stacks]
    assert kinds == [False, False, True]
    assert all(b.ffn_in is not None for st in m.nhits.stacks[:-1] for b in st.blocks)


# ---------------------------------------------------------------- Residual Cycle Forecasting (RCF)
from timeseriesssm.model.time_encoding import freq_id_from_name  # noqa: E402

def _rcf_pair(w=24):
    kw = dict(hidden_dim=16, rank_dim=8, patch_len=16, quantile_levels=(), cascade="nhits", context_len=96,
              horizon_len=48, time_mix="causal_decay", time_context_bidirectional=True, use_calendar=False,
              mse_weight=1.0, mae_weight=0.0, pinball_weight=0.0)
    torch.manual_seed(0)
    plain = TimeSeriesSSM(TimeSeriesSSMConfig(**kw))
    torch.manual_seed(0)
    rcf = TimeSeriesSSM(TimeSeriesSSMConfig(rcf_cycle=w, **kw))
    g = torch.randn(400, C)
    plain.fit_channel_graph(g); rcf.fit_channel_graph(g)
    missing, unexpected = rcf.load_state_dict(plain.state_dict(), strict=False)  # graph encoder is built lazily
    assert missing == ["rcf_q"] and not unexpected
    return plain.eval(), rcf.eval()


def _phase(b, w, start=None):
    start = torch.randint(0, w, (b,)) if start is None else start
    return (start.view(-1, 1) + torch.arange(96 + 48)) % w


def test_rcf_zero_init_equals_plain_model():
    plain, rcf = _rcf_pair()
    assert rcf.rcf_q.shape == (24, C) and torch.count_nonzero(rcf.rcf_q) == 0
    past, ph = torch.randn(3, 96, C), _phase(3, 24)
    a, _ = plain.forward_window(past, 48)
    b, _ = rcf.forward_window(past, 48, cycle_phase=ph)
    assert torch.allclose(a, b, atol=1e-6)
    with pytest.raises(ValueError):
        rcf.forward_window(past, 48)


def test_rcf_adds_phase_aligned_cycle_and_gets_gradients():
    _, rcf = _rcf_pair()
    with torch.no_grad():
        rcf.rcf_q.copy_(torch.randn(24, C))
    past, ph = torch.randn(2, 96, C), _phase(2, 24)
    out, _ = rcf.forward_window(past, 48, cycle_phase=ph)
    # remove the cycle's contribution by hand: normalized-space forecast minus Q[horizon phases]
    xn, mean, std = rcf.revin.normalize(past)
    z = rcf.nhits(xn - rcf.rcf_q[ph[:, :96]], rcf._z_channel(C, past.device),
                  rcf.time_encoding(rcf.nhits.n_tokens, freq_id=freq_id_from_name(None), device=past.device))
    ref = rcf.revin.denormalize(z + rcf.rcf_q[ph[:, 96:]], mean, std)
    assert torch.allclose(out, ref, atol=1e-5)
    rcf.train()
    loss = rcf.loss(past, torch.randn(2, 48, C), cycle_phase=ph)
    loss.backward()
    assert rcf.rcf_q.grad is not None and rcf.rcf_q.grad.abs().sum() > 0


def test_rcf_no_leakage_and_equivariance_with_permuted_q():
    _, rcf = _rcf_pair()
    with torch.no_grad():
        rcf.rcf_q.copy_(torch.randn(24, C))
    past, ph = torch.randn(2, 96, C), _phase(2, 24)
    base, _ = rcf.forward_window(past, 48, cycle_phase=ph)
    # horizon values never enter; only horizon *phases* (calendar) do
    out, _ = rcf.forward_window(past.clone(), 48, cycle_phase=ph)
    assert torch.allclose(out, base)
    perm = torch.randperm(C)
    zc = rcf._z_channel(C, past.device)
    xn = rcf.revin.normalize(past)[0]
    zt = rcf.time_encoding(rcf.nhits.n_tokens, freq_id=freq_id_from_name(None), device=past.device)
    a = rcf.nhits(xn - rcf.rcf_q[ph[:, :96]], zc, zt) + rcf.rcf_q[ph[:, 96:]]
    qp = rcf.rcf_q[:, perm]
    b = rcf.nhits(xn[..., perm] - qp[ph[:, :96]], zc[perm], zt) + qp[ph[:, 96:]]
    assert torch.allclose(a[..., perm], b, atol=1e-5)


def test_rcf_single_stack_path_matches_normalized_space_definition():
    kw = dict(hidden_dim=16, rank_dim=8, patch_len=16, quantile_levels=(), head="flatten", context_len=96,
              horizon_len=48, time_mix="causal_decay", time_context_bidirectional=True, use_calendar=False,
              mse_weight=1.0, mae_weight=0.0, pinball_weight=0.0)
    torch.manual_seed(0)
    plain = TimeSeriesSSM(TimeSeriesSSMConfig(**kw))
    rcf = TimeSeriesSSM(TimeSeriesSSMConfig(rcf_cycle=24, **kw))
    g = torch.randn(400, C)
    plain.fit_channel_graph(g); rcf.fit_channel_graph(g)
    rcf.load_state_dict(plain.state_dict(), strict=False)
    plain.eval(); rcf.eval()
    past, ph = torch.randn(2, 96, C), _phase(2, 24)
    a, _ = plain.forward_window(past, 48)
    b, _ = rcf.forward_window(past, 48, cycle_phase=ph)
    assert torch.allclose(a, b, atol=1e-5)  # zero Q -> identical to the plain model
    with torch.no_grad():
        rcf.rcf_q.copy_(torch.randn(24, C))
        rcf.revin.weight.copy_(torch.rand(C) + 0.5)
        rcf.revin.bias.copy_(torch.randn(C) * 0.1)
        for prm in rcf.forecast_head.parameters():
            prm.zero_()  # backbone forecast = 0 in normalized space
    out, _ = rcf.forward_window(past, 48, cycle_phase=ph)
    _, mean, std = rcf.revin.normalize(past)
    # CycleNet definition: y_norm = backbone(x_norm - Q) + Q, then RevIN-denormalize -> here backbone = 0
    ref = rcf.revin.denormalize(rcf.rcf_q[ph[:, 96:]], mean, std)
    assert torch.allclose(out, ref, atol=1e-5)
    with torch.no_grad():
        rcf.load_state_dict(plain.state_dict(), strict=False)
        rcf.rcf_q.copy_(torch.randn(24, C))
    rcf.train()
    loss = rcf.loss(past, torch.randn(2, 48, C), cycle_phase=ph)
    loss.backward()
    assert rcf.rcf_q.grad is not None and rcf.rcf_q.grad.abs().sum() > 0


@pytest.mark.parametrize("use_revin", [True, False])
def test_channel_configuration_is_stable_across_forwards(use_revin):
    """Regression: with use_revin=False every forward used to rebuild the channel encoder and RCF table."""
    cfg = TimeSeriesSSMConfig(hidden_dim=16, rank_dim=8, patch_len=16, quantile_levels=(), cascade="nhits",
                              context_len=96, horizon_len=48, use_revin=use_revin, rcf_cycle=24, use_calendar=False)
    m = TimeSeriesSSM(cfg)
    m.fit_channel_graph(torch.randn(400, C))
    m.eval()  # the channel encoder has dropout
    enc, q = m.channel_encoder, m.rcf_q
    z_before = m._z_channel(C, torch.device("cpu")).clone()
    for _ in range(2):
        m.forward_window(torch.randn(2, 96, C), 48, cycle_phase=_phase(2, 24))
    assert m.channel_encoder is enc and m.rcf_q is q
    assert torch.allclose(m._z_channel(C, torch.device("cpu")), z_before)


def test_freq_loss_weight_zero_is_time_loss_and_positive_adds_spectral_term():
    """FreDF: a=0 must reproduce the time-domain loss exactly; a>0 mixes in the rFFT L1 term."""
    torch.manual_seed(0)
    kw = dict(hidden_dim=16, rank_dim=8, patch_len=16, quantile_levels=(), cascade="nhits", context_len=96,
              horizon_len=48, use_calendar=False, mse_weight=1.0, mae_weight=0.0, pinball_weight=0.0)
    m = TimeSeriesSSM(TimeSeriesSSMConfig(**kw))
    m.fit_channel_graph(torch.randn(400, C))
    m.eval()  # after fit_channel_graph: modules it creates start in train mode (dropout)
    x, y = torch.randn(2, 96, C), torch.randn(2, 48, C)
    point, _ = m.forward_window(x, 48)
    mse = torch.nn.functional.mse_loss(point, y)
    spec = (torch.fft.rfft(point, dim=1) - torch.fft.rfft(y, dim=1)).abs().mean()
    assert torch.allclose(m.loss(x, y), mse, atol=1e-6)
    m.config.freq_loss_weight = 0.5
    assert torch.allclose(m.loss(x, y), 0.5 * mse + 0.5 * spec, atol=1e-5)


def _v4(**over):
    kw = dict(hidden_dim=16, rank_dim=8, patch_len=16, quantile_levels=(), cascade="nhits", context_len=96,
              horizon_len=48, use_calendar=False, time_mix="causal_decay", use_glu_ffn=False, rcf_cycle=24)
    kw.update(over)
    torch.manual_seed(0)
    m = TimeSeriesSSM(TimeSeriesSSMConfig(**kw))
    m.fit_channel_graph(torch.randn(400, C))
    m.eval()
    with torch.no_grad():  # non-trivial RevIN affine and RCF table
        m.revin.weight.uniform_(0.5, 1.5); m.revin.bias.uniform_(-0.3, 0.3); m.rcf_q.normal_(0, 0.3)
    return m


def test_v4_defaults_build_exactly_the_v3_parameter_set():
    v3 = _v4()
    v4 = _v4(channel_mix_scale=1.0, channel_mix_learn=False, linear_path=False, series_tokens=False)
    assert v3.state_dict().keys() == v4.state_dict().keys()
    v4.load_state_dict(v3.state_dict())
    x, ph = torch.randn(2, 96, C), _phase(2, 24)
    assert torch.equal(v3.forward_window(x, 48, cycle_phase=ph)[0], v4.forward_window(x, 48, cycle_phase=ph)[0])


def test_linear_path_with_closed_gate_is_the_rlinear_map_plus_rcf():
    m = _v4(linear_path=True, deep_gate_init=0.0)
    x, ph = torch.randn(2, 96, C), _phase(2, 24)
    xn, mean, std = m.revin.normalize(x)
    xn = xn - m.rcf_q[ph[:, :96]]
    lin = m.linear_head(xn.transpose(1, 2)).transpose(1, 2) + m.rcf_q[ph[:, 96:]]
    ref = m.revin.denormalize(lin, mean, std)
    assert torch.allclose(m.forward_window(x, 48, cycle_phase=ph)[0], ref, atol=1e-5)
    # the gate is a parameter, so the deep cascade starts contributing after the first update
    m.train(); m.loss(x, torch.randn(2, 48, C), cycle_phase=ph).backward()
    assert m.deep_gate.grad is not None and m.deep_gate.grad.abs().sum() > 0


def test_series_tokens_start_as_the_model_without_them():
    base = _v4()
    st = _v4(series_tokens=True, series_dim=16, series_heads=4)
    missing = st.load_state_dict(base.state_dict(), strict=False).missing_keys
    assert missing and all(k.startswith(("series.", "series_gate")) for k in missing)
    x, ph = torch.randn(2, 96, C), _phase(2, 24)
    assert torch.allclose(base.forward_window(x, 48, cycle_phase=ph)[0], st.forward_window(x, 48, cycle_phase=ph)[0],
                          atol=1e-6)
    st.train(); st.loss(x, torch.randn(2, 48, C), cycle_phase=ph).backward()
    assert st.series_gate.grad.abs().sum() > 0  # the branch opens through its gate


def test_series_token_mixer_is_channel_permutation_equivariant():
    from timeseriesssm.model.series_tokens import SeriesTokenMixer
    torch.manual_seed(0)
    s = SeriesTokenMixer(96, 48, dim=16, layers=2, heads=4, dropout=0.0, n_stacks=3, hidden_dim=8).eval()
    for p in s.ctx:
        torch.nn.init.normal_(p.weight)
    x, perm = torch.randn(2, 96, C), torch.randperm(C)
    f, ctx = s(x)
    fp, ctxp = s(x[..., perm])
    assert torch.allclose(f[..., perm], fp, atol=1e-5)
    assert all(torch.allclose(a[:, perm], b, atol=1e-5) for a, b in zip(ctx, ctxp))


def test_channel_mix_scale_zero_blocks_all_cross_channel_flow():
    m = _v4(channel_mix_scale=0.0, linear_path=True, deep_gate_init=1.0)
    x, ph = torch.randn(2, 96, C), _phase(2, 24)
    y0 = m.forward_window(x, 48, cycle_phase=ph)[0]
    x2 = x.clone(); x2[..., 0] += torch.randn(2, 96)
    y1 = m.forward_window(x2, 48, cycle_phase=ph)[0]
    assert torch.allclose(y0[..., 1:], y1[..., 1:], atol=1e-6) and not torch.allclose(y0[..., 0], y1[..., 0])


def test_channel_mix_scale_one_fixed_equals_v3_and_learned_gate_trains():
    m = _v4(channel_mix_learn=True, channel_mix_scale=1.0)
    gates = [mod.mix_gate for mod in m.modules() if hasattr(mod, "mix_gate")]
    assert gates and all(isinstance(g, torch.nn.Parameter) for g in gates)
    v3 = _v4()
    v3.load_state_dict({k: v for k, v in m.state_dict().items() if "mix_gate" not in k})
    x, ph = torch.randn(2, 96, C), _phase(2, 24)
    assert torch.allclose(m.forward_window(x, 48, cycle_phase=ph)[0], v3.forward_window(x, 48, cycle_phase=ph)[0],
                          atol=1e-6)
    m.train(); m.loss(x, torch.randn(2, 48, C), cycle_phase=ph).backward()
    assert any(g.grad is not None and g.grad.abs() > 0 for g in gates)


def test_channel_subset_forward_matches_full_forward_without_mixing():
    m = _v4(channel_mix_scale=0.0, linear_path=True, deep_gate_init=0.7)
    x, ph = torch.randn(2, 96, C), _phase(2, 24)
    idx = torch.tensor([3, 0, 4])
    full = m.forward_window(x, 48, cycle_phase=ph)[0]
    sub = m.forward_window(x[..., idx], 48, cycle_phase=ph, channel_index=idx)[0]
    assert torch.allclose(full[..., idx], sub, atol=1e-5)
    assert m._n_channels == C  # the subset forward did not reconfigure the channel set
    # with mixing and series tokens the subset forward runs and stays configured on the full set
    m2 = _v4(series_tokens=True, series_dim=16)
    assert m2.forward_window(x[..., idx], 48, cycle_phase=ph, channel_index=idx)[0].shape == (2, 48, 3)
    assert m2._n_channels == C


def test_bf16_autocast_training_step_is_finite_and_scan_runs_in_fp32():
    """bf16 autocast training (memory saving): finite loss/grads, decay scan kept in fp32. On CUDA when available
    (some CPUs have no bf16 backward kernels)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _v4(linear_path=True, series_tokens=True, series_dim=16, freq_loss_weight=0.5).to(dev)
    m.train()
    x, y, ph = torch.randn(2, 96, C, device=dev), torch.randn(2, 48, C, device=dev), _phase(2, 24).to(dev)
    seen = []
    blk = next(mod for mod in m.modules() if isinstance(mod, CausalDecaySSM))
    h = blk.register_forward_hook(lambda mod, inp, out: seen.append(out.dtype))
    with torch.autocast(dev, dtype=torch.bfloat16):
        loss = m.loss(x, y, cycle_phase=ph)
    h.remove()
    assert torch.isfinite(loss) and seen and all(d == torch.float32 for d in seen)
    if dev == "cuda":
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


# ---------------------------------------------------------------------------------------------------- v5 candidates
def test_v5_switches_off_build_exactly_the_v4_model():
    v4 = _v4(linear_path=True, series_tokens=True, series_dim=16)
    v5 = _v4(linear_path=True, series_tokens=True, series_dim=16, label_pca_weight=0.0, lowpass_frac=0.0,
             stat_head=False)
    assert v4.state_dict().keys() == v5.state_dict().keys()
    v5.load_state_dict(v4.state_dict())
    x, ph = torch.randn(2, 96, C), _phase(2, 24)
    assert torch.equal(v4.forward_window(x, 48, cycle_phase=ph)[0], v5.forward_window(x, 48, cycle_phase=ph)[0])


def test_dct_lowpass_is_an_orthogonal_projection_and_full_band_is_identity():
    from timeseriesssm.model.timeseries_ssm import _dct_lowpass
    p = _dct_lowpass(48, 0.25).double()
    assert torch.allclose(p @ p, p, atol=1e-5) and torch.allclose(p, p.T, atol=1e-6)
    assert torch.allclose(_dct_lowpass(48, 1.0), torch.eye(48), atol=1e-5)
    smooth = torch.linspace(-1, 1, 48).double()  # a linear trend is (almost) entirely low-frequency
    assert (p @ smooth - smooth).abs().max() < 0.1


def test_lowpass_full_band_equals_model_without_it():
    base = _v4()
    lp = _v4(lowpass_frac=1.0)
    lp.load_state_dict(base.state_dict())
    x, ph = torch.randn(2, 96, C), _phase(2, 24)
    assert torch.allclose(base.forward_window(x, 48, cycle_phase=ph)[0], lp.forward_window(x, 48, cycle_phase=ph)[0],
                          atol=1e-5)


def test_stat_head_starts_as_revin_and_learns():
    base = _v4()
    sh = _v4(stat_head=True)
    missing = sh.load_state_dict(base.state_dict(), strict=False).missing_keys
    assert missing and all(k.startswith("stat_mlp") for k in missing)
    x, ph = torch.randn(2, 96, C) * 3 + 1, _phase(2, 24)
    assert torch.allclose(base.forward_window(x, 48, cycle_phase=ph)[0], sh.forward_window(x, 48, cycle_phase=ph)[0],
                          atol=1e-5)
    sh.train(); sh.loss(x, torch.randn(2, 48, C), cycle_phase=ph).backward()
    assert sh.stat_mlp[-1].weight.grad.abs().sum() > 0
    idx = torch.tensor([4, 1])  # channel subsets work with the stat head
    assert sh.forward_window(x[..., idx], 48, cycle_phase=ph, channel_index=idx)[0].shape == (2, 48, 2)


def test_haar_transform_is_orthonormal_and_wavelet_loss_combines():
    from timeseriesssm.model.timeseries_ssm import haar_coeffs
    x = torch.randn(3, 96, C, dtype=torch.float64)
    co = haar_coeffs(x, 4)
    assert co.shape[1] == 96 and torch.allclose(co.pow(2).sum(1), x.pow(2).sum(1))  # Parseval (even lengths)
    assert haar_coeffs(torch.randn(2, 45, C), 4).shape[0] == 2  # odd lengths are padded, no crash
    m = _v4(wavelet_loss_weight=0.3, freq_loss_weight=0.2, mse_weight=1.0, mae_weight=0.0)
    x, y, ph = torch.randn(2, 96, C), torch.randn(2, 48, C), _phase(2, 24)
    point = m.forward_window(x, 48, cycle_phase=ph)[0]
    mse = torch.nn.functional.mse_loss(point, y)
    spec = (torch.fft.rfft(point, dim=1) - torch.fft.rfft(y, dim=1)).abs().mean()
    wav = haar_coeffs(point - y, 4).abs().mean()
    assert torch.allclose(m.loss(x, y, cycle_phase=ph), 0.5 * mse + 0.2 * spec + 0.3 * wav, atol=1e-5)


def test_round2_losses_off_by_default_and_well_behaved():
    from timeseriesssm.model.timeseries_ssm import bures_wasserstein, patch_structural_loss
    torch.manual_seed(0)
    z = torch.randn(16, 30, C, dtype=torch.float64)
    assert bures_wasserstein(z, z).abs() < 1e-4                      # zero for identical distributions
    assert bures_wasserstein(z, z + 1.0) > 0.9                       # a mean shift of 1 per dim -> ~1 per dim
    assert patch_structural_loss(z, z, 8).abs() < 1e-4
    assert patch_structural_loss(2 * z + 1, z, 8) > 0.5              # scale/offset changes are penalized
    m = _v4(mse_weight=1.0, mae_weight=0.0)
    x, y, ph = torch.randn(2, 96, C), torch.randn(2, 48, C), _phase(2, 24)
    base = m.loss(x, y, cycle_phase=ph)
    for over in ({"horizon_weight": "arctan"}, {"distdf_weight": 0.2}, {"ps_loss_weight": 0.2}):
        for k, v in over.items():
            setattr(m.config, k, v)
        l2 = m.loss(x, y, cycle_phase=ph)
        assert torch.isfinite(l2) and not torch.allclose(l2, base)
        m.train(); m.zero_grad(); l2.backward(); m.eval()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        for k in over:  # reset to the defaults
            setattr(m.config, k, TimeSeriesSSMConfig().__dict__[k] if k in TimeSeriesSSMConfig().__dict__ else 0.0)
    assert torch.allclose(m.loss(x, y, cycle_phase=ph), base)        # defaults restore the original loss


def test_stats_head_v2_starts_as_revin_and_aux_supervises():
    base = _v4()
    sh = _v4(stat_head=True, stat_head_linear=True, stat_head_aux=0.5)
    sh.load_state_dict(base.state_dict(), strict=False)
    x, y, ph = torch.randn(2, 96, C) * 2 + 1, torch.randn(2, 48, C), _phase(2, 24)
    assert torch.allclose(base.forward_window(x, 48, cycle_phase=ph)[0], sh.forward_window(x, 48, cycle_phase=ph)[0],
                          atol=1e-5)
    sh.train()
    loss = sh.loss(x, y, cycle_phase=ph)
    sh.config.stat_head_aux = 0.0
    assert loss > sh.loss(x, y, cycle_phase=ph)  # the statistics loss is active
    sh.config.stat_head_aux = 0.5
    sh.zero_grad(); sh.loss(x, y, cycle_phase=ph).backward()
    assert sh.stat_lin.weight.grad.abs().sum() > 0 and sh.stat_gate.grad is not None


def test_fan_branch_shapes_aux_and_topk():
    from timeseriesssm.model.timeseries_ssm import topk_frequency_part
    t = torch.arange(96, dtype=torch.float32)
    pure = (torch.sin(2 * math.pi * 4 * t / 96) + 0.5 * torch.cos(2 * math.pi * 9 * t / 96)).view(1, 96, 1)
    assert torch.allclose(topk_frequency_part(pure, 2), pure, atol=1e-4)  # 2 components recover a 2-tone signal
    m = _v4(fan_k=3, fan_aux=1.0)
    x, y, ph = torch.randn(2, 96, C), torch.randn(2, 48, C), _phase(2, 24)
    assert m.forward_window(x, 48, cycle_phase=ph)[0].shape == (2, 48, C)
    m.train(); m.zero_grad(); m.loss(x, y, cycle_phase=ph).backward()
    assert m.fan_out.weight.grad.abs().sum() > 0
    idx = torch.tensor([2, 0])
    assert m.forward_window(x[..., idx], 48, cycle_phase=ph, channel_index=idx)[0].shape == (2, 48, 2)


# ---------------------------------------------------------------------------------------------------- v6: STAR
def test_star_mixer_equivariant_in_eval_and_stochastic_in_training():
    from timeseriesssm.model.series_tokens import SeriesTokenMixer
    torch.manual_seed(0)
    for mixer in ("star", "both"):
        s = SeriesTokenMixer(96, 48, dim=16, layers=2, heads=4, dropout=0.0, n_stacks=3, hidden_dim=8, mixer=mixer)
        for p in s.ctx:
            torch.nn.init.normal_(p.weight)
        s.eval()
        x, perm = torch.randn(2, 96, C), torch.randperm(C)
        f, ctx = s(x)
        fp, ctxp = s(x[..., perm])
        assert torch.allclose(f[..., perm], fp, atol=1e-5)
        assert all(torch.allclose(a[:, perm], b, atol=1e-5) for a, b in zip(ctx, ctxp))
        s.train()
        torch.manual_seed(1); a1 = s(x)[0]
        torch.manual_seed(2); a2 = s(x)[0]
        assert not torch.allclose(a1, a2)  # stochastic pooling samples channels during training
        a1.sum().backward()
        assert all(p.grad is not None for p in s.layers.parameters())


def test_star_core_is_a_softmax_weighted_average_in_eval():
    from timeseriesssm.model.series_tokens import _STARLayer
    layer = _STARLayer(8, 4, 0.0).eval()
    h = torch.randn(3, C, 4)
    w = torch.softmax(h, dim=1)
    assert torch.allclose(layer.core(h), (w * h).sum(1, keepdim=True))


def test_v6_star_switch_off_is_the_v5_model_and_on_trains():
    v5 = _v4(series_tokens=True, series_dim=16)
    v6 = _v4(series_tokens=True, series_dim=16, series_mixer="attention")
    assert v5.state_dict().keys() == v6.state_dict().keys()
    star = _v4(series_tokens=True, series_dim=16, series_mixer="star")
    x, ph = torch.randn(2, 96, C), _phase(2, 24)
    star.train(); star.loss(x, torch.randn(2, 48, C), cycle_phase=ph).backward()
    assert star.series_gate.grad.abs().sum() > 0
    idx = torch.tensor([4, 1, 2])  # channel subsets work with STAR
    star.eval()
    assert star.forward_window(x[..., idx], 48, cycle_phase=ph, channel_index=idx)[0].shape == (2, 48, 3)


# ---------------------------------------------------------------------------------------------------- speed: exact
@pytest.mark.parametrize("bidirectional", [False, True])
def test_parallel_scan_matches_loop_forward_and_gradients(bidirectional):
    torch.manual_seed(0)
    m = CausalDecaySSM(H, Z, R, bidirectional_context=bidirectional).double()
    with torch.no_grad():  # a wide range of decays, including near-0 and near-1
        m.A_log.uniform_(-6.0, 3.0)
    x, z = _inputs(3)
    x, z = x.double().requires_grad_(), z.double()
    vis = torch.ones(B, P, C, dtype=torch.bool); vis[:, 5:] = False
    outs, grads = {}, {}
    for mode in ("loop", "parallel"):
        CausalDecaySSM.SCAN = mode
        m.zero_grad(); x.grad = None
        y = m(x, z, visible=vis, n_context=5)
        y.pow(2).sum().backward()
        outs[mode] = y.detach()
        grads[mode] = [x.grad.clone()] + [p.grad.clone() for p in m.parameters() if p.grad is not None]
    CausalDecaySSM.SCAN = "parallel"
    assert torch.allclose(outs["loop"], outs["parallel"], rtol=1e-10, atol=1e-12)
    assert all(torch.allclose(a, b, rtol=1e-9, atol=1e-11) for a, b in zip(grads["loop"], grads["parallel"]))


def test_parallel_scan_float32_close_to_loop_in_full_model():
    m = _v4(time_mix="causal_decay")
    x, ph = torch.randn(4, 96, C), _phase(4, 24)
    CausalDecaySSM.SCAN = "loop"; ref = m.forward_window(x, 48, cycle_phase=ph)[0]
    CausalDecaySSM.SCAN = "parallel"; new = m.forward_window(x, 48, cycle_phase=ph)[0]
    assert (ref - new).abs().max() < 1e-5


def test_gradient_checkpointing_never_changes_outputs_or_gradients():
    """Checkpointing only trades memory for recomputation (RNG state restored for dropout): bit-identical results."""
    grads = []
    for ckpt in (False, True):
        torch.manual_seed(0)
        m = TimeSeriesSSM(TimeSeriesSSMConfig(hidden_dim=16, rank_dim=8, patch_len=16, quantile_levels=(),
                                              cascade="nhits", context_len=96, horizon_len=48, use_calendar=False,
                                              time_mix="causal_decay", dropout=0.2, grad_checkpoint=ckpt,
                                              series_tokens=True, series_dim=16, series_mixer="star"))
        m.fit_channel_graph(torch.randn(400, C, generator=torch.Generator().manual_seed(1)))
        m.train()
        g = torch.Generator().manual_seed(2)
        x, y = torch.randn(2, 96, C, generator=g), torch.randn(2, 48, C, generator=g)
        torch.manual_seed(3)
        loss = m.loss(x, y)
        loss.backward()
        grads.append((loss.detach(), [p.grad.clone() for p in m.parameters() if p.grad is not None]))
    (l0, g0), (l1, g1) = grads
    assert torch.equal(l0, l1) and len(g0) == len(g1) and all(torch.equal(a, b) for a, b in zip(g0, g1))


def test_barron_rho_limits_and_partition_function():
    from timeseriesssm.model.timeseries_ssm import _barron_rho, _log_partition
    x = torch.linspace(-5, 5, 101, dtype=torch.float64)
    one = torch.tensor(1.0, dtype=torch.float64)
    assert torch.allclose(_barron_rho(x, one), torch.sqrt(x * x + 1) - 1, atol=1e-12)        # alpha=1: pseudo-Huber
    near2 = torch.tensor(1.999, dtype=torch.float64)
    assert torch.allclose(_barron_rho(x, near2), 0.5 * x * x, rtol=1e-2, atol=1e-6)  # alpha->2: x^2/2 (0.5% at |x|=5)
    # log Z: alpha->2 is a unit Gaussian (log sqrt(2 pi)); alpha=1 by direct quadrature
    assert abs(float(_log_partition(near2)) - 0.5 * math.log(2 * math.pi)) < 2e-3
    xs = torch.linspace(-200, 200, 400001, dtype=torch.float64)
    z1 = torch.trapezoid(torch.exp(-(torch.sqrt(xs * xs + 1) - 1)), xs)
    assert abs(float(_log_partition(one)) - float(torch.log(z1))) < 1e-6


def test_adaptive_time_loss_learns_alpha_and_scale_and_default_is_unchanged():
    m_fixed = _v4(mse_weight=1.0, mae_weight=0.0)
    m_ad = _v4(mse_weight=1.0, mae_weight=0.0, time_loss="adaptive", adaptive_alpha_init=1.5)
    assert m_fixed.loss_alpha_raw is None and m_ad.loss_alpha_raw.shape == (C,)
    a, lc = m_ad.adaptive_loss_params(C)
    assert torch.allclose(a, torch.full((C,), 1.5), atol=1e-6) and torch.all(lc == 0)
    x, y, ph = torch.randn(2, 96, C), torch.randn(2, 48, C), _phase(2, 24)
    m_ad.train(); loss = m_ad.loss(x, y, cycle_phase=ph); loss.backward()
    assert torch.isfinite(loss) and m_ad.loss_alpha_raw.grad.abs().sum() > 0 and m_ad.loss_log_c.grad.abs().sum() > 0
    idx = torch.tensor([1, 3])  # channel subsets index the per-channel loss parameters
    assert torch.isfinite(m_ad.loss(x[..., idx], y[..., idx], cycle_phase=ph, channel_index=idx))


def test_parallel_scan_chunked_matches_full():
    """The OOM fallback (row-chunked parallel scan, evaluation only) equals the one-shot parallel scan."""
    from timeseriesssm.model.causal_ssm import CausalDecaySSM
    g = torch.Generator().manual_seed(0)
    kv = torch.randn(3, 5, 17, 4, generator=g, dtype=torch.float64)
    log_a = -torch.rand(3, 5, 17, 4, generator=g, dtype=torch.float64)
    full = CausalDecaySSM._parallel_fwd(kv, log_a)
    chunked = CausalDecaySSM._parallel_fwd_chunked(kv, log_a, max_bytes=17 * 17 * 4 * 8 * 2)  # 2 rows per chunk
    assert torch.allclose(full, chunked, rtol=0, atol=1e-12)


@pytest.mark.parametrize("per_window", [False, True])
def test_calendar_per_window_makes_forecasts_batch_independent(per_window):
    """Batch-averaged calendar codes couple the windows of a batch; per-window codes do not."""
    m = _v4(use_calendar=True, calendar_per_window=per_window)
    with torch.no_grad():
        m.time_encoding.cal_proj.weight.normal_(0, 1.0)
    past, ph = torch.randn(4, 96, C), _phase(4, 24)
    ts = 3600.0 * (torch.arange(96).view(1, -1) + 37 * torch.arange(4).view(-1, 1)).double()
    with torch.no_grad():
        batch = m.forward_window(past, 48, timestamps=ts, cycle_phase=ph)[0]
        alone = m.forward_window(past[:1], 48, timestamps=ts[:1], cycle_phase=ph[:1])[0]
    same = torch.allclose(batch[:1], alone, atol=1e-5)
    assert same if per_window else not same


def test_canonical_sign_makes_recomputed_eigenvectors_permutation_equivariant():
    from timeseriesssm.model.channel_graph import ChannelGraphEncoder
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1000, 15, generator=g) @ torch.randn(15, 15, generator=g)
    perm = torch.randperm(15, generator=g)
    a = ChannelGraphEncoder(15, 12, sign="canonical").fit(x).builder.eigvecs
    b = ChannelGraphEncoder(15, 12, sign="canonical").fit(x[:, perm]).builder.eigvecs
    assert torch.allclose(a[perm], b, atol=1e-4)
    top = a.abs().argmax(dim=0)
    assert torch.all(a[top, torch.arange(a.shape[1])] > 0)


def test_spectral_gssc_matches_explicit_kernel():
    from timeseriesssm.global_ssm import SpectralGSSC
    g = torch.Generator().manual_seed(0)
    d, m, f = 6, 3, 9
    op = SpectralGSSC(hidden_dim=H, num_eig=d, num_filters=m).double()
    x, z = torch.randn(B, P, f, H, generator=g).double(), torch.randn(f, d * m, generator=g).double()
    zz = z.view(f, d, m)
    kern = torch.einsum("ajh,bjh->abh", op.u_q(zz), op.u_k(zz))          # [F, F', H]: one kernel per feature
    self_w = (op.u_sq(zz) * op.u_sk(zz)).sum(1)
    ref = torch.einsum("abh,nkbh->nkah", kern, op.u_o(x)) + self_w * op.u_s(x)
    assert torch.allclose(op(x, z), ref, atol=1e-10)


def test_spectral_gssc_is_invariant_to_eigenvector_signs():
    from timeseriesssm.global_ssm import SpectralGSSC
    from timeseriesssm.positional_encoding import EigenAugmentedPE
    g = torch.Generator().manual_seed(1)
    d, m, f = 6, 4, 11
    pe = EigenAugmentedPE(pe_dim=d, num_filters=m, norm="global_rms").double()
    op = SpectralGSSC(hidden_dim=H, num_eig=d, num_filters=m).double()
    lam, vecs = torch.rand(d, generator=g).double(), torch.randn(f, d, generator=g).double()
    flip = torch.tensor([1.0, -1.0, -1.0, 1.0, -1.0, 1.0]).double()
    x = torch.randn(B, P, f, H, generator=g).double()
    assert torch.allclose(op(x, pe(lam, vecs)), op(x, pe(lam, vecs * flip)), atol=1e-12)
    # the default (normalised) codes and the rank-R kernel are not sign-invariant, hence the canonical sign rule
    pe_n = EigenAugmentedPE(pe_dim=d, num_filters=m).double()
    assert not torch.allclose(pe_n(lam, vecs), pe_n(lam, vecs * flip).view(f, d, m).mul(flip[None, :, None])
                              .reshape(f, -1), atol=1e-6)


def test_spectral_kernel_model_is_permutation_equivariant_with_solver_signs():
    """Prop. 2(ii) without a sign rule: codes recomputed from permuted data, solver signs, spectral kernel."""
    L_, H_, C_ = 96, 24, 9
    cfg = TimeSeriesSSMConfig(cascade="nhits", context_len=L_, horizon_len=H_, hidden_dim=32, rank_dim=16,
                              patch_len=16, time_mix="causal_decay", use_glu_ffn=False, series_tokens=True,
                              series_mixer="star", series_dim=32, stat_head=True, rcf_cycle=24, quantile_levels=(),
                              channel_kernel="spectral", use_selective=False, channel_graph_sign="solver")
    torch.manual_seed(3)
    train = torch.randn(400, C_).cumsum(0)
    ma = TimeSeriesSSM(cfg)
    ma.fit_channel_graph(train.float())
    ma = ma.double().eval()
    with torch.no_grad():
        for p_ in ma.parameters():
            p_.add_(0.05 * torch.randn_like(p_))
    X = torch.randn(2, L_, C_).double()
    ts = (torch.arange(L_ + H_) * 3600.0).repeat(2, 1)
    phase = (torch.arange(L_ + H_) % 24).repeat(2, 1)
    perm = torch.randperm(C_)
    with torch.no_grad():
        oa = ma.forward_window(X, H_, timestamps=ts[:, :L_], freq="h", quantile_levels=(), cycle_phase=phase)[0]
        mb = TimeSeriesSSM(cfg)
        mb.fit_channel_graph(train[:, perm].float())         # its own eigendecomposition (solver signs)
        mb = mb.double().eval()
        sd = {k: v.clone() for k, v in ma.state_dict().items() if ".builder." not in k}
        for k in ("revin.weight", "revin.bias"):
            sd[k] = sd[k][perm]
        sd["rcf_q"] = sd["rcf_q"][:, perm]
        mb.load_state_dict(sd, strict=False)
        ob = mb.forward_window(X[..., perm], H_, timestamps=ts[:, :L_], freq="h", quantile_levels=(),
                               cycle_phase=phase)[0]
        # and flipping the stored eigenvectors of a trained model leaves its forecast unchanged
        ma.channel_encoder.builder.eigvecs.mul_(-1)
        oc = ma.forward_window(X, H_, timestamps=ts[:, :L_], freq="h", quantile_levels=(), cycle_phase=phase)[0]
    assert torch.allclose(ob, oa[..., perm], atol=1e-5), float((ob - oa[..., perm]).abs().max())
    assert torch.allclose(oc, oa, atol=1e-10)


def test_spectral_kernel_rejects_selective_codes():
    with pytest.raises(ValueError):
        TimeSeriesSSMBlock(hidden_dim=H, channel_z_dim=12, time_z_dim=Z, rank_dim=R, use_selective=True,
                           channel_kernel="spectral", channel_num_eig=6, channel_num_filters=2)


# ---- review v3 controls ---------------------------------------------------------------------------------------------
def test_complete_graph_random_basis_is_orthonormal_complement_and_seeded():
    from timeseriesssm.model.channel_graph import ChannelGraphEncoder
    train = torch.randn(300, 13)
    encs = []
    for seed in (2021, 2021, 2022):
        torch.manual_seed(seed)
        enc = ChannelGraphEncoder(num_channels=13, pe_dim=6, method="constant_random").fit(train)
        encs.append(enc)
    e = encs[0].builder.eigvecs.double()
    assert torch.allclose(e.T @ e, torch.eye(6, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(e.sum(0), torch.zeros(6, dtype=torch.float64), atol=1e-6)  # orthogonal to the constant
    assert torch.equal(encs[0].builder.eigvecs, encs[1].builder.eigvecs)              # same seed, same basis
    assert not torch.allclose(encs[0].builder.eigvecs, encs[2].builder.eigvecs)       # another seed, another basis
    # every orthonormal basis of the complement is an eigenbasis of the complete-graph Laplacian
    lap = encs[0].builder.laplacian.double()
    assert torch.allclose(lap @ e, e * encs[0].builder.eigvals.double(), atol=1e-5)
    assert encs[0].learned_basis is None


def test_complete_graph_learned_codes_train():
    cfg = dict(channel_graph="constant_learned")
    m = _v4(**cfg)
    m.fit_channel_graph(torch.randn(300, C))
    lb = m.channel_encoder.learned_basis
    assert isinstance(lb, torch.nn.Parameter) and any(p is lb for p in m.parameters())
    x, ph = torch.randn(2, 96, C), _phase(2, 24)
    m.train(); m.loss(x, torch.randn(2, 48, C), cycle_phase=ph).backward()
    assert lb.grad is not None and lb.grad.abs().sum() > 0


def test_stat_head_without_revin_starts_as_the_model_without_it():
    def build(**over):
        kw = dict(hidden_dim=16, rank_dim=8, patch_len=16, quantile_levels=(), cascade="nhits", context_len=96,
                  horizon_len=48, use_calendar=False, time_mix="causal_decay", use_glu_ffn=False, rcf_cycle=24,
                  use_revin=False, **over)
        torch.manual_seed(0)
        m = TimeSeriesSSM(TimeSeriesSSMConfig(**kw))
        m.fit_channel_graph(torch.randn(400, C))
        return m.eval()
    base, sh = build(), build(stat_head=True)
    assert sh.revin is None
    missing = sh.load_state_dict(base.state_dict(), strict=False).missing_keys
    assert missing and all(k.startswith("stat_mlp") for k in missing)
    x, ph = torch.randn(2, 96, C) * 3 + 1, _phase(2, 24)
    assert torch.allclose(base.forward_window(x, 48, cycle_phase=ph)[0], sh.forward_window(x, 48, cycle_phase=ph)[0],
                          atol=1e-5)
    sh.train(); sh.loss(x, torch.randn(2, 48, C), cycle_phase=ph).backward()
    assert sh.stat_mlp[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kernel", ["attention", "star"])
def test_content_only_channel_mixers_are_equivariant_and_train(kernel):
    blk = TimeSeriesSSMBlock(hidden_dim=H, channel_z_dim=Z, time_z_dim=Z, rank_dim=R, use_selective=False,
                             channel_kernel=kernel).eval()
    x, zt = _inputs(5)
    zc = torch.randn(C, Z)
    perm = torch.randperm(C)
    with torch.no_grad():
        y = blk(x, zc, zt)
        yp = blk(x[:, :, perm], zc[perm], zt)
        y_other_codes = blk(x, torch.randn(C, Z), zt)
    assert torch.allclose(yp, y[:, :, perm], atol=1e-5)
    assert torch.allclose(y, y_other_codes, atol=1e-6)  # content-only: the codes are ignored
    m = _v4(channel_kernel=kernel, use_selective=False)
    m.fit_channel_graph(torch.randn(300, C))
    x, ph = torch.randn(2, 96, C), _phase(2, 24)
    m.train(); m.loss(x, torch.randn(2, 48, C), cycle_phase=ph).backward()
    with pytest.raises(ValueError):
        TimeSeriesSSMBlock(hidden_dim=H, channel_z_dim=Z, time_z_dim=Z, rank_dim=R, use_selective=True,
                           channel_kernel=kernel)


@pytest.mark.parametrize("pools,ratios,blocks", [((1, 1, 1), (1, 1, 1), 1), ((1,), (1,), 3)])
def test_single_resolution_and_one_stack_cascades_keep_series_branch_and_stat_head(pools, ratios, blocks):
    m = _v4(nhits_pool_kernels=pools, nhits_downsample=ratios, nhits_blocks_per_stack=blocks, series_tokens=True,
            series_mixer="star", series_dim=16, stat_head=True)
    m.fit_channel_graph(torch.randn(300, C))
    assert len(m.nhits.stacks) == len(pools) and m.series is not None and m.stat_mlp is not None
    x, ph = torch.randn(2, 96, C), _phase(2, 24)
    assert m.forward_window(x, 48, cycle_phase=ph)[0].shape == (2, 48, C)
    m.train(); m.loss(x, torch.randn(2, 48, C), cycle_phase=ph).backward()
