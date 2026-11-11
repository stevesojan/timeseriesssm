"""Tests of the Forecaster API and of the shipped presets (CPU, a few seconds)."""
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timeseriesssm import Forecaster, TimeSeriesSSM, TimeSeriesSSMConfig, list_presets, load_preset  # noqa: E402

# parameter counts of the selected models at H=96, as reported in the paper (Table 3 / Appendix G)
PAPER_PARAMS = {"ETTh1": (7, 441914), "ETTh2": (7, 1241113), "ETTm1": (7, 687042), "ETTm2": (7, 383269),
                "Weather": (21, 705018), "ECL": (321, 2965901), "Traffic": (862, 839612)}


def test_presets_listed():
    assert list_presets() == sorted(PAPER_PARAMS)


@pytest.mark.parametrize("name", sorted(PAPER_PARAMS))
def test_preset_parameter_count_matches_paper(name):
    p = load_preset(name)
    c, n = PAPER_PARAMS[name]
    net = TimeSeriesSSM(TimeSeriesSSMConfig(**{**p["model"], "quantile_levels": ()}, context_len=p["seq_len"],
                                            horizon_len=96))
    net.configure_channels(c)
    assert sum(q.numel() for q in net.parameters()) == n


def _toy(n=1200, c=4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = np.sin(2 * np.pi * t / 24)[:, None] * np.linspace(1, 2, c)[None]
    x = 10 + 3 * base + 0.1 * rng.standard_normal((n, c))
    stamps = np.datetime64("2020-01-01T00:00") + t.astype("timedelta64[h]")
    return x, stamps


SMALL = dict(patch_len=16, hidden_dim=32, rank_dim=16, num_blocks=2, dropout=0.0, cascade="nhits",
             nhits_blocks_per_stack=1, time_mix="causal_decay", use_glu_ffn=False, rcf_cycle=24,
             mse_weight=0.0, mae_weight=1.0, horizon_weight="arctan", freq_loss_weight=0.5, stat_head=True,
             series_tokens=True, series_dim=32, series_layers=1, series_mixer="star", quantile_levels=())


def _fit(tmp_path=None, epochs=2):
    x, s = _toy()
    f = Forecaster(SMALL, {"epochs": epochs, "batch_size": 32, "lr": 1e-3, "patience": 3}, horizon=24, seq_len=96,
                   freq="h", device="cpu")
    f.fit(x[:800], x[800 - 96:1000], s[:800], s[800 - 96:1000], verbose=False)
    return f, x, s


def test_fit_predict_shapes_and_units():
    f, x, s = _fit()
    one = f.predict(x[1000:1096], s[1000:1096])
    batch = f.predict(np.stack([x[1000:1096], x[1010:1106]]), np.stack([s[1000:1096], s[1010:1106]]))
    assert one.shape == (24, 4) and batch.shape == (2, 24, 4)
    np.testing.assert_allclose(batch[0], one, rtol=1e-5, atol=1e-5)
    assert abs(one.mean() - x.mean()) < 3.0  # returned in the original units
    m = f.evaluate(x[1000 - 96:], s[1000 - 96:])
    assert m["windows"] == len(x) - 1000 - 24 + 1 and np.isfinite(m["mse"])
    assert len(f.history) >= 1


def test_save_load_roundtrip(tmp_path):
    f, x, s = _fit()
    path = tmp_path / "model.pt"
    f.save(path)
    g = Forecaster.load(path, device="cpu")
    np.testing.assert_allclose(g.predict(x[1000:1096], s[1000:1096]), f.predict(x[1000:1096], s[1000:1096]),
                               rtol=1e-6, atol=1e-6)


def test_channel_permutation_equivariance():
    """Proposition 2(i): relabelling the channels with their channel-specific parameters relabels the forecast."""
    torch.manual_seed(0)
    f, x, s = _fit(epochs=1)
    perm = torch.tensor([2, 0, 3, 1])
    g = f.copy()
    with torch.no_grad():
        g.mean, g.std = g.mean[perm.numpy()], g.std[perm.numpy()]
        sd = g.net.state_dict()
        for k, v in sd.items():
            if k in ("revin.weight", "revin.bias"):
                v.copy_(v[perm])
            elif k == "rcf_q":
                v.copy_(v[:, perm])
        enc = g.net.channel_encoder.builder
        enc.eigvecs.copy_(enc.eigvecs[perm])
        assert not torch.allclose(sd["revin.weight"], torch.ones(4))  # trained, so the permutation is tested
    ctx, ts = x[1000:1096], s[1000:1096]
    np.testing.assert_allclose(g.predict(ctx[:, perm.numpy()], ts), f.predict(ctx, ts)[:, perm.numpy()],
                               rtol=1e-4, atol=1e-4)


def test_timestamps_required_for_cycle_removal():
    x, s = _toy()
    f = Forecaster(SMALL, {"epochs": 1}, horizon=24, device="cpu")
    with pytest.raises(ValueError):
        f.fit(x[:800], verbose=False)


def test_cycle_phase_from_float64_stamps_matches_index_phase():
    """The cycle phase is computed from float64 seconds, so it stays exact far from the epoch (float32 would not)."""
    for step, w, freq in ((600.0, 144, "10min"), (3600.0, 168, "h")):
        f = Forecaster({**SMALL, "rcf_cycle": w}, {"epochs": 1}, horizon=24, freq=freq, device="cpu")
        f.step = step
        stamps = 2 * 7 * 86400 + step * np.arange(60000, dtype=np.float64)
        for start in (0, 1, 12345, 59000):
            ph = f._phase(torch.from_numpy(stamps[start:start + 144]).unsqueeze(0))[0]
            ref = (torch.arange(start, start + 144) + int(stamps[0] // step)) % w
            assert torch.equal(ph, ref)
