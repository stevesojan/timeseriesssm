"""Forecast a synthetic multivariate hourly series with TimeSeriesSSM (runs on a CPU in about a minute).

    python examples/quickstart.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from timeseriesssm import Forecaster  # noqa: E402

# 8 coupled channels: a shared daily cycle, a channel-specific phase and noise
rng = np.random.default_rng(0)
n, c = 4000, 8
idx = pd.date_range("2024-01-01", periods=n, freq="h")
hour = np.arange(n)
shared = np.sin(2 * np.pi * hour / 24)[:, None]
series = 50 + 10 * shared * np.linspace(0.5, 1.5, c) + np.cos(2 * np.pi * hour / 168)[:, None] \
    + rng.normal(0, 1, (n, c))

train, val, test = slice(0, 2800), slice(2800 - 96, 3400), slice(3400 - 96, n)

model = dict(  # a small configuration; Forecaster.from_preset("ETTh1") gives the configuration of the paper
    patch_len=16, hidden_dim=64, rank_dim=32, num_blocks=2, dropout=0.1,
    cascade="nhits", nhits_blocks_per_stack=1, use_glu_ffn=False,
    time_mix="causal_decay",           # decaying scan over the patches of each channel
    channel_mix_scale=1.0,             # graph state-space convolution across channels
    rcf_cycle=24,                      # remove a learned daily cycle
    series_tokens=True, series_dim=64, series_layers=1, series_mixer="star",
    stat_head=True, mae_weight=1.0, mse_weight=0.0, horizon_weight="arctan", freq_loss_weight=0.5,
)
fc = Forecaster(model, {"epochs": 5, "batch_size": 32, "lr": 1e-3, "schedule": "cosine"}, horizon=24, freq="h",
                device="cpu")
fc.fit(series[train], series[val], idx[train], idx[val])

print("test (normalised scale):", fc.evaluate(series[test], idx[test]))
forecast = fc.predict(series[-96:], idx[-96:])   # next 24 hours, original units, shape [24, 8]
print("forecast shape:", forecast.shape)
fc.save("quickstart_model.pt")
restored = Forecaster.load("quickstart_model.pt", device="cpu")
assert np.allclose(restored.predict(series[-96:], idx[-96:]), forecast, atol=1e-5)
