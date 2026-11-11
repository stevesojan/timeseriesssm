"""Training and inference API for TimeSeriesSSM.

``Forecaster`` trains one model for one horizon with the recipe of the paper:

- lookback windows of ``seq_len`` steps, every sliding window of the training series, shuffled;
- the objective of ``TimeSeriesSSM.loss`` (horizon-weighted absolute error plus the FreDF term);
- Adam (AdamW when weight decay is set), a cosine or halving learning-rate schedule;
- an exponential moving average of the weights, used for validation, early stopping and prediction;
- the epoch with the lowest validation MSE is kept.

The channel graph is fitted on the training series only.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import WEEK, to_seconds
from .model import TimeSeriesSSM, TimeSeriesSSMConfig

PRESET_DIR = Path(__file__).resolve().parent / "presets"


def list_presets() -> list[str]:
    """Names of the selected configurations shipped with the package (one per TSLib dataset)."""
    return sorted(p.stem for p in PRESET_DIR.glob("*.json"))


def load_preset(name: str) -> dict:
    """The selected configuration of a TSLib dataset: ``{"model": ..., "train": ..., "freq": ..., ...}``."""
    path = PRESET_DIR / f"{name}.json"
    if not path.exists():
        raise ValueError(f"unknown preset {name!r}; available: {', '.join(list_presets())}")
    return json.loads(path.read_text())


@dataclass
class TrainConfig:
    lr: float = 1e-3
    schedule: str = "cosine"     # "cosine": 0.5 (1 + cos(pi e / E)); "type1": halve every epoch (TSLib)
    epochs: int = 10
    patience: int = 3            # epochs without validation improvement before stopping
    batch_size: int = 32
    weight_decay: float = 0.0    # > 0 uses AdamW
    grad_clip: float = 0.0       # > 0 clips the global gradient norm
    ema_decay: float = 0.995     # > 0 keeps an exponential moving average of the weights
    eval_batch_size: int = 64
    seed: int = 2021
    grad_checkpoint: bool = False  # trades compute for memory; does not change the result


def _lr_at(epoch: int, cfg: TrainConfig) -> float:
    if cfg.schedule == "type1":
        return cfg.lr * 0.5 ** epoch
    if cfg.schedule == "cosine":
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, cfg.epochs)))
    if cfg.schedule == "constant":
        return cfg.lr
    raise ValueError(f"unknown schedule {cfg.schedule!r}")


class _Windows:
    """All sliding windows (x [L, C], y [H, C], stamps [L + H]) of one series, gathered on the device."""

    def __init__(self, values: np.ndarray, stamps: np.ndarray, seq_len: int, horizon: int, device) -> None:
        self.vals = torch.as_tensor(values, dtype=torch.float32).to(device)
        self.stamps = torch.as_tensor(stamps, dtype=torch.float64)
        self.n = len(values) - seq_len - horizon + 1
        if self.n <= 0:
            raise ValueError(f"series of length {len(values)} is shorter than seq_len + horizon "
                             f"= {seq_len + horizon}")
        self.off_x = torch.arange(seq_len, device=device)
        self.off_y = torch.arange(seq_len, seq_len + horizon, device=device)
        self.off_t = torch.arange(seq_len + horizon)

    def batches(self, batch_size: int, shuffle: bool):
        for i in DataLoader(range(self.n), batch_size=batch_size, shuffle=shuffle, drop_last=False):
            ig = i.to(self.vals.device)
            yield (self.vals[ig[:, None] + self.off_x], self.vals[ig[:, None] + self.off_y],
                   self.stamps[i[:, None] + self.off_t])


class Forecaster:
    """Train, evaluate and apply a TimeSeriesSSM forecaster for one horizon.

    Args:
        model: fields of ``TimeSeriesSSMConfig`` (``context_len`` and ``horizon_len`` are set from ``seq_len``
            and ``horizon``).
        train: a ``TrainConfig`` or a dict of its fields.
        horizon: number of future steps predicted at once.
        seq_len: lookback length.
        freq: sampling interval as a pandas-style string ("h", "15min", "10min", ...).
        standardize: fit a per-channel standard scaler on the training series and return predictions in the
            original units. Set it to False for series that are already normalised (as in the TSLib protocol).
    """

    def __init__(self, model: dict | None = None, train: TrainConfig | dict | None = None, *, horizon: int = 96,
                 seq_len: int = 96, freq: str = "h", standardize: bool = True, device: str | None = None) -> None:
        self.model_kwargs = {k: v for k, v in (model or {}).items() if k not in ("context_len", "horizon_len")}
        if "quantile_levels" in self.model_kwargs:
            self.model_kwargs["quantile_levels"] = tuple(self.model_kwargs["quantile_levels"])
        self.train_cfg = train if isinstance(train, TrainConfig) else TrainConfig(**(train or {}))
        self.horizon, self.seq_len, self.freq, self.standardize = int(horizon), int(seq_len), freq, standardize
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net: TimeSeriesSSM | None = None
        self.mean = self.std = None
        self.time_offset = 0.0
        self.step = None
        self.history: list[dict] = []

    @classmethod
    def from_preset(cls, name: str, horizon: int = 96, **kwargs) -> "Forecaster":
        """The selected configuration of a TSLib dataset (``list_presets()``), e.g. ``from_preset("ETTh1")``."""
        p = load_preset(name)
        train = {**p["train"], **kwargs.pop("train_overrides", {})}
        return cls(p["model"], train, horizon=horizon, seq_len=p["seq_len"], freq=kwargs.pop("freq", p["freq"]),
                   **kwargs)

    # ------------------------------------------------------------------ building blocks
    def _build(self, n_channels: int) -> TimeSeriesSSM:
        cfg = TimeSeriesSSMConfig(**self.model_kwargs, context_len=self.seq_len, horizon_len=self.horizon,
                                  grad_checkpoint=self.train_cfg.grad_checkpoint)
        net = TimeSeriesSSM(cfg)
        net.configure_channels(n_channels)
        return net

    def _uses_time(self) -> bool:
        return bool(self.model_kwargs.get("rcf_cycle", 0)) or bool(self.model_kwargs.get("use_calendar", False))

    def _stamps(self, timestamps, n: int) -> np.ndarray:
        if timestamps is None:
            if self._uses_time():
                raise ValueError("this configuration uses timestamps (cycle removal or calendar codes): "
                                 "pass the timestamps of the series")
            return np.arange(n, dtype=np.float64) * (self.step or 1.0)
        t = to_seconds(timestamps)
        if len(t) != n:
            raise ValueError(f"got {len(t)} timestamps for a series of length {n}")
        return t - self.time_offset

    def _phase(self, t: torch.Tensor) -> torch.Tensor | None:
        w = int(self.model_kwargs.get("rcf_cycle", 0))
        if w <= 0:
            return None
        return torch.remainder(torch.floor(t.double() / self.step), w).long().to(self.device)

    def _norm(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std if self.standardize else x

    def _forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net.forward_window(x, self.horizon, timestamps=t.float().to(self.device), freq=self.freq,
                                       quantile_levels=(), cycle_phase=self._phase(t))[0]

    def _swap(self, weights: dict) -> dict:
        old = {}
        with torch.no_grad():
            for k, p in self.net.named_parameters():
                old[k] = p.detach().clone()
                p.copy_(weights[k])
        return old

    def _errors(self, win: _Windows) -> tuple[float, float]:
        self.net.eval()
        se = ae = n = 0.0
        with torch.no_grad():
            for x, y, t in win.batches(self.train_cfg.eval_batch_size, shuffle=False):
                err = self._forward(x, t) - y
                se += float((err ** 2).sum()); ae += float(err.abs().sum()); n += err.numel()
        return se / n, ae / n

    # ------------------------------------------------------------------ public API
    def fit(self, train, val=None, train_timestamps=None, val_timestamps=None, verbose: bool = True) -> "Forecaster":
        """Train on ``train`` ([T, C] array) and keep the epoch with the lowest MSE on ``val``.

        ``val`` should include the ``seq_len`` steps before its first target (TSLib convention). Without ``val``,
        the weights of the last epoch are kept. Timestamps (datetime-like or seconds, one per row) are needed when
        the configuration uses cycle removal or calendar codes, which every preset does.
        """
        cfg = self.train_cfg
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        train = np.asarray(train, dtype=np.float64)
        if train.ndim == 1:
            train = train[:, None]
        if train_timestamps is not None:
            t_raw = to_seconds(train_timestamps)
            self.time_offset = float(np.floor(t_raw[0] / WEEK) * WEEK)  # keeps the phase, fits float32
            self.step = float(np.median(np.diff(t_raw)))
        elif self._uses_time():
            raise ValueError("this configuration uses timestamps: pass train_timestamps (and val_timestamps)")
        else:
            self.step = 1.0
        if self.standardize:
            self.mean, self.std = train.mean(axis=0), train.std(axis=0)
            self.std = np.where(self.std == 0, 1.0, self.std)
        tr = _Windows(self._norm(train), self._stamps(train_timestamps, len(train)), self.seq_len, self.horizon,
                      self.device)
        va = None
        if val is not None:
            val = np.asarray(val, dtype=np.float64)
            val = val[:, None] if val.ndim == 1 else val
            va = _Windows(self._norm(val), self._stamps(val_timestamps, len(val)), self.seq_len, self.horizon,
                          self.device)
        self.net = self._build(train.shape[1])
        self.net.fit_channel_graph(torch.from_numpy(self._norm(train).astype(np.float32)))  # training rows only
        self.net.to(self.device)
        opt = (torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
               if cfg.weight_decay > 0 else torch.optim.Adam(self.net.parameters(), lr=cfg.lr))
        ema = {k: v.detach().clone() for k, v in self.net.named_parameters()} if cfg.ema_decay > 0 else None
        best, best_state, bad, self.history = float("inf"), None, 0, []
        t0 = time.time()
        for ep in range(cfg.epochs):
            for g in opt.param_groups:
                g["lr"] = _lr_at(ep, cfg)
            self.net.train()
            tot, nb = 0.0, 0
            for x, y, t in tr.batches(cfg.batch_size, shuffle=True):
                opt.zero_grad(set_to_none=True)
                loss = self.net.loss(x, y, timestamps=t.float().to(self.device), freq=self.freq,
                                     cycle_phase=self._phase(t))
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
                opt.step()
                if ema is not None:
                    with torch.no_grad():
                        keys = list(ema)
                        torch._foreach_mul_([ema[k] for k in keys], cfg.ema_decay)
                        torch._foreach_add_([ema[k] for k in keys],
                                            [p.detach() for _, p in self.net.named_parameters()],
                                            alpha=1.0 - cfg.ema_decay)
                tot += loss.item(); nb += 1
            live = self._swap(ema) if ema is not None else None  # validate (and keep) the averaged weights
            val_mse = self._errors(va)[0] if va is not None else float("nan")
            self.history.append({"epoch": ep, "train_loss": tot / max(1, nb), "val_mse": val_mse,
                                 "seconds": time.time() - t0})
            if verbose:
                print(f"epoch {ep}: train loss {tot / max(1, nb):.4f}  val mse {val_mse:.4f}", flush=True)
            stop = False
            if va is None or val_mse < best:
                best, bad = (val_mse if va is not None else best), 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                stop = bad >= cfg.patience
            if live is not None:
                self._swap(live)  # back to the live weights for training
            if stop:
                break
        self.net.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        self.net.eval()
        return self

    def evaluate(self, series, timestamps=None) -> dict:
        """MSE and MAE over every sliding window of ``series`` ([T, C], including the ``seq_len`` steps before its
        first target), on the scale the model works on (normalised when ``standardize`` is True)."""
        self._check_fitted()
        series = np.asarray(series, dtype=np.float64)
        series = series[:, None] if series.ndim == 1 else series
        win = _Windows(self._norm(series), self._stamps(timestamps, len(series)), self.seq_len, self.horizon,
                       self.device)
        mse, mae = self._errors(win)
        return {"mse": mse, "mae": mae, "windows": win.n}

    def predict(self, context, timestamps=None) -> np.ndarray:
        """Forecast the next ``horizon`` steps.

        Args:
            context: [L, C] or [B, L, C] lookback values with ``L = seq_len``, in the units of the training data.
            timestamps: the timestamps of the lookback ([L] or [B, L]); the horizon steps follow at the sampling
                interval of the training data. Needed when the configuration uses cycle removal or calendar codes.
        Returns:
            [H, C] or [B, H, C] forecasts in the units of the training data.
        """
        self._check_fitted()
        x = np.asarray(context, dtype=np.float64)
        single = x.ndim == 2
        x = x[None] if single else x
        if x.shape[1] != self.seq_len:
            raise ValueError(f"context must have seq_len = {self.seq_len} steps, got {x.shape[1]}")
        b = x.shape[0]
        if timestamps is None:
            if self._uses_time():
                raise ValueError("this configuration uses timestamps: pass the timestamps of the context")
            t = np.tile(np.arange(self.seq_len + self.horizon, dtype=np.float64), (b, 1))
        else:
            ts = np.asarray(timestamps)
            ts = ts[None] if ts.ndim == 1 else ts
            past = np.stack([to_seconds(row) for row in ts]) - self.time_offset
            future = past[:, -1:] + self.step * np.arange(1, self.horizon + 1)
            t = np.concatenate([past, future], axis=1)
        xn = torch.as_tensor(self._norm(x), dtype=torch.float32, device=self.device)
        self.net.eval()
        with torch.no_grad():
            out = self._forward(xn, torch.as_tensor(t, dtype=torch.float64)).double().cpu().numpy()
        if self.standardize:
            out = out * self.std + self.mean
        return out[0] if single else out

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        torch.save({"model_kwargs": self.model_kwargs, "train": asdict(self.train_cfg), "horizon": self.horizon,
                    "seq_len": self.seq_len, "freq": self.freq, "standardize": self.standardize,
                    "mean": self.mean, "std": self.std, "time_offset": self.time_offset, "step": self.step,
                    "n_channels": self.net._n_channels, "state_dict": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "Forecaster":
        s = torch.load(path, map_location="cpu", weights_only=False)
        f = cls(s["model_kwargs"], s["train"], horizon=s["horizon"], seq_len=s["seq_len"], freq=s["freq"],
                standardize=s["standardize"], device=device)
        f.mean, f.std, f.time_offset, f.step = s["mean"], s["std"], s["time_offset"], s["step"]
        f.net = f._build(s["n_channels"])
        f.net.load_state_dict(s["state_dict"])
        f.net.to(f.device).eval()
        return f

    @property
    def n_params(self) -> int:
        self._check_fitted()
        return sum(p.numel() for p in self.net.parameters())

    def _check_fitted(self) -> None:
        if self.net is None:
            raise RuntimeError("call fit() (or Forecaster.load) first")

    def copy(self) -> "Forecaster":
        return copy.deepcopy(self)
