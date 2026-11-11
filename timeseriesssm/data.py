"""Time-Series-Library (TSLib) long-term forecasting protocol (thuml/Time-Series-Library, features='M').

- ETTh*: 12/4/4 months of hourly data; ETTm*: the same in 15-minute steps.
- Other datasets (Weather, ECL, Traffic, ...): 70% train, 20% test, the rest validation.
- Validation and test windows start ``seq_len`` steps before their border, so the first target is the first row of
  the split.
- A standard scaler is fitted on the training rows only; errors are reported on this normalised scale.
- Every sliding window is scored (``n - seq_len - pred_len + 1`` windows per split).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

WEEK = 7 * 86400


@dataclass
class TSLibSplits:
    values: np.ndarray      # [T, C] normalised series
    timestamps: np.ndarray  # [T] seconds (float64)
    borders: dict           # split -> (start, end) row indices; windows are taken within [start, end)
    columns: list

    def split(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.borders[name]
        return self.values[a:b], self.timestamps[a:b]


def load_tslib(csv_path: str | Path, seq_len: int = 96) -> TSLibSplits:
    """Load one TSLib csv (first column ``date``) with the official splits and training-split standardisation.

    The dataset kind is taken from the file name: ``ETTh*``, ``ETTm*`` or any other (70/10/20 split).
    """
    path = Path(csv_path)
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    values = df.drop(columns=["date"]).to_numpy(dtype=np.float64)
    n, name = len(df), path.stem
    if name.startswith(("ETTh", "ETTm")):
        m = 30 * 24 * (4 if name.startswith("ETTm") else 1)
        b1 = [0, 12 * m - seq_len, 12 * m + 4 * m - seq_len]
        b2 = [12 * m, 12 * m + 4 * m, 12 * m + 8 * m]
    else:
        num_train, num_test = int(n * 0.7), int(n * 0.2)
        num_vali = n - num_train - num_test
        b1 = [0, num_train - seq_len, n - num_test - seq_len]
        b2 = [num_train, num_train + num_vali, n]
    train = values[b1[0]:b2[0]]
    mean, std = train.mean(axis=0), train.std(axis=0)  # sklearn StandardScaler (ddof=0)
    std = np.where(std == 0, 1.0, std)
    norm = ((values - mean) / std).astype(np.float32)
    secs = dates.astype("int64").to_numpy() / 1e9
    return TSLibSplits(norm, secs, {"train": (b1[0], b2[0]), "val": (b1[1], b2[1]), "test": (b1[2], b2[2])},
                       list(df.columns[1:]))


def to_seconds(timestamps) -> np.ndarray:
    """Datetime-like (pandas / numpy datetime64 / strings) or numeric seconds -> float64 seconds."""
    arr = np.asarray(timestamps)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.float64)
    return pd.to_datetime(arr).astype("int64").to_numpy() / 1e9
