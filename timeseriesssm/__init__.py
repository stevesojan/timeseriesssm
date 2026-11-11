"""TimeSeriesSSM: multivariate long-term forecasting with graph state-space convolution across channels.

The model is ``TimeSeriesSSM`` with its configuration ``TimeSeriesSSMConfig``; ``Forecaster`` wraps it with the
training recipe of the paper (windowing, objective, weight averaging, schedules, early stopping) and the
selected configuration of every TSLib dataset (``list_presets`` / ``Forecaster.from_preset``).
"""

from __future__ import annotations

from .forecaster import Forecaster, TrainConfig, list_presets, load_preset
from .model import TimeSeriesSSM, TimeSeriesSSMConfig

__version__ = "1.0.0"

__all__ = ["Forecaster", "TrainConfig", "TimeSeriesSSM", "TimeSeriesSSMConfig", "list_presets", "load_preset",
           "__version__"]
