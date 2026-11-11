"""Incremental feature-graph statistics (used by ``feature_graph.FeatureGraphBuilder``)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

__all__ = ["StreamingFeatureGraph"]


class StreamingFeatureGraph:
    """
    Incremental first/second moments for the feature graph.

    Uses Chan et al.'s pairwise co-moment update, which is the numerically
    stable counterpart of a naive ``sum(x)`` / ``sum(x x^T)`` accumulator and is
    invariant to chunk boundaries. ``finalize_adjacency`` reproduces
    ``FeatureGraphBuilder._build_adjacency`` for the data-dependent methods.
    """

    def __init__(self, num_features: int, eps: float = 1e-6) -> None:
        if num_features < 2:
            raise ValueError("num_features must be >= 2.")
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> "StreamingFeatureGraph":
        self._count = 0
        self._mean = np.zeros(self.num_features, dtype=np.float64)
        self._m2 = np.zeros((self.num_features, self.num_features), dtype=np.float64)
        return self

    @property
    def count(self) -> int:
        return int(self._count)

    def update(self, chunk: np.ndarray | torch.Tensor) -> "StreamingFeatureGraph":
        """Fold one chunk of rows ``[n, F]`` into the running statistics."""
        if isinstance(chunk, torch.Tensor):
            block = chunk.detach().to(device="cpu", dtype=torch.float64).numpy()
        else:
            block = np.asarray(chunk, dtype=np.float64)
        if block.ndim == 3:
            block = block.reshape(-1, block.shape[-1])
        if block.ndim != 2:
            raise ValueError("chunk must have shape [n, F] or [b, n, F].")
        if block.shape[1] != self.num_features:
            raise ValueError("chunk feature dimension does not match num_features.")

        n_b = int(block.shape[0])
        if n_b == 0:
            return self

        mean_b = block.mean(axis=0)
        centered = block - mean_b
        m2_b = centered.T @ centered

        if self._count == 0:
            self._count = n_b
            self._mean = mean_b
            self._m2 = m2_b
            return self

        n_a = self._count
        n_t = n_a + n_b
        delta = mean_b - self._mean
        self._m2 = self._m2 + m2_b + np.outer(delta, delta) * (n_a * n_b / n_t)
        self._mean = (n_a * self._mean + n_b * mean_b) / n_t
        self._count = n_t
        return self

    def covariance(self) -> np.ndarray:
        if self._count < 2:
            raise RuntimeError("At least 2 rows are required to finalize feature-graph statistics.")
        return self._m2 / float(self._count - 1)

    def finalize_adjacency(self, method: str) -> np.ndarray:
        """
        Produce the symmetric, zero-diagonal adjacency for ``method``.

        Mirrors ``FeatureGraphBuilder._build_adjacency`` including the
        ``clamp_min(eps)`` guard applied to per-feature standard deviations.
        """
        if method not in {"correlation", "mutual_information"}:
            raise ValueError(
                f"StreamingFeatureGraph only accumulates data-dependent methods; got {method!r}."
            )
        cov = self.covariance()
        std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        std = np.maximum(std, self.eps)
        corr = cov / np.outer(std, std)

        if method == "correlation":
            adj = np.abs(corr)
        else:
            adj = np.sqrt(np.clip(np.abs(corr), 0.0, None))

        adj = adj.astype(np.float32, copy=False)
        np.fill_diagonal(adj, 0.0)
        return 0.5 * (adj + adj.T)

    def state_dict(self) -> dict[str, Any]:
        """Serializable statistics, so incremental learning can resume later."""
        return {
            "num_features": self.num_features,
            "eps": self.eps,
            "count": int(self._count),
            "mean": self._mean.copy(),
            "m2": self._m2.copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> "StreamingFeatureGraph":
        self.num_features = int(state["num_features"])
        self.eps = float(state["eps"])
        self._count = int(state["count"])
        self._mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self._m2 = np.asarray(state["m2"], dtype=np.float64).copy()
        return self


# ---------------------------------------------------------------------------
# Memory monitoring and adaptive ingestion
# ---------------------------------------------------------------------------
