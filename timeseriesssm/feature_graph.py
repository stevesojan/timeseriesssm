"""Feature-graph construction and cached Laplacian eigendecomposition."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from .streaming import StreamingFeatureGraph


GraphMethod = Literal["correlation", "mutual_information", "learnable", "constant"]

# Methods whose adjacency depends on the data and can therefore be accumulated
# incrementally from streamed chunks.
_DATA_DEPENDENT_METHODS = {"correlation", "mutual_information"}


def canonical_sign(evecs: torch.Tensor) -> torch.Tensor:
    """Flip each eigenvector (column) so that its largest-magnitude entry is positive. The rule follows the rows
    under a permutation, so for simple eigenvalues and a unique largest entry the eigenvectors of P^T L P are exactly
    the permuted eigenvectors of L (the solver's own sign choice is arbitrary)."""
    if evecs.numel() == 0:
        return evecs
    idx = evecs.abs().argmax(dim=0, keepdim=True)
    sgn = torch.sign(torch.gather(evecs, 0, idx))
    return evecs * torch.where(sgn == 0, torch.ones_like(sgn), sgn)


class FeatureGraphBuilder(nn.Module):
    """
    Builds a feature graph and caches Laplacian eigenpairs.

    The cached eigenvectors/eigenvalues are non-trainable buffers and are meant
    to be computed once during dataset initialization.
    """

    def __init__(
        self,
        num_features: int,
        pe_dim: int,
        method: GraphMethod = "correlation",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_features < 2:
            raise ValueError("num_features must be >= 2.")
        if pe_dim < 1:
            raise ValueError("pe_dim must be >= 1.")

        self.num_features = num_features
        self.pe_dim = pe_dim
        self.method = method
        self.eps = eps

        if method == "learnable":
            init = torch.randn(num_features, num_features) * 0.02
            self.learnable_logits = nn.Parameter(init)
        else:
            self.register_parameter("learnable_logits", None)

        # Pre-size buffers so inference can load_state_dict without calling fit() first
        # (torch.empty(0) cannot receive [F,F] weights from a checkpoint).
        zf = torch.zeros(num_features, num_features, dtype=torch.float32)
        self.register_buffer("adjacency", zf)
        self.register_buffer("laplacian", zf.clone())
        self.register_buffer("eigvals", torch.zeros(pe_dim, dtype=torch.float32))
        self.register_buffer("eigvecs", torch.zeros(num_features, pe_dim, dtype=torch.float32))

        # Streaming accumulator is plain state, never part of state_dict, so
        # checkpoints written before streaming existed still load unchanged.
        self._stream: Optional[StreamingFeatureGraph] = None

    def _build_adjacency(self, x: torch.Tensor) -> torch.Tensor:
        """Build a symmetric feature adjacency matrix."""
        if self.method == "correlation":
            centered = x - x.mean(dim=0, keepdim=True)
            std = centered.std(dim=0, keepdim=True).clamp_min(self.eps)
            z = centered / std
            corr = (z.transpose(0, 1) @ z) / max(1, z.shape[0] - 1)
            adj = corr.abs()
        elif self.method == "mutual_information":
            # Lightweight MI-like proxy using non-linear correlation magnitude.
            # This avoids heavyweight dependencies and remains deterministic.
            centered = x - x.mean(dim=0, keepdim=True)
            std = centered.std(dim=0, keepdim=True).clamp_min(self.eps)
            z = centered / std
            corr = (z.transpose(0, 1) @ z) / max(1, z.shape[0] - 1)
            adj = torch.sqrt(corr.abs().clamp_min(0.0))
        elif self.method == "constant":
            n = self.num_features
            device, dtype = x.device, x.dtype
            one = torch.ones(n, n, dtype=dtype, device=device)
            adj = one - torch.eye(n, dtype=dtype, device=device)
            return adj
        elif self.method == "learnable":
            logits = 0.5 * (self.learnable_logits + self.learnable_logits.transpose(0, 1))
            adj = torch.sigmoid(logits)
        else:
            raise ValueError(f"Unsupported graph construction method: {self.method}")

        adj = adj - torch.diag_embed(torch.diagonal(adj))
        adj = 0.5 * (adj + adj.transpose(0, 1))
        return adj

    def fit(self, x: torch.Tensor) -> "FeatureGraphBuilder":
        """
        Fit feature graph from data.

        Args:
            x: Tensor of shape [N, F] or [B, N, F].
        """
        if x.dim() == 3:
            x_2d = x.reshape(-1, x.shape[-1])
        elif x.dim() == 2:
            x_2d = x
        else:
            raise ValueError("x must have shape [N, F] or [B, N, F].")

        if x_2d.shape[1] != self.num_features:
            raise ValueError("x feature dimension does not match num_features.")

        with torch.no_grad():
            adj = self._build_adjacency(x_2d)
            self._apply_adjacency(adj)
        return self

    def _apply_adjacency(self, adj: torch.Tensor) -> None:
        """Normalize an adjacency into the cached Laplacian eigenpairs."""
        deg = adj.sum(dim=1)
        inv_sqrt_deg = (deg + self.eps).rsqrt()
        norm_adj = inv_sqrt_deg.unsqueeze(1) * adj * inv_sqrt_deg.unsqueeze(0)
        lap = torch.eye(self.num_features, device=adj.device, dtype=adj.dtype) - norm_adj

        evals, evecs = torch.linalg.eigh(lap)
        evals = evals.clamp_min(0.0)

        start = 1 if evals.shape[0] > 1 else 0
        end = min(self.num_features, start + self.pe_dim)
        evals = evals[start:end]
        evecs = evecs[:, start:end]
        if getattr(self, "canonical_sign", False):
            evecs = canonical_sign(evecs)

        if evals.numel() < self.pe_dim:
            pad = self.pe_dim - evals.numel()
            evals = torch.nn.functional.pad(evals, (0, pad), value=0.0)
            evecs = torch.nn.functional.pad(evecs, (0, pad), value=0.0)

        self.adjacency.detach().copy_(adj)
        self.laplacian.detach().copy_(lap)
        self.eigvals.detach().copy_(evals.to(dtype=self.eigvals.dtype))
        self.eigvecs.detach().copy_(evecs.to(dtype=self.eigvecs.dtype))

    # ------------------------------------------------------------------
    # Streaming / incremental interface
    # ------------------------------------------------------------------

    def supports_streaming(self) -> bool:
        """Whether this graph method needs data and can be accumulated incrementally."""
        return self.method in _DATA_DEPENDENT_METHODS

    def reset_stream(self) -> "FeatureGraphBuilder":
        """Start a fresh incremental accumulation of feature statistics."""
        self._stream = StreamingFeatureGraph(num_features=self.num_features, eps=self.eps)
        return self

    def update(self, chunk: torch.Tensor) -> "FeatureGraphBuilder":
        """Fold one streamed chunk ``[n, F]`` into the running statistics."""
        if self._stream is None:
            self.reset_stream()
        assert self._stream is not None
        self._stream.update(chunk)
        return self

    def stream_row_count(self) -> int:
        return 0 if self._stream is None else self._stream.count

    def finalize_adjacency(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return the adjacency implied by the accumulated statistics."""
        target = device if device is not None else self.adjacency.device
        if self.method == "constant":
            one = torch.ones(self.num_features, self.num_features, dtype=torch.float32, device=target)
            return one - torch.eye(self.num_features, dtype=torch.float32, device=target)
        if self.method == "learnable":
            if self.learnable_logits is None:
                raise RuntimeError("learnable_logits missing for method='learnable'.")
            logits = 0.5 * (self.learnable_logits + self.learnable_logits.transpose(0, 1))
            adj = torch.sigmoid(logits).to(device=target)
            adj = adj - torch.diag_embed(torch.diagonal(adj))
            return 0.5 * (adj + adj.transpose(0, 1))
        if self._stream is None:
            raise RuntimeError("No streamed statistics available. Call reset_stream()/update() first.")
        adj_np = self._stream.finalize_adjacency(self.method)
        return torch.from_numpy(adj_np).to(device=target, dtype=torch.float32)

    def finalize_stream(self, device: Optional[torch.device] = None) -> "FeatureGraphBuilder":
        """Close the incremental pass and install the resulting eigenpairs."""
        with torch.no_grad():
            self._apply_adjacency(self.finalize_adjacency(device=device))
        return self

    def stream_state_dict(self) -> Optional[dict]:
        """Accumulator state, so an incremental run can be resumed across sessions."""
        return None if self._stream is None else self._stream.state_dict()

    def load_stream_state_dict(self, state: dict) -> "FeatureGraphBuilder":
        self._stream = StreamingFeatureGraph(num_features=self.num_features, eps=self.eps)
        self._stream.load_state_dict(state)
        return self

    def is_fitted(self) -> bool:
        """Return whether graph eigenpairs look initialized (fit or loaded from checkpoint)."""
        if self.eigvecs.shape != (self.num_features, self.pe_dim):
            return False
        return bool(self.eigvecs.detach().norm().item() > 1e-12)

    def get_eigenpairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached (eigenvalues, eigenvectors)."""
        if not self.is_fitted():
            raise RuntimeError("FeatureGraphBuilder is not fitted. Call fit() first.")
        return self.eigvals, self.eigvecs
