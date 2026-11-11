"""Channel / variate graph with optional lag-aggregated correlation."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from ..feature_graph import FeatureGraphBuilder
from ..positional_encoding import EigenAugmentedPE

ChannelGraphMethod = Literal["correlation", "lag_correlation", "learnable", "constant", "provided",
                             "constant_random", "constant_learned"]


def random_complement_basis(n: int, k: int, seed: int) -> torch.Tensor:
    """[n, k] orthonormal columns orthogonal to the constant vector, drawn uniformly (QR of a centred Gaussian
    matrix). For the complete graph, every such basis is a valid set of non-constant Laplacian eigenvectors."""
    g = torch.Generator().manual_seed(int(seed) % (2**63))
    a = torch.randn(n, k, generator=g, dtype=torch.float64)
    a = a - a.mean(dim=0, keepdim=True)
    q, r = torch.linalg.qr(a)
    return (q * torch.sign(torch.diagonal(r)).unsqueeze(0)).to(torch.float32)


def lag_correlation_adjacency(x: torch.Tensor, max_lag: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Max |corr| over lags ``0..max_lag`` for columns of ``x`` [T, C].
    """
    if x.dim() != 2:
        raise ValueError("x must have shape [T, C].")
    t, c = x.shape
    adj = x.new_zeros(c, c)
    for lag in range(0, int(max_lag) + 1):
        a = x[lag:] if lag else x
        b = x[: t - lag] if lag else x
        if a.shape[0] < 2:
            break
        ac = a - a.mean(dim=0, keepdim=True)
        bc = b - b.mean(dim=0, keepdim=True)
        sa = ac.std(dim=0, keepdim=True).clamp_min(eps)
        sb = bc.std(dim=0, keepdim=True).clamp_min(eps)
        za = ac / sa
        zb = bc / sb
        corr = (za.transpose(0, 1) @ zb) / max(1, a.shape[0] - 1)
        adj = torch.maximum(adj, corr.abs())
    adj = adj - torch.diag_embed(torch.diagonal(adj))
    return 0.5 * (adj + adj.transpose(0, 1))


class ChannelGraphEncoder(nn.Module):
    """Fit a variate graph on train series and emit EigenAugmentedPE encodings."""

    def __init__(
        self,
        num_channels: int,
        pe_dim: int,
        num_filters: int = 4,
        method: ChannelGraphMethod = "lag_correlation",
        max_lag: int = 4,
        dropout: float = 0.0,
        sign: Literal["solver", "canonical"] = "solver",
        code_norm: str = "layer",
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.pe_dim = int(pe_dim)
        self.method = method
        self.max_lag = int(max_lag)
        builder_method = "correlation" if method == "lag_correlation" else method
        # complete-graph controls: the eigensolver's basis of the repeated eigenvalue is arbitrary (and localised), so
        # "constant_random" replaces it by a random orthonormal basis of the same eigenspace (seeded by the run seed)
        # and "constant_learned" starts from that basis and trains it (graph-free learned channel codes)
        self.basis = {"constant_random": "random", "constant_learned": "learned"}.get(method, "solver")
        if self.basis != "solver":
            builder_method = "constant"
        self.learned_basis: nn.Parameter | None = None
        if builder_method not in {"correlation", "learnable", "constant", "mutual_information"}:
            builder_method = "correlation"
        self.builder = (
            FeatureGraphBuilder(
                num_features=num_channels,
                pe_dim=pe_dim,
                method=builder_method,  # type: ignore[arg-type]
            )
            if num_channels >= 2
            else None
        )
        if self.builder is not None:
            self.builder.canonical_sign = sign == "canonical"
        self.pe = EigenAugmentedPE(pe_dim=pe_dim, num_filters=num_filters, dropout=dropout, norm=code_norm)
        # method="provided": a known sensor graph (e.g. road network) replaces the estimated one.
        self._provided_adj: torch.Tensor | None = None

    def set_adjacency(self, adj: torch.Tensor) -> "ChannelGraphEncoder":
        """Store a known [C, C] adjacency (symmetrized, zero diagonal) for method='provided'."""
        adj = torch.as_tensor(adj, dtype=torch.float32)
        if adj.shape != (self.num_channels, self.num_channels):
            raise ValueError(f"adjacency must be [{self.num_channels}, {self.num_channels}], got {tuple(adj.shape)}.")
        adj = 0.5 * (adj.abs() + adj.abs().transpose(0, 1))
        self._provided_adj = adj - torch.diag_embed(torch.diagonal(adj))
        return self

    @property
    def z_dim(self) -> int:
        return self.pe.output_dim

    def fit(self, x: torch.Tensor) -> "ChannelGraphEncoder":
        """
        Args:
            x: [T, C] or [B, T, C] train values of the *joint* channel stack.
        """
        if self.builder is None:
            return self
        if x.dim() == 3:
            x2 = x.reshape(-1, x.shape[-1])
        elif x.dim() == 2:
            x2 = x
        else:
            raise ValueError("x must have shape [T, C] or [B, T, C].")
        if self.method == "provided":
            if self._provided_adj is None:
                raise RuntimeError("method='provided' needs set_adjacency(adj) before fit().")
            with torch.no_grad():
                self.builder._apply_adjacency(self._provided_adj.to(x2.device))
        elif self.method == "lag_correlation":
            adj = lag_correlation_adjacency(x2, max_lag=self.max_lag)
            with torch.no_grad():
                self.builder._apply_adjacency(adj)
        else:
            self.builder.fit(x2)
        if self.basis != "solver":
            ev = self.builder.eigvecs
            k = min(self.num_channels - 1, self.pe_dim)
            basis = torch.zeros_like(ev)
            basis[:, :k] = random_complement_basis(self.num_channels, k, torch.initial_seed()).to(ev.device)
            with torch.no_grad():
                self.builder.eigvecs.copy_(basis)
            if self.basis == "learned":
                self.learned_basis = nn.Parameter(basis.clone())
        return self

    def is_fitted(self) -> bool:
        if self.builder is None:
            return True
        return self.builder.is_fitted()

    def encodings(self) -> torch.Tensor:
        """Return [C, Z] channel positional encodings."""
        if self.builder is None:
            z = torch.zeros(self.num_channels, self.z_dim)
            if next(self.pe.parameters(), None) is not None:
                z = z.to(next(self.pe.parameters()).device)
            return z
        eigvals, eigvecs = self.builder.get_eigenpairs()
        if self.learned_basis is not None:
            eigvecs = self.learned_basis
        return self.pe(eigvals, eigvecs)
