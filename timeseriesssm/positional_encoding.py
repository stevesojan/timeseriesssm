"""Eigenvalue-augmented positional encoding for feature nodes."""

from __future__ import annotations

import torch
import torch.nn as nn


class GlobalRMSScale(nn.Module):
    """Divide all codes [F, Z] by one scalar, their root-mean-square over features and entries."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z / torch.sqrt(z.pow(2).mean() + self.eps)


class EigenAugmentedPE(nn.Module):
    """
    Construct z_f = [phi_1(lambda) * p_f, ..., phi_m(lambda) * p_f].

    The filter generator phi(.) is shared across eigenvalue coordinates, making
    it permutation-equivariant over the eigenvalue axis.
    """

    def __init__(
        self,
        pe_dim: int,
        num_filters: int = 4,
        phi_hidden_dim: int = 32,
        dropout: float = 0.0,
        norm: str = "layer",
    ) -> None:
        super().__init__()
        self.pe_dim = pe_dim
        self.num_filters = num_filters

        self.phi_mlp = nn.Sequential(
            nn.Linear(1, phi_hidden_dim),
            nn.GELU(),
            nn.Linear(phi_hidden_dim, num_filters),
        )
        self.dropout = nn.Dropout(dropout)
        # "layer": LayerNorm over each flattened code (the reported models). Its mean and offset do not change sign
        # with an eigenvector, so the spectral kernel uses "global_rms": all codes divided by one scalar, their
        # root-mean-square, which is invariant to eigenvector signs and to the channel order
        if norm == "layer":
            self.norm = nn.LayerNorm(pe_dim * num_filters)
        elif norm == "global_rms":
            self.norm = GlobalRMSScale()
        else:
            raise ValueError(f"unknown code norm {norm!r}")

    @property
    def output_dim(self) -> int:
        return self.pe_dim * self.num_filters

    def forward(self, eigvals: torch.Tensor, eigvecs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eigvals: [d]
            eigvecs: [F, d]
        Returns:
            z: [F, d * m]
        """
        if eigvals.dim() != 1:
            raise ValueError("eigvals must have shape [d].")
        if eigvecs.dim() != 2:
            raise ValueError("eigvecs must have shape [F, d].")
        if eigvecs.shape[1] != eigvals.shape[0]:
            raise ValueError("eigvecs.shape[1] must match eigvals.shape[0].")

        filters = self.phi_mlp(eigvals.unsqueeze(-1))  # [d, m]
        z = torch.einsum("fd,dm->fdm", eigvecs, filters).reshape(
            eigvecs.shape[0], -1
        )
        z = self.dropout(z)
        return self.norm(z)
