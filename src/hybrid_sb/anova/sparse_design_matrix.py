"""Sparse second-order ANOVA basis on a canonical active edge set."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .design_matrix import build_full_pair_basis, pair_indices

Edge = tuple[int, int]


def canonicalize_edges(
    edges: Iterable[Edge] | torch.Tensor,
    binary_dim: int,
    *,
    reject_duplicates: bool = True,
) -> tuple[Edge, ...]:
    raw = edges.tolist() if isinstance(edges, torch.Tensor) else list(edges)
    canonical: list[Edge] = []
    for edge in raw:
        if len(edge) != 2:
            raise ValueError("each edge must contain two indices")
        i, j = int(edge[0]), int(edge[1])
        if i == j:
            raise ValueError("diagonal edges are invalid")
        if min(i, j) < 0 or max(i, j) >= binary_dim:
            raise ValueError("edge index out of range")
        canonical.append((min(i, j), max(i, j)))
    if reject_duplicates and len(set(canonical)) != len(canonical):
        raise ValueError("duplicate edges are invalid")
    return tuple(sorted(set(canonical)))


def number_of_active_terms(binary_dim: int, edges: Iterable[Edge]) -> int:
    return 1 + binary_dim + len(canonicalize_edges(edges, binary_dim))


def build_sparse_pair_basis(actions: torch.Tensor, edges: Iterable[Edge]) -> torch.Tensor:
    if actions.ndim != 2:
        raise ValueError("actions must be rank 2")
    canonical = canonicalize_edges(edges, actions.shape[1])
    columns = [torch.ones_like(actions[:, :1]), actions]
    if canonical:
        columns.append(torch.stack([actions[:, i] * actions[:, j] for i, j in canonical], 1))
    return torch.cat(columns, 1)


@dataclass(frozen=True)
class SparseANOVACoefficients:
    intercept: torch.Tensor
    unary: torch.Tensor
    pair: torch.Tensor
    edges: tuple[Edge, ...]

    @property
    def binary_dim(self) -> int:
        return self.unary.shape[-1]

    def dense_pair(self) -> torch.Tensor:
        shape = self.pair.shape[:-1] + (self.binary_dim, self.binary_dim)
        result = torch.zeros(shape, dtype=self.pair.dtype, device=self.pair.device)
        for k, (i, j) in enumerate(self.edges):
            result[..., i, j] = self.pair[..., k]
            result[..., j, i] = self.pair[..., k]
        return result


def unpack_sparse_coefficients(
    vector: torch.Tensor, binary_dim: int, edges: Iterable[Edge]
) -> SparseANOVACoefficients:
    canonical = canonicalize_edges(edges, binary_dim)
    if vector.shape[-1] != 1 + binary_dim + len(canonical):
        raise ValueError("coefficient shape mismatch")
    return SparseANOVACoefficients(
        vector[..., 0], vector[..., 1 : 1 + binary_dim],
        vector[..., 1 + binary_dim :], canonical,
    )


def pack_sparse_coefficients(coefficients: SparseANOVACoefficients) -> torch.Tensor:
    return torch.cat(
        (coefficients.intercept.unsqueeze(-1), coefficients.unary, coefficients.pair), -1
    )


def evaluate_sparse_surrogate(
    actions: torch.Tensor, coefficients: SparseANOVACoefficients
) -> torch.Tensor:
    return (build_sparse_pair_basis(actions, coefficients.edges) *
            pack_sparse_coefficients(coefficients)).sum(-1)


def sparse_anova_expectation(
    spin: torch.Tensor, coefficients: SparseANOVACoefficients
) -> torch.Tensor:
    value = coefficients.intercept + (coefficients.unary * spin).sum(-1)
    for k, (i, j) in enumerate(coefficients.edges):
        value = value + coefficients.pair[..., k] * spin[..., i] * spin[..., j]
    return value


def sparse_anova_gradient(
    spin: torch.Tensor, coefficients: SparseANOVACoefficients
) -> torch.Tensor:
    gradient = coefficients.unary + torch.zeros_like(spin)
    for k, (i, j) in enumerate(coefficients.edges):
        coefficient = coefficients.pair[..., k]
        gradient[..., i] = gradient[..., i] + coefficient * spin[..., j]
        gradient[..., j] = gradient[..., j] + coefficient * spin[..., i]
    return gradient


def assert_full_basis_compatibility(actions: torch.Tensor) -> bool:
    edges = pair_indices(actions.shape[1])
    return torch.equal(build_sparse_pair_basis(actions, edges), build_full_pair_basis(actions))
