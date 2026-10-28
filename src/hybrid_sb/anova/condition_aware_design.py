"""Select the smallest stable sparse binary query design."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .query_design import generate_binary_design
from .sparse_design_matrix import build_sparse_pair_basis, canonicalize_edges


@dataclass(frozen=True)
class SparseDesignDiagnostics:
    rows: int
    terms: int
    rank: int
    condition_number: float
    minimum_singular_value: float
    maximum_gram_correlation: float
    duplicate_rate: float


@dataclass(frozen=True)
class SelectedSparseDesign:
    actions: torch.Tensor
    diagnostics: SparseDesignDiagnostics
    row_multiplier: int
    automatic_augmentations: int


def sparse_design_diagnostics(actions: torch.Tensor, edges) -> SparseDesignDiagnostics:
    X = build_sparse_pair_basis(actions, edges)
    singular = torch.linalg.svdvals(X)
    rank = int(torch.linalg.matrix_rank(X))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    normalized = X / torch.linalg.vector_norm(X, dim=0).clamp_min(1e-12)
    gram = normalized.T @ normalized
    off = gram - torch.diag_embed(gram.diagonal())
    duplicate_rate = 1.0 - len(torch.unique(actions, dim=0)) / len(actions)
    return SparseDesignDiagnostics(
        len(actions), X.shape[1], rank, condition, float(singular[-1]),
        float(off.abs().max()), duplicate_rate,
    )


def select_condition_aware_design(
    binary_dim: int,
    edges,
    seed: int,
    *,
    condition_threshold: float = 100.0,
    row_multipliers=(2, 3, 4),
    method: str = "random_rademacher",
    dtype=torch.float64,
    device="cpu",
) -> SelectedSparseDesign:
    canonical = canonicalize_edges(edges, binary_dim)
    p = 1 + binary_dim + len(canonical)
    attempts = []
    for offset, multiplier in enumerate(row_multipliers):
        actions = generate_binary_design(
            binary_dim, multiplier * p, seed + offset, method, dtype=dtype, device=device
        )
        diagnostic = sparse_design_diagnostics(actions, canonical)
        attempts.append(diagnostic)
        if diagnostic.rank == p and diagnostic.condition_number <= condition_threshold:
            return SelectedSparseDesign(actions, diagnostic, multiplier, offset)
    detail = ", ".join(
        f"{d.rows} rows: rank={d.rank}/{p}, cond={d.condition_number:.3g}" for d in attempts
    )
    raise RuntimeError(f"no acceptable sparse OLS design ({detail})")


@dataclass(frozen=True)
class FittedSparseANOVA:
    vector: torch.Tensor
    coefficients: object
    diagnostics: SparseDesignDiagnostics


def fit_sparse_anova(actions, values, edges, *, method="ols", ridge_lambda=1e-6):
    from .sparse_design_matrix import unpack_sparse_coefficients
    X = build_sparse_pair_basis(actions, edges)
    diagnostic = sparse_design_diagnostics(actions, edges)
    if method == "ols":
        if diagnostic.rank != X.shape[1]:
            raise ValueError("rank-deficient sparse OLS design")
        vector = torch.linalg.lstsq(X, values).solution
    elif method == "ridge":
        penalty = ridge_lambda * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
        penalty[0, 0] = 0
        vector = torch.linalg.solve(X.T @ X + penalty, X.T @ values)
    else:
        raise ValueError("method must be ols or ridge")
    if not torch.isfinite(vector).all():
        raise FloatingPointError("non-finite sparse coefficient")
    return FittedSparseANOVA(
        vector, unpack_sparse_coefficients(vector, actions.shape[1], edges), diagnostic
    )
