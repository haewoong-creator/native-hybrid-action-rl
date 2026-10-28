import itertools

import torch


def enumerate_binary_actions(
    binary_dim: int,
    *,
    max_exact_binary_dim: int = 16,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Enumerate {-1,+1}^b in lexicographic order."""
    if not isinstance(binary_dim, int) or binary_dim < 1:
        raise ValueError("binary_dim must be a positive integer")
    if binary_dim > max_exact_binary_dim:
        raise ValueError(
            f"exact enumeration disabled for binary_dim={binary_dim}; maximum is "
            f"{max_exact_binary_dim}"
        )
    return torch.tensor(
        list(itertools.product((-1.0, 1.0), repeat=binary_dim)),
        dtype=dtype,
        device=device,
    )

