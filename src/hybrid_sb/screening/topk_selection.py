"""Deterministic canonical top-K edge selection."""
import torch


def deterministic_topk_edges(scores: torch.Tensor, k: int):
    if scores.ndim not in (2, 3) or scores.shape[-1] != scores.shape[-2]:
        raise ValueError("scores must have shape (b,b) or (batch,b,b)")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("scores contain NaN/Inf")
    b = scores.shape[-1]
    pairs = [(i, j) for i in range(b) for j in range(i + 1, b)]
    if not 0 <= k <= len(pairs):
        raise ValueError("invalid k")
    flat = scores.unsqueeze(0) if scores.ndim == 2 else scores
    outputs = []
    # Python's stable sort provides an explicit lexicographic tie break.
    for batch_scores in flat:
        ranked = sorted(pairs, key=lambda edge: (-float(batch_scores[edge].detach()), edge))
        outputs.append(ranked[:k])
    edges = torch.tensor(outputs, dtype=torch.long, device=scores.device)
    return edges[0] if scores.ndim == 2 else edges
