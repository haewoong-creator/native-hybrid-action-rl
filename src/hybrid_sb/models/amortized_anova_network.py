"""Context- and edge-conditioned sparse ANOVA coefficient predictor."""
import torch
from torch import nn


class AmortizedANOVANetwork(nn.Module):
    def __init__(self, state_dim: int, continuous_dim: int, binary_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.binary_dim = binary_dim
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + continuous_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.intercept = nn.Linear(hidden_dim, 1)
        self.unary = nn.Linear(hidden_dim, binary_dim)
        self.bit_embedding = nn.Embedding(binary_dim, hidden_dim // 2)
        self.pair_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, continuous_latent, edges, edge_mask=None):
        context = self.encoder(torch.cat((state, continuous_latent), -1))
        if edges.numel() == 0:
            pair = context.new_empty((len(state), 0))
            return self.intercept(context), self.unary(context), pair
        if edges.ndim == 2:
            edges = edges.unsqueeze(0).expand(len(state), -1, -1)
        left, right = self.bit_embedding(edges[..., 0]), self.bit_embedding(edges[..., 1])
        edge_feature = torch.cat((left + right, (left - right).abs()), -1)
        expanded = context[:, None, :].expand(-1, edges.shape[1], -1)
        pair = self.pair_head(torch.cat((expanded, edge_feature), -1)).squeeze(-1)
        if edge_mask is not None:
            pair = pair * edge_mask.to(pair.dtype)
        return self.intercept(context), self.unary(context), pair
