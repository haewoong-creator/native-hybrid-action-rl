"""Canonical [1, main, i<j pair] basis and coefficient conversion."""
import torch
from .exact_projection import ANOVACoefficients
def number_of_full_pair_terms(binary_dim):
    if not isinstance(binary_dim,int) or binary_dim<1: raise ValueError("binary_dim must be positive")
    return 1+binary_dim+binary_dim*(binary_dim-1)//2
def pair_indices(binary_dim): return [(i,j) for i in range(binary_dim) for j in range(i+1,binary_dim)]
def build_full_pair_basis(binary_actions):
    if not isinstance(binary_actions,torch.Tensor) or binary_actions.ndim!=2: raise ValueError("actions must be rank 2")
    pairs=[(binary_actions[:,i]*binary_actions[:,j])[:,None] for i,j in pair_indices(binary_actions.shape[1])]
    return torch.cat((torch.ones(len(binary_actions),1,dtype=binary_actions.dtype,device=binary_actions.device),binary_actions,*pairs),1)
def unpack_anova_coefficients(coefficients,binary_dim):
    if coefficients.shape!=(number_of_full_pair_terms(binary_dim),): raise ValueError("coefficient shape mismatch")
    pair=torch.zeros(binary_dim,binary_dim,dtype=coefficients.dtype,device=coefficients.device); offset=1+binary_dim
    for k,(i,j) in enumerate(pair_indices(binary_dim)): pair[i,j]=pair[j,i]=coefficients[offset+k]
    return ANOVACoefficients(coefficients[0],coefficients[1:offset],pair)
def pack_anova_coefficients(coefficients):
    b=len(coefficients.main); return torch.cat((coefficients.intercept.reshape(1),coefficients.main,torch.stack([coefficients.pair[i,j] for i,j in pair_indices(b)])))
