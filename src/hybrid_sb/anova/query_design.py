"""Local deterministic binary query designs and diagnostics."""
from dataclasses import dataclass,asdict
import math,torch
from .design_matrix import build_full_pair_basis,number_of_full_pair_terms
@dataclass
class DesignDiagnostics:
    rank:int; rank_deficiency:int; condition_number:float; minimum_singular_value:float
    maximum_gram_correlation:float; duplicate_rate:float; bit_balance:float; pair_balance:float
def _hadamard(order,dtype):
    H=torch.ones(1,1,dtype=dtype)
    while len(H)<order: H=torch.cat((torch.cat((H,H),1),torch.cat((H,-H),1)),0)
    return H
def generate_binary_design(binary_dim,num_rows,seed,method,dtype=torch.float64,device="cpu"):
    if method not in ("hadamard","fractional_factorial","random_rademacher"): raise ValueError("unknown design method")
    if num_rows<1: raise ValueError("num_rows must be positive")
    generator=torch.Generator().manual_seed(seed)
    if method=="hadamard":
        order=1
        while order<max(num_rows,binary_dim+1): order*=2
        H=_hadamard(order,dtype); actions=H[:min(num_rows,order),1:binary_dim+1]
        if len(actions)<num_rows: actions=torch.cat((actions,(2*torch.randint(0,2,(num_rows-len(actions),binary_dim),generator=generator)-1).to(dtype)),0)
    elif method=="fractional_factorial":
        index=torch.arange(num_rows); bits=((index[:,None]>>torch.arange(binary_dim))&1); actions=(2*bits-1).to(dtype)
        # Deterministic seeded row sign permutation avoids a fixed lexicographic corner.
        actions=actions*torch.where(torch.rand(binary_dim,generator=generator)>.5,1.,-1.)
    else: actions=(2*torch.randint(0,2,(num_rows,binary_dim),generator=generator)-1).to(dtype)
    # Full-pair estimation needs resolution beyond a classical strength-2 OA.
    # Deterministically augment aliased designs when enough rows were requested.
    p=number_of_full_pair_terms(binary_dim)
    if num_rows>=p and int(torch.linalg.matrix_rank(build_full_pair_basis(actions)))<p:
        for attempt in range(128):
            candidate=(2*torch.randint(0,2,(num_rows,binary_dim),generator=generator)-1).to(dtype)
            if int(torch.linalg.matrix_rank(build_full_pair_basis(candidate)))==p:
                actions=candidate; break
    return actions.to(device)
def design_diagnostics(actions):
    X=build_full_pair_basis(actions); singular=torch.linalg.svdvals(X); rank=int(torch.linalg.matrix_rank(X)); p=X.shape[1]
    condition=float(singular[0]/singular[-1]) if singular[-1]>0 else float("inf")
    normalized=X/torch.linalg.vector_norm(X,dim=0).clamp_min(1e-12); gram=normalized.T@normalized; off=gram-torch.diag_embed(gram.diagonal())
    unique=len(torch.unique(actions,dim=0)); pairs=X[:,1+actions.shape[1]:]
    return DesignDiagnostics(rank,p-rank,condition,float(singular[-1]),float(off.abs().max()),1-unique/len(actions),
        float(actions.mean(0).abs().mean()),float(pairs.mean(0).abs().mean()) if pairs.numel() else 0.)
