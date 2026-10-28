from dataclasses import dataclass
import torch
@dataclass
class ANOVACoefficients:
    intercept:torch.Tensor
    main:torch.Tensor
    pair:torch.Tensor
def exact_uniform_projection(actions,values):
    if actions.ndim!=2 or values.shape!=(len(actions),): raise ValueError("invalid action/value table")
    if len(actions)!=2**actions.shape[1]: raise ValueError("full binary enumeration required")
    intercept=values.mean(); main=(values[:,None]*actions).mean(0)
    pair=torch.einsum("n,ni,nj->ij",values,actions,actions)/len(actions)
    pair=pair-torch.diag_embed(pair.diagonal()); pair=.5*(pair+pair.T)
    return ANOVACoefficients(intercept,main,pair)
