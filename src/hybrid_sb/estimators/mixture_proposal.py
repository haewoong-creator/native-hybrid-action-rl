from dataclasses import dataclass
import torch
@dataclass
class MixtureSamples:
    actions:torch.Tensor
    target_probability:torch.Tensor
    proposal_probability:torch.Tensor
    log_target_probability:torch.Tensor
    log_proposal_probability:torch.Tensor
def action_log_probability(actions,spin):
    if actions.shape[-1]!=spin.shape[-1] or torch.any(spin.abs()>1): raise ValueError("invalid action/spin")
    probabilities=(1+actions*spin)/2
    return torch.log(probabilities).sum(-1)
def support_masked_weighted_log_probability(coefficient,actions,spin):
    """Evaluate c log mu only on c != 0, with the exact convention 0 log 0 = 0.

    A zero probability carrying a nonzero coefficient is not removable and is
    therefore reported as a genuine inconsistency rather than hidden.
    """
    if actions.shape[-1]!=spin.shape[-1] or torch.any(spin.abs()>1): raise ValueError("invalid action/spin")
    if coefficient.shape!=actions.shape[:-1]: raise ValueError("coefficient/action shape mismatch")
    probabilities=(1+actions*spin)/2
    support=coefficient!=0
    invalid=support & (probabilities<=0).any(-1)
    if invalid.any(): raise FloatingPointError("zero target probability has nonzero score coefficient")
    contribution=torch.zeros_like(coefficient)+spin.sum()*0
    if support.any(): contribution[support]=coefficient[support]*torch.log(probabilities[support]).sum(-1)
    return contribution
def sample_mu(spin,num_samples,generator):
    uniforms=torch.rand((num_samples,spin.shape[-1]),generator=generator,dtype=spin.dtype,device="cpu").to(spin.device)
    return torch.where(uniforms<(1+spin)/2,torch.ones_like(uniforms),-torch.ones_like(uniforms))
def mixture_probability(actions,spin,epsilon):
    if not 0<=epsilon<=1: raise ValueError("epsilon must lie in [0,1]")
    target=torch.exp(action_log_probability(actions,spin)); return (1-epsilon)*target+epsilon*(.5**spin.shape[-1])
def sample_mixture(spin,num_samples,epsilon,generator):
    if not 0<=epsilon<=1: raise ValueError("epsilon must lie in [0,1]")
    choose=torch.rand(num_samples,generator=generator,dtype=spin.dtype)<epsilon
    actions=sample_mu(spin,num_samples,generator)
    count=int(choose.sum())
    if count:
        uniform=(2*torch.randint(0,2,(count,spin.shape[-1]),generator=generator)-1).to(dtype=spin.dtype,device=spin.device); actions[choose.to(spin.device)]=uniform
    log_target=action_log_probability(actions,spin); target=torch.exp(log_target); proposal=(1-epsilon)*target+epsilon*(.5**spin.shape[-1])
    return MixtureSamples(actions,target,proposal,log_target,torch.log(proposal))
