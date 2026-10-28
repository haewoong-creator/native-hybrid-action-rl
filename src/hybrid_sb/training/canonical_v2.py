"""Canonical-v2 pure-tensor Full graph primitives."""
from dataclasses import dataclass
import torch
from torch import nn
from hybrid_sb.models.spin_network import StateConditionedSpinNetwork
from hybrid_sb.models.amortized_anova_network import AmortizedANOVANetwork
from hybrid_sb.models.structured_hard_critic import TwinStructuredHardActionCritic
from hybrid_sb.reference.bistable_continuous import BistableContinuousReference
from hybrid_sb.screening.topk_selection import deterministic_topk_edges
from hybrid_sb.estimators.mixture_proposal import sample_mixture,support_masked_weighted_log_probability
from hybrid_sb.config.canonical_v2 import CANONICAL_V2_FULL
from hybrid_sb.anova.condition_aware_design import select_condition_aware_design
class CanonicalActor(nn.Module):
 def __init__(self,state_dim,continuous_dim,binary_dim,hidden=64):
  super().__init__();self.continuous_dim=continuous_dim;self.binary_dim=binary_dim;self.net=nn.Sequential(nn.Linear(state_dim,hidden),nn.SiLU(),nn.Linear(hidden,continuous_dim+binary_dim))
 def forward(self,s):return self.net(s).split((self.continuous_dim,self.binary_dim),-1)
def mean_field_policy(J,h,control_logits):
 ref=h+torch.einsum("bij,bj->bi",J,torch.tanh(h));logits=ref+control_logits;return logits,torch.tanh(logits)
def hard_bits(spin):return torch.where(spin>=0,torch.ones_like(spin),-torch.ones_like(spin))
def surrogate_value(intercept,unary,pair,bits,edges):
 out=intercept.squeeze(-1)+(unary*bits).sum(-1)
 if edges.numel():out=out+(pair*(bits[:,edges[:,0]]*bits[:,edges[:,1]])).sum(-1)
 return out
def analytic_surrogate(intercept,unary,pair,spin,edges):
 out=intercept.squeeze(-1)+(unary*spin).sum(-1)
 if edges.numel():out=out+(pair*(spin[:,edges[:,0]]*spin[:,edges[:,1]])).sum(-1)
 return out
def discrete_acv_score_loss(F,intercept,unary,pair,spin,edges,epsilon,M,generator,alpha=1.):
 if len(spin)!=1:raise ValueError("canonical estimator primitive currently requires one context per call")
 samples=sample_mixture(spin[0],M,epsilon,generator);bits=samples.actions
 g=surrogate_value(intercept.expand(M,-1),unary.expand(M,-1),pair.expand(M,-1),bits,edges)
 w=(samples.target_probability/samples.proposal_probability).detach()
 analytic=analytic_surrogate(intercept,unary,pair,spin,edges).mean()
 coefficient=w*(F.detach()-g.detach())
 target=analytic+support_masked_weighted_log_probability(coefficient,bits,spin[0]).mean()
 return -target/alpha,{"is_application_count":1,"acv_application_count":1,"anova_application_count":1,"reward_scale_count":1}
class CanonicalV2Full(nn.Module):
 def __init__(self,state_dim,continuous_dim,binary_dim,hidden=64,config=CANONICAL_V2_FULL):
  super().__init__();self.state_dim=state_dim;self.continuous_dim=continuous_dim;self.binary_dim=binary_dim;self.config=config
  self.actor=CanonicalActor(state_dim,continuous_dim,binary_dim,hidden);self.ising=StateConditionedSpinNetwork(state_dim,binary_dim,(hidden,hidden));self.anova=AmortizedANOVANetwork(state_dim,continuous_dim,binary_dim,hidden);self.critic=TwinStructuredHardActionCritic(state_dim,continuous_dim,binary_dim,hidden_dim=hidden,residual_weight=0.);self.reference=BistableContinuousReference(1.,1.);self.exact_enumeration_call_count=0;self.bistable_call_count=0;self.ou_call_count=0
 def policy(self,state):
  cctrl,bctrl=self.actor(state);J,h=self.ising(state);logits,spin=mean_field_policy(J,h,bctrl);self.bistable_call_count+=1;c=torch.tanh(self.reference.controlled_drift(torch.zeros_like(cctrl),cctrl));return c,J,h,logits,spin,hard_bits(spin)
 def active_edges(self,state,c):
  all_edges=torch.triu_indices(self.binary_dim,self.binary_dim,1).T.to(state.device);_,_,pair=self.anova(state,c,all_edges);scores=pair.detach().abs().mean(0);matrix=scores.new_zeros(self.binary_dim,self.binary_dim);matrix[all_edges[:,0],all_edges[:,1]]=scores;edges=deterministic_topk_edges(matrix,self.config.top_k(self.binary_dim));return edges
 def oa_design(self,edges,seed=42,dtype=torch.float32,device="cpu"):
  return select_condition_aware_design(self.binary_dim,[tuple(x) for x in edges.tolist()],seed,dtype=dtype,device=device)
 def actor_discrete_loss(self,state,generator):
  if len(state)!=1:raise ValueError("one context per structured query group")
  c,J,h,logits,spin,_=self.policy(state);edges=self.active_edges(state,c);intercept,unary,pair=self.anova(state,c,edges);design=self.oa_design(edges,dtype=state.dtype,device=state.device)
  generator_state=generator.get_state();bits=sample_mixture(spin[0],self.config.proposal_samples(self.binary_dim),self.config.epsilon,generator).actions
  F=self.critic.min_q(state.expand(len(bits),-1),c.expand(len(bits),-1),bits).squeeze(-1)
  generator.set_state(generator_state)
  loss,counts=discrete_acv_score_loss(F,intercept.detach(),unary.detach(),pair.detach(),spin,edges,self.config.epsilon,len(bits),generator,self.config.alpha)
  return loss,{**counts,"oa_query_count":len(design.actions),"exact_enumeration_call_count":self.exact_enumeration_call_count,"edges":edges.detach().cpu().tolist()}
