"""Single-component ablations of the frozen Canonical-V2 Full trainer."""
from __future__ import annotations
import hashlib,types
import torch
from hybrid_sb.training.canonical_v2_trainer import CanonicalV2Trainer
from hybrid_sb.training.canonical_v2 import analytic_surrogate
from hybrid_sb.estimators.mixture_proposal import sample_mixture,support_masked_weighted_log_probability
from hybrid_sb.anova.condition_aware_design import SelectedSparseDesign,sparse_design_diagnostics

VARIANTS=("ABL_NO_ISING_J","ABL_NO_PAIR_CRITIC","ABL_OA_TO_MC","ABL_NO_ACV")

def _without_pair(self,state,continuous_action,binary_action):
 z=self.context(torch.cat((state,continuous_action),-1));unary=self.unary(z);pair=self.pair(z)
 value=self.intercept(z)+(unary*binary_action).sum(-1,keepdim=True)/(self.binary_dim**.5)
 residual=self.residual(torch.cat((z,binary_action),-1))
 return value+self.residual_weight*residual,unary,pair,residual

class ComponentAblationTrainer(CanonicalV2Trainer):
 def __init__(self,*args,variant:str,environment:str,**kwargs):
  if variant not in VARIANTS:raise ValueError(f"unknown ablation {variant}")
  self.variant=variant;self.environment=environment;self.design_calls=0
  super().__init__(*args,**kwargs)
  if variant=="ABL_NO_ISING_J":
   original=self.model.ising.forward
   def no_j(module,state):
    J,h=original(state);return torch.zeros_like(J),h
   self.model.ising.forward=types.MethodType(no_j,self.model.ising)
  elif variant=="ABL_NO_PAIR_CRITIC":
   for twin in (self.model.critic,self.target_critic):
    twin.q1.components=types.MethodType(_without_pair,twin.q1);twin.q2.components=types.MethodType(_without_pair,twin.q2)
   self.model.active_edges=types.MethodType(lambda model,state,c:torch.empty((0,2),dtype=torch.long,device=state.device),self.model)
  elif variant=="ABL_OA_TO_MC":
   original=self.model.oa_design
   def mc_design(model,edges,seed=42,dtype=torch.float32,device="cpu"):
    structured=original(edges,seed=seed,dtype=dtype,device=device);self.design_calls+=1
    token=f"{self.environment}|{self.seed}|{self.counters['training_step']}|{self.design_calls}".encode();derived=int.from_bytes(hashlib.sha256(token).digest()[:8],'little')%(2**63-1)
    generator=torch.Generator().manual_seed(derived)
    actions=(2*torch.randint(0,2,(len(structured.actions),model.binary_dim),generator=generator)-1).to(device=device,dtype=dtype)
    return SelectedSparseDesign(actions,sparse_design_diagnostics(actions,[tuple(x) for x in edges.tolist()]),structured.row_multiplier,0)
   self.model.oa_design=types.MethodType(mc_design,self.model)
  elif variant=="ABL_NO_ACV":
   def no_acv(model,state,generator):
    if len(state)!=1:raise ValueError("one context per structured query group")
    c,J,h,logits,spin,_=model.policy(state);edges=model.active_edges(state,c);intercept,unary,pair=model.anova(state,c,edges);design=model.oa_design(edges,dtype=state.dtype,device=state.device)
    samples=sample_mixture(spin[0],model.config.proposal_samples(model.binary_dim),model.config.epsilon,generator);bits=samples.actions
    F=model.critic.min_q(state.expand(len(bits),-1),c.expand(len(bits),-1),bits).squeeze(-1);w=(samples.target_probability/samples.proposal_probability).detach()
    target=support_masked_weighted_log_probability(w*F.detach(),bits,spin[0]).mean()
    return -target/model.config.alpha,{"is_application_count":1,"acv_application_count":0,"anova_application_count":1,"reward_scale_count":1,"oa_query_count":len(design.actions),"exact_enumeration_call_count":model.exact_enumeration_call_count,"edges":edges.detach().cpu().tolist()}
   self.model.actor_discrete_loss=types.MethodType(no_acv,self.model)
