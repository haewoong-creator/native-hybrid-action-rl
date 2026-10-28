"""End-to-end no-environment trainer for CANONICAL_V2_FULL."""
from __future__ import annotations
import copy,hashlib,json,random
import numpy as np
import torch
from torch.nn import functional as F
from hybrid_sb.config.canonical_v2 import CANONICAL_V2_FULL
from hybrid_sb.training.canonical_v2 import CanonicalV2Full,surrogate_value
V2_SHA="c6478f5e1a715975f4319f52ff1db1d2a3f997dfa7907c81d9149ecde498b4dc"
class CanonicalReplay:
 def __init__(self,capacity,state_dim,continuous_dim,binary_dim,seed=0):self.capacity=capacity;self.dims=(state_dim,continuous_dim,binary_dim);self.rows=[];self.generator=torch.Generator().manual_seed(seed)
 def add(self,state,continuous,binary,reward,next_state,terminated=False,truncated=False):
  b=torch.as_tensor(binary).detach().cpu()
  if b.shape!=(self.dims[2],) or not torch.all((b==-1)|(b==1)):raise ValueError("replay requires hard binary action")
  row=tuple(torch.as_tensor(x).detach().cpu().clone() for x in (state,continuous,b,reward,next_state,torch.tensor(terminated),torch.tensor(truncated)))
  self.rows.append(row)
  if len(self.rows)>self.capacity:self.rows.pop(0)
 def sample(self,n,device="cpu",dtype=torch.float32):
  if n>len(self.rows):raise ValueError("insufficient replay")
  ids=torch.randperm(len(self.rows),generator=self.generator)[:n].tolist();cols=list(zip(*(self.rows[i] for i in ids)))
  values=[torch.stack(x).to(device) for x in cols];values[:5]=[x.to(dtype) for x in values[:5]];return dict(zip(("state","continuous","binary","reward","next_state","terminated","truncated"),values))
 def state_dict(self):return {"rows":self.rows,"generator":self.generator.get_state(),"capacity":self.capacity,"dims":self.dims}
 def load_state_dict(self,d):self.rows=d["rows"];self.generator.set_state(d["generator"])
 def __len__(self):return len(self.rows)
class CanonicalV2Trainer:
 def __init__(self,state_dim,continuous_dim,binary_dim,*,seed=42,config=CANONICAL_V2_FULL,dtype=torch.float32,device="cpu"):
  if config.config_id!="CANONICAL_V2_FULL" or config.exact_enumeration:raise ValueError("invalid Full config")
  torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);self.seed=seed;self.config=config;self.device=torch.device(device);self.dtype=dtype
  self.model=CanonicalV2Full(state_dim,continuous_dim,binary_dim,config=config).to(self.device,dtype);self.target_critic=self.model.critic.make_target().to(self.device,dtype)
  self.optimizers={"actor":torch.optim.Adam(self.model.actor.parameters(),lr=config.actor_lr),"critic":torch.optim.Adam(self.model.critic.parameters(),lr=config.critic_lr),"ising":torch.optim.Adam(self.model.ising.parameters(),lr=config.ising_lr),"anova":torch.optim.Adam(self.model.anova.parameters(),lr=config.anova_lr)}
  self.replay=CanonicalReplay(config.replay_capacity,state_dim,continuous_dim,binary_dim,seed+91);self.generator=torch.Generator().manual_seed(seed+700);self.counters={k:0 for k in ("training_step","critic_steps","actor_steps","hJ_steps","anova_steps","target_updates")}
 def optimizer_owners(self):
  owners={}
  for name,opt in self.optimizers.items():
   for group in opt.param_groups:
    for p in group["params"]:owners.setdefault(id(p),[]).append(name)
  return owners
 def warmup_action(self,state):
  c=2*torch.rand(self.model.continuous_dim,generator=self.generator,dtype=self.dtype)-1;b=2*torch.randint(0,2,(self.model.binary_dim,),generator=self.generator,dtype=torch.long)-1;return c.to(state.device),b.to(state.device,dtype=self.dtype)
 def add_transition(self,*args,**kwargs):return self.replay.add(*args,**kwargs)
 def _clip_step(self,opt,params):
  norm=torch.nn.utils.clip_grad_norm_(list(params),self.config.gradient_clip);opt.step();return float(norm)
 def _target_action(self,s):
  with torch.no_grad():return self.model.policy(s)[0],self.model.policy(s)[5]
 def critic_step(self,b):
  with torch.no_grad():
   nc,nb=self._target_action(b["next_state"]);q=self.target_critic.min_q(b["next_state"],nc,nb).squeeze(-1);mask=(~b["terminated"].bool()&~b["truncated"].bool()).to(q.dtype);y=b["reward"].reshape(-1)+.99*mask.reshape(-1)*q
  q1,q2=self.model.critic(b["state"],b["continuous"],b["binary"]);loss=F.mse_loss(q1.squeeze(-1),y)+F.mse_loss(q2.squeeze(-1),y);opt=self.optimizers["critic"];opt.zero_grad(set_to_none=True);loss.backward();gn=self._clip_step(opt,self.model.critic.parameters());self.counters["critic_steps"]+=1;return loss.detach(),gn
 def anova_step(self,state):
  c=self.model.policy(state[:1])[0].detach();edges=self.model.active_edges(state[:1],c);design=self.model.oa_design(edges,dtype=self.dtype,device=self.device);bits=design.actions
  with torch.no_grad():target=self.model.critic.min_q(state[:1].expand(len(bits),-1),c.expand(len(bits),-1),bits).squeeze(-1)
  intercept,unary,pair=self.model.anova(state[:1],c,edges);pred=surrogate_value(intercept.expand(len(bits),-1),unary.expand(len(bits),-1),pair.expand(len(bits),-1),bits,edges);loss=F.mse_loss(pred,target);opt=self.optimizers["anova"];opt.zero_grad(set_to_none=True);loss.backward();gn=self._clip_step(opt,self.model.anova.parameters());self.counters["anova_steps"]+=1;return loss.detach(),gn
 def _actor_objective(self,state):
  c,J,h,logits,spin,bits=self.model.policy(state[:1]);dloss,counts=self.model.actor_discrete_loss(state[:1],self.generator);cctrl,_=self.model.actor(state[:1])
  for p in self.model.critic.parameters():p.requires_grad_(False)
  cq=self.model.critic.min_q(state[:1],c,bits).mean();loss=.5*cctrl.square().mean()+self.config.lambda_q*(1-spin.square()).square().mean()-cq/self.config.alpha+dloss
  for p in self.model.critic.parameters():p.requires_grad_(True)
  return loss,counts
 def actor_step(self,state):
  for p in self.model.ising.parameters():p.requires_grad_(False)
  loss,counts=self._actor_objective(state);opt=self.optimizers["actor"];opt.zero_grad(set_to_none=True);loss.backward();gn=self._clip_step(opt,self.model.actor.parameters());
  for p in self.model.ising.parameters():p.requires_grad_(True)
  self.counters["actor_steps"]+=1;return loss.detach(),gn,counts
 def hj_step(self,state):
  for p in self.model.actor.parameters():p.requires_grad_(False)
  loss,_=self._actor_objective(state);opt=self.optimizers["ising"];opt.zero_grad(set_to_none=True);loss.backward();gn=self._clip_step(opt,self.model.ising.parameters());
  for p in self.model.actor.parameters():p.requires_grad_(True)
  self.counters["hJ_steps"]+=1;return loss.detach(),gn
 @torch.no_grad()
 def polyak(self):
  t=self.config.polyak_tau
  for target,online in zip(self.target_critic.parameters(),self.model.critic.parameters()):target.mul_(1-t).add_(online,alpha=t)
  self.counters["target_updates"]+=1
 def update(self,batch):
  self.counters["training_step"]+=1;cl,cgn=self.critic_step(batch);self.polyak();scheduled=self.counters["training_step"]%self.config.actor_delay==0;out={"critic_loss":float(cl),"critic_grad_norm":cgn,"did_target_update":True,"did_actor_update":False,"did_hJ_update":False,"did_anova_update":False}
  if scheduled:
   al,agn=self.anova_step(batch["state"]);acl,acgn,counts=self.actor_step(batch["state"]);hl,hgn=self.hj_step(batch["state"]);out.update(anova_loss=float(al),anova_grad_norm=agn,actor_loss=float(acl),actor_grad_norm=acgn,hJ_loss=float(hl),hJ_grad_norm=hgn,did_actor_update=True,did_hJ_update=True,did_anova_update=True,**counts)
  return out
 def checkpoint(self):
  return copy.deepcopy({"canonical_v2_sha":V2_SHA,"config":self.config.to_dict(),"seed":self.seed,"model":self.model.state_dict(),"target_critic":self.target_critic.state_dict(),"optimizers":{k:v.state_dict() for k,v in self.optimizers.items()},"counters":self.counters,"replay":self.replay.state_dict(),"generator":self.generator.get_state(),"torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state(),"python_rng":random.getstate(),"spec":{"state_dim":self.model.state_dim,"continuous_dim":self.model.continuous_dim,"binary_dim":self.model.binary_dim}})
 def load_checkpoint(self,d):
  if d["canonical_v2_sha"]!=V2_SHA:raise ValueError("canonical v2 SHA mismatch")
  if d["config"]["config_id"]!="CANONICAL_V2_FULL":raise ValueError("config mismatch")
  self.model.load_state_dict(d["model"]);self.target_critic.load_state_dict(d["target_critic"])
  for k,v in self.optimizers.items():v.load_state_dict(d["optimizers"][k])
  self.counters=copy.deepcopy(d["counters"]);self.replay.load_state_dict(d["replay"]);self.generator.set_state(d["generator"]);torch.set_rng_state(d["torch_rng"]);np.random.set_state(d["numpy_rng"]);random.setstate(d["python_rng"])
