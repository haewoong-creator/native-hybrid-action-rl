"""Domain-agnostic observation and PopArt scale stabilization for canonical v2."""
from __future__ import annotations
import copy
import torch
from torch.nn import functional as F
from hybrid_sb.training.canonical_v2_trainer import CanonicalV2Trainer

EPS=1e-6
class RunningMoments:
 def __init__(self,shape,dtype=torch.float64):self.count=0;self.mean=torch.zeros(shape,dtype=dtype);self.m2=torch.zeros(shape,dtype=dtype)
 def update(self,x):
  x=torch.as_tensor(x,dtype=self.mean.dtype).reshape(-1,*self.mean.shape);n=len(x)
  if not n:return
  bm=x.mean(0);bm2=((x-bm).square()).sum(0)
  if self.count==0:self.count=n;self.mean=bm;self.m2=bm2;return
  total=self.count+n;delta=bm-self.mean;self.mean=self.mean+delta*n/total;self.m2=self.m2+bm2+delta.square()*self.count*n/total;self.count=total
 @property
 def variance(self):return self.m2/max(self.count,1)
 @property
 def std(self):return torch.sqrt(self.variance+EPS)
 def normalize(self,x):return (x-self.mean.to(x.device,x.dtype))/self.std.to(x.device,x.dtype)
 def state_dict(self):return {'count':self.count,'mean':self.mean,'m2':self.m2}
 def load_state_dict(self,d):self.count=d['count'];self.mean=d['mean'];self.m2=d['m2']

class ScaleStabilizedV2Trainer(CanonicalV2Trainer):
 def __init__(self,*args,observation_standardization=False,value_normalization=False,**kwargs):
  super().__init__(*args,**kwargs);self.observation_standardization=bool(observation_standardization);self.value_normalization=bool(value_normalization);self.obs_moments=RunningMoments((self.model.state_dim,));self.q_moments=RunningMoments(())
 def observe(self,state):
  if self.observation_standardization:self.obs_moments.update(torch.as_tensor(state).reshape(1,-1))
 def normalize_state(self,state):return self.obs_moments.normalize(state) if self.observation_standardization else state
 def physical_q(self,qhat):
  if not self.value_normalization:return qhat
  return self.q_moments.std.to(qhat.device,qhat.dtype)*qhat+self.q_moments.mean.to(qhat.device,qhat.dtype)
 def normalized_q(self,q):
  if not self.value_normalization:return q
  return (q-self.q_moments.mean.to(q.device,q.dtype))/self.q_moments.std.to(q.device,q.dtype)
 @staticmethod
 def _preserve_structured_output(qnet,ratio,offset):
  # Qhat is the additive intercept + unary/sqrt(d) + pair/sqrt(E).
  # Scale every additive head, but apply the affine offset exactly once.
  for head in (qnet.intercept,qnet.unary,qnet.pair):
   head.weight.mul_(ratio);head.bias.mul_(ratio)
  qnet.intercept.bias.add_(offset)
  if qnet.residual_weight:
   qnet.residual[-1].weight.mul_(ratio);qnet.residual[-1].bias.mul_(ratio)
 def _popart_update(self,y):
  old_mu=self.q_moments.mean.clone();old_sigma=self.q_moments.std.clone();self.q_moments.update(y.detach().cpu());new_mu=self.q_moments.mean;new_sigma=self.q_moments.std
  with torch.no_grad():
   for twin in (self.model.critic,self.target_critic):
    for qnet in (twin.q1,twin.q2):
     ratio=(old_sigma/new_sigma).to(qnet.intercept.weight);offset=((old_mu-new_mu)/new_sigma).to(qnet.intercept.bias);self._preserve_structured_output(qnet,ratio,offset)
 def critic_step(self,b):
  with torch.no_grad():
   nc,nb=self._target_action(b['next_state']);qhat=self.target_critic.min_q(b['next_state'],nc,nb).squeeze(-1);q=self.physical_q(qhat);mask=(~b['terminated'].bool()&~b['truncated'].bool()).to(q.dtype);y=b['reward'].reshape(-1)+.99*mask.reshape(-1)*q
  if self.value_normalization:self._popart_update(y);target=self.normalized_q(y)
  else:target=y
  q1,q2=self.model.critic(b['state'],b['continuous'],b['binary']);loss=F.mse_loss(q1.squeeze(-1),target)+F.mse_loss(q2.squeeze(-1),target);opt=self.optimizers['critic'];opt.zero_grad(set_to_none=True);loss.backward();gn=self._clip_step(opt,self.model.critic.parameters());self.counters['critic_steps']+=1;self.last_scale_telemetry={'physical_td_abs_median':float(y.abs().median()),'normalized_td_abs_median':float(target.abs().median()),'physical_q_abs_median':float(self.physical_q(torch.minimum(q1,q2).squeeze()).abs().median()),'normalized_q_abs_median':float(torch.minimum(q1,q2).squeeze().abs().median()),'q_mean':float(self.q_moments.mean),'q_std':float(self.q_moments.std)};return loss.detach(),gn
 def update(self,batch):
  b=dict(batch);b['state']=self.normalize_state(b['state']);b['next_state']=self.normalize_state(b['next_state']);out=super().update(b);out.update(getattr(self,'last_scale_telemetry',{}));return out
 def checkpoint(self):
  d=super().checkpoint();d['scale_stabilization']={'observation_standardization':self.observation_standardization,'value_normalization':self.value_normalization,'epsilon':EPS,'obs_moments':self.obs_moments.state_dict(),'q_moments':self.q_moments.state_dict()};return d
 def load_checkpoint(self,d):
  super().load_checkpoint(d);s=d['scale_stabilization'];self.observation_standardization=s['observation_standardization'];self.value_normalization=s['value_normalization'];self.obs_moments.load_state_dict(s['obs_moments']);self.q_moments.load_state_dict(s['q_moments'])
