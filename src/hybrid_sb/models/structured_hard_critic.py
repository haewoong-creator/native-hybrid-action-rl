"""Factorized hard-action critic with optional residual and sparse pair heads."""
from __future__ import annotations
import copy,math,torch
from torch import nn

class StructuredHardActionCritic(nn.Module):
    def __init__(self,state_dim,continuous_dim,binary_dim,hidden_dim=128, *,
                 pair_mode="full",active_edges=None,residual_weight=1.):
        super().__init__();self.state_dim=state_dim;self.continuous_dim=continuous_dim
        self.binary_dim=binary_dim;self.pair_mode=pair_mode;self.residual_weight=residual_weight
        if pair_mode not in ("full","active"):raise ValueError("pair_mode must be full or active")
        edges=[(i,j) for i in range(binary_dim) for j in range(i+1,binary_dim)] if pair_mode=="full" else list(active_edges or ())
        canonical=sorted((min(i,j),max(i,j)) for i,j in edges)
        if len(set(canonical))!=len(canonical) or any(i==j for i,j in canonical):raise ValueError("invalid active edges")
        edge_tensor=torch.tensor(canonical,dtype=torch.long).reshape(-1,2);self.register_buffer("edges",edge_tensor)
        self.context=nn.Sequential(nn.Linear(state_dim+continuous_dim,hidden_dim),nn.LayerNorm(hidden_dim),nn.SiLU(),
                                   nn.Linear(hidden_dim,hidden_dim),nn.SiLU())
        self.intercept=nn.Linear(hidden_dim,1);self.unary=nn.Linear(hidden_dim,binary_dim)
        self.pair=nn.Linear(hidden_dim,len(canonical))
        self.residual=nn.Sequential(nn.Linear(hidden_dim+binary_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,1))
        nn.init.normal_(self.residual[-1].weight,std=1e-4);nn.init.zeros_(self.residual[-1].bias)

    def components(self,state,continuous_action,binary_action):
        z=self.context(torch.cat((state,continuous_action),-1));unary=self.unary(z)
        pair=self.pair(z);value=self.intercept(z)+(unary*binary_action).sum(-1,keepdim=True)/math.sqrt(self.binary_dim)
        if len(self.edges):
            products=binary_action[:,self.edges[:,0]]*binary_action[:,self.edges[:,1]]
            value=value+(pair*products).sum(-1,keepdim=True)/math.sqrt(len(self.edges))
        residual=self.residual(torch.cat((z,binary_action),-1))
        return value+self.residual_weight*residual,unary,pair,residual

    def forward(self,state,continuous_action,binary_action):return self.components(state,continuous_action,binary_action)[0]

    def pair_matrix(self,state,continuous_action):
        z=self.context(torch.cat((state,continuous_action),-1));values=self.pair(z)
        matrix=values.new_zeros(len(state),self.binary_dim,self.binary_dim)
        if len(self.edges):
            matrix[:,self.edges[:,0],self.edges[:,1]]=values;matrix[:,self.edges[:,1],self.edges[:,0]]=values
        return matrix

class TwinStructuredHardActionCritic(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__();self.q1=StructuredHardActionCritic(*args,**kwargs);self.q2=StructuredHardActionCritic(*args,**kwargs)
    def forward(self,*inputs):return self.q1(*inputs),self.q2(*inputs)
    def min_q(self,*inputs):return torch.minimum(*self(*inputs))
    def make_target(self):
        target=copy.deepcopy(self);target.requires_grad_(False);target.eval();return target
