"""State-conditioned bounded Ising couplings and external fields."""
import math,torch
from torch import nn
class StateConditionedSpinNetwork(nn.Module):
    def __init__(self,state_dim,binary_dim,hidden_dims=(128,128),bounded=True):
        super().__init__(); self.state_dim=state_dim; self.binary_dim=binary_dim; self.bounded=bounded
        self.register_buffer("edge_i",torch.triu_indices(binary_dim,binary_dim,1)[0])
        self.register_buffer("edge_j",torch.triu_indices(binary_dim,binary_dim,1)[1])
        dims=[state_dim,*hidden_dims]; layers=[]
        for din,dout in zip(dims[:-1],dims[1:]):
            layer=nn.Linear(din,dout); nn.init.kaiming_uniform_(layer.weight,a=math.sqrt(5)); nn.init.zeros_(layer.bias)
            layers.extend((layer,nn.SiLU()))
        self.encoder=nn.Sequential(*layers); self.coupling_head=nn.Linear(dims[-1],len(self.edge_i)); self.field_head=nn.Linear(dims[-1],binary_dim)
        for head in (self.coupling_head,self.field_head):
            nn.init.normal_(head.weight,std=1e-3); nn.init.zeros_(head.bias)
    def forward(self,state):
        p=next(self.parameters())
        if not isinstance(state,torch.Tensor) or state.ndim!=2 or state.shape[1]!=self.state_dim: raise ValueError("invalid state shape")
        if state.dtype!=p.dtype or state.device!=p.device: raise ValueError("state dtype/device mismatch")
        z=self.encoder(state); edges=self.coupling_head(z); field=self.field_head(z)
        if self.bounded: edges,field=torch.tanh(edges),torch.tanh(field)
        J=torch.zeros(len(state),self.binary_dim,self.binary_dim,dtype=state.dtype,device=state.device)
        J[:,self.edge_i,self.edge_j]=edges; J[:,self.edge_j,self.edge_i]=edges
        return J,field
