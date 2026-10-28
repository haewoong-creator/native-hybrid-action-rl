import copy,torch
from hybrid_sb.models.structured_hard_critic import StructuredHardActionCritic,TwinStructuredHardActionCritic
def test_structured_shape_symmetry_residual_and_twin_independence(tmp_path):
 model=StructuredHardActionCritic(4,2,8,32,residual_weight=0.).double();s=torch.randn(5,4,dtype=torch.float64);c=torch.randn(5,2,dtype=torch.float64);a=torch.where(torch.rand(5,8)>.5,1.,-1.).double()
 out=model(s,c,a);matrix=model.pair_matrix(s,c)
 assert out.shape==(5,1) and torch.equal(matrix,matrix.transpose(1,2)) and torch.equal(matrix.diagonal(dim1=1,dim2=2),torch.zeros(5,8,dtype=torch.float64))
 twin=TwinStructuredHardActionCritic(4,2,8,32);assert next(twin.q1.parameters()).data_ptr()!=next(twin.q2.parameters()).data_ptr()
 path=tmp_path/"s.pt";torch.save(model.state_dict(),path);clone=copy.deepcopy(model);clone.load_state_dict(torch.load(path,weights_only=True));assert torch.equal(out,clone(s,c,a))
def test_residual_weight_zero_ignores_residual_head():
 model=StructuredHardActionCritic(2,1,3,8,residual_weight=0.);s=torch.randn(2,2);c=torch.randn(2,1);a=torch.ones(2,3)
 before=model(s,c,a);model.residual[-1].bias.data.fill_(100);assert torch.equal(before,model(s,c,a))
