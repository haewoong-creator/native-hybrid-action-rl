import torch
from hybrid_sb.estimators.mixture_proposal import mixture_probability,sample_mixture
from hybrid_sb.exact import enumerate_binary_actions
def test_normalization_actions_epsilon_zero_and_weights():
    spin=torch.tensor([.2,-.4,.1],dtype=torch.float64); actions=enumerate_binary_actions(3)
    for epsilon in (0.,.1,1.): assert torch.allclose(mixture_probability(actions,spin,epsilon).sum(),torch.tensor(1.,dtype=torch.float64))
    samples=sample_mixture(spin,100,.2,torch.Generator().manual_seed(4))
    assert torch.all((samples.actions==-1)|(samples.actions==1)); assert torch.isfinite(samples.target_probability/samples.proposal_probability).all()
    zero=sample_mixture(spin,100,0.,torch.Generator().manual_seed(5)); assert torch.equal(zero.target_probability,zero.proposal_probability)
