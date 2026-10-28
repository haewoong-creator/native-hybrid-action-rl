import torch
from hybrid_sb.models.spin_network import StateConditionedSpinNetwork
def test_structure_variation_overfit_and_checkpoint(tmp_path):
    torch.manual_seed(1); net=StateConditionedSpinNetwork(4,4,(16,16)).double(); x=torch.randn(20,4,dtype=torch.float64)
    J,h=net(x); assert J.shape==(20,4,4) and h.shape==(20,4)
    assert torch.equal(J,J.transpose(-1,-2)); assert torch.equal(J.diagonal(dim1=-2,dim2=-1),torch.zeros_like(h))
    target=torch.tanh(x); optimizer=torch.optim.Adam(net.parameters(),lr=.01); initial=((h-target)**2).mean().item()
    for _ in range(100):
        _,prediction=net(x); loss=((prediction-target)**2).mean(); optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert loss.item()<.2*initial and not torch.equal(net(x[:1])[0],net(x[1:2])[0])
    path=tmp_path/"spin.pt"; torch.save(net.state_dict(),path); loaded=StateConditionedSpinNetwork(4,4,(16,16)).double(); loaded.load_state_dict(torch.load(path,weights_only=True))
    assert torch.equal(net(x)[0],loaded(x)[0]) and torch.equal(net(x)[1],loaded(x)[1])
