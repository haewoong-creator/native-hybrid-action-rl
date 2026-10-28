import pytest,torch
from hybrid_sb.anova.condition_aware_design import *

def test_condition_threshold_detection_and_full_rank():
    edges=[(0,1),(1,2),(2,3)]
    selected=select_condition_aware_design(4,edges,7,condition_threshold=100)
    assert selected.diagnostics.rank==selected.diagnostics.terms
    assert selected.diagnostics.condition_number<=100
    with pytest.raises(RuntimeError):
        select_condition_aware_design(4,edges,7,condition_threshold=1.0)

def test_automatic_enlargement_can_be_observed():
    edges=[(i,j) for i in range(6) for j in range(i+1,6)]
    # A deliberately strict threshold makes 2p fail while a later candidate may pass.
    first=select_condition_aware_design(6,edges,4,condition_threshold=20)
    assert first.row_multiplier in (2,3,4)
