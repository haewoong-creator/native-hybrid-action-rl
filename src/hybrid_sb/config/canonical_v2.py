from dataclasses import dataclass,asdict
@dataclass(frozen=True)
class CanonicalV2Config:
    config_id:str="CANONICAL_V2_FULL";epsilon:float=.10;alpha:float=1.;lambda_q:float=.01;actor_delay:int=2;polyak_tau:float=.01;gradient_clip:float=1.;warmup_steps:int=1000;batch_size:int=32;replay_capacity:int=10000;actor_lr:float=3e-4;critic_lr:float=3e-4;ising_lr:float=1e-3;anova_lr:float=3e-3;exact_enumeration:bool=False
    def top_k(self,b):return min(b,b*(b-1)//2)
    def coefficient_count(self,b):return 1+b+self.top_k(b)
    def proposal_samples(self,b):return max(16,2*self.coefficient_count(b))
    def to_dict(self):return asdict(self)
CANONICAL_V2_FULL=CanonicalV2Config()
