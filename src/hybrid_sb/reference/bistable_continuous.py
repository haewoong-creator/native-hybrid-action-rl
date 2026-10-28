"""Canonical-v1 continuous double-well reference (pure tensor API)."""
from __future__ import annotations
from dataclasses import dataclass
import torch
@dataclass(frozen=True)
class BistableContinuousReference:
    rho: torch.Tensor | float
    well: torch.Tensor | float
    def _parameters(self,c):
        rho=torch.as_tensor(self.rho,dtype=c.dtype,device=c.device)
        well=torch.as_tensor(self.well,dtype=c.dtype,device=c.device)
        if torch.any(rho<=0) or torch.any(well<=0):raise ValueError("rho and well must be positive")
        return rho,well
    def potential(self,c):
        rho,well=self._parameters(c);return (rho/4*(c.square()-well.square()).square()).sum(-1)
    def reference_drift(self,c):
        rho,well=self._parameters(c);return -rho*c*(c.square()-well.square())
    def controlled_drift(self,c,control):
        if c.shape!=control.shape:raise ValueError("c/control shape mismatch")
        return self.reference_drift(c)+control
