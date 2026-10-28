# Native Hybrid-Action Actor–Critic

Research portfolio for reinforcement learning with coupled binary and continuous actions. The project studies how to retain native hybrid controls while avoiding exhaustive treatment of all \(2^b\) binary configurations.

## What is included

- A state-conditioned binary actor and continuous reference policy
- Structured twin critics for hard hybrid actions
- Sparse, condition-aware interaction design and mixture-proposal utilities
- Canonical training and scale-stabilized trainer implementations
- Curated aggregate results from HardMove, Cogen-BC, and GymInvMgmt
- Paper and supplement PDFs with their LaTeX sources
- Focused unit tests for the released core components

## Evidence at a glance

| Domain | Action scaling studied | Main comparison retained |
|---|---|---|
| HardMove | 4–10 binary variables (16–1,024 combinations) | Ours and multiple hybrid-action baselines, ablations, runtime |
| Cogen-BC | Native binary/continuous plant controls | Ours vs. HyAR-TD3, P-TD3, HHQN-TD3; stabilization ablation |
| GymInvMgmt | Fixed topology with C6/C8/C10 control sets | Ours vs. HyAR-TD3, seed stability and compute |

The tables report the archived experimental outputs without reselecting seeds or recomputing metrics. See [`results/`](results/) and [`docs/RESULTS.md`](docs/RESULTS.md).

## Install and test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
```

The released code is the method core, not a bundled benchmark distribution. External environments, raw datasets, checkpoints, and large logs are deliberately excluded. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Paper

- [`paper/main_final.pdf`](paper/main_final.pdf)
- [`paper/supplement_final.pdf`](paper/supplement_final.pdf)

## Scope

This repository is a curated research artifact and portfolio. Results support empirical claims in the evaluated settings; they do not establish universal superiority or a convergence guarantee.
