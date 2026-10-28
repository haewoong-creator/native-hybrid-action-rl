# Curated experimental results

The repository preserves the final aggregate and seed-level evidence used in the research write-up.

## HardMove

The study covers HM4–HM10, corresponding to 16–1,024 binary configurations. The directory includes final return, success, paired comparison, per-seed, runtime, and core ablation tables.

## Cogen-BC

The final archived comparison reports mean return over five training seeds and 50 independent test days per seed: Ours-C06-OQ −3.927M, HyAR-TD3 −32.471M, P-TD3 −61.003M, and HHQN-TD3 −69.854M. Lower-magnitude negative return is better in this cost-derived task. Full precision, seed values, runtime, action-use diagnostics, and stabilization qualifications are retained in `results/cogen/`.

## GymInvMgmt

On the fixed-topology study, Ours versus HyAR-TD3 mean returns were 2015.52 vs. 2029.04 (C6), 1722.11 vs. 1852.52 (C8), and 1562.29 vs. 1455.05 (C10). The paired 95% confidence intervals all include zero, so these data do not justify a blanket superiority claim. Seed-level stability and compute tables are retained.

## Interpretation boundary

Training seeds—not evaluation episodes—are treated as the independent statistical units. Environment-specific stabilization did not transfer uniformly, and the GymInvMgmt experiments show seed instability at larger control sets. Read the domain reports before using a single headline number.
