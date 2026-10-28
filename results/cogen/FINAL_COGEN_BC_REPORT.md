# Final SustainGym Cogen-BC confirmatory report

**COGEN_BC_FINAL_EXPERIMENT_COMPLETE = YES**

Cogen-BC is an action-restricted SustainGym Cogen-derived variant. It fixes the original `CT_NrBays` 12-way actuator a priori to 12 for every method and preserves the source transition and reward functions and the six binary plus eight bounded continuous agent-controlled actuators. This is not a result on original unrestricted SustainGym Cogen.

Ours-C06-OQ uses the same core hybrid-action algorithm as HardMove Ours-C06, with Cogen-specific observation standardization and PopArt-style Q normalization selected in the earlier validation ablation. One identical training configuration was not used across benchmarks.

All five Ours seeds 200–204 completed 500,000 training interactions, passed checkpoint and numerical checks, and were evaluated on the 50 independent sealed final-test days only after training. Compatible baselines used the same Cogen-BC action wrapper, training budget, fresh seeds, validation-day split, and final-test days. Their HPO used only validation days. The frozen prior protocol uses `renewables_magnitude=0` for training and `renewables_magnitude=300` for validation and final-test for every method; this train/evaluation uncertainty shift is disclosed rather than changed after results.

| Method | Final return mean ± sample SD | Validation AUC 10k–500k mean ± SD | Runtime upper bound, seconds, mean ± SD |
|---|---:|---:|---:|
| Ours-C06-OQ | -3,927,165 ± 352,308 | -4,050,146 ± 226,621 | 9,673 ± 43 |
| HyAR-TD3 | -32,471,141 ± 14,677,394 | -34,113,057 ± 16,107,973 | 26,618 ± 10 |
| P-TD3 | -61,002,638 ± 31,700,657 | -64,204,009 ± 30,975,886 | 18,597 ± 14 |
| HHQN-TD3 | -69,853,871 ± 10,575,978 | -72,941,766 ± 13,365,571 | 25,008 ± 64 |
| HPPO | -31,263,058 ± 10,638,006 | -31,636,243 ± 10,442,248 | 4,162 ± 21 |

| Method | Dynamic constraint cost mean ± SD | Non-delivery cost mean ± SD | Continuous saturation mean ± SD | Binary support mean ± SD |
|---|---:|---:|---:|---:|
| Ours-C06-OQ | 2,956,784 ± 452,127 | 950,960 ± 173,782 | 0.0000 ± 0.0000 | 8.2 ± 7.8 |
| HyAR-TD3 | 29,721,820 ± 15,943,743 | 2,725,788 ± 5,490,660 | 0.5344 ± 0.1630 | 6.8 ± 13.0 |
| P-TD3 | 46,997,739 ± 36,291,831 | 13,985,035 ± 21,318,103 | 1.0000 ± 0.0000 | 1.0 ± 0.0 |
| HHQN-TD3 | 47,131,918 ± 21,850,230 | 22,703,820 ± 27,486,680 | 1.0000 ± 0.0000 | 1.0 ± 0.0 |
| HPPO | 26,196,968 ± 6,275,770 | 5,023,108 ± 5,678,754 | 0.6769 ± 0.0246 | 1.0 ± 0.0 |

SustainGym source commit: `eac2a4d5ce4ccf44b13c290edac72532c68e30f5`. Baseline source commit: `ece95621b8c49f154f96cf7d395b95362a3b3d4e`. Campaign elapsed wall time from the first Ours seed config timestamp: 14.15 hours. Sum of final training process-time upper bounds: 116.75 hours; concurrent runs make this larger than elapsed wall time.

P-TD3, HHQN-TD3, and HyAR-TD3 were interrupted and resumed from checkpoints. Their trainer summaries record only the resumed segment. The runtime table adds a bounded initial segment: lower bound from the original run-log birth to its last persisted evidence, upper bound from birth to the first observation that the processes were absent. The displayed runtime is the upper bound; lower bounds appear in `runtime_summary.csv` and seed-level CSV. Thus runtime is not exact for these methods, with a maximum initial-segment interval width of 507 seconds per seed. The interruption was not counted as a scientific training failure.

The AUC uses the common 10k–500k validation support because baseline source runs first evaluate at 10k. n=5 training seeds; no significance claim is made. The 50 days are paired across methods and are not counted as extra training seeds. Baseline source code selects CUDA when available, while the frozen Ours runner uses CPU; the runtime column uses observed or bounded reconstructed wall time under these method-specific backends and is not a controlled hardware-speed comparison.

Raw training-seed final returns:
- Ours-C06-OQ: -3,987,818, -3,635,720, -4,160,532, -4,346,236, -3,505,520
- HyAR-TD3: -35,560,680, -53,245,266, -12,174,469, -31,754,764, -29,620,524
- P-TD3: -71,796,693, -32,913,500, -69,849,035, -26,495,429, -103,958,535
- HHQN-TD3: -64,326,630, -82,728,820, -55,074,672, -71,980,230, -75,159,001
- HPPO: -42,238,020, -41,698,323, -29,258,734, -25,497,892, -17,622,322

Reference policies on the same 50 sealed days:
- all_off_midpoint: -5,194,081
- all_on_midpoint: -13,022,588
- random_hybrid: -19,088,456
- midpoint_random_binary: -8,913,279

HPO and representation-data exposure are disclosed separately:
- Ours-C06-OQ: counted validation/HPO main interactions 3,000,000; HPO representation interactions 0; final-seed representation interactions 0. 1,720,000 Cogen HPO + 1,280,000 stabilization ablation; excludes forensics
- HyAR-TD3: counted validation/HPO main interactions 700,000; HPO representation interactions 960,000; final-seed representation interactions 2,400,000. 8×50k Stage 1 + top-2 Stage 2; seed-100 trajectories resume from 50k to 100k; HyAR precollection reported separately
- P-TD3: counted validation/HPO main interactions 700,000; HPO representation interactions 0; final-seed representation interactions 0. 8×50k Stage 1 + top-2 Stage 2; seed-100 trajectories resume from 50k to 100k; HyAR precollection reported separately
- HHQN-TD3: counted validation/HPO main interactions 700,000; HPO representation interactions 0; final-seed representation interactions 0. 8×50k Stage 1 + top-2 Stage 2; seed-100 trajectories resume from 50k to 100k; HyAR precollection reported separately
- HPPO: counted validation/HPO main interactions 700,000; HPO representation interactions 0; final-seed representation interactions 0. 8×50k Stage 1 + top-2 Stage 2; seed-100 trajectories resume from 50k to 100k; HyAR precollection reported separately

Ours had more prior Cogen-specific validation exposure than any single baseline under this counted scope; therefore this comparison does not establish equal HPO budgets. Earlier forensic and exploratory work is additional historical evidence and is not counted in the table above. Baselines were given fixed, limited eight-configuration searches rather than result-driven retuning. HPPO has four distinct effective settings because source 50-step epochs make the predeclared batch sizes 32 and 64 equivalent. Ours uses Cogen-selected observation and Q normalization, while the pinned baselines retain source-native preprocessing; performance differences therefore do not isolate the core algorithm from scale handling.

PDQN-TD3 was excluded before Cogen performance inspection because its pinned parameter actor is unbounded and the frozen Cogen-BC contract forbids adding a baseline-only squash or clip. See its compatibility report.

Numerical failures, retries, and any incomplete candidate runs are retained in method run logs; only complete final five-seed runs are aggregated. See `status.json` and per-method artifacts for exact checkpoints and seed-level values.

## Answers to the prespecified scientific questions

**Q1 — Reproduction:** All five fresh seeds completed 500k interactions. Ours mean sealed final-test return was -3,927,165 ± 352,308 sample SD across those seeds. Validation AUC and each seed's return are reported above and in the CSVs.
**Q2 — References:** Every Ours seed exceeded both the random hybrid and all-off midpoint reference returns under the frozen gate. The raw seed and reference values are reported above.
**Q3 — Scale stability:** No recurrence was detected in the final five Ours runs under the recorded checks: complete checkpoints=True, finite critic gradients=True, NaN/Inf zero=True, no continuous-saturation pathology=True.
**Q4 — Compatible baselines:** Ours ranked 1/5 by mean sealed final-test episodic return among the five reported methods. This descriptive ranking is not a significance claim.
**Q5 — HPO fairness:** The environment, wrapper, training budget, fresh training seeds, validation split, sealed days, and evaluation rule were common. HPO and preprocessing exposure were unequal: Ours had at least 3,000,000 counted Cogen-specific prior interactions and selected Cogen-specific normalization, while each baseline had 700,000 main HPO interactions and source-native preprocessing. The results therefore do not isolate algorithm superiority from HPO or scale handling, and they do not support an equal-search-budget claim.
**Q6 — Representation data:** HyAR representation precollection is counted separately in `hpo_compute_summary.csv` for HPO and final seeds; it is not hidden in the 500k policy-training budget.
**Q7 — Variant identity:** These are Cogen-BC action-restricted variant results, not original unrestricted SustainGym Cogen results. The fixed `CT_NrBays=12` rule is shared by every method.
**Q8 — Cross-benchmark configuration:** HardMove used C06; the Cogen campaign used C06-OQ with Cogen-specific observation and Q normalization. They are distinct benchmark configurations.
