# Domain-agnostic scale stabilization ablation

## Frozen scope and validity

This is a 2×2 mechanistic ablation rooted at frozen Ours-C06. The Cogen-BC environment, reward, dynamics, data, forecast, action mapping, CT_NrBays=12 rule, architecture, optimizer, exploration, OA/pair machinery, and all C06 hyperparameters were unchanged. The sealed final-test split was never used. All selection and reporting here use the pre-existing HPO validation split.

Observation moments use population Welford updates on the training stream only, with `epsilon=1e-6`, no clipping, and frozen statistics during evaluation. Value normalization uses one shared TD-target moment stream for both online twins and target twins. For old `(mu,sigma)` and new `(mu',sigma')`, every additive structured-Q output head is multiplied by `sigma/sigma'`; the intercept alone additionally receives `(mu-mu')/sigma'`. This preserves `sigma*Qhat+mu` while avoiding applying the affine offset once per additive head. Replay and reward remain raw.

All 17 required normalization tests passed. Maximum measured physical-Q change across a PopArt update was 8.93e-4 in float precision.

## Stage B result (100k, validation, seeds 0–2)

| Variant | Return mean ± sample SD | Raw seed returns | AUC | Saturation | Critic grad | Critic clip | Binary support | Q–MC Pearson / Spearman |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| C06 | -15,417,577 ± 1,283,225 | [-16322576.711053161, -13949007.695671234, -15981147.031914368] | -30,509,221 | 0.996 | 1.771e+13 | 1.000 | 2.67 | 0.023 / -0.255 |
| C06-O | -13,155,401 ± 476,607 | [-12619763.724203415, -13313792.563160095, -13532646.02320343] | -22,183,249 | 0.861 | 4.579e+11 | 1.000 | 17.67 | -0.137 / 0.171 |
| C06-Q | -12,572,216 ± 2,880,754 | [-15896000.01085724, -10795933.800424041, -11024712.714477234] | -19,281,324 | 0.034 | 6.726e+00 | 0.959 | 2.33 | -0.170 / -0.376 |
| C06-OQ | -3,894,038 ± 170,419 | [-4041585.8529844666, -3707503.3718751525, -3933025.1724821473] | -8,665,292 | 0.000 | 3.075e-01 | 0.322 | 7.33 | 0.504 / 0.321 |

The validation references are random hybrid -20,430,283, all-off midpoint -5,409,074, all-on midpoint -13,426,502, and midpoint plus random binary -9,116,930. C06-OQ beats all four references in every seed; its worst seed is -4.04M, above the all-off mean -5.41M. No significance claim is made at n=3.

## Mechanistic findings

Observation standardization is supported for early training: with only the single reset observation available, step-0 saturation is 1.000 (and therefore not improved), but by step 250 it falls to zero versus 0.957 for C06, remains zero through the 1k warmup boundary, broadens binary support, and causes no numerical failure. C06-O alone later returns to substantial saturation and large gradients. Value stabilization is supported for optimization conditioning: C06-Q reduces critic gradients from about 1e13 to single digits and continuous saturation to 0.034. Its critic clipping fraction remains high and Q–MC rank calibration does not improve, so calibration support for V2 alone is absent.

The combined C06-OQ interaction is strongest. It reaches zero final continuous saturation, sub-unit actor/critic gradients, much lower clipping fractions, higher binary support than C06, much smaller across-seed return spread, and the only clearly positive Q–MC Pearson correlation (0.504). Its return improves by about 11.52M over C06. This supports multiple interacting scale mechanisms. Binary support recovers partially (7.33 versus 2.67 combinations on average), but remains far below all 64 actions, so scale stabilization does not establish complete recovery from binary collapse.

Q–MC uses 36 candidates per variant (3 validation initial days × 12 actions). Correlation estimates are diagnostic and small-sample; top-1 agreement is 1/3 for C06-OQ, so calibration is improved but not solved.

## HardMove transfer

The exact C06-OQ rules were transferred without environment-specific settings or HPO. Each cell uses 250k interactions, evaluations every 10k over 20 episodes, and seeds 0–2, matching the existing C06 Stage-1 source. Results are in `hardmove/summary.csv` and raw rows in `hardmove/seed_level_results.csv`.

| Environment | Original C06 success | C06-OQ success | Original success AUC | C06-OQ success AUC |
|---|---:|---:|---:|---:|
| HardMove4 | 0.467 ± 0.029 | 0.000 ± 0.000 | 0.436 | 0.006 |
| HardMove6 | 0.567 ± 0.029 | 0.000 ± 0.000 | 0.383 | 0.005 |
| HardMove8 | 0.417 ± 0.236 | 0.017 ± 0.029 | 0.255 | 0.008 |
| HardMove10 | 0.467 ± 0.058 | 0.000 ± 0.000 | 0.249 | 0.004 |

The transfer is numerically stable but does not preserve HardMove performance: 11 of 12 C06-OQ endpoint successes are zero and the remaining cell is 0.05, while original C06 has substantially positive mean success across the suite. Thus Cogen evidence supports a scale-conditioning remedy in that domain, but the unchanged normalization cannot be claimed as a generally performance-preserving replacement for canonical C06.

## Answers and verdicts

1. Observation standardization resolves early tanh saturation: **yes from step 250; no at literal step 0, when only one observation defines the moments**.
2. Value normalization resolves critic-gradient scale: **yes**; standalone Q–MC calibration: **no**.
3. Effects are complementary: **yes**, with the combined variant substantially stronger.
4. C06-OQ is stable and better than C06 on all three Cogen seeds: **yes**.
5. Binary support recovers without exploration changes: **partially**.
6. Q–MC calibration improves: **for C06-OQ, partially**.
7. It beats random hybrid: **yes, every seed**.
8. It beats the all-off midpoint mean: **yes, every seed**.
9. It preserves HardMove performance: **no**.
10. Evidence supports domain-agnostic rules curing the measured Cogen scale pathology, but HardMove refutes a broad claim that they preserve performance across domains. A defensible claim must state both findings.

`COGEN_BC_STABILIZATION_ABLATION_COMPLETE = YES`  
`OBSERVATION_STABILIZATION_SUPPORTED = YES`  
`VALUE_STABILIZATION_SUPPORTED = YES`  
`GENERAL_STABILIZATION_PROMISING = YES`  
`COGEN_BC_STABILIZED_LEARNABLE = YES`  
`HARDMOVE_STABILIZATION_CHECK_COMPLETE = YES`  
`HARDMOVE_PERFORMANCE_PRESERVED = NO`
