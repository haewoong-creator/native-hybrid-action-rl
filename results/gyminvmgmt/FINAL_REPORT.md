# FT10 고정 topology 확장성 실험 최종 보고

본 실험은 이전 W10 결과를 본 뒤 설계한 후속 통제 실험이다. 동일 물리 W10·수요 경로에서 직접 제어 행동 집합만 C6/C8/C10으로 바꿨다. HyAR는 초기 150k random representation precollection을 각 500k 예산에 포함했다. MB-PPO는 신규 학습 대상이 아니며 과거 W10에서 Ours보다 우수했던 한계를 유지한다. 사전검사에서 원래 예정한 final 수요 3개의 배열을 읽은 절차 오류가 있어, 본학습 전 기록한 `protocol/FINAL_TEST_AMENDMENT_001.json`에 따라 전체 최종 테스트를 미열람 stream 20261015, episode 24000–24049로 교체했다. 이는 pilot 성능에 따른 선택이 아니다.

| 조건 | Ours return ± seed SD | HyAR-TD3 return ± seed SD | Ours−HyAR paired 95% CI |
|---|---:|---:|---:|
| FT10-C6 | 2015.52 ± 33.22 | 2029.04 ± 73.56 | -13.52 [-133.03, +105.98] |
| FT10-C8 | 1722.11 ± 218.40 | 1852.52 ± 110.46 | -130.41 [-509.82, +249.00] |
| FT10-C10 | 1562.29 ± 227.47 | 1455.05 ± 48.15 | +107.24 [-150.13, +364.61] |

차원별 상대 격차 변화: Δ8−Δ6=-116.89 (paired 95% CI [-531.92, +298.15]), Δ10−Δ8=+237.65 (paired 95% CI [-116.20, +591.50]). 5개 training seed가 독립 단위이므로 CI는 넓고, 평가 episode 50개를 추가 seed로 취급하지 않았다. 행동 마스크가 바뀌므로 이는 순수 수학적 차원 효과가 아니다.

원자료·시드별 결과·안정성·계산비용·그림은 각각 `evaluation/RAW_RESULTS.csv`, `aggregate/SEEDWISE_RESULTS.csv`, `aggregate/STABILITY_METRICS.csv`, `aggregate/COMPUTE_RESULTS.csv`, `aggregate/figures/`에 있다. Ours의 과거 critic Q 궤적은 로깅되지 않아 최종 checkpoint의 사후 Q만 보고하고, 이를 학습 당시 계측으로 해석하지 않는다. 별도 actor/critic update 시간·OA 호출 수·HyAR VAE 학습 시간·GPU 메모리는 이 실행에서 완전하게 계측되지 않았으므로 `COMPUTE_RESULTS.csv`에 빈 값 또는 CPU-only로 명시했다. 낮은 평균/분산, 초기 공급 OFF, HyAR 사전수집과 Ours의 기존 MB-PPO 열세를 모두 해석에 포함한다. 결과 확인 후 checkpoint/설정은 변경하지 않았다.
