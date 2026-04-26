## Forward-greedy / linear-additivity bound on support sizes (tau=0.005, 3 seeds)

**Columns:** HRNet-backward (paper), LINEAR-backward (optimal under additivity → forward-greedy = backward-greedy on the 1-D thresholding problem), RANDOM-order expected (closed-form under the same linear model).

| Region | HRNet mean±std | LINEAR mean±std | RANDOM E[|S|] | HRNet − LINEAR | RANDOM − HRNet (savings) |
| --- | ---: | ---: | ---: | ---: | ---: |
| contour | 84.0 ± 1.7 | 82.3 ± 1.5 | 90.9 | +1.7 | +6.9 |
| left_brow | 57.3 ± 7.0 | 41.3 ± 16.3 | 88.1 | +16.0 | +30.8 |
| right_brow | 48.7 ± 9.1 | 43.0 ± 14.0 | 87.8 | +5.7 | +39.2 |
| nose | 63.7 ± 4.7 | 30.0 ± 16.1 | 66.7 | +33.7 | +3.0 |
| left_eye | 56.7 ± 18.5 | 43.3 ± 17.6 | 89.4 | +13.3 | +32.7 |
| right_eye | 50.7 ± 12.7 | 31.7 ± 14.6 | 88.1 | +19.0 | +37.5 |
| mouth_outer | 53.0 ± 7.8 | 39.0 ± 10.6 | 91.0 | +14.0 | +38.0 |
| mouth_inner | 52.7 ± 9.8 | 31.7 ± 4.0 | 92.0 | +21.0 | +39.3 |

### Per-seed detail (HRNet / LINEAR / RANDOM)

| Region | seed 42 | seed 7 | seed 13 |
| --- | --- | --- | --- |
| contour | 85 / 81 / 89.8 | 82 / 84 / 91.7 | 85 / 82 / 91.1 |
| left_brow | 50 / 34 / 87.6 | 58 / 60 / 91.0 | 64 / 30 / 85.9 |
| right_brow | 50 / 43 / 88.0 | 57 / 57 / 89.6 | 39 / 29 / 86.0 |
| nose | 62 / 13 / 40.6 | 60 / 45 / 80.4 | 69 / 32 / 79.0 |
| left_eye | 38 / 45 / 89.7 | 57 / 60 / 90.7 | 75 / 25 / 87.8 |
| right_eye | 41 / 47 / 90.4 | 65 / 30 / 88.6 | 46 / 18 / 85.5 |
| mouth_outer | 49 / 31 / 90.4 | 62 / 51 / 91.7 | 48 / 35 / 90.9 |
| mouth_inner | 47 / 28 / 91.9 | 64 / 36 / 91.5 | 47 / 31 / 92.5 |

**Interpretation (see paper.md §Forward-greedy optimality bound):**
- `HRNet − LINEAR`: negative means the linear model predicts a larger support set than HRNet actually needs (HRNet is more forgiving than additive theory predicts); positive means the linear model is optimistic and HRNet enforces a bigger set.
- `RANDOM − HRNet`: savings of influence-ordered greedy over uninformed random masking. Large positive numbers here are the claim-1 evidence that ordering encodes structure.
