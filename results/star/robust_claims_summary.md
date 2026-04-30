# T2: STAR cross-architecture robust-claim check

## Per-region support cardinality (HRNet vs STAR, WFLW-test, r=8, τ=0.005)

| region       | HRNet |S| | STAR |S| |
|--------------|----------:|---------:|
| contour      |        85 |       94 |
| left_brow    |        50 |       56 |
| right_brow   |        50 |       69 |
| nose         |        62 |       72 |
| left_eye     |        38 |       63 |
| right_eye    |        41 |       71 |
| mouth_outer  |        49 |       76 |
| mouth_inner  |        47 |       76 |

## Claims

| claim | HRNet | STAR |
|-------|:-----:|:----:|
| (i) every \|S\| < 98 | OK | OK |
| (ii) contour largest | OK | OK |
| (iii) contour > nose | OK | OK |

Largest-support region — HRNet: **contour** (85);  STAR: **contour** (94).