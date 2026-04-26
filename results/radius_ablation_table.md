## Occlusion-radius ablation: support-set sizes |S| at tau=0.005

**Grid:** 3 seeds (42, 7, 13) x 3 radii (4, 8, 12 px on 256x256 input).

### Seed 42
| Region | r=4 | r=8 | r=12 | range (max-min) |
| --- | ---: | ---: | ---: | ---: |
| contour | 55 | 85 | 93 | 38 |
| left_brow | 15 | 50 | 67 | 52 |
| right_brow | 13 | 50 | 72 | 59 |
| nose | 16 | 62 | 72 | 56 |
| left_eye | 18 | 38 | 58 | 40 |
| right_eye | 18 | 41 | 60 | 42 |
| mouth_outer | 28 | 49 | 65 | 37 |
| mouth_inner | 27 | 47 | 62 | 35 |

### Seed 7
| Region | r=4 | r=8 | r=12 | range (max-min) |
| --- | ---: | ---: | ---: | ---: |
| contour | 63 | 82 | 92 | 29 |
| left_brow | 26 | 58 | 60 | 34 |
| right_brow | 38 | 57 | 72 | 34 |
| nose | 40 | 60 | 74 | 34 |
| left_eye | 23 | 57 | 72 | 49 |
| right_eye | 19 | 65 | 73 | 54 |
| mouth_outer | 24 | 62 | 69 | 45 |
| mouth_inner | 35 | 64 | 74 | 39 |

### Seed 13
| Region | r=4 | r=8 | r=12 | range (max-min) |
| --- | ---: | ---: | ---: | ---: |
| contour | 55 | 85 | 89 | 34 |
| left_brow | 16 | 64 | 81 | 65 |
| right_brow | 13 | 39 | 66 | 53 |
| nose | 17 | 69 | 84 | 67 |
| left_eye | 23 | 75 | 84 | 61 |
| right_eye | 18 | 46 | 66 | 48 |
| mouth_outer | 33 | 48 | 63 | 30 |
| mouth_inner | 29 | 47 | 58 | 29 |

### Cross-seed summary (mean +/- std across seeds {42, 7, 13})
| Region | r=4 mean+/-std | r=8 mean+/-std | r=12 mean+/-std | rank r=4 | rank r=8 | rank r=12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| contour | 57.7 +/- 4.6 | 84.0 +/- 1.7 | 91.3 +/- 2.1 | 8 | 8 | 8 |
| left_brow | 19.0 +/- 6.1 | 57.3 +/- 7.0 | 69.3 +/- 10.7 | 2 | 6 | 4 |
| right_brow | 21.3 +/- 14.4 | 48.7 +/- 9.1 | 70.0 +/- 3.5 | 3 | 1 | 5 |
| nose | 24.3 +/- 13.6 | 63.7 +/- 4.7 | 76.7 +/- 6.4 | 5 | 7 | 7 |
| left_eye | 21.3 +/- 2.9 | 56.7 +/- 18.5 | 71.3 +/- 13.0 | 4 | 5 | 6 |
| right_eye | 18.3 +/- 0.6 | 50.7 +/- 12.7 | 66.3 +/- 6.5 | 1 | 2 | 3 |
| mouth_outer | 28.3 +/- 4.5 | 53.0 +/- 7.8 | 65.7 +/- 3.1 | 6 | 4 | 2 |
| mouth_inner | 30.3 +/- 4.2 | 52.7 +/- 9.8 | 64.7 +/- 8.3 | 7 | 3 | 1 |

### Robust claims (true at every (seed, radius) cell)
- |S| < 98 in every cell: **True**
- contour has the largest |S| in every cell: **True**
- contour > nose in every cell: **True**

### Fragile claims (radius-dependent)
Below contour, the rank order shifts across radii. The 'eyes need least context' reading is a r >= 8 phenomenon; at r = 4, eyebrows are typically the smallest support set.

