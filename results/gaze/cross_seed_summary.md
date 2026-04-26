## Cross-seed gaze regression on MPIIFaceGaze (15-fold LOPO)

| Seed | |S_eye| | all98 mean ± std | eye_support mean ± std | non_support mean ± std | gap eye-all98 | gap non-all98 | Wilcoxon p (eye) | Wilcoxon p (non) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed42 | 59 | 8.34 ± 2.85° | 8.12 ± 2.15° | 10.99 ± 1.80° | -0.22° | +2.65° | 0.561 | 0.0084 |
| seed7 | 85 | 8.63 ± 2.67° | 8.68 ± 2.61° | 11.04 ± 1.85° | +0.05° | +2.41° | 0.524 | 0.0125 |
| seed13 | 84 | 8.97 ± 2.81° | 8.73 ± 2.73° | 11.01 ± 1.85° | -0.23° | +2.04° | 0.035 | 0.0181 |

- **Eye-support gap, mean across seeds:** -0.134° (max |gap| over seeds: 0.232°)
- **Non-support gap, mean across seeds:** +2.368° (min over seeds: +2.043°)
