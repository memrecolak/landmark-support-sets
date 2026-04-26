# Radius-ablation run status

Driver: `scripts/radius_ablation_runner.sh` (idempotent, resumable across power cuts).
Progress log (append-only): `results/radius_ablation/STATUS.log`.

## Grid (3 seeds × 2 new radii = 6 pairs)

r=8 already done in main results tree (seed 42 / 7 / 13) — that is the
anchor column; only r=4 and r=12 are the new runs.

| Seed | Radius | Influence | Elimination |
| ---: | ---: | --- | --- |
| 42 | 4  | `seed42_r4/.influence.done`  | `seed42_r4/.elimination.done` |
| 42 | 12 | `seed42_r12/.influence.done` | `seed42_r12/.elimination.done` |
| 7  | 4  | `seed7_r4/.influence.done`   | `seed7_r4/.elimination.done` |
| 7  | 12 | `seed7_r12/.influence.done`  | `seed7_r12/.elimination.done` |
| 13 | 4  | `seed13_r4/.influence.done`  | `seed13_r4/.elimination.done` |
| 13 | 12 | `seed13_r12/.influence.done` | `seed13_r12/.elimination.done` |

A sentinel file (`.<phase>.done`) is touched **only after the phase
writes its artifact successfully**. If a file exists without its
sentinel, the runner assumes a crash mid-write and re-runs the phase
(deleting the stale artifact first).

## To check status at a glance

```bash
ls results/radius_ablation/*/.influence.done   2>/dev/null | wc -l   # expect 6 when all influence done
ls results/radius_ablation/*/.elimination.done 2>/dev/null | wc -l   # expect 6 when all elim done
tail -20 results/radius_ablation/STATUS.log
```

## To resume after a power cut

```bash
bash scripts/radius_ablation_runner.sh > results/radius_ablation/runner.log 2>&1 &
```

The runner will re-read the sentinels and skip everything that is
already complete, picking up at the phase that was undergoing when
power was lost.
