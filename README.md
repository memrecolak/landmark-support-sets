# landmark-support-sets

Code, results, and trained-model artifacts for an intervention-based study of which facial landmarks a 98-point heatmap detector actually consults to localise each anatomical region.

For each ordered pair `(k, j)` of WFLW landmarks we replace a circular disk around landmark `k`'s ground-truth location with constant fill and record the change in NME at landmark `j`, yielding a 98×98 importance matrix. From that matrix we extract, per region, the smallest landmark subset whose context keeps target NME within tolerance of baseline (the per-region *minimum support set*) via influence-ordered backward-greedy elimination.

Three findings hold on every cell of a 3-seed × 3-radius × 8-tolerance robustness grid:

1. Every region's support set is strictly smaller than 98.
2. The contour region requires the largest support.
3. Contour support exceeds nose support.

A downstream check on MPIIFaceGaze (15-fold leave-one-person-out) shows the eye-region support set tracks the all-98 input on every seed (max gap 0.23°), while the complement degrades by 2.0–2.6°. A paired Wilcoxon over the 15 fold-pairs confirms the asymmetry: complement-vs-all-98 is significant on every seed (p ≤ 0.018); eye-support-vs-all-98 is indistinguishable on Seeds A and B, and significantly *negative* (support better) on Seed C.

A training-time corroboration retrains HRNet-W18 with the loss masked to the 59 eye-support landmarks on three independent seeds; eye-region NME matches the all-98 baseline to within ≤ 1σ (mean Δ = +0.0008 left_eye, −0.0004 right_eye). A cross-architecture sanity check on STAR-Loss (Stacked Hourglass + AAM) on the same WFLW crops reproduces all three load-bearing claims (`|S| < 98`, contour-largest, contour > nose), and the per-landmark heatmap σ from STAR correlates with the HRNet support cardinality at Spearman ρ = −0.52 (95% bootstrap CI [−0.65, −0.37], p_perm ≈ 1e-4, N = 98).

## Layout

```
configs/      training configs (HRNet-W18 on WFLW, three seeds)
results/      influence matrix, support-set tables, gaze results, per-attribute breakdown
scripts/      entry-point scripts (data prep, train, ablate, analyse)
src/          library code
```

`results/` ships with the artifacts compiled into the manuscript:

- `influence_matrix.npz` — the Seed-A 98×98 mean-Δ-on-Δ importance matrix (~92 MB)
- `mpiifacegaze_landmarks.npz` — predicted landmarks for every MPIIFaceGaze frame, Seed-A detector (~30 MB)
- `cross_seed_tau_table.{json,md}` — τ-sweep × seed × radius support-set sizes
- `radius_ablation_table.{json,md}` — radius sweep at τ = 0.005
- `forward_greedy_bound.{json,md}` — linear-additivity surrogate vs. backward-greedy comparison
- `dark_decoding_eval.{json,md}` — per-region NME with DARK decoding
- `gaze/cross_seed_summary.{json,md}` — cross-seed gaze table + paired-Wilcoxon p-values
- `gaze/{,seed7/,seed13/}gaze_results.json` — per-seed 15-fold LOPO gaze results
- `gaze/seed{7,13}/landmarks.npz` — per-seed MPIIFaceGaze landmark predictions (~30 MB each)
- `attribute_analysis/`, `elimination/`, `radius_ablation/`, `seed7/`, `seed13/` — supporting per-cell outputs
- `star/influence_matrix_star.npz` — STAR-Loss 98×98 influence matrix on the same WFLW crops (cross-architecture sanity check)
- `star/sigma_per_landmark.npz`, `star/sigma_vs_support_stats.json` — STAR per-landmark heatmap σ + Spearman ρ stats vs. HRNet support cardinality
- `star/trajectories_star.json`, `star/robust_claims_summary.md` — STAR per-region elimination trajectories and a check that the three load-bearing HRNet claims also hold on STAR

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows bash; use .venv/bin/activate on Linux/macOS
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Verified on Python 3.12 with PyTorch 2.11+cu128. CUDA 12.8 wheels work on both Ada (RTX 4070 Ti SUPER) and Blackwell (RTX 5070 Ti) cards. CPU-only inference works for the gaze and analysis pipelines but not for the WFLW training step.

## Datasets

Neither dataset is bundled (license-restricted). Download from the original sources:

- **WFLW** — https://wywu.github.io/projects/LAB/WFLW.html (place under `data/wflw/`)
- **MPIIFaceGaze** — https://www.perceptualui.org/research/datasets/MPIIFaceGaze/ (place under `data/mpiifacegaze/`)

Then run `python scripts/prepare_wflw.py` and `python scripts/prepare_mpiifacegaze.py` to generate the index files used by training and inference.

## Reproducing the paper results

The full pipeline (3 seeds × 3 radii) takes roughly 45 GPU-hours on a single RTX 4070 Ti SUPER. Numbers below are per seed.

```bash
# 1. Train HRNet-W18 (~2.5 h per seed)
python scripts/run_influence.py --config configs/wflw_hrnet_w18.yaml          # seed 42
python scripts/run_influence.py --config configs/wflw_hrnet_w18_seed7.yaml    # seed 7
python scripts/run_influence.py --config configs/wflw_hrnet_w18_seed13.yaml   # seed 13

# 2. Compute the 98x98 influence matrix (~45 min per (seed, radius) cell)
#    Sweep r in {4, 8, 12} via the shell runner:
bash scripts/radius_ablation_runner.sh

# 3. Backward-greedy elimination per region (~5.5 h per seed)
python scripts/run_elimination.py --seed 42

# 4. Cross-seed and tau-sweep aggregation
python scripts/aggregate_cross_seed_tau.py
python scripts/aggregate_radius_ablation.py
python scripts/forward_greedy_bound.py

# 5. Downstream gaze regression on MPIIFaceGaze (~10 min per seed)
#    Per-seed predict + run_gaze, then aggregate with Wilcoxon:
python scripts/predict_landmarks_mpiifacegaze.py \
    --checkpoint experiments/wflw_hrnet_w18_baseline/best.pth \
    --mpii-root data/mpiifacegaze --out results/gaze/landmarks.npz
python scripts/run_gaze.py \
    --landmarks results/gaze/landmarks.npz \
    --trajectories results/elimination/trajectories.json \
    --out results/gaze
# (repeat for seeds 7 and 13, then)
python scripts/aggregate_cross_seed_gaze.py

# 6. Cross-architecture sanity check on STAR-Loss (Stacked Hourglass + AAM)
#    Confirms the three load-bearing HRNet claims (|S| < 98, contour-largest,
#    contour > nose) on a different architecture trained on the same WFLW crops.
python scripts/run_influence_star.py
python scripts/run_elimination_star.py
python scripts/extract_star_sigma.py
python scripts/check_robust_claims_star.py
python scripts/scatter_star_sigma_vs_support.py   # writes sigma_vs_support_stats.json

# 7. Training-time corroboration: HRNet-W18 with loss masked to the eye-support set.
#    Three independent seeds; eye-region NME should match the all-98 baseline
#    to within ~1σ (Δ = +0.0008 left_eye, −0.0004 right_eye).
python -m src.training.train_subset --config configs/wflw_hrnet_w18_eyesupport_seed7381.yaml
python -m src.training.train_subset --config configs/wflw_hrnet_w18_eyesupport_seed9321.yaml
python -m src.training.train_subset --config configs/wflw_hrnet_w18_eyesupport_seed6027.yaml
```

The pre-computed Seed-A outputs in `results/` let you skip steps 1–3 and reproduce the numerical claims directly. The pre-computed STAR artifacts in `results/star/` let you skip step 6's training pass and verify the cross-architecture claims directly.

## License

Code is released under the [MIT License](LICENSE). Trained-model checkpoints follow the same license; WFLW and MPIIFaceGaze remain under their respective dataset licenses.

## Citation

A pre-print of the manuscript will be linked here once available. Until then, cite this repository directly:

```
@misc{colak_landmark_support_sets,
  author       = {Çolak, M. Emre},
  title        = {landmark-support-sets: Per-Region Minimum Support Sets for Facial Landmark Detection},
  year         = {2026},
  howpublished = {\url{https://github.com/memrecolak/landmark-support-sets}}
}
```
