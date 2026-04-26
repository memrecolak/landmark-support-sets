# landmark-support-sets

Code, results, and trained-model artifacts for an intervention-based study of which facial landmarks a 98-point heatmap detector actually consults to localise each anatomical region.

For each ordered pair `(k, j)` of WFLW landmarks we replace a circular disk around landmark `k`'s ground-truth location with constant fill and record the change in NME at landmark `j`, yielding a 98×98 importance matrix. From that matrix we extract, per region, the smallest landmark subset whose context keeps target NME within tolerance of baseline (the per-region *minimum support set*) via influence-ordered backward-greedy elimination.

Three findings hold on every cell of a 3-seed × 3-radius × 8-tolerance robustness grid:

1. Every region's support set is strictly smaller than 98.
2. The contour region requires the largest support.
3. Contour support exceeds nose support.

A downstream check on MPIIFaceGaze (15-fold leave-one-person-out) shows the eye-region support set tracks the all-98 input on every seed (max gap 0.23°), while the complement degrades by 2.0–2.6°. A paired Wilcoxon over the 15 fold-pairs confirms the asymmetry: complement-vs-all-98 is significant on every seed (p ≤ 0.018); eye-support-vs-all-98 is indistinguishable on seeds 42 and 7, and significantly *negative* (support better) on seed 13.

## Layout

```
configs/      training configs (HRNet-W18 on WFLW, three seeds)
figures/      PNG figures used in the manuscript
results/      influence matrix, support-set tables, gaze results, per-attribute breakdown
scripts/      entry-point scripts (data prep, train, ablate, analyse, plot)
src/          library code
```

`results/` ships with the artifacts compiled into the manuscript:

- `influence_matrix.npz` — the seed-42 98×98 mean-Δ-on-Δ importance matrix (~92 MB)
- `mpiifacegaze_landmarks.npz` — predicted landmarks for every MPIIFaceGaze frame, seed-42 detector (~30 MB)
- `cross_seed_tau_table.{json,md}` — τ-sweep × seed × radius support-set sizes
- `radius_ablation_table.{json,md}` — radius sweep at τ = 0.005
- `forward_greedy_bound.{json,md}` — linear-additivity surrogate vs. backward-greedy comparison
- `dark_decoding_eval.{json,md}` — per-region NME with DARK decoding
- `gaze/cross_seed_summary.{json,md}` — cross-seed gaze table + paired-Wilcoxon p-values
- `gaze/{,seed7/,seed13/}gaze_results.json` — per-seed 15-fold LOPO gaze results
- `gaze/seed{7,13}/landmarks.npz` — per-seed MPIIFaceGaze landmark predictions (~30 MB each)
- `attribute_analysis/`, `elimination/`, `radius_ablation/`, `seed7/`, `seed13/` — supporting per-cell outputs

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

# 6. Plots
python scripts/plot_influence_matrix.py
python scripts/plot_elimination_trajectories.py
python scripts/plot_radius_ablation.py
python scripts/plot_gaze.py
```

The pre-computed seed-42 outputs in `results/` let you skip steps 1–3 and reproduce the figures and tables directly.

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
