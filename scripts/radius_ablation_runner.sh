#!/usr/bin/env bash
# Radius-ablation driver: 3 seeds × 2 new radii = 6 (influence + elimination)
# pairs. Resumable across power cuts via per-phase sentinel files.
#
# State layout under results/radius_ablation/seed${S}_r${R}/:
#   influence_matrix.npz   # written by run_influence
#   .influence.done        # sentinel: influence completed
#   elimination/trajectories.json
#   elimination/trajectories.png
#   .elimination.done      # sentinel: elimination completed
#
# On restart: any phase whose .done sentinel exists is skipped.
# Cumulative progress is appended (not overwritten) to
# results/radius_ablation/STATUS.log so we can always see what is done
# and what was undergoing at the moment of interruption.
set -euo pipefail

PY=".venv/Scripts/python.exe"
TAU=0.005
STATUS="results/radius_ablation/STATUS.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$STATUS"
}

declare -A CKPT=(
    [42]="experiments/wflw_hrnet_w18_baseline/best.pth"
    [7]="experiments/wflw_hrnet_w18_seed7/best.pth"
    [13]="experiments/wflw_hrnet_w18_seed13/best.pth"
)

log "=== runner start (pid=$$) ==="

for seed in 42 7 13; do
    for r in 4 12; do
        outdir="results/radius_ablation/seed${seed}_r${r}"
        mkdir -p "$outdir"
        inf="${outdir}/influence_matrix.npz"
        inf_done="${outdir}/.influence.done"
        elim="${outdir}/elimination"
        elim_done="${outdir}/.elimination.done"
        ck="${CKPT[$seed]}"
        tag="seed=${seed} r=${r}"

        if [[ -f "$inf_done" ]]; then
            log "SKIP influence ${tag} (sentinel present)"
        else
            log "START influence ${tag}"
            # Clean up a partial influence file if a prior run crashed
            # mid-write (no sentinel means we cannot trust the file).
            rm -f "$inf"
            "$PY" -m scripts.run_influence \
                --checkpoint "$ck" \
                --out "$inf" \
                --radius "$r" \
                --batch-size 32
            touch "$inf_done"
            log "DONE  influence ${tag}"
        fi

        if [[ -f "$elim_done" ]]; then
            log "SKIP elimination ${tag} (sentinel present)"
        else
            log "START elimination ${tag}"
            rm -rf "$elim"
            "$PY" -m scripts.run_elimination \
                --checkpoint "$ck" \
                --influence "$inf" \
                --out-dir "$elim" \
                --radius "$r" \
                --batch-size 32 \
                --tolerance "$TAU"
            touch "$elim_done"
            log "DONE  elimination ${tag}"
        fi
    done
done

log "=== runner finished: all 6 pairs complete ==="
