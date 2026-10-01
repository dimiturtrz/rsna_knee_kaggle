#!/usr/bin/env bash
# Cell D of the 2x2 — HIS 44-slot corpus, HIS recipe, OUR trunk. The one cell nobody has run.
#
# reading: raptor_ft_ourtrunk44.pt
# bead: knee-s39kk (roster), C149 §2 (the ask)
#
# WHY THIS AND WHY NOW. D146 §2 measured that both foreign families share ONE trunk
# (`coatnet_rmlp_2_rw_384`), so the corpus-versus-trunk decomposition it proposed rests on n=2 and
# cannot separate the two terms — D147 §2 says so in writing: *"Corpus moves distance-to-us; trunk sets
# the floor they share. Both terms are live and I cannot separate them at n=2."* Every cell run so far
# holds the trunk fixed and moves the corpus. This one holds the CORPUS fixed and moves the trunk, and
# it is the only arrangement that separates them.
#
# WHAT IT COSTS AND WHAT IT BUYS. ~1.5-1.8 h of card (cell A' ran 16 epochs at ~400 s/epoch on a
# coatnet at 384; dinov2-small at 336 is lighter), and NO decode at all — the corpus is on disk at
# `/d/data/volumetric/mri/processed/raptor/raptor-corpus`, which is what prices `deep32` (a 95 GB cache build) out of the same
# slot. It produces a candidate member as a by-product, so a null still leaves a scored seat.
#
# PRE-REGISTERED, before it starts. D147 §2 reads the shared-trunk floor at nn-corr 0.89-0.91 between
# the two foreign families and `repl2x2`'s distance to OUR pool at 0.812-0.827. Our trunk is the pool's
# trunk, so this member shares with our pool exactly what the two foreign families share with each
# other. **Predicted nn-corr to our pool 0.87-0.92, i.e. AT OR ABOVE `hisrecipe` (0.883-0.888) and well
# above `repl2x2` (0.82)** — the corpus alone should not buy the 2.6th percentile once the trunk is
# ours. D35 (`experiments/README.md`, the seat gate) then prices it near +0.0012, a pool-median seat.
#
#   - If it lands at 0.87-0.92: trunk carries the decorrelation, the corpus term is small, and C149 §2's
#     geometry arm is priced DOWN accordingly. That is the expensive thing this cheap arm protects.
#   - If it lands below 0.85: the corpus carries it, the geometry arm is the right next spend, and the
#     D35 seat is ~+0.0033. This is the outcome that would make a second axis real.
#   - Either way it is an INSTRUMENT first and a member second. It is not queued as an improvement.
#
# WHICH CORPUS IT ACTUALLY OPENS, asserted rather than assumed. His `_find` (T:86) walks
# `/kaggle/input`, which `foreign_run --kaggle-input` re-roots at `$RAPTOR_CORPUS` (the 44-slot part;
# `raptor-maxspan` = 64 sits beside it). The walk returns the first match and nothing in his log records
# which one it read, so a mis-rooted walk is a silently mislabelled cell. The guard below replicates
# `_find` over that root and refuses unless it resolves to the 44-slot part. That is the same class of defect as the cache-key collisions `slots.py` records escaping three
# times.
#
# --norm imagenet IS NOT OPTIONAL. Cell A' was launched without it, fed [0,1] to a timm-pretrained
# trunk, and was DISCARDED at epoch 3 (`logs/repl2x2_a.full.log:20`). Same trunk-init path here.
set -u

REPO="D:/personal_projects/rsna_knee_kaggle"
# THROUGH `uv`, AS AN ARRAY: `arm_check` RUNS every arm file to read its flags, and the only thing
# making that a lint is a stub shadowing `uv` alone — a direct interpreter walks past it and the lint
# becomes a card job (measured 2026-09-09). `--project` pins the env whatever the caller's cwd; it resolves the same `$REPO/.venv` this line used to name, since this arm does not source
# `experiments/lib.sh` and so nothing points `uv` at `.venv-arm`. An array, because the invocation is
# several words and `nohup $PY` would split it wrong.
PY_RUN=(env UV_NO_SYNC=1 uv run --project "$REPO" python)
SCRIPT="$REPO/external/raptor/knee-mri-training-the-twelve-finding-model.py"
LOG="$REPO/logs/ourtrunk44.full.log"
RAPTOR_CORPUS="D:/data/volumetric/mri/processed/raptor/raptor-corpus"

export PYTHONPATH="$REPO"

RAPTOR_CORPUS="$RAPTOR_CORPUS" "${PY_RUN[@]}" - <<'GUARD' >> "$LOG" 2>&1 || { echo "ARM REFUSED ourtrunk44 (wrong corpus resolved)" >> "$LOG"; exit 1; }
import os, sys
def find(name, root=os.environ["RAPTOR_CORPUS"]):
    for d, _, fs in os.walk(root):
        if name in fs:
            return os.path.join(d, name)
    return None
path = find("all_vols.npy")
print("corpus resolves to", path, flush=True)
if path is None or "raptor-corpus" not in path.replace("\\", "/"):
    sys.exit(1)
import numpy as np
vols = np.load(path, mmap_mode="r")
print("shape", vols.shape, flush=True)
sys.exit(0 if vols.shape[1] == 44 else 1)
GUARD

nohup "${PY_RUN[@]}" -m kneemri.experiments.foreign_run \
  --script "$SCRIPT" \
  --arch vit_small_patch14_dinov2.lvd142m \
  --res 336 --epochs 16 --bs 8 --k 12 --k_eval 24 --grad_ckpt --ckpt timm --norm imagenet \
  --tag ourtrunk44 \
  --kaggle-input "$RAPTOR_CORPUS" \
  --labels "D:/data/volumetric/mri/processed/raptor/labels/labels_llm_soft.parquet" \
  --folds 1 --fold 0 --fold_file "$REPO/external/raptor/fold0_909.json" \
  --workers 4 >> "$LOG" 2>&1 &
echo "ourtrunk44 launched as pid $! at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
wait
echo "ARM DONE ourtrunk44" >> "$LOG"
