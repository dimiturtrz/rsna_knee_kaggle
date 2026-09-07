#!/usr/bin/env bash
# hisrecipe_distill — A′ (his trunk, his script, his recipe, his labels; fold 0 held out) with OUR
# distillation folded into the LABEL TABLE. Not one line of his script changes.
#
# WHY A LABEL TABLE AND NOT A LOSS EDIT. His loss is `nn.BCEWithLogitsLoss(pos_weight=pw)` on a float
# soft-label vector (`...twelve-finding-model.py:458`, step :483), and BCE is LINEAR in its target:
#   BCE(z, (1-w)·y + w·t) = (1-w)·BCE(z, y) + w·BCE(z, t).
# So training him on the blended table IS a distillation term toward our teacher at weight w, exact,
# with his file untouched (`foreign_run`'s contract; `external/` is a checkout). What differs from our
# `DistilledObjective` (`kneemri/training/objective.py:394-436`) is the MECHANISM: ours imitates the
# teacher's pairwise RANKS, this imitates its VALUES through BCE. Name that in the ledger line.
#
# WHAT IS BLENDED. Only the 3453 studies he trains on. The 896 fold-0 studies keep his raw values, so
# the OOF truth his writer records (:545) never carries a teacher value, and the teacher's fold-0 rows
# (from `oof_fold0_s2026`, clean) are never read at all. The blend prints the same recipe line A′ did
# (`logs/repl2x2_a.full.log:4`): `[cv] fold 0/1: train 3453 | OOF held-out 896 | fold_file fold0_909.json`.
# If that line reads anything else, the held-out set moved — stop.
#
# WHY w = 0.3. Per-column marginal of the blended target against his raw y on the 3453 train rows
# (researcher letter 2026-09-07 16:07Z, table): at 0.3 the positive rate at his 0.5 moves by at most +0.04 absolute
# (MedMen .385→.427, Contusion .155→.182; nine columns unchanged to 3 dp), so his table is still his
# table. At 0.7 Effusion's positive rate goes .251→.396 and PF OA .332→.396 — a relabel, not a
# distillation (mastermind pin 1). The corr penalty barely separates the two weights (corr model
# 0.836 vs 0.843 → 0.0004 in the fit), so the marginal decides.
#
# SECOND, SMALLER MOVING VARIABLE INSIDE THE ONE FILE: his `pw = clip((1-prev)/prev, 1, 10)` (:406-407)
# is recomputed from the blended means, so pos_weight drops from A′'s
#   [4.17 7.51 1.36 3.65 2.63 4.07 1.81 2.07 1.23 4.88 3.10 7.70] to
#   [3.41 5.98 1.27 3.25 2.22 3.26 1.59 1.68 1.36 3.59 2.75 5.90].
#
# CALIBRATED LEVELS DROPPED. `LevelCalibration` fits one rate per unique value on GOLD cells
# (`kneemri/evaluation/level_calibration.py:72`; README:8622-8627); his table holds 14 distinct values
# and none of the 58 gold studies, so the fit has zero cells. There is no calibration step to port.
#
# PRICE. A′: 16 epochs in 6826 s (`logs/repl2x2_a.full.log:68`) = 1.90 h, plus best/top-k/SWA OOF
# passes (effv2l's OOF f0 landed 15:45Z and its SWA OOF 15:49Z) → ~2.0-2.1 h of card.
#
# PRE-REGISTRATION (researcher, D36 job 2). Ruler L75mid on the 896. A′ = 0.8787 solo, corr 0.830,
# honest 1/3 seat +0.0027 observed. Fit (D35): honest seat = −0.0046 + 0.313·(solo − 0.8865)
# + 0.0557·(1 − corr), corr modelled 0.836.
#   solo ≥ 0.895  → ship-candidate: honest 1/3 ≈ +0.007; full-fit twin, then a slot.
#   0.885..0.895  → directional (+0.004..+0.007); compose with lever 1, no slot on its own.
#   ≤ 0.885       → our knobs do not transfer into his recipe; the knob is ONE (the blended table), so the
#                   thing that failed is BCE-toward-values at w=0.3 on his 14-level table — say so, and
#                   the per-column table says which columns the teacher pulled the wrong way.
#
# ARM_CHECK NOTE. Nothing here calls `run_train`, so under `stub_lib.sh` both `uv run` steps are no-ops
# and `arm_check` SKIPS this arm (prints nothing, does not fail). The flags below are A′'s verbatim
# (`experiments/after_cellc.sh:39-46`) with two changes: `--tag` and `--labels`.
#
# deliberate: --labels labels_llm_soft_distill0.3 (the single moving variable vs A′; his script unchanged)
# deliberate: weight 0.3 (chosen on the marginal, see above; 0.7 is a relabel)
# deliberate: --tag hisrecipe_distill (his OOF writer names the artefact from it)
# reading: raptor_oof_hisrecipe_distill_fold0.npz
# bead: knee-qtjb4
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

SCRIPT="$REPO/external/raptor/knee-mri-training-the-twelve-finding-model.py"
FOLDS="$REPO/external/raptor/fold0_909.json"
HIS_LABELS="D:/data/external/raptor/labels/labels_llm_soft.csv"
TEACHER="$PROCESSED/teacher_clean_full.npz"
BLEND="D:/data/external/raptor/labels/labels_llm_soft_distill0.3"
require "$SCRIPT"
require "$FOLDS"
require "$HIS_LABELS"
require "$TEACHER"

uv run --no-sync --project "$REPO" python -m kneemri.experiments.his_labels_blend \
  --his "$HIS_LABELS" --teacher "$TEACHER" --fold-file "$FOLDS" --fold 0 \
  --weight 0.3 --out "$BLEND.csv"

hold_card
export PYTHONPATH="$REPO"
# STATUS IS PROPAGATED BY HAND, and it has to be. `lib.sh` runs under `set -u` and NOT `set -e`, so a
# dead trainer falls through to `release_card` and the script exits with THAT command's status: this arm
# died twice on 2026-09-07 (`ModuleNotFoundError: No module named 'sklearn'`, one second in) and the
# chain recorded `returned 0` both times, with `queue_check` reading OK over an idle card. The trainer's
# own status is captured before anything else can overwrite it, the card is released either way, and the
# arm then exits with it.
set +e
uv run --no-sync --project "$REPO" python -m kneemri.experiments.foreign_run \
  --script "$SCRIPT" \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
  --res 384 --epochs 16 --bs 8 --k 12 --k_eval 24 --grad_ckpt --ckpt timm \
  --tag hisrecipe_distill \
  --labels "$BLEND.parquet" \
  --folds 1 --fold 0 --fold_file "$FOLDS" \
  --workers 4
trainer_status=$?
release_card

# A ZERO STATUS IS NOT EVIDENCE THE TRAINER TRAINED. His script answers a labels file it cannot find
# by printing `_NO_LABELS` and calling `SystemExit(0)` (T:379-384), and `foreign_run.py:71` does not
# translate a death into a non-zero status either — so this arm died twice tonight reading `returned 0`
# with `queue_check` OK over an idle card. The OOF artefact is the one signal only a run that actually
# ran can produce, so it is what the status is taken from.
OOF="$MODELS/raptor_ft_hisrecipe_distill.holdout.npz"
if [ ! -f "$OOF" ]; then
  echo "hisrecipe_distill wrote no $OOF - the trainer exited $trainer_status without training" >&2
  exit 1
fi
exit "$trainer_status"
