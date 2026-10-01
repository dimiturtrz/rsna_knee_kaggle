#!/usr/bin/env bash
# hisrecipe_distill_full — the SHIPPABLE twin of `hisrecipe_distill`: his recipe, his script, our
# distillation in the label table, and NOTHING held out. Armed only on that arm's GO.
#
# WHY A TWIN AND NOT THE ARM ITSELF. `hisrecipe_distill` holds his fold 0 out so the seat can be PRICED
# (the 896 are the ruler). A member that trains on 3453 of 4349 studies is a member built to be read,
# not to be shipped: every seat in the champion is fitted with `--folds 0` on all of them, and packaging
# the held-out fit would seat a member that saw 79% of what its siblings saw. So the reading arm and the
# shipping arm are two runs, and this is the second one. Same relationship as `calib_dense_hold` ->
# `calib_dense_hold_ff`.
#
# THE ONLY TWO DIFFERENCES FROM `hisrecipe_distill.sh`, both forced by "nothing held out":
#   1. `--folds 0` (his default, `...twelve-finding-model.py:350`) instead of `--folds 1 --fold 0
#      --fold_file`. At `folds > 0` his script carves `held` out of `train_ids` and writes
#      `raptor_oof_<tag>_fold<n>.npz` (:391-397, :533-547); at `folds 0` that whole block is skipped, no
#      OOF file is written and every study trains. **So this arm produces NO holdout record and can never
#      be scored offline** — which is why it is dark by construction and why it may only be armed AFTER
#      its held-out twin has already returned a GO. It is `# unmeasurable:` on purpose, not by omission.
#   2. The blend covers ALL rows, via `--fold -1` to `his_labels_blend`. `Folds.members` returns the
#      studies assigned to the fold (`his_labels_blend.py:36-40`) and `fold0_909.json` assigns only the
#      value 0 (verified: `sorted(set(fmap.values())) == [0]`), so `-1` selects the empty set and every
#      study is blended at w = 0.3. In the reading arm the 896 keep his RAW values so his OOF truth never
#      carries a teacher value; here there is no OOF truth to protect, and leaving 896 rows unblended
#      would ship a member trained on two different target definitions.
#      `teacher_clean_full.npz` covers all 909 fold-0 studies (verified: 909 of 909), so the blend has a
#      teacher value for every row it needs one for.
#
# WHAT IS IN-SAMPLE HERE, SAID PLAINLY. The teacher's values on the fold-0 rows are clean OOF
# (`oof_fold0_s2026`), but this member trains on those rows, so it can never be read on the 896 or the
# 909 again — no future ruler may score it, and `foreign_overlap` must mark it 100% in-sample. That is
# the standing contract for every `_ff` member the project ships (README:8822); it is written here
# because this is the first foreign-recipe member to take it.
#
# PRICE. `hisrecipe_distill` = 16 epochs over 3453 studies in ~1.9 h plus its OOF passes (~2.0-2.1 h of
# card). This trains 4349 studies (+26%) and skips every OOF pass: 16 x ~1.9 h x 4349/3453 ~= 2.4 h.
# 384 res, bs 8, grad-ckpt — the same 12 GB class, and it must not share the card.
#
# GAIN, AND THE GATE THAT IS NOT IN THIS FILE. This arm is priced by its twin, not by itself. Under D35
# the honest 1/3 seat = -0.0046 + 0.313*(solo - 0.8865) + 0.0557*(1 - corr): at the twin's ship-candidate
# band (solo >= 0.895, corr modelled 0.836) that is ~ +0.007 offline, ~ +0.004 board at 0.55x. The
# full-fit lift over its held-out twin is the standard +0.004..+0.006 solo the project measures for the
# extra 26% of data, and it is NOT re-measured here because it cannot be.
#
# ARM ONLY IF `hisrecipe_distill` READS A SWAP GAIN >= +0.0010 AGAINST THE RAPTOR 1/3 SEAT WITH THE CORR
# GUARD HOLDING (mastermind 17:05Z branch tree). A solo in the 0.885..0.895 directional band is NOT a GO:
# 2.4 h of card for a seat that composes with something else is the wrong ratio while the corr axis is
# still open.
#
# deliberate: --folds 0 (nothing held out; the shipping fit, and the reason this arm is unreadable)
# deliberate: --fold -1 to the blend (empty held set => every row blended at w=0.3, one target definition)
# deliberate: --tag hisrecipe_distill_full (his writer names the checkpoint from it; must not collide
#             with the twin's artefacts in the same directory)
# unmeasurable: nothing is held out, so no holdout record exists and no ledger row can price this arm;
#               it is priced by `hisrecipe_distill`'s reading on the 896 and shipped on that reading.
# bead: knee-az22y
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

SCRIPT="$REPO/external/raptor/knee-mri-training-the-twelve-finding-model.py"
RAPTOR_CORPUS="D:/data/volumetric/mri/processed/raptor/raptor-corpus"
FOLDS="$REPO/external/raptor/fold0_909.json"
HIS_LABELS="D:/data/volumetric/mri/processed/raptor/labels/labels_llm_soft.csv"
TEACHER="$PROCESSED/teacher_clean_full.npz"
BLEND="D:/data/volumetric/mri/processed/raptor/labels/labels_llm_soft_distill0.3_full"
require "$SCRIPT"
require "$FOLDS"
require "$HIS_LABELS"
require "$TEACHER"

uv run --no-sync --project "$REPO" python -m kneemri.experiments.his_labels_blend \
  --his "$HIS_LABELS" --teacher "$TEACHER" --fold-file "$FOLDS" --fold -1 \
  --weight 0.3 --out "$BLEND.csv"

hold_card
export PYTHONPATH="$REPO"
uv run --no-sync --project "$REPO" python -m kneemri.experiments.foreign_run \
  --script "$SCRIPT" \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
  --res 384 --epochs 16 --bs 8 --k 12 --k_eval 24 --grad_ckpt --ckpt timm \
  --tag hisrecipe_distill_full \
  --kaggle-input "$RAPTOR_CORPUS" \
  --labels "$BLEND.parquet" \
  --folds 0 \
  --workers 4
# The trainer's status, captured before `release_card` overwrites it — `lib.sh` runs under `set -u` and
# not `set -e`, so without this a dead trainer exits 0 (see the sibling arm, which died twice that way).
trainer_status=$?
release_card
exit "$trainer_status"
