#!/usr/bin/env bash
# v3six.sh — the family swap applied to ALL SIX members at once, not to a seventh seat.
#
# WHY THIS ARM EXISTS, AND WHY IT IS THE ONLY ONE LEFT. Seven board entries now read 0.936-0.937 with
# count (dark6/dark7), member identity, fold status (allfit/lastseat), regularisation (augseat), encoder
# family at ONE seat (v3seat, DINOv3-base) and architecture family OUTSIDE the slice-then-pool trunk
# (farseat, V-JEPA2 tubelet transformer, nearest-neighbour agreement 0.8210 against a seated band of
# 0.8801-0.9143) each moved alone. Adding the most distant available member is worth what adding the
# nearest one is worth. That is not a diversity deficit; it says the SIX already determine the answer and
# a seventh vote of any provenance is averaged into a decision that does not depend on it.
#
# The untested object is therefore the six themselves. This arm rebuilds the whole roster in a different
# encoder family rather than seating one member of it.
#
# WHAT IS COPIED AND WHAT MOVES. Each recipe is `final_fit_six.sh`'s verbatim, carrying the same
# `$FIT` = `--fold -1 --ema` and the same per-member axis (calibrated targets, grade ladder, through
# plane, plain, pooling, focal tail). Exactly two things move: `--variant v3base` on every member, and
# the slice count.
#
# THE SLICE COUNT MOVES BECAUSE THE CARD IS SHARED, and it is named rather than buried. `v3base_dense`
# peaked at 25.38 GB allocated at 20 slices; biohub holds 10.2 GB of this 32.6 GB card by arrangement, so
# 20 slices is denied while it runs. The derived cap prices this shape at 22.7 GB at 20 slices, hence
# ~13.6 at 12, which clears the 22.4 GB available. `band_span` measured the span main effect at BOTH
# densities (+0.0039 at 12 slices, +0.0038 at 20) and the density main effect at both spans (+0.0014,
# +0.0013), so what 12 slices forfeits is the density term: +0.0013, a fifth of the 0.0062 floor, against
# a family term this arm is built to measure. Attribution is family + density, and density is bounded.
#
# ONE AXIS IS NOT A COPY. `base_dense_ff` was distinguished by `--variant base` (DINOv2-base), and under
# a family swap that axis collapses into the swap itself and would duplicate `v3_ws_ff`'s shape. It is
# replaced by `--slice-pooling attend` — the slice combiner every other member holds at `max`, so it keeps six DISTINCT
# recipes without diluting the family. Six of one family is the whole point; five and a stranger is the
# thing seven entries already refuted.
#
# NONE OF THESE SIX CAN EVER BE SCORED OFFLINE. Each sees every holdout study under `--fold -1`, so no
# output may be seat-scored, curve-scored or distance-scored, and none may enter a pool reporting an
# offline mean — the same rule `final_fit_six.sh` states. This roster ships blind against dark6.
#
# COST, PRICED BEFORE RUNNING. `final_fit` measured ~425 s/epoch for DINOv2-small at 20 slices over 4,406
# studies; 12 slices is ~0.6 of the tokens and v3base is 86M against 22M parameters, so ~510 s/epoch is
# the estimate and six twelve-epoch fits is ~10 h of a 20 h window, before biohub contention. The gain
# this could plausibly deliver is the only one left above the floor, at maybe one-in-five — an expected
# value of roughly 0.002, which is poor arithmetic in isolation and is taken anyway for two reasons the
# ratio does not see: the alternative use of this card is ZERO, and a FLAT reading closes the ensemble
# axis permanently and redirects tomorrow's five slots to data and labels instead of roster tweaks.
#
# THE FALSIFIABLE PART, written before the run and read on the BOARD, never offline:
#   * >=0.945 against dark6's 0.937 — the family of the SIX is the plateau, and the next roster is built
#     in whichever family wins rather than by seating dissenters into this one.
#   * 0.940-0.944 — the family carries but does not close the gap to 0.950; a mixed six (three of each)
#     is then the arm that prices whether it is family or simply CHANGE.
#   * 0.933-0.939 FLAT — the plateau does not live in the encoder family at all. The ensemble axis is
#     finished in the strong form: neither the marginal seat NOR the whole roster's family moves it, and
#     every remaining path runs through data or labels. This is the expected reading.
#   * <=0.932 — DINOv3-base does not survive all-data fitting at this depth, and the harness is the first
#     suspect: check that `--variant v3base` reached the checkpoint config on every member before
#     concluding anything about the family.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

HF_TOKEN="$(grep -oE 'hf_[A-Za-z0-9_-]+' /d/life/creds/machine/hfpat.txt | head -1)"
export HF_TOKEN
[ -n "$HF_TOKEN" ] || { echo "MISSING: no Hugging Face token, and DINOv3 is a gated repository"; exit 1; }

FIT="--fold -1 --ema"
SPAN="--slices 12 --band-low 0.1 --band-high 0.9"
V3="--variant v3base --batch 6 --backbone-lr 4e-5"

run_train v3_calib_ff \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  $V3 $SPAN --calibrated-targets \
  --labels "$LABELS" --out "$MODELS/v3_calib_ff.pt"

run_train v3_grade_ff \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  $V3 $SPAN --grade-ladder \
  --labels "$LABELS" --out "$MODELS/v3_grade_ff.pt"

run_train v3_through_ff \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  $V3 $SPAN --through-plane \
  --labels "$LABELS" --out "$MODELS/v3_through_ff.pt"

# The 0.20-0.80 band is `lr4e-5_ws_ff`'s own, kept rather than harmonised: it is the one member of the
# six that reads a different pixel group, and harmonising it would remove a diversity axis while
# claiming to hold the recipe fixed.
run_train v3_ws_ff \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  $V3 --slices 12 --band-low 0.2 --band-high 0.8 \
  --labels "$LABELS" --out "$MODELS/v3_ws_ff.pt"

run_train v3_attend_ff \
  $RECIPE $FIT --epochs 12 --slice-pooling attend --whole-stack $THROUGHPUT \
  $V3 $SPAN \
  --labels "$LABELS" --out "$MODELS/v3_attend_ff.pt"

run_train v3_focal_ff \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  $V3 $SPAN --pool cls_mean_focal \
  --labels "$LABELS" --out "$MODELS/v3_focal_ff.pt"
