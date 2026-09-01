#!/usr/bin/env bash
# bead: knee-8009
# small_feat_fit — stage two: `band_dense`, unchanged, started from the feature-distilled encoder.
#
# THE PAIR THIS READS AGAINST is `band_dense_s2026` = 0.8578, and the comparison is only worth making
# because this line is band_dense's line with ONE argument added. Every other two-stage arm in this
# project failed that discipline in some way: the logit pair's control was cold BY DESIGN and so could
# never separate the distilled trunk from the warm start, and the six warm-start arms varied rate or
# schedule alongside the initialisation. Here the rate, the schedule, the epochs, the slices, the band,
# the pooling, the variant and the image size are identical; `--encoder-weights` is the variable.
#
# PRE-REGISTRATION, written before the number, against 0.8578 and the 0.0062 paired floor:
#
#   - 0.8640 or above  — feature-level distillation pretrains a trunk that logit-level distillation could
#     not (-0.0077 on swinbase, measured), and it does so on the trunk the champion is built from. This
#     is the first pretraining signal the project owns. It goes to the roster, and the next question is
#     whether a SECOND stage-one epoch budget buys more, since stage one is undertrained by construction.
#   - 0.8516 to 0.8640 — flat. Feature distillation is not a pretraining mechanism at this scale either,
#     and knee-8009(b) closes with BOTH levels measured rather than one measured and one assumed. That is
#     a real closure, not a null: the bead has been open on the strength of "untested" since it was filed.
#   - below 0.8516     — it actively hurts, which is the same sign the logit pair read. Two levels, same
#     sign, four trunks of warm-start evidence: distillation-as-initialisation is dead here, and the card
#     goes to the corpus axis with the question settled rather than parked.
#
# WHAT NO BRANCH LICENSES. Nothing here speaks to distillation as a FINE-TUNE term, which is separately
# measured and separately weak (`distill_pairs` +0.0016, `raptor_distill` +0.0035, `merged_distill_cf`
# +0.0000, `stack_increments` +0.0008) — that is a different attachment point and it stays where it is.
# Nothing here speaks to a teacher other than `raptor_v4_swa`, whose features these are and which is
# 98.6% in-sample on the holdout, capping the ceiling.
#
# PRICE: ~64 min (12 epochs at ~320 s), 0 slots. The pair costs ~1.4h against the swinbase pair's 3.3h.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"
require "$PROCESSED/small_feat_pre_s2026.pt"

uv run --no-sync --project "$REPO" python -m kneemri.training.fitted_encoder \
  "$PROCESSED/small_feat_pre_s2026.pt" "$PROCESSED/encoders/small_feat_pre.enc.pt" || exit 1

run_train small_feat_fit \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --encoder-weights "$PROCESSED/encoders/small_feat_pre.enc.pt" \
  --labels "$LABELS" --out "$PROCESSED/small_feat_fit_s2026.pt"
