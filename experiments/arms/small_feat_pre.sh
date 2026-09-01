#!/usr/bin/env bash
# bead: knee-8009
# small_feat_pre — knee-8009(b) on the trunk that actually scores. Stage one of two.
#
# WHY THIS TRUNK AND NOT THE ONE (b) WAS FIRST ARMED ON. `raptor_feat_pre`/`_fit` asked whether matching
# a teacher's REPRESENTATION pretrains a trunk, and asked it on swinbase, whose cold control is 0.8331 —
# the second-weakest trunk in the monoculture table. That was the right place to ASK, because the
# swinbase pair inherits the logit arm's control and so reads as a clean A/B against it. It is the wrong
# place to WIN: a mechanism worth the 1.5% house bar has to move the trunk the champion is built out of,
# and every seated member here is DINOv2. A +0.006 on a 0.8331 trunk buys a member nobody seats.
#
# WHAT IS HELD. Everything in `band_dense`, character for character — the same 12 epochs, the same
# 4e-5 backbone rate, the same 20 slices, the same 0.1-0.9 band, the same whole-stack max pooling, the
# same default variant and image size. `band_dense_s2026` at 0.8578 is therefore a MATCHED cold control
# and not merely a nearby number. That matters more here than anywhere else in this chain: the logit
# two-stage's own interpretation says its control was cold by design, so it could never separate
# "distilled trunk" from "warm start at all", and a warm start on its own is measured at -0.0072,
# -0.0057, -0.0044 and +0.0024 across four trunks. If this pair moves, the ONLY thing that moved is what
# the encoder was pretrained on.
#
# WHAT VARIES, and it is one term: this stage adds `--feature-weight` against the teacher's pooled
# representation and trains for 4 epochs, and stage two starts from the encoder that produces. No label
# information reaches the student that is not already in `report_labels_v2.csv`.
#
# EXPECTED UNDERTRAINED AT 4 EPOCHS. Its own holdout number is a probe, not a target — the logit stage
# one read 0.8177 still rising, and the feature stage one read 0.8201 the same way. A caveat on the SIZE
# of the pretraining signal, never on its SIGN.
#
# THE TEACHER IS 98.6% IN-SAMPLE (`foreign_overlap` on `raptor_v4_swa`), which caps what this can reach,
# not which way it points. A teacher that has seen the holdout can only make the student's ceiling look
# better than it is; it cannot manufacture a mechanism that is not there.
#
# THE ONE THING TO CHECK ON THE LOG, and the reason this arm exists at all: `feature_weight: 20.0` with
# a non-null `teacher_features`. Until 2026-09-01 `context_width()` returned the head's INPUT width, so
# the projection was built as nn.Linear(2048, 1024) and fed a 256-wide tensor — every feature
# distillation arm died at its first step with `mat1 and mat2 shapes cannot be multiplied`, BEFORE epoch
# 0, which left no missing epoch line for anyone to notice. Verify the flag landed; do not assume it.
#
# PRICE: ~21 min (4 epochs at ~320 s), 0 slots.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"
require "$PROCESSED/raptor_features.npz"

run_train small_feat_pre \
  $RECIPE --epochs 4 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --feature-weight 20.0 --teacher-features "$PROCESSED/raptor_features.npz" \
  --labels "$LABELS" --out "$PROCESSED/small_feat_pre_s2026.pt"
