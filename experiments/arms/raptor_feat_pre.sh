#!/usr/bin/env bash
# raptor_feat_pre.sh — stage one of knee-8009 (b): FEATURE distillation used as pretraining.
#
# THE QUESTION, AND WHY IT IS NOT THE ONE (a) ASKED. `swin_distill_pre`/`_fit` distilled the teacher's
# twelve LOGITS into the trunk and the warm fit read -0.0077 against a cold control. The pre-registered
# suspect for that number was the LEVEL of the target, and this arm is that suspect tested: the student is
# pulled onto what the teacher READ from the study — its pooled representation, `d` wide — rather than
# onto the twelve directions its own head splits that reading along. A logit is a dot product against one
# learned direction per finding; a representation is the thing those directions are read out of, and it is
# the only level at which an unlabelled corpus could ever be used.
#
# COSINE, THROUGH A LEARNED PROJECTION, AND NOTHING PAIRWISE. The ranking machinery (a) used exists
# because the metric reads ORDER over studies and discards calibration — an argument about LOGITS, which
# does not transfer to a representation: there is no ordering in a `d`-wide vector and no metric reading
# it. The term is `1 - cos(W x_student, x_teacher)` with `W` a bias-free linear map from the student's
# context width to the teacher's, because the two widths differ by trunk and must never be assumed equal,
# and because a bias could satisfy a cosine by translating the whole cloud onto the teacher's mean
# direction while carrying nothing per study.
#
# THE LABEL TERM IS A PROBE HERE, NOT A TARGET. `--feature-weight 20` is what makes this a PRETRAIN: the
# twelve heads still run, so the log has a diagnostic, but they are a small part of the gradient the way
# `--distill-weight 20` made them one in (a). What this stage produces is a TRUNK; stage two is the member.
#
# EXPECTED TO BE UNDERTRAINED AT FOUR EPOCHS, and that is a caveat on the SIZE of the final number and
# NOT on its sign. (a)'s stage one had exactly this property — 0.7578 -> 0.7923 -> 0.8123 -> 0.8177, still
# rising at the last epoch — and the interpretation says so in writing. A stage that has not converged
# hands stage two a trunk that is less shaped than the mechanism allows, so it can only understate the
# effect; it cannot manufacture one, and it cannot flip a positive to a negative.
#
# THE TEACHER IS THE FOREIGN RAPTOR, AND IT IS IN-SAMPLE. This header claimed until 2026-08-31 that it
# "was never fitted on this corpus, so it cannot be memorising the holdout". That is measurably wrong:
# `foreign_overlap` puts raptor at 98.6% in-sample here, which is what voided `raptor_solo`'s 0.8938. The
# checkpoints under `external/raptor/` carry no recorded recipe, so `raptor_features.honest()` cannot
# refuse them the way it refuses a `fold: -1` of ours — it passes by absence of evidence, not by evidence.
# What that does and does not cost is argued in full on `raptor_feat_fit.sh`, which is where the control
# and the branches live; in short it caps the CEILING and not the SIGN, because both arms of the read fit
# the same labels on the same folds against the same cold control. `raptor_features.npz` is written by
# `python -m kneemri.kaggle.raptor_features`, the feature sibling of `raptor_teacher`, reading the pooled
# vector rather than the head's twelve.
#
# cost   4 epochs at 384 px on an 87M trunk, priced off swin_supervised's 12 epochs at ~90 min = ~30 min.
#        Zero slots. Stage two is another ~3.3 h.
#
# NOT PRE-REGISTERED HERE, for (a)'s stage-one reason: a run whose gradient is 20 parts cosine to one part
# labels produces no member worth reading, and its holdout is a diagnostic (did the trunk learn ANYTHING)
# rather than a result. The branches live on `raptor_feat_fit`, which is where the control is. The one
# thing to check on this log before stage two runs is that the run recorded `feature_weight: 20.0` and a
# non-null `teacher_features` — a teacher file that joins on nothing trains a plain labelled run and hands
# stage two a control wearing an arm's name.
#
# bead: knee-8009
# deliberate: --feature-weight   (20, representation-dominant BY CONSTRUCTION — this is the varied term)
# deliberate: --teacher-features
# deliberate: --epochs           (4 — a shaping stage, not a fit; matched to (a)'s stage one)
# deliberate: --variant          (swinbase, matched to the control on disk)
# deliberate: --train-size       (384, pinned by the window tables — see swin_supervised.sh, do not sweep)
# deliberate: --backbone-lr
# deliberate: --batch
# deliberate: --accumulate
# deliberate: --vram-gb
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"
require "$PROCESSED/raptor_features.npz"

run_train raptor_feat_pre \
  $RECIPE --epochs 4 --batch 1 --accumulate 6 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant swinbase --backbone-lr 1e-4 --slices 20 --band-low 0.1 --band-high 0.9 --train-size 384 \
  --feature-weight 20.0 --teacher-features "$PROCESSED/raptor_features.npz" \
  --vram-gb 8 \
  --labels "$LABELS" --out "$MODELS/raptor_feat_pre_s2026.pt"
