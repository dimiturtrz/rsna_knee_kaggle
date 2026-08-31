#!/usr/bin/env bash
# swin_distill_pre.sh — stage one of knee-8009's cheap instrument (a): distillation used as PRETRAINING.
#
# THE QUESTION, AND WHY IT IS NOT THE QUESTION THE TWO PRIOR DISTILL ARMS ASKED. `distill_pairs` (+0.0016)
# and `raptor_distill` (+0.0035) both mixed a teacher term into a NORMAL labelled fit at weight 0.3, read
# the member that came out, and priced its seat. Both landed inside the 0.0062 floor. The owner's stated
# order is the other one: distil FIRST, into the trunk, then fit the labels warm from that trunk. That
# changes what the teacher is for. At weight 0.3 the teacher nudges an ordering the labels are already
# setting; at weight 20 it is most of the gradient, and the twelve label logits are a probe the trunk is
# shaped through rather than the objective.
#
# WHAT IT IS BOUGHT FOR, said plainly. This is an INSTRUMENT and it is priced as one. knee-8009's real
# build (b) is FEATURE distillation — matching the teacher's REPRESENTATION, label-free, which is the
# route external unlabelled data could ever be used on. (b) is a build; this is two runs of code that
# already exists. If a teacher-dominant first stage moves the warm fit at all, the representation channel
# is live and (b) is licensed. If it reads flat, (b) is NOT refuted: ordering-over-twelve-targets was
# always the wrong signal for a pretraining stage, and saying so afterwards would be a rationalisation,
# so it is said here, before the run.
#
# THE STUDENT IS SWIN, AND THAT IS THE WHOLE REASON THIS ARM IS AFFORDABLE. `swin_supervised` is the
# matched control and it is ALREADY ON DISK: 12 epochs, 384 px, swinbase, `--backbone-lr 1e-4`, every
# other flag identical to stage two. So this instrument costs one extra run rather than two, and the
# comparison is single-variable in the strict sense `mismatched` can check — the only difference between
# `swin_distill_fit` and `swin_supervised` is which weights the trunk started from.
#
# AND SWIN IS THE MEMBER THE ANSWER MATTERS MOST FOR. It reads 0.8331 solo and was refused a seat at
# -0.0012, yet its nearest-neighbour rank correlation is 0.8346 against every seated member's 0.8772 to
# 0.9225 — the most decorrelated member this project owns, refused purely on solo strength. A member at
# ~0.83 correlation needs roughly +0.017 to +0.020 solo to buy a seat, which is far outside what any knob
# has delivered here; the only thing that could is a better TRUNK.
#
# THE TEACHER IS THE ONE THAT ALREADY TRANSFERRED. `raptor_teacher.npz` is a foreign 0.923-solo model's
# ordering of the training studies, never fitted on this corpus, so it cannot be memorising the holdout —
# the assumption `DistilledObjective` says it cannot check for itself. It is also the stronger of the two
# teachers tried (+0.0035 against +0.0016 for our own pool), which is the mechanism's own prediction
# holding: an ordering from outside the family carries more than one from inside it.
#
# cost   4 epochs at 384 px on an 87M trunk, priced off swin_supervised's 12 epochs at ~90 min = ~30 min.
#        Zero slots. Stage two is another ~90 min.
#
# NOT PRE-REGISTERED HERE. This stage produces no member worth reading: a run whose gradient is 20 parts
# teacher ordering to one part labels is not competing with anything on the board, and its holdout score
# is a diagnostic (did the trunk learn ANYTHING) rather than a result. The branches live on
# `swin_distill_fit`, which is where the control is. The one thing to check on this log before stage two
# runs is that the run recorded `distill_weight: 20.0` and a non-null `teacher_targets` — a teacher file
# that joins on nothing trains a plain labelled run and hands stage two a control wearing an arm's name.
#
# bead: knee-8009
# bead: knee-7ybf
# deliberate: --distill-weight  (20, teacher-dominant BY CONSTRUCTION — this is the varied term)
# deliberate: --teacher-targets
# deliberate: --epochs          (4 — a shaping stage, not a fit)
# deliberate: --variant         (swinbase, matched to the control on disk)
# deliberate: --train-size      (384, pinned by the window tables — see swin_supervised.sh, do not sweep)
# deliberate: --backbone-lr
# deliberate: --batch
# deliberate: --accumulate
# deliberate: --vram-gb
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"
require "$PROCESSED/raptor_teacher.npz"

run_train swin_distill_pre \
  $RECIPE --epochs 4 --batch 1 --accumulate 6 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant swinbase --backbone-lr 1e-4 --slices 20 --band-low 0.1 --band-high 0.9 --train-size 384 \
  --distill-weight 20.0 --teacher-targets "$PROCESSED/raptor_teacher.npz" \
  --vram-gb 8 \
  --labels "$LABELS" --out "$PROCESSED/swin_distill_pre_s2026.pt"
