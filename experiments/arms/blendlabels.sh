#!/usr/bin/env bash
# blendlabels.sh — the control's recipe trained on a finer-grained label table, and nothing else moved.
#
# bead: knee-w7xa
#
# WHAT MOVES. Two flags. `--labels` points at `blend_three_v1.csv` — our own table averaged with two
# public LLM readers by `python -m core.data.labels.table_blend` — and `--eval-labels` pins the holdout
# ruler to `report_labels_v2.csv`, the table every other fold-0 row was scored against.
#
# THE SECOND FLAG IS THE ARM. Our fold-0 holdout is scored against v2 itself. An arm that trains on a
# different table and is scored unpinned would measure DISAGREEMENT WITH V2 and call it quality: a
# member fitted on the blend necessarily predicts the blend, so it necessarily scores worse against v2
# the more the blend differs from it. Pinning the ruler is what makes the two arms comparable at all.
# Without it this arm answers a question nobody asked.
#
# THE MECHANISM, and it is NOT the one this investigation started with. `report_labels_v2` was first
# read as the WORST mainstream table — 0.8658 against the best public table's 0.8927 on the 58
# radiologist-annotated studies. That reading was wrong. Quantise the public table onto our own five
# levels and it falls to 0.8445, BELOW ours: at equal resolution our reader wins by +0.021. The gap is
# TIE MASS. `Supervision.MEASURED_AGREEMENT` emits five levels and ties 38.5% of cell pairs; an 85-level
# LLM probability ties 17.3%, and AUC breaks no ties.
#
# So the claim this arm tests is not "their reader is better". It is that a FIVE-LEVEL BCE target
# carries less ordering information into the gradient than an 85-level one — a training claim the gold
# ruler cannot settle, because on the gold set the tie-breaking is worth +0.0216 to the LABELS whether
# or not it is worth anything to a MODEL fitted on them.
#
# control  native224 = 0.8502 — same twelve slices, same 0.20-0.80 band, same whole-stack, same 4e-5,
#          same twelve epochs, same 224px read. The ONLY difference is which table the loss reads.
# cost     ~20 min at 224px/12sl, the control's own cost. No cache is built: the blend is a CSV.
#
# WHY 224 AND NOT THE 336 CONTROL. native224 read 0.8502 against lr4e-5_ws's 0.8525 — inside the 0.0062
# floor, so the pixel axis is flat at this geometry and the cheaper control is as good a control. The
# claim under test is about the LABEL scale's resolution in the gradient, which is independent of the
# pixel grid, and running it at 224 costs 20 minutes instead of an hour. Read this arm against 0.8502.
#
# THE FALSIFIABLE PART, written before the run.
#   * >= +0.015 over the control. Target resolution was costing every member on disk, and the whole
#     roster should be refitted on a finer table before anything else is spent.
#   * +0.006 to +0.015. Directional under the house bar. Keep it as a member — a model fitted on a
#     different supervision signal is a diversity axis, which is what the seat argument prices.
#   * within +/- 0.006. Five levels are enough for the gradient, the tie mass is a property of the
#     RULER and not of the training signal, and this axis closes. Note that this is a real and useful
#     answer: it would retire every future "get better labels" proposal that argues from resolution.
#   * <= -0.015. The public readers are injecting noise our own reader does not have, which argues for
#     the TIEBREAK mode (base ordering dominant) over MEAN, and that is the follow-up arm.
#
# AND A PER-LABEL PREDICTION, added 2026-08-25 18:58Z from the supervision_gap split, before the run.
# The diagnostic divides the twelve findings in two: six where the model already BEATS the report labels
# (Effusion +0.1304, Contusion +0.0891, Synovitis +0.0508, Medial OA +0.0419, Baker's +0.0226, Fracture
# +0.0160) and six where it trails, among them all four internal-derangement structures. A finer label
# table is a supervision improvement, so its gain can only land on the first group — where the reader is
# the weak part. On ACL, MCL and the two menisci the reader already reads 0.84-0.99 and the model is what
# trails, so a better table has nothing to give there.
#
# So `run_comparison --control native224_s2026.pt --arms blendlabels_s2026.pt` is the reading, not the
# mean alone. A mean that moves with the gain spread evenly, or concentrated on the four the mechanism
# says it CANNOT touch, is a mean that moved for some other reason and the arm is unread. A mean that
# moves with the gain on the six reader-bottlenecked findings is the mechanism firing.
#
# AND THE HARNESS CHECK CAUGHT ITS THIRD FLAG, 2026-08-25 18:59Z, on this arm's first launch. The header
# above said `deliberate: --image-size 224` and the run_train line passed no --image-size at all, so the
# recipe inherited $RECIPE's 336 and the log printed "image_size": "336". Every band below was written
# against native224's 0.8502; at 336 the arm is an hour instead of twenty minutes and its matched control
# is a different row. Nothing about the arm was wrong except that its own command did not say what its
# own comment said — which is why the check reads the RECIPE LINE and not the header. Killed at 60s and
# relaunched with the flag. `base224` and `native224` both pass it explicitly; this arm alone did not.
#
# HARNESS CHECK BEFORE READING ANY OF THAT. The printed recipe must name blend_three_v1.csv on "labels"
# AND report_labels_v2.csv on "eval_labels" AND 224 on "image_size" — a null here is the failure mode that makes the whole run
# unreadable, and it is the flag most likely to be silently dropped. The log must load 4,406 studies and
# hold out 909, the same holdout every fold-0 row was scored on.
#
# deliberate: --image-size 224 (the control native224 is itself the 224 read of lr4e-5_ws, and the
# deliberate: pixel axis measured flat at -0.0023; this arm is 20 min there instead of an hour)
# deliberate: --eval-labels report_labels_v2.csv (the arm's whole point — the ruler must stay the
# deliberate: table every other fold-0 row was scored against, or the holdout measures disagreement)
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

BLEND="$PROCESSED/labels/blend_three_v1.csv"
require "$BLEND"
require "$LABELS"

run_train blendlabels \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 --image-size 224 \
  --labels "$BLEND" --eval-labels "$LABELS" --out "$MODELS/blendlabels_s2026.pt"
