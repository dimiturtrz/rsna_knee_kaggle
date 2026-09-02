#!/usr/bin/env bash
# clip224.sh — native224's recipe with `--grad-clip 1.0` and nothing else moved. The first arm this
# project has ever run with a bounded optimiser step.
#
# bead: knee-s9cr
#
# WHY IT DID NOT EXIST UNTIL NOW. `clip_grad` appeared nowhere in this tree: every one of the ~190 arms
# in the ledger took whatever step the batch it drew happened to ask for. That is not a decision anybody
# recorded — it is the reference recipe this project started from, and that recipe has since moved out
# from under it (the backbone rate went 8e-6 -> 4e-5 on the rate sweep). The flag is new code, added
# with this arm: `kneemri/training/gradient_clip.py`, applied between the last backward of a step and
# `scaler.step`, with the scaler unscaled first so the norm is a fact about the model rather than about
# the loss scale.
#
# THE MECHANISM, AND THE MEASUREMENT THAT ASKS FOR IT. `native224_long` ran the same recipe for twelve
# extra epochs and slid −0.0165 while its training loss kept falling. A model that is still fitting its
# training set and losing holdout AUC is either memorising or being thrown around; the second has a
# standard test, and this is it. Adam normalises by a running second moment, which bounds the size of a
# TYPICAL step but not of an outlier one: a batch whose gradient is many times the running scale moves
# the weights many times as far, and at 4e-5 on an encoder that is five times the distance it moved
# under the old rate. Clipping bounds the distance one batch may push; it changes nothing about the
# typical step, which is why the whole effect should live in the late epochs.
#
# WHY 1.0. The value the ViT recipes this trunk came from use (DINOv2's own training uses 3.0 on a much
# larger batch; the ImageNet ViT and MAE fine-tuning recipes use 1.0), and it is the value that leaves
# the median step untouched while cutting the tail — our gradients are not measured, so the honest
# choice is the literature's default rather than a number derived from nothing. `clip224_tight.sh` takes
# the second point at 0.5, which is where clipping starts to bind on the ordinary step too.
#
# THE RESOLUTION CONNECTION. This is a 224 arm rather than a dense one because the instability was
# MEASURED at 224 — `native224_long` is the run that slid — and because a 19-minute arm is the cheapest
# place in this project to ask a question about late training. If it clears, the same flag is worth a
# confirm at 336 where the family's members are fitted, and that is a separate slot.
#
# control  native224 = 0.8502, best at epoch 7, converged not truncated. 12 epochs, 12 slices, 0.20-0.80
#          band, whole-stack, backbone 4e-5, 224 px, same labels, same 909-study holdout, unbounded
#          steps. The single variable is --grad-clip.
# cost     ~19 min for twelve epochs at 94 s. Same cache as native224. One extra norm over the
#          parameters per optimiser step — sub-millisecond against a 94-second epoch.
# price    19 min for P(clears the 0.0062 floor on the MEAN) = 0.20. The mechanism is generic and
#          well-attested, but the failure it addresses is a LATE-epoch one and this control peaks at
#          epoch 7 of 12: if the slide only bites after the peak, clipping protects epochs whose
#          checkpoints are discarded anyway and the arm reads flat by construction. That asymmetry is
#          why the value here is partly instrumental — see the FLAT branch.
#
# HARNESS CHECK BEFORE READING ANY NUMBER. The printed recipe must say "grad_clip": "1.0" and
# "image_size": "224", and the log must load 4,406 studies and hold out 909. The landed check compares
# the trainer's `gradient_clip_norm` against the flag, so a value that never reached `Trainer` refuses
# before the card is spent — which matters more here than anywhere: an unlanded clip is a byte-identical
# rerun of the control, the one failure that reads exactly like a null.
#
# THE FALSIFIABLE PART, written before the run.
#   CONFIRMED   the mean peak clears 0.8564 (control + floor). Bounded steps are worth more than every
#               aggregation switch measured at dense, they cost nothing, and every arm in the queue
#               after this one should carry the flag. That would also mean ~190 ledger rows were
#               measured under an avoidable variance source, which reopens the long-run family.
#   DIRECTIONAL positive under the floor. Kept — free, mechanism-sound, standard practice — and folded
#               into the recipe; not a finding on its own.
#   FLAT, AND STILL INFORMATIVE. Clipping is the standard first answer to a late slide; a flat reading
#               at the peak plus a SMALLER late-epoch slide (read the epoch curve, not just the peak)
#               says the instability is real but priced into epochs the best-epoch rule already
#               discards, and it licenses the twelve-epoch budget as the right one. Flat with the slide
#               unchanged says the slide is overfitting rather than step size, and points at --mixup and
#               --weight-decay instead — which is the next arm in this family either way.
#   REFUTED     the mean falls by more than the floor. Clipping at 1.0 that binds on ordinary steps is
#               a learning-rate cut in disguise, applied unevenly across the run; that would say our
#               gradients live above 1.0 routinely, which is worth knowing and makes 0.5 pointless.
#
# deliberate: --grad-clip 1.0 (the arm's thesis — bound how far one batch may move the weights)
# deliberate: --image-size 224 (native224's control, per that row's standing instruction)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train clip224 \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 --grad-clip 1.0 \
  --labels "$LABELS" --out "$MODELS/clip224_s2026.pt"
