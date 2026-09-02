#!/usr/bin/env bash
# headlr_dense.sh — restore the head:backbone rate RATIO the reference argued for, now that half of it
# has been re-measured and the other half has not.
#
# THE CONSTANT THIS ARM ATTACKS IS ONE THE CODE ITSELF FLAGS AS UNMEASURED. `Adaptation` in trainer.py
# carries three numbers — head_lr 1e-3, backbone_lr 8e-6, unfreeze_last 6 — and its own docstring says
# of the first two: "That argument sets the 125:1 default ratio, and it was inherited from the reference
# rather than measured." It then says the three "stopped being three constants" because depth and rate
# are entangled. Two of the three have since been opened: the backbone rate moved to 4e-5 and is a large
# part of the +0.0306 this ledger has added since `rank24_s2026`, and unfreeze depth was bracketed by
# `deep24`/`shallow24`. The head rate is the one that was never touched.
#
# AND MOVING THE OTHER HALF SILENTLY CHANGED IT. The ratio the reference argued for is 125:1, and the
# argument behind it is about KIND rather than magnitude — the head is randomly initialised and has to
# travel, the encoder is pretrained and needs only to be nudged off its solution. That argument does not
# stop applying when the backbone rate goes up. But the backbone rate DID go up five-fold while head_lr
# stayed at 1e-3, so every arm since is running at 25:1: not a measured ratio, not the inherited one,
# just the arithmetic left over from changing one number of a pair that was only ever justified as a
# pair. --head-lr 5e-3 restores 125:1 at the settled backbone rate.
#
# WHY THIS IS WORTH A CARD SLOT AND NOT A KNOB. Rate is the lever family that has actually paid here —
# the backbone-rate change is the single largest recipe component this ledger owns — and this is the
# last member of the adaptation triple with no number on it. It is also the cheapest kind of finding to
# keep: a rate costs nothing per step, so a win is free forever and a loss removes a live question.
#
# THE SHAPE OF THE RISK IS NAMED, because it is the thing that would make a null uninformative. 5e-3 is
# hot for a head, but it is a PEAK rather than a constant: OneCycleLR takes both groups' max_lr and this
# run spends its first 15% warming up, so the head does not see 5e-3 until the encoder has settled. The
# distinguishing evidence is in the curve rather than the peak — a diverged head shows as a first-epoch
# AUC at or near 0.5 and a loss that does not come down, which is not the same result as a head that
# trained fine and simply did not help. The clause below reads them apart rather than pooling them.
#
# control  band_dense = 0.8578 (best epoch 8 of 12, backbone 4e-5, head 1e-3, 20 slices, band 0.10-0.90,
#          max pooling, whole stack, batch 12 — read out of the recorded recipe).
#          Single variable: --head-lr 5e-3. Same schedule, same operating point, same cache.
# seat     comp/champion.json greedy4_v1 = 0.8679 over the 909-study fold-0 holdout.
# cost     about 1.2 h. A rate change adds no per-step work, so this is the control's cost exactly.
#
# PRE-REGISTERED, three outcomes:
#   PREDICTION — the peak mean clears 0.8578 by more than the 0.0062 holdout floor. Then the ratio
#     argument survives its own re-measurement, the recipe gains a rate the way it gained the backbone
#     one, and every arm after this inherits it for free.
#   REFUTATION — the peak mean is at or below 0.8578 AND the first epoch trained normally (epoch-0 AUC
#     well above 0.5, loss falling). Then 125:1 is a reference artefact rather than a property of this
#     head, 25:1 was accidentally right, and the adaptation triple is closed with a measurement instead
#     of an inheritance. That is worth 1.2 h on its own: it retires the last unmeasured constant.
#   THIRD OUTCOME — the first epoch shows the divergence signature (AUC at or near 0.5, flat loss). Then
#     this says nothing about the ratio, only that 5e-3 is past the head's stability edge at batch 12,
#     and the honest follow-up is the midpoint (2e-4 to 1e-3 is 25:1; 5e-3 is 125:1; 1.6e-3 is 40:1)
#     rather than a verdict. Written down in advance so a diverged run cannot be read as a refutation.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train headlr_dense \
  $RECIPE --head-lr 5e-3 --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --labels "$LABELS" --out "$MODELS/headlr_dense_s2026.pt"
