#!/usr/bin/env bash
# slot_dropout.sh — the first arm on this project's SUPERVISION axis rather than its decode axis.
#
# Every regulariser this project has perturbs one of two things: `--mixup` perturbs the LABEL, and
# `--intensity-aug` perturbs the APPEARANCE. Neither changes which acquisitions a study is made of,
# and that is the axis this corpus actually varies on:
#
#   AX_FLUID_FS 100%   SAG_FLUID_FS 98%   COR_FLUID_FS 93%
#   SAG_FLUID_NOFS 55% COR_T1 60%         SAG_T1 57%
#
# Nearly half of all studies reach inference carrying slots that no member has ever been trained
# without. `--slot-dropout` withholds a PRESENT slot at random, moving the pixels and the presence
# mask together, and never empties a study.
#
# WHAT WAS ALREADY RIGHT, AND WHY THAT NARROWS THE CLAIM. The head does NOT zero-fill an absent slot:
# `SlotScheme.presence` reports a per-slot coverage figure and `--slot-pooling attend` gives each
# diagnosis its own query over the PRESENT slots. So the defect is not that absence is misrepresented
# at inference — it is that absence was never in the training distribution, which is a different and
# smaller claim than the one this arm was first sketched under.
#
# THE RATE IS DERIVED, NOT PICKED. Mean fill across the six slots is 0.772, so a study drawn from this
# corpus is missing 0.23 of its slots on average. 0.25 matches the training-time withholding rate to
# the absence rate the corpus already exhibits, which makes the training distribution a superset of
# the serving one rather than a differently-shaped guess. Chrysochoou et al. 2026 used 0.4 on a
# four-sequence brain protocol; ours is a six-slot protocol with one slot always present, so matching
# our own measured rate is the better-argued constant.
#
# PRECEDENT, AT ITS REAL STRENGTH — AND IT DOES NOT PREDICT A SOLO GAIN. Grovik et al. 2021 (npj
# Digital Medicine) report input-level dropout holding **AUC exactly flat, 0.989 against 0.989**,
# while Dice rose 0.021 and false positives per patient fell 7.0 -> 3.6. Chrysochoou et al. 2026 (npj
# Precision Oncology) report dropout beating zero-fill and generative synthesis on every
# missing-sequence cell, per-patient Dice never dropping more than 0.03-0.04. Both are SEGMENTATION,
# and the ONE paper that measures our metric measured no gain on it. This arm is therefore not armed
# as a solo-score lever and no seat is pre-registered.
#
# WHY RUN IT ANYWAY. Seven board flats closed seating; combiner, pooling, member-count, serving and
# decode are closed from six independent directions; `radnet_family` is closing the trunk lane at the
# convolutional end. `competence_axis` reads 81.7-82% of what separates 105 members as ONE scalar,
# which is what a pool of members differing only in decode looks like. A member trained to survive a
# withheld acquisition cannot rely on the same evidence as one that never saw one, so this is the
# cheapest change on the list that could move a member OFF that axis. The reading that matters is the
# error correlation against the champion members, which costs no slot.
#
# control  `band_dense` at 0.8578, this arm's recipe with `--slot-dropout` absent. Matched and already
#          on disk, so no control arm is run. Floor 0.0062, board resolution 0.0015.
# cost     ~1.5 h at 12 epochs; dropout is a mask multiply and adds no memory.
#
# PRE-REGISTERED.
#   IT PAYS ON ITS OWN — solo clears 0.8640 (control + floor). Then absence-in-training is a real
#     regulariser on this corpus and the rate becomes a one-dimensional sweep, 0.15 and 0.35 next.
#   FLAT AND DECORRELATED — solo within 0.0062 of the control. This is the EXPECTED branch and it is
#     the one the precedent predicts. The reading moves to card-free error correlation against every
#     champion member: if this member's holdout errors are less correlated with them than any
#     incumbent pair is, a supervision-space change bought diversity that a decode-space change could
#     not, which is the first new seat mechanism since the seating axis closed. If it is flat AND
#     correlated, the axis is filled with a negative and closes with a number.
#   IT COSTS — solo below 0.8516 (control - floor). Then withholding evidence at 0.25 removes more
#     than it regularises at this corpus size, and the honest follow-up is 0.10 rather than a
#     defence of 0.25 — with the arm re-read for whether the loss falls on the studies whose slots
#     were dropped or on the ones that were always complete.
#
# bead: knee-dcko
#
# deliberate: --slot-dropout
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train slot_dropout \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --slot-dropout 0.25 \
  --labels "$LABELS" --out "$MODELS/slot_dropout_s2026.pt"
