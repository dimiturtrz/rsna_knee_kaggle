#!/usr/bin/env bash
# disagreed_ramp.sh — the empty cell of the difficulty 2x2, and the only one whose neighbours disagree.
#
# Difficulty has two signals here and two ways to spend one, and three of the four cells are measured
# on this corpus at twelve epochs on one recipe:
#
#                     static tilt      schedule
#   label certainty        -            0.8584   (certainty_ramp, 6 of 12 epochs)
#   member disagreement  0.8471           ?      (disagreed_full at alpha 1.0)
#
# with `disagreed_uniform` at 0.8507 as the matched alpha-zero control for the whole disagreement row.
#
# THE TWO NEIGHBOURS POINT OPPOSITE WAYS. Tilting the draw toward disagreement from the first step
# COSTS 0.0036 against its own control. Ramping label trust over the first half of the run PAYS 0.0077,
# above the 0.0062 holdout floor. Both are supervision-axis changes on the same recipe, so the
# difference between them is not the signal — it is WHEN the intervention is applied. This arm applies
# the schedule to the signal that lost.
#
# WHY THE MECHANISM IS PLAUSIBLE RATHER THAN A COIN FLIP. `sampling.py` records why the weight is
# disagreement and not loss: ~87% of the residual's mass is aleatoric, so a loss-ranked draw mines label
# noise. But `residual_split` also puts ~70% of the pool's errors in the band that most-but-not-all
# members get wrong, and a model that has not yet learned the easy structure cannot use a hard study for
# anything except memorising it. A tilt from step one spends its whole first pass there. That is exactly
# the failure a curriculum is FOR, and it is the reading that makes the static arm's loss informative
# rather than a dead end.
#
# PRECEDENT. Zhou et al. ISBI 2023 (PMC10602195), elbow-fracture radiographs, 1,865 patients: a
# VoG-ordered easy-to-hard curriculum moves binary AUC 0.8407 -> 0.8652, +0.0245, with no human
# difficulty annotation. It is the only intervention on the supervision axis with a published AUC delta
# on a medical classification task, which is why it outranks slot dropout — whose one AUC-measuring
# precedent (Grovik et al. 2021) moved AUC by exactly 0.000.
#
# THE RAMP LENGTH IS TAKEN, NOT PICKED. `certainty_ramp` spends 6 of 12 epochs reaching full trust and
# is the only schedule that has paid here; this spends 6 of 12 reaching full tilt. Alpha 1.0 is
# `disagreed_full`'s, so the pair differs in the schedule and in nothing else — a matched control that
# is already on disk.
#
# control  `disagreed_full` at 0.8471 (same alpha, no schedule) and `disagreed_uniform` at 0.8507 (same
#          draw, no tilt). Both on disk. No control arm runs.
# cost     ~1.5 h at 12 epochs; a weighted draw of the same size costs what the uniform one did.
#
# PRE-REGISTERED.
#   THE SCHEDULE CARRIES THE SIGNAL — clears 0.8569 (uniform control + floor). Then the static arm's
#     loss was a timing failure and not a verdict on disagreement, the difficulty row re-opens, and the
#     next question is the ramp length rather than the signal.
#   THE SCHEDULE RESCUES BUT DOES NOT PAY — lands between 0.8471 and 0.8569. Then the schedule is worth
#     what it recovers and nothing more; the reading moves to error correlation against the champion
#     members, since a member trained on a different study ORDER is a supervision-space seat candidate
#     whether or not it is a better solo.
#   THE SIGNAL IS WRONG — at or below 0.8471. Then disagreement-weighted drawing is dead on this corpus
#     at any schedule, both cells of that row are filled with a negative, and the difficulty axis closes
#     with certainty as its only usable signal.
#
# bead: knee-zer5
#
# deliberate: --sample-ramp
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
SPREAD="$PROCESSED/train_spread_honest12.npz"
require "$SPREAD"

run_train disagreed_ramp \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --sample-spread "$SPREAD" --sample-alpha 1.0 --sample-ramp 6 \
  --labels "$LABELS" --out "$PROCESSED/disagreed_ramp_s2026.pt"
