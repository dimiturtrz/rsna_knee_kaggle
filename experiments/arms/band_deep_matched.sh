#!/usr/bin/env bash
# bead: knee-zt3l6
# reading: logs/band_deep_matched_ledger.log
# also: knee-zsyp4
# deliberate: --unfreeze-last 12 --layer-decay 0.7 --backbone-lr 9.8e-5 --batch 2 --accumulate 6 — the ladder IS the arm, and the anchor + batch are its forced companions (derived in the header).
# band_deep_matched.sh — the layer-decay arm the ledger says the depth axis is OWED.
#
# WHY THIS IS NOT A REPEAT. `band_dense_llrd` (2026-08-21) ran `--unfreeze-last 12 --layer-decay 0.7`
# at `--backbone-lr 4e-5` and read 0.8483 against `band_dense`'s 0.8578. The ledger REFUSED that arm's
# own BELOW branch and said why: `StudyModel.parameter_groups` anchors the ladder at the TOP opened
# block — `backbone_lr * layer_decay ** (len(opened) - 1 - depth)` — so opening twelve blocks does not
# merely add six whisper blocks under the control's six. It ALSO re-grades the control's own six from a
# flat 4e-5 down to 4e-5, 2.8e-5, 2.0e-5, 1.37e-5, 9.6e-6, 6.7e-6, a geometric mean of `0.7^2.5` =
# **0.41 of the control's rate**. The arm changed two things and the training loss said so: it floored
# ~0.13 high and was still descending at e10, the signature of too little rate rather than of wrecked
# early blocks. `experiments/README.md:1681` states the consequence in one line — "a re-run at
# `--layer-decay 0.7` is a different experiment, not a repeat, and it is the one the axis is owed."
# This is that re-run, and nothing else in the tree has it.
#
# THE COMPENSATION, derived rather than tuned. To leave the control's six blocks at the rate they
# actually shipped at, the anchor has to absorb the ladder's own geometric mean over those six:
#
#     backbone_lr = 4e-5 / 0.7^2.5 = 4e-5 / 0.4100 = 9.76e-5 -> 9.8e-5
#
# Blocks 6-11 then run at 1.64e-5, 2.35e-5, 3.36e-5, 4.80e-5, 6.86e-5, 9.80e-5 — geometric mean
# 4.0e-5, the control's flat rate — while blocks 0-5 run from 1.15e-5 down to 1.9e-6, the whisper the
# original arm was built to give them. So the ONE thing that moves against `band_dense` is that the
# encoder's bottom half opens at a decayed rate; the top half's aggregate rate is held. There is no
# setting where "only the grading changes" is exactly true — a ladder differs from a flat rate by
# construction — and matching the geometric mean over the shared six is the closest a single arm gets.
# It is stated here so the read is made against what was actually held, not against a claim of purity.
#
# WHY THE AXIS IS WORTH RE-OPENING NOW, and this is the part that is new since August. Every
# foreign-trunk arm this project has run inherits `unfreeze_last 6` at `offset 0` and `layer_decay 1.0`
# — decay is OFF everywhere, on every trunk, and has never been varied except in the confounded arm
# above. The published fine-tuning literature for out-of-distribution ViTs runs 0.65-0.85 as a matter
# of course. If depth pays once the rate confound is removed, the recipe under `radiodino_corpus`
# (0.8214), `ortho_dense` (0.8452), `coatnet_hybrid` (0.8259) and `mricore_corpus` is the wrong recipe,
# and every one of those readings is a reading of our schedule rather than of their pretraining. That
# is the instrument case for this arm: it does not have to beat anything to be worth two hours, it has
# to tell us whether an entire family of negative results was measured through a recipe that never let
# the trunk move. Q-RECIPE, sent to Kimi 2026-09-05T12:53Z, asks the literature side of the same thing.
#
# THE BATCH IS THE SAME FORCED SECOND VARIABLE, priced identically to the arm this replaces: twelve
# trainable blocks retain twice the activation stack, so `--batch 2 --accumulate 6` holds the effective
# batch at 12. `small_accum` measured that substitution at 0.8579 against `small_batch6`'s 0.8598 and
# the control's 0.8578 — three points inside a 0.0020 spread, a third of the 0.0062 floor.
#
# READ IT ON, in order, against control `band_dense` = 0.8578 (best epoch 8) and the confounded
# `band_dense_llrd` = 0.8483:
#   * ABOVE 0.8640 — depth pays once the rate is held, the flat rate was the problem all along, and the
#     whole depth axis plus every foreign-trunk negative was measured through a broken instrument.
#     This is the branch that licenses re-running the trunk family, and it is the reason for the arm.
#   * 0.8516 to 0.8640 — inside the floor of the control. Decay is NEUTRAL at matched mean rate, the
#     +0.0008 depth bracket stands as a real null rather than a confounded one, and `band_dense_llrd`'s
#     -0.0095 is explained ENTIRELY by the rate cut. That last part is itself a finding about the
#     ladder anchor and is worth the arm even though it moves no score.
#   * 0.8421 to 0.8516 — reproduces the confounded arm despite the compensation, which says the rate
#     cut was NOT the explanation and the early blocks resist at any rate. Closes the depth axis, and
#     retires the suspicion that hangs over every foreign-trunk row.
#   * BELOW 0.8421 — worse than the confounded arm while running its shared blocks 2.4x faster. Suspect
#     the harness first: `trainer.py:166` records `encoder_rate_decay` off the optimiser's top two
#     backbone groups rather than off the flag, so assert it landed at 0.7 and that the top group's
#     rate reads ~9.8e-5 before believing any of the above.
# The TRAINING LOSS is the second instrument again, and it is the one that caught the confound: this
# arm's loss should now TRACK `band_dense`'s 1.2162 -> 0.5780 rather than flooring 0.13 high. A loss
# that still floors high means the compensation did not reach the optimiser, not that depth failed.
#
# control  band_dense = 0.8578 (best epoch 8), six blocks at a flat 4e-5.
# control  band_dense_llrd = 0.8483, the same twelve blocks at an UNCOMPENSATED anchor.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train band_deep_matched \
  $RECIPE --epochs 12 --batch 2 --accumulate 6 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 9.8e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --unfreeze-last 12 --layer-decay 0.7 \
  --labels "$LABELS" --out "$MODELS/band_deep_matched_s2026.pt"
