#!/usr/bin/env bash
# bead: knee-62sol
# reading: logs/band_deep224_ledger.log
# also: knee-zt3l6 knee-zsyp4
# deliberate: --unfreeze-last 12 --layer-decay 0.7 --backbone-lr 9.8e-5 --batch 2 --accumulate 6 --image-size 224 — the ladder is the arm, the anchor and batch are its forced companions, and 224 is the control's own resolution.
# band_deep224.sh — the depth arm re-run at the resolution that made it cheap, on the instrument that
# makes it readable.
#
# WHY THIS IS NOT A REPEAT OF `band_deep_matched`. That arm ran, converged 12/12, and read 0.8487 — the
# 0.8421-0.8516 branch its own header pre-registered as "closes the depth axis, and retires the
# suspicion that hangs over every foreign-trunk row". The branch was REFUSED, and by the arm's own
# second pre-registered instrument: its training loss floored at 0.7667 against `band_dense`'s 0.5780,
# 0.189 high — worse than the 0.13 that disqualified the confounded `band_dense_llrd` it was built to
# replace. Its header names the exact signature: "A loss that still floors high means the compensation
# did not reach the optimiser, not that depth failed."
#
# AND NOTHING COULD SAY WHICH. `Landed` witnessed `layer_decay` — the RATIO between two backbone
# groups — and a ratio lands at exactly the asked 0.7 whether the rate it divides is 9.8e-5 or the 4e-5
# the recipe inherits. `backbone_lr` itself was written off in `NOT_A_SWITCH` as "a rate, not a switch",
# so the single most consequential number in the arm had no positive witness at all. The header's BELOW
# branch told the reader to "assert the top group's rate reads ~9.8e-5 before believing any of the
# above"; there was no reading to take. The finished checkpoint cannot answer it either — checkpoints
# save no optimiser state — so the arm is owed a re-run rather than a re-read (knee-62sol).
#
# WHAT IS DIFFERENT NOW is one commit, not one flag: `Trainer.backbone_rate` captures the anchor off the
# built optimiser groups and `Landed` compares it as `encoder_top_rate`. This arm therefore CANNOT start
# at the inherited rate: the pre-flight refuses before the card is spent. That is the whole reason the
# depth axis is worth a second forty minutes — not a new hypothesis, a first honest measurement of the
# old one.
#
# WHY 224. `band224` (2026-09-05) read 0.8529 against `band_dense`'s 0.8578 at 336 — a delta of 0.0049,
# inside the 0.0062 floor — at 142 s/epoch against 450. The pixel axis is flat and the arm is 3.2x
# cheaper, which is what turns a 90-minute re-run into a 28-minute one. The control moves WITH the arm:
# everything below is read against `band224` = 0.8529, not against the 336 number.
#
# THE COMPENSATION IS UNCHANGED, because the rate it compensates is unchanged. `band224` runs the same
# flat 4e-5 as `band_dense`, so the anchor still has to absorb the ladder's geometric mean over the
# control's shared six blocks: 4e-5 / 0.7^2.5 = 9.8e-5. Blocks 6-11 then run 1.64e-5 -> 9.80e-5,
# geometric mean 4.0e-5; blocks 0-5 get the whisper. The one thing that moves against the control is
# that the bottom half opens at a decayed rate.
#
# THE BATCH IS THE SAME FORCED SECOND VARIABLE: twelve trainable blocks retain twice the activation
# stack, so `--batch 2 --accumulate 6` holds the effective batch at 12. `small_accum` priced that
# substitution at 0.8579 against `small_batch6`'s 0.8598 and the control's 0.8578 — a 0.0020 spread, a
# third of the floor.
#
# READ IT ON, in order, against control `band224` = 0.8529 (best epoch 6). The brackets are
# `band_deep_matched`'s, shifted by the control's own 0.0049:
#   * ABOVE 0.8591 — depth pays once the rate is held, and every foreign-trunk negative was measured
#     through a recipe that never let the trunk move. This branch re-opens the trunk family and is the
#     reason for the arm.
#   * 0.8467 to 0.8591 — inside the control's floor. Decay is NEUTRAL at matched mean rate, the depth
#     bracket stands as a real null rather than a confounded one, and `band_dense_llrd`'s -0.0095 is
#     explained ENTIRELY by the rate cut.
#   * 0.8372 to 0.8467 — reproduces the confounded arm despite a compensation now WITNESSED to have
#     landed. The early blocks resist at any rate: closes the depth axis and retires the suspicion over
#     ~25 foreign-trunk rows. This is the branch `band_deep_matched` scored into and could not claim.
#   * BELOW 0.8372 — worse than the confounded arm while running its shared blocks 2.4x faster. Suspect
#     the harness, but no longer the anchor: `encoder_top_rate` is compared before the first step.
# THE TRAINING LOSS IS THE SECOND INSTRUMENT and it is now a CLEAN one, because the anchor is separately
# witnessed. It should track `band224`'s 0.5914 rather than flooring high. If it floors high anyway with
# the anchor confirmed at 9.8e-5, that is a finding about what a decayed bottom half costs the fit — not
# an unreadable arm.
#
# control  band224 = 0.8529 (best epoch 6), six blocks at a flat 4e-5, 224 px.
# control  band_dense_llrd = 0.8483, twelve blocks at an UNCOMPENSATED anchor, 336 px.
# refused  band_deep_matched = 0.8487, the same arm at 336 px with the anchor unwitnessed.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train band_deep224 \
  $RECIPE --epochs 12 --batch 2 --accumulate 6 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 9.8e-5 --slices 20 --band-low 0.1 --band-high 0.9 --image-size 224 \
  --unfreeze-last 12 --layer-decay 0.7 \
  --labels "$LABELS" --out "$MODELS/band_deep224_s2026.pt"
