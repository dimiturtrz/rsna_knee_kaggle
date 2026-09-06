#!/usr/bin/env bash
# seat_distill_lr4e5.sh — the FIRST champion seat retrained under the OOF teacher. This one number
# decides whether seven hours of card go to retraining the other five seats.
#
# WHY A SEAT AND NOT A SEVENTH MEMBER. `oof_teacher` read 0.8681 against `band_dense`'s 0.8578, the
# best solo this project owns. A solo reaches the board as one seat among seven and is diluted; a
# treatment carried across EVERY seat passes through at 92-147% (measured, this project's uniform
# treatments). 0.944 + 0.0107 is the only arithmetic on the table that reaches first place without a
# new architecture. But a pass-through is a claim about seats, and no seat has been retrained yet —
# so the claim is priced on ONE seat before the other five are queued.
#
# WHY THIS SEAT. `lr4e-5_ws` is the weakest of the six at 0.8525 and its recipe is the closest to
# `band_dense`'s: same rate 4e-5, same whole-stack pooling, same throughput, same epochs. It differs
# only by the band crop and slice count (`band_dense` narrows to 0.1-0.9 at 20 slices; this one takes
# the $RECIPE default). Fewest changed variables between the arm that produced the +0.0107 prior and
# the arm that has to transfer it, which is exactly the seat mastermind named (22:25Z).
#
# THE CHAMPION SEAT ITSELF CANNOT BE READ, and this is the reason the arm is spelled with the twin.
# All six champion members are `--fold -1` fits (`final_fit_six.sh`) — they hold nothing out, so no
# holdout row can price them and "seat holdout + 0.008" is not a number that exists for `lr4e-5_ws_ff`.
# `lr4e-5_ws_s2026` is that seat's fold-0 twin: same recipe, same rate, one fold held. Its 0.8525 is
# the only readable denominator the seat has, so the bar is stated against it and the all-data refit
# follows only if this clears.
#
# ONE VARIABLE against `lr4e-5_ws`: --teacher-targets + --distill-weight. Weight 0.7 HELD from
# `foreign_distill_mix07` and from `oof_teacher`, both of which found it the best point on the
# monotone line. Rate, slices, pooling, whole-stack, throughput and epochs all held.
#
# THE TEACHER IS ALREADY PRICED AND ITS SOURCE IS NOT THE MECHANISM. `oof_teacher` (ours, out-of-fold)
# 0.8681 and `foreign_distill_mix07` (foreign) 0.8656 sit 0.0025 apart, inside the pair's 0.0062 floor.
# Distillation is the lever; whose posterior does the teaching is not. OOF is used here because it is
# the higher of the two and because its fold coverage is verified (3498 rows, fold 0 excluded so the
# holdout cannot leak).
#
# PRE-REGISTERED READING, before the arm runs.
#   * >= 0.8605 (0.8525 + 0.008) — the treatment transfers to a seat recipe that is NOT band_dense.
#     Queue the other five seats back to back, then the all-data refits. This is the first-place path.
#   * 0.8565-0.8605 — a real gain that does not reach the bar. The treatment is band_dense-specific in
#     SIZE, not in sign. Read the per-label shape: if the gain is still on MCL / Lateral Meniscus /
#     ACL / Contusion, the mechanism held and the seat recipe damped it; retrain the seat whose recipe
#     is NEXT closest (`through_dense`, 0.8558) before committing the remaining four.
#   * <= 0.8565 — the +0.0107 is a property of band_dense's band-and-slice geometry rather than of
#     distillation, and the six-seat plan is not licensed. The next question is then which geometry,
#     not which teacher.
#
# COST ~75 min of card, 12 epochs, zero submission slots. The teacher file already exists.
#
# deliberate: --teacher-targets (the treatment under test; ours out-of-fold, 3498 rows, fold 0 out)
# deliberate: --distill-weight (0.7, HELD from oof_teacher and foreign_distill_mix07)
# deliberate: --slices 12 --band-low 0.2 --band-high 0.8 (the control's geometry, written out rather
# deliberate: than inherited: `rate_wholestack.sh` states none of the three and took the pre-settlement
# deliberate: defaults in silence (knee-aixw), so an inherited default here would only be matched to
# deliberate: 0.8525 by accident of when the arm ran. These are those values, pinned.)
# bead: knee-ozpyq
# reading: seat_distill_lr4e5_s2026.holdout.npz
# control  lr4e-5_ws = 0.8525
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
require "$PROCESSED/teacher_oof_corpus.npz"

run_train seat_distill_lr4e5 \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --distill-weight 0.7 --teacher-targets "$PROCESSED/teacher_oof_corpus.npz" \
  --labels "$LABELS" --out "$MODELS/seat_distill_lr4e5_s2026.pt"
