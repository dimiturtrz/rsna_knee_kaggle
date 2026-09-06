#!/usr/bin/env bash
# foreign_dissent.sh — the third use of the same teacher file, after the first two were measured.
#
# WHAT THE TEACHER FILE HAS ALREADY BOUGHT AND NOT BOUGHT. `teacher_foreign_corpus.npz` is
# mattiaangeli's five released DINOv3 folds, fold-matched so every study is read by the one fold that
# never trained on it (0.8929 over 4406 rows). Three things can be done with a teacher's ordering:
#
#   SEAT it      — served directly as a member. +0.0178 at 6x on the served-shape base of 8, +0.0218 at
#                  6x on the greedyseat base of 7. Shipped today at 6/13 of the vote.
#   ABSORB it    — distil it into a student. `foreign_distill` read solo 0.8595 (flat against band_dense
#                  0.8578) and seat NEGATIVE at every weight on both rulers, -0.0004 at 1x to -0.0027 at
#                  4x on RankAverage. A student trained toward an ordering ends up agreeing with our
#                  pool, because our labels hold 0.7 of its loss. Absorbing is lossy where seating is not.
#   DISSENT from it — spend the capacity where that ordering FAILS. This arm. It is the only one of the
#                  three that can raise ensemble ambiguity without raising member error, because the
#                  pair mass moves but the direction of every pair is still the reports'.
#
# WHY THE 2026-08-24 REFUTATION DOES NOT COVER THIS. `dissent_seat` ran this exact flag at the same
# strength and read -0.0069, 38th of 66 in greedy, and the ledger recorded the objective axis as not a
# seat lever. Two things separate that arm from this one, and both are on the record rather than argued:
#   * ITS TEACHER WAS OURS. `teacher_fold0.npz` is our own fold-0 twins, so "the pairs the teacher gets
#     wrong" were the corpus's hard pairs, which our pool is already spending its capacity on. The
#     complement of a pool to ITSELF is noise. The complement of our pool to a family that now carries
#     0.4615 of the served vote is a different set of pairs, and nothing has measured it.
#   * IT WAS KILLED AT EPOCH 8 OF 12. `convergence_check`'s TRUNCATED class: its -0.0069 is a FLOOR, not
#     a landing. The refutation it carries is therefore weaker than the sentence it was written into.
# The teacher is the single variable this arm moves against `dissent_seat`; the recipe is byte-identical
# to it, which makes `dissent_seat` the matched control as well as the thing being re-opened.
#
# ONE VARIABLE against `dissent_seat`: --teacher-targets, ours -> theirs. Rate, slices, band, pooling,
# whole-stack, epochs, --rank-weight 0 and --dissent-weight 1.0 all held.
#
# HOW IT IS READ, written before the number. The reading is the SEAT, not the solo — same as
# `foreign_distill`, and for the same reason: a member whose pair mass is deliberately spent off the
# corpus mean is not expected to top a solo table.
#   * seat POSITIVE beside the foreign block on the served-shape base of 8 — the complement is real, the
#     objective axis re-opens on the teacher rather than on the loss, and the next arm is the weight.
#   * seat FLAT (interval spans zero) — the foreign family's failures are the corpus's failures; nothing
#     family-specific is recoverable from where it is wrong either. Together with `foreign_distill` that
#     closes every non-seating use of a foreign teacher, which is a real closure and worth the card.
#   * seat NEGATIVE and solo well below `dissent_seat`'s 0.8509 floor — the dissent term is fighting the
#     labels rather than re-weighting them, and 1.0 is too large for a teacher this distant.
#
# THE RULER, AND ITS KNOWN GAP. `seat_weight` DROPS foreign records passed to --base (knee: filed today),
# so the served shape has to be built the way `logs/seat_weight_served.log` was built, not by passing the
# foreign records as a plain base. A run that prints "base of 7" has silently priced the wrong ensemble.
#
# COST. ~70 min of card, 12 epochs on the shared recipe. ZERO submission slots. Teacher file on disk.
#
# deliberate: --teacher-targets (the single moving variable; theirs, not ours)
# deliberate: --dissent-weight (1.0, HELD from dissent_seat so strength is not a second variable)
# deliberate: --rank-weight (0, HELD from dissent_seat; the dissent term IS the pair term)
# bead: knee-1vdcz
# reading: foreign_dissent_s2026.holdout.npz
# control  dissent_seat = 0.8509 (TRUNCATED at e8, a floor) and band_dense = 0.8578 (flat pair mass).
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
require "$PROCESSED/teacher_foreign_corpus.npz"

run_train foreign_dissent \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --rank-weight 0 --dissent-weight 1.0 --teacher-targets "$PROCESSED/teacher_foreign_corpus.npz" \
  --labels "$LABELS" --out "$MODELS/foreign_dissent_s2026.pt"
