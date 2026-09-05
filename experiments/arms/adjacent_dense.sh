#!/usr/bin/env bash
# bead: knee-e6sj
# reading: logs/adjacent_dense_ledger.log
# deliberate: --spacing adjacent — the ONE flag that moves against band_dense, and the arm is nothing else.
# adjacent_dense.sh — the contiguity re-test at the ONE operating point knee-e6sj left it open at.
#
# THIS IS NOT A REPEAT, AND THE ARM IT LOOKS LIKE WAS REFUTED. `adjacent224` (2026-08-27) read 0.8462
# against `native224` 0.8502 — MEAN -0.0040, inside the 0.0062 floor, and the axis was closed on it.
# The close note is what licenses this arm, and it named the re-open condition before the result was
# known: "it does not rule out adjacency at a DENSER stack where span is not the thing being spent. If
# merged_dense_cf (20 slices, 0.1-0.9) pays, adjacency is worth one re-test there and only there."
# That condition is now MET and was not met in August. `adjacent224` ran at 12 slices over 0.20-0.80;
# `band_dense` at 20 slices over 0.10-0.90 reads 0.8578, is the best single member the project owns,
# and `merged_blend_cf` sits in the champion. The operating point moved underneath a closed axis.
#
# WHY THE OPERATING POINT IS THE WHOLE ARGUMENT. Contiguity is not free: a contiguous run of N slices
# covers less of the band than N spread across it, so adjacency is always bought with SPAN. At twelve
# slices in a 0.6 band that purchase cost a third of the span, and the ledger read the damage exactly
# where the trade predicts — Regional -0.0073 against focal -0.0018, broad rather than confined. The
# same twenty slices in a 0.8 band buy the same contiguity out of a much larger in-band slice count, so
# the span spent per slice of contiguity is smaller. That ratio is the mechanism, it is the only thing
# that changed since the refutation, and it is measured rather than assumed: see the coverage figure
# recorded with this arm in `logs/adjacent_dense_ledger.log`.
#
# WHAT CONTIGUITY IS FOR, unchanged from the bead. Under `SPREAD` no two sampled slices are ever
# adjacent in millimetres, at ANY band width — `Spacing`'s own docstring says so. A meniscal tear is
# 1-3 mm of signal that has to touch an articular surface on two CONSECUTIVE slices, so under spread
# the model cannot see that continuity at all. The rival's claim (forum topic 737597, self-reported)
# is +0.018 for centre-9 adjacent over 9 spread across 24, at 288 px / 140 mm / 5-fold — a different
# operating point from ours in every term, which is why their number sizes the axis and settles nothing.
#
# THE PRICE. COST one cache build (55.6 GB decoded, already on disk before this arm is queued) plus 12
# epochs at `band_dense`'s step time, about 2h of card. GAIN: the rival sizes it at +0.018 and our own
# refutation at this budget sizes it at -0.0040, so the honest prior is wide and centred near zero.
# P(clears +0.010) ~ 0.25. It is bought as a DATA axis rather than a knob — the one upstream choice
# about WHICH slices the model sees at a fixed slice budget — and the band-width work sizes that family
# at 0.0138 (0.10-0.90 = 0.8565 against 0.00-1.00 = 0.8427), which is the evidence that slice CHOICE at
# this operating point moves numbers of the size we need.
#
# READ IT ON, against control `band_dense` = 0.8578 (best epoch 8 of 12 — converged, not a floor).
# The bead pre-registered these bands in August and they are carried over UNCHANGED, because a band
# rewritten after a refutation is not a pre-registration:
#   * ABOVE +0.010 (0.8678) — contiguity is a real lever at density, the August refutation was an
#     artefact of spending span at twelve slices, and the axis re-opens with a cache to rebuild on its
#     account. This also clears the 0.874 single-member target the board arithmetic asks for.
#   * +0.006 to +0.010 (0.8638 to 0.8678) — directional. Keep as a diversity member and re-test; do not
#     report it as a finding and do not plan around it.
#   * within +/- 0.006 (0.8518 to 0.8638) — the axis CLOSES for good. Adjacency does not transfer at any
#     operating point we can reach, and the rival's +0.018 is their 5-fold OOF ruler rather than a
#     mechanism. This is the branch that retires the suspicion, and it is worth the card for that alone.
#   * BELOW -0.006 (0.8518) — spread is LOAD-BEARING and the wide-band result is why. Reproduces
#     `adjacent224` at a second, denser operating point, which is a stronger closure than one point.
# THE PER-FINDING READ IS THE SECOND INSTRUMENT, and it caught the mechanism last time. `adjacent224`
# fired and cost: Medial Meniscus -0.0136*, the one marked delta, on precisely the finding contiguity
# was written for. If the mean is flat here but Medial Meniscus is now POSITIVE and marked, the span
# trade is confirmed as the August cost and the axis is open at a denser stack still. If Medial
# Meniscus is negative again, the mechanism is refuted on its own chosen finding twice and nothing
# about a third operating point should be believed.
#
# control  band_dense = 0.8578 (best epoch 8), identical in every flag but `--spacing`.
# refuted   adjacent224 = 0.8462 vs native224 0.8502 at 12 slices / 0.20-0.80 / 224 px.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train adjacent_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --spacing adjacent \
  --labels "$LABELS" --out "$MODELS/adjacent_dense_s2026.pt"
