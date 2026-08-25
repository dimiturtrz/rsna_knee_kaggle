#!/usr/bin/env bash
# confident_span.sh — the LABEL CONTENT axis, measured against a control that already exists.
#
# WHY THIS ARM, AND WHY IT DISPLACED A ROSTER ARM ON THE CARD. Seven board entries read 0.936-0.937
# while member count, member identity, fold status, regularisation, encoder family and architecture
# family each moved alone. The ensemble axis is finished. Every remaining path runs through the DATA or
# the TARGETS, and the ledger says the target path was never actually measured:
#
#   confident_labels.py: "Our own label experiment measured -0.0008 between two tables and was run
#   once, against a noise floor since measured at 0.006 - so it could not have detected an effect in
#   either direction, and we do not actually know whether label content moves our score at all."
#
# That is an OPEN axis wearing a closed axis's clothes, and it has sat there while five slots bought
# five confirmations of a plateau.
#
# WHAT THE CENSUS SAYS, and it is worse than the 27.3% headline. `UNK` share per target:
#   Synovitis 84.2%   Fracture 56.4%   Baker's 45.9%   Lateral OA 33.1%   Medial OA 25.5%
#   Contusion 20.8%   PF OA 18.5%      Lateral Meniscus 10.0%  Effusion 9.9%  MCL 9.8%
#   ACL 8.1%          Medial Meniscus 5.7%
# Synovitis is supervised by 694 asserted cells out of 4,406. The rest sit at UNCERTAIN_LEVEL = 0.28,
# and `--weight-floor` (in $RECIPE, on every member we ship) admits them at CONFIDENCE_FLOOR = 0.25 of
# a full weight rather than dropping them. So most of that label's loss mass is a constant.
#
# AND THE CONSTANT IS NOT NEUTRAL. `targets.MEASURED_AGREEMENT` records the radiologist agreeing with
# the 0.28 level at 0.136 — the one level that asserts nothing is nonetheless pulling every unmentioned
# finding toward a value twice the rate anyone endorses.
#
# THE SINGLE VARIABLE. `band_span`'s recipe verbatim — 12 slices, 0.10-0.90, 4e-5, 12 epochs, max
# pooling, whole-stack, fold 0 — with `--labels confident.csv` in place of `report_labels_v2.csv`.
# `confident_labels` clears the VALUE of every `UNK` cell rather than zeroing its confidence, so
# `Targets.load` skips the target and gives it weight 0.0: "we do not know" instead of "we weakly
# believe 0.28". Nothing else moves. This trains on strictly LESS supervision, and the question is
# whether less-and-honest beats more-and-wrong.
#
# control  band_span = 0.8565 (0.10-0.90 at 12 slices, same fold, same recipe). Already on the ledger,
#          so this arm needs no matched control of its own and costs one run rather than two.
# cost     ~12 epochs at the control's step time; ~1 h. 12 slices, not band_dense's 20, because 20
#          reserves 25.4 GB and biohub holds 10.2 GB of this card by arrangement.
#
# PREDICTION, written before the run. The four labels whose UNK share exceeds a third — Synovitis,
# Fracture, Baker's, Lateral OA — are where the removed mass lives, so they are where an effect must
# appear if there is one. Three of those four are in our weakest half by AUC.
#
# THE FALSIFIABLE PART:
#   * >= 0.8627 (+0.0062, one floor) — label content moves our score, the axis reopens as the first
#     live lever in days, and the expensive version (porting a real multilingual report reader, which
#     is where the strongest public notebooks spend their lines) becomes worth costing.
#   * 0.8503-0.8627 — within the floor either way. NOT a null: it bounds the effect of removing 27% of
#     the supervision at under a floor, which is itself the answer to "is our label noise the ceiling".
#     If removing a quarter of the targets changes nothing, the targets are not what binds us.
#   * <= 0.8503 (-0.0062) — the UNK cells carry real signal at their 0.25 weight and silence IS weak
#     evidence of absence, exactly as CONFIDENCE_FLOOR's docstring argues. Then the honest move is the
#     opposite one: trust silence MORE, not less, and `--calibrated-targets` (0.28 -> 0.136) is the
#     already-built lever to push further rather than remove.
#
# Whatever it reads, it is a fact about supervision rather than about seating, which is the axis that
# has been flat for seven entries.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

CONFIDENT="$PROCESSED/labels/confident.csv"
require "$CONFIDENT"

run_train confident_span \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.1 --band-high 0.9 \
  --labels "$CONFIDENT" --out "$PROCESSED/confident_span_s2026.pt"
