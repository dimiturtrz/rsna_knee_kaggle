#!/usr/bin/env bash
# labels_v4hybrid.sh — knee-wko44's pre-registered arm: band_dense's recipe with ONLY --labels moved.
#
# bead: knee-wko44
#
# WOLLEK'S RULER, NOT OURS. arXiv:2306.05997 changed the LABELER and read downstream AUC: rule-based
# parser 0.858, learned labeler 0.939, on one architecture and one corpus. Our reader IS the rule-based
# row. The bead's own note records why the cheap gate was abandoned — a 58-study panel cannot resolve
# the effect — so this axis runs Wollek's way, on a full candidate table and the 909-study holdout.
#
# THE TABLE, chosen on evidence rather than on the mean. A gold-58 census of every full-corpus table on
# disk (2026-09-06, logs/table_gold_census.log) ranks `blend_hybrid_fx_v4b` first at 0.8995 against our
# shipped v2's 0.8658. `blend_hybrid_fxrub` is within 0.0009 of it on the mean and strictly worse on
# DIRECTION, which is the tiebreak that matters: v4b cuts Effusion's false-positive rate from 69.6% to
# 43.5% of gold negatives and improves Contusion in both directions, while fxrub leaves Effusion exactly
# where v2 had it. A table is picked for the columns it fixes, never for its aggregate.
#
# `blendlabels` IS NOT THIS ARM, and the difference is the point. That one trained on a MEAN of three
# tables (`blend_three_v1`, 0.8912) — averaging back toward the lexicon — and it composed +0.0034. This
# one trains on the single best candidate, +0.0337 of gold agreement over the control's table.
#
# THE PINNED RULER IS THE SECOND HALF OF THE ARM. `--eval-labels report_labels_v2.csv` keeps the holdout
# scored against the table every other fold-0 row was scored on. Without it a member fitted on v4b would
# be scored against v2 and measure DISAGREEMENT WITH V2 as if it were quality — necessarily worse the
# more the tables differ, which is the one number this arm must not produce.
#
# WHAT THE ARM CAN AND CANNOT MOVE, written before it runs, from the per-column direction of the two
# tables on the gold 58 (`supervision_gap --labels <table>`, the confusion block):
#
#   CAN MOVE — v4b's supervision is measurably closer to the radiologist here:
#     Effusion   FP 69.6% -> 43.5%   (FN 2.9% -> 17.1%)
#     Contusion  FP 51.3% -> 41.0%   (FN 15.8% -> 10.5%)
#     Fracture   FN 27.8% -> 22.2%
#   SHOULD READ FLAT — the tables are within noise of each other on these:
#     Medial OA, Baker's, Lateral OA, PF OA
#   CANNOT MOVE, and this one is a CONTROL:
#     Synovitis. Not because the column is broken — v4b reads it at gold AUC 0.8190 against v2's 0.6870,
#     the largest per-column repair any blend on disk carries, and it is already IN the table this arm
#     trains on. The reason it is a control is that the arm changes nothing further about it: there is no
#     second treatment here, so a synovitis move is attributable to the same table swap as everything
#     else and adds no independent evidence. (The confusion rates ARE identical across thirteen tables —
#     9.7% FP / 51.9% FN — and that identity was misread once as the readers being blind. It is a
#     THRESHOLD artefact: 84.2% of studies carry `Synovitis__verdict == UNK` at a 0.21-0.30 prior, so the
#     silent mass sits below the 0.5 cut however well it is ordered inside itself. A column at AUC 0.819
#     posting 51.9% FN is miscalibrated, not uninformed. Filling those cells from Effusion was measured
#     and REFUTED: it imports Effusion's own over-call and costs -0.0753 on the column.)
#   AND THE FOUR THE MECHANISM CANNOT REACH — ACL, MCL, both menisci. `supervision_gap` reads the
#     reader at 0.84-0.99 on these and the MODEL as the trailing half; better supervision has nothing
#     to give where the supervision was never the weak part. A gain concentrated here is unread.
#
# PRE-REGISTERED READING, per mastermind 20:45Z and 20:52Z.
#   * v2 holdout mean FLAT or DOWN is EXPECTED and is not a kill — a model fitted on a better table
#     agrees less with the worse ruler. The mean is reported, never used as the verdict.
#   * The verdict is the gold-58 per-label read against `holdannot_s2026`'s 0.8932 mean: >= +0.010, or
#     the three CAN-MOVE findings up by >= +0.015 between them, opens the axis and licenses ARM B
#     (`--fold -1`, the board solo instrument, shipped against holdsolo's 0.928).
#   * Below that, with the three findings flat, the labeler axis is priced on OUR corpus and closes
#     regardless of Wollek's +0.081 — which is a chest X-ray, single-label, German-report number and
#     was never our expected gain.
#
# COST ~80 min of card, zero slots, no cache (the table is a CSV; the pixels are band_dense's).
# CONTROL `margin_seat_s2026` / `band_dense` at 0.8578, identical recipe, same fold, same holdout.
#
# THE GOLD RULER ONLY WORKS ON A HELD FIT, which is why `--hold-annotations` is on this line (mastermind
# 21:48Z). The deciding bar above is stated on the gold 58, and every table on this project weights the
# annotated studies at 3.0 — so without the flag the gold read would score the arm's own training cells
# and the whole 80 minutes would produce a fold-0 proxy the report ruler cannot see the treated columns
# on. That is exactly the defect that voided `raptor_solo`'s 0.8938 (98.6% in-sample, 2026-08-29). The
# flag also MOVES THE CONTROL: `band_dense` did not hold annotations, so it remains the control for the
# v2 holdout mean and CANNOT be the control for the gold read. The matched gold control is
# `holdannot_s2026` at 0.8932 — same holding, v2 table — and the bar is stated against that number.
#
# deliberate: --hold-annotations (the gold 58 is the deciding ruler; an unheld fit scores its own cells)
# deliberate: --labels blend_hybrid_fx_v4b.csv (the arm; best of thirteen tables on the gold 58, and
# deliberate: picked on per-column direction rather than on its aggregate)
# deliberate: --eval-labels report_labels_v2.csv (the ruler stays where every other fold-0 row was
# deliberate: scored, or the holdout measures disagreement with v2 instead of quality)
# reading: labels_v4hybrid_s2026.holdout.npz
#
# THE SUCCESSOR IS NAMED HERE, BEFORE THE ARM RUNS, so it is not re-decided from mood at 03:00Z
# (mastermind 21:03Z). If this clears its bar, the next arm is NOT a seventh member: it is the same
# table plus the silent-cell level map, retrained on the BEST base (labelwise-20-slice, 0.8613) — the
# board is paid by a uniform treatment carried across the champion's seats, never by one more seat.
# The level map's UNK prior per finding is derived, not tuned: the gold-58 positive rate among the gold
# cells the reader left UNK, denominator printed, one number per column.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

CANDIDATE="$PROCESSED/labels/blend_hybrid_fx_v4b.csv"
require "$CANDIDATE"
require "$LABELS"

run_train labels_v4hybrid \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --hold-annotations \
  --labels "$CANDIDATE" --eval-labels "$LABELS" --out "$MODELS/labels_v4hybrid_s2026.pt"
