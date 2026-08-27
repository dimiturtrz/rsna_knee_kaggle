#!/usr/bin/env bash
# bead: knee-07wb
# confident224.sh — the LABEL CONTENT axis, at the corner that fits a shared card.
#
# WHY THIS AND NOT `confident_span.sh`, WHICH ASKS THE SAME QUESTION. That arm is written at the 336 px
# corner and `vram_need` reads 17997 MiB for it. celltrack holds this card by arrangement and leaves
# 17147 MiB free, so it misses by 850 MiB and would park until the card empties. This is the same single
# variable at 224 px: 8124 MiB, which runs beside a co-tenant holding up to 24 GB, and ~94 s/epoch
# instead of ~290, so ~19 minutes rather than an hour.
#
# WHAT IT GATES. knee-pju4 proposes a local open-weights multilingual reader over the 14,445 UNK
# (study,finding) pairs — 27.3% of all supervision. That costs a model download, deps this machine does
# not have (vllm / accelerate / gptqmodel are all absent), the card for inference, and a retrain of every
# seat. Call it a day. It rests on the UNTESTED premise that label CONTENT moves our score at all.
#
# THIS ASKS THAT PREMISE FROM THE CHEAP SIDE. `confident.csv` clears every UNK VALUE rather than filling
# it, so `Targets.load` skips the target and gives the cell weight 0.0: "we do not know" in place of "we
# weakly believe 0.28". It trains on strictly LESS supervision. If removing a quarter of the targets
# moves nothing, the targets are not what binds us, and the reader project is dead before it costs a day.
#
# WHAT THE OTHER TWO THIRDS OF knee-pju4 TURNED OUT TO BE, audited 2026-08-27 and recorded in its notes:
# the VALUE the UNK constant takes is already corrected (`MEASURED_AGREEMENT` maps 0.28 -> 0.136, the
# rate the radiologist actually agreed at, plus a per-finding shift in `FINDING_OFFSETS`), and the WEIGHT
# it pulls with is already floored (`--weight-floor` is in $RECIPE, so a UNK cell trains at
# 0.25 + 0.75*conf = 0.2875 against a YES cell's 0.9625). Only the FILLING is open, and only this gates it.
#
# THE SINGLE VARIABLE. `native224`'s recipe verbatim — 12 slices, band 0.2–0.8, --image-size 224, 12
# epochs, backbone-lr 4e-5, max pooling, whole-stack, fold 0 — with `--labels confident.csv` in place of
# $LABELS. Nothing else moves.
#
# control  native224 = 0.8502, best epoch 7, convergence_check clean. Already on the ledger, so this arm
#          needs no matched control of its own and costs ONE run.
# cost     ~19 min, zero slots.
#
# THE FALSIFIABLE PART, against the 0.0062 three-seed paired floor:
#   * >= 0.8564 — label content moves the score. knee-pju4's reader is LIVE and worth the day, and the
#     expensive multilingual read becomes the thing to build rather than a thing to argue about.
#   * 0.8440–0.8564 — within the floor either way. NOT a null: it bounds the effect of removing 27.3% of
#     the supervision at under a floor, which IS the answer to "is our label noise the ceiling". This is
#     the branch that saves the most, because it retires a day of work on a measurement.
#   * <= 0.8440 — the UNK cells carry real signal at their 0.2875 weight, and silence IS weak evidence of
#     absence exactly as CONFIDENCE_FLOOR's docstring argues. Then the honest move is the OPPOSITE one:
#     trust silence MORE, and `--calibrated-targets` is the already-built lever to push rather than remove.
#
# CAVEAT STATED BEFORE THE NUMBER. This is the 224/12-slice corner, weaker than the champion's dense one.
# A null bounds the effect HERE; a positive would want confirming at 336 before a seat is retrained. It is
# not an unrepresentative corner: knee-rh4t reads the two best self-reported single-model rivals, 0.942
# and 0.938, as both at 224 px.
#
# HARNESS CHECK BEFORE READING ANY OF THAT. The recipe line the run prints must say "image_size": "224"
# and must name confident.csv, and the log must load 4,406 studies and hold out 909 — the same holdout
# every fold-0 row was scored on. Three runs have entered this ledger as facts about an architecture when
# they were facts about a flag that never landed.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

CONFIDENT="$PROCESSED/labels/confident.csv"
require "$CONFIDENT"

run_train confident224 \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 \
  --labels "$CONFIDENT" --out "$PROCESSED/confident224_s2026.pt"
