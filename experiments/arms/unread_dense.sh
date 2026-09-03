#!/usr/bin/env bash
# unread_dense.sh — the 27.3% of supervision that is a CONSTANT, given an order. `knee-s6s9r`.
#
# THE DEFECT IS COUNTED, NOT HYPOTHESISED. 14,445 of 52,872 (study, finding) pairs carry the UNK verdict,
# which `report_reader` assigns when the report does not name the structure at all — Synovitis on 84.2% of
# studies, Fracture 56.4%, Baker's 45.9%. Every one of those cells trains toward the SAME number, so they
# carry no order, and order is the whole of what the metric reads. `supervision_gap` prices the consequence
# on the 58 annotated studies: labels - model = -0.0470 [-0.0717, -0.0218] over 696 cells, with Synovitis
# +0.1075 and Fracture +0.0715. A model fitted on those constants out-reads them. This arm turns that into
# supervision.
#
# THE TEACHER IS NOT OURS, ON PURPOSE. The values come from the published pool — twenty checkpoints, four
# seeds over five folds whose oof id-sets partition all 4,407 studies — so every cell is scored by a model
# that did not train on it (`PublishedRuler`, the same arrays supervision_gap's -0.0470 was measured with).
# Using our own member would be self-distillation through a shared trunk on cells it was itself fitted on:
# a training-dynamics experiment wearing a label experiment's clothes.
#
# ONE FLAG AGAINST THE CONTROL. Every flag below is `band_dense_hold`'s, which exists precisely to be the
# held control at this corner, except `--unread-targets`. It reads band_dense's own pixel cache, so no
# decode is owed. `--hold-annotations` STAYS: `annotated_readings` refuses a checkpoint that trained on the
# 58 annotation cells, and the gold read is this arm's primary number, not its secondary.
#
# SCORE ONLY, AND MEAN-PRESERVING. The override replaces what an unread cell pulls TOWARD and never how
# hard — the weight `--weight-floor` gave it is untouched, so the experiment stays one variable wide. The
# rank is taken inside each finding's UNK block, so it is uniform by construction and the band's mean is
# the constant those cells already carry. Only ORDER is added. Moving the mean would be a second change,
# and a prevalence shift is the one thing a twelve-head trunk transmits between heads.
#
# THE BAND IS DERIVED. Its half-width is the NEARER gap from the unread level to a neighbouring level in
# `MEASURED_AGREEMENT`, on the axis the label table is written in. This arm runs on the UNCALIBRATED
# report_labels_v2.csv that lib.sh points every arm at, so the centre is 0.28, the neighbours are 0.08 and
# 0.68, the half-width is 0.20 and the band is [0.08, 0.48] — so an override never asserts
# UNK cells cannot be measured corpus-wide (only 58 studies carry an annotation), so the size of the claim
# is capped by something that WAS measured rather than by a number chosen to make the arm interesting.
#
# THE PROXY IS UNINTERPRETABLE HERE, and that is registered before the run rather than argued after it. The
# report holdout scores against report-derived labels, whose UNK cells ARE the constants this arm replaces.
# On 27.3% of cells the ruler and the treatment disagree by construction, so a proxy DROP is the expected
# reading and is not evidence against the mechanism. Quote it, never decide on it.
#
# PRE-REGISTERED on GOLD, before the arm runs. Gold delta against band_dense_hold_s2026 (0.8956), paired
# half-width +/-0.015:
#
#   >= +0.015   THE MECHANISM LIVES. Semi-supervised targeting of never-measured cells beats the constant.
#               Roster it, and only then price a slot. The next arms are the dose (how wide the band) and
#               the weight-side variant, not a second seating.
#   within +/-0.015   NULL ON THIS RULER. Directional only, no slot, and do NOT tune the band — a band
#               fitted until the number moves is the hack this project's rules forbid.
#   <= -0.015   REFUTED FOR THIS APPLICATION, and it does NOT rule out: the weight-side variant, which
#               reaches the same cells through --weight-floor rather than the score; nor restriction to the
#               five findings where supervision_gap is positive (Synovitis, Fracture, Lateral Meniscus,
#               Baker's), leaving MCL at -0.0692 alone. Report the PER-LABEL rows before writing anything.
#
# WHAT WOULD MAKE ME DISBELIEVE ANY OF IT. That the flag never landed. `--unread-targets` is a PATH, and
# `Supervision.taught` DROPS a row naming a study the table has no place for — by design, so a smoke can
# run on sixty studies — which makes a wrong path and a legitimately-subset table the same silent zero.
# Check the run's recipe block prints `"unread_targets"` at the table's path, and check the table's row
# count against the `unread cells written` line the generator logged, BEFORE reading any score.
#
# THE SECOND READING IS OWED, and knee-v78o1 is why that sentence is here. `recal_dense` sat four days with
# its pre-registered gold number never computed and every gate green, because a reading that lives in an
# arm's prose and needs a separate command has nothing asserting it happened. This arm's gold read is
# `kneemri.evaluation.annotated_readings --against <full PATH to band_dense_hold_s2026.pt>` and it is not
# optional: without it this arm has no deciding number at all.
#
# deliberate: --unread-targets (the arm's thesis, and its only difference from band_dense_hold)
# deliberate: --hold-annotations (band_dense_hold's read exactly — the control, and what makes gold readable)
#
# bead: knee-s6s9r
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
# control  band_dense_hold_s2026 = 0.8573 proxy / 0.8956 gold (declared 2026-09-03, knee-s6s9r)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
UNREAD="$PROCESSED/unread_targets.csv"
require "$LABELS"
require "$UNREAD"

run_train unread_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --hold-annotations \
  --unread-targets "$UNREAD" \
  --labels "$LABELS" --out "$MODELS/unread_dense_s2026.pt"
