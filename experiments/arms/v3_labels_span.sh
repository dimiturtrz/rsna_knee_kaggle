#!/usr/bin/env bash
# The first member this project has ever trained on a label table made AFTER the competition's criteria
# entered the reader. Everything else in the pool was fitted on `report_labels_v2.csv`.
#
# WHAT D343/D344 MEASURED, AND WHY THIS IS NOT A KNOB. `report_labels_v2.csv` was written 2026-08-11.
# The host's criteria entered the reading prompt on 2026-08-27 (`ec4d3f72`). So v2 is the LEXICON-era
# table: keyword matching over nine languages by a pipeline that predated the rubric it was supposed to
# encode. The criteria say in so many words that degeneration without discontinuity is negative for ACL
# and that intrasubstance signal not reaching the surface is negative for the menisci; those sentences
# were nowhere in the pipeline that produced the table 286 arm invocations have trained on.
#
# The damage is measured, one-directional, and large. Of 94 blind human adjudications of v2-vs-cluster
# dissent, v2 is wrong on 94.2% of the 86 cells where it OVER-calls and right on 87.5% of the 8 where it
# under-calls. That is ~1,955 false positives corpus-wide, 12.1% of all 16,128 positive labels,
# concentrated on the tear columns by a named mechanism: the extractor reads degeneration as tear.
#
# THE FIX ALREADY EXISTS. `report_labels_v3.csv` is the criteria-era reader's table. Scored on the same
# 94 verdicts it reads the degeneration line at 0.709 against v2's 0.058, with the under-call rate
# unchanged at 0.875 — it corrects the over-call without over-correcting into the other direction, which
# is the failure mode D343's own table let us name in advance. Corpus-wide it differs from v2 on 6,461
# cells (12.2%), every one of them v2=1 -> 0 and not a single flip the other way, deleting 40% of v2's
# positive labels.
#
# THE ARM. `band_span`'s recipe verbatim — same epochs, same pooling, same span, same fold, same cached
# decode — with ONE substitution: `--labels` names v3. Nothing else moves, so a delta is the table.
#
# THE RULER, AND THE TRAP IN IT. `--eval-labels` is pinned to v2 ON PURPOSE. It keeps the holdout truth
# identical to `band_span`'s so the ledger number is comparable, and it makes the eval-on-own-generator
# loop explicit rather than hidden: a v3-trained member is being scored by a table whose errors it was
# trained not to reproduce, so the v2-ruler number is a FLOOR by construction and is not the read. The
# read is the corrected T1 rescore off this run's holdout npz, which is what D343 built and what the
# adjudication is hardening.
#
# control  band_span 0.8565 midpoint / 0.8592 T1, same recipe, same fold, v2 labels.
# cost     the control's step time; 12 epochs; ~70 min on the 5090. No decode, no new artefact.
#
# PREDICTION, written before the run. On the v2 ruler this arm reads AT OR BELOW band_span — the ruler
# rewards reproducing the 12.1% it was built from. On the corrected T1 ruler it reads ABOVE, and the
# clearing condition is the owner's floor rather than a knob-sized delta: T1 >= 0.8592 + 0.015 makes the
# lexicon table the single largest measured defect in the project and re-opens every member in the pool
# as mis-trained. Between +0.005 and +0.015 on T1 is directional and licenses a v3 re-fit of the top
# seats, not a claim. A T1 delta at or below zero says the extraction layer is not the ceiling and
# D330's report-silence bias underneath it is, which is a harder result and a more useful one.
#
# The refutation I am most alert to is PLUMBING: v3 must actually be the training table. Assert the
# recipe's `labels` field names v3 and its `eval_labels` names v2 before any number here is written down.
#
# reading: holdout909
# bead: knee-clzzs
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

V3="$PROCESSED/labels/report_labels_v3.csv"
require "$V3"
require "$LABELS"

run_train v3_labels_span \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.1 --band-high 0.9 \
  --labels "$V3" --eval-labels "$LABELS" --out "$MODELS/v3_labels_span_s2026.pt"
