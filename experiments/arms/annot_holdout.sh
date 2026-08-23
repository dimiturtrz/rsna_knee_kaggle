#!/usr/bin/env bash
# bead: knee-zc2t
# annot_holdout.sh — a member that never trains on the 58 image-read annotations, so the only
# radiologist-truth ruler in this project can finally be pointed at the CURRENT architecture.
#
# THIS ARM IS AN INSTRUMENT AND SCORES NOTHING. It is not expected to beat its control and its holdout
# number is not the deliverable. What it produces is a set of predictions that never saw the
# annotations, which is the one input `kneemri.evaluation.supervision_gap` needs and cannot get from
# any member in the roster.
#
# WHY IT IS WORTH A CARD HOUR. Eleven axes have now read flat inside the 0.0062 floor (knee-jk3):
# sampling, label re-read, encoder family, decode/pooling/width, batch/accumulate/ranking, resolution,
# severity, field of view, aggregation, SSL-as-init, crop centring, reformats, volumetric trunk. Eleven
# flat axes is what a LABEL CEILING looks like, and knee-7d1 already names the ruler as the top
# remaining suspect. Every local number we have is agreement with LLM-read REPORT labels; the board
# scores radiologist annotations, and where the report says YES the radiologist agrees 65.3% of the
# time.
#
# THE NUMBER IT PROTECTS. `supervision_gap`, run 2026-08-14 on the published members' out-of-fold
# predictions (interpretations/kneemri/2026-08-14_labels-or-pixels.md), read aggregate labels 0.8639
# against model 0.8391 — with SIX labels where the report describes the finding well and the model
# cannot reproduce it, mean gap ~0.11. Six of twelve labels at 0.11 is ~0.055 on the metric, the only
# quantity in this repository that is the size of the frontier gap it would have to explain. That read
# is nine days and several architecture changes old, and it cannot be repeated: it needs predictions
# untainted by the annotations, and it will never get them from a member fitted after it.
#
# WHAT BLOCKS IT, which is the finding this arm exists to remove. `core/data/labels/targets.py`
# overrides those 58 studies' scores with the annotations and weights them at `ANNOTATED_WEIGHT` = 3.0
# in EVERY member we fit. 696 cells at weight 3.0 is ~0.66% of total training weight — and in exchange
# we spend the only radiologist-truth ruler in the project, because every member is then scored against
# a table it trained on. That is a bad trade and it has been running silently since the annotations
# were wired in.
#
# THE RULER DOES NOT MOVE. `--hold-annotations` sets `Supervision.annotations` false, which
# `moves_the_ruler()` reports, so `Targets.pair` re-reads the HOLDOUT table under a plain supervision
# that keeps the annotations. This arm therefore trains without them and is scored on the unchanged
# ruler — every comparison against every ledger row stays valid, and the arm earns no self-agreement
# credit either way. `Targets.supervised()` records the switch, so the pre-flight landed check proves
# the flag fired rather than the run assuming it.
#
# ONE VARIABLE against the control: `--hold-annotations`. Everything else is byte-identical to
# `band_dense`, whose 0.8578 (best epoch 8) is already fitted and needs no matched rerun.
#
# PRE-REGISTERED, BEFORE THE NUMBER. This is a read on the arm's own holdout, and it is a claim about
# the WEIGHT, not about the method:
#   within 0.0062 of 0.8578 — expected. 0.66% of training weight cannot move a mean, and the
#              instrument is delivered: run `supervision_gap` on this member's holdout predictions.
#   a drop LARGER than the floor — a finding in itself, and not a failed arm: it prices what those 58
#              studies actually carry at 3x, which nothing has ever measured. It would say the
#              annotation upweight is doing real work and make the trade a genuine decision rather than
#              an oversight.
#   a RISE larger than the floor — suspect the plumbing first. The ruler is supposed to be identical
#              between the two arms; a rise means it is not, and the landed check plus the recorded
#              supervision block are where to look before anything is written down.
#
# COST. ~72 min on the card (12 epochs at ~361 s, small at the settled operating point), for zero
# direct score. What it buys is the only measurement that can distinguish "our architecture has hit its
# ceiling" from "our LABELS have", which is the question eleven flat axes have left standing.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train annot_holdout \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --hold-annotations \
  --labels "$LABELS" --out "$PROCESSED/annot_holdout_s2026.pt"
