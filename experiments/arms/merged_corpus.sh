#!/usr/bin/env bash
# merged_corpus.sh — the control's recipe over a corpus with a SECOND SITE in it, and nothing else moved.
#
# bead: knee-2jg6
#
# WHAT MOVES. One flag: `--scheme merged`, which points the trainer at
# images_336px_12sl_merged.npy — our 4,407 studies followed by kneeMRI Rijeka's 736, on the canonical
# six-slot axis. A foreign exam occupies SLOTS[0] and its other five slots are zero with the mask
# saying so, so the head width, the slot attention and the member shape are the control's exactly.
# The label table moves with it (`merged_labels.csv`, the column union), because a foreign row can
# only be scored on what ITS table named: `Targets.pair` gives an absent cell weight 0.0, so the 736
# supervise ACL and are silent on the other eleven findings rather than being labelled zero.
#
# WHY IT IS COMPARABLE TO A RUN THAT ALREADY EXISTS. `ReportFolds.assign` sends an identifier
# `train.csv` has never heard of to TRAIN_ONLY = -2. Measured on the real merged index: 736 train-only,
# fold 0's holdout 909 — OUR studies, the same 909 every fold-0 arm in the ledger was scored on. So the
# control  lr4e-5_ws = 0.8525 (declared 2026-09-03 from this header own prose, knee-sgzgc)
# control is not run again. It is `lr4e-5_ws = 0.8525`: same 12 slices, same 0.20-0.80 band (the
# merged cache carries the historic band, which is why the control is NOT band_span), same
# whole-stack, same 4e-5 backbone rate, same twelve epochs, same label table for our rows.
#
# PRICED BEFORE RUNNING, AND THE PRICE IS BAD IF YOU READ IT OFFLINE. +736 studies is +16.7% of the
# corpus, and they carry ONE of twelve findings. If the whole gain landed on ACL and nowhere else it
# is worth a twelfth of its size on the averaged holdout — an ACL gain of 0.02 shows up as 0.0017 on
# the mean, a quarter of the 0.0062 floor. This arm will therefore almost certainly read FLAT offline,
# and that is not its refutation.
#
# THE BET IS NOT THE HOLDOUT. The metric that decides this competition is over a HIDDEN test set that
# is not our corpus, and every member we have ever fitted has seen exactly one site. A trunk that has
# fitted knees from a second scanner population is the only lever on the table that acts on
# distribution shift rather than on our own 909. That is unmeasurable offline by construction, which
# is why the offline read below is written as a WIRING check and the real read is a board slot.
#
# COST. The control's step time scaled by studies: ~330 s/epoch at 12 slices over 4,406 becomes
# ~385 s over 5,143, so twelve epochs is ~1.3 h. Against it: the only untried axis that is DATA rather
# than a knob, after four consecutive decode/head knobs returned nothing.
#
# DO NOT LET THIS BECOME AN SSL ARM. It is a normal supervised fit. Nothing is pretrained, nothing is
# distilled, no foreign row is scored.
#
# THE FALSIFIABLE PART, written before the run.
#   * ACL rises against the control by more than its own paired interval — the second site reached the
#     fit and taught the trunk something about the one finding it labels. The arm then earns a board
#     slot as a member swap, and the board is what prices the generalisation claim.
#   * ACL flat and the mean flat — the DEFAULT, and the honest reading is "16.7% more studies on one
#     label buys nothing measurable on our own holdout", which does not refute the hidden-test
#     argument and does not support it either. It buys a member that reads a different corpus, which
#     is a diversity axis no seated member has.
#   * The mean FALLS below 0.8463 (control minus the 0.0062 floor) — the foreign block is hurting the
#     eleven findings it is silent on. That would mean the zero slots or the NaN weighting are leaking
#     into the loss, and the HARNESS is the first suspect, not the idea: check that `weights[said]`
#     really zeroed those cells before concluding a second site is harmful.
#   * ACL rises AND the mean falls — the trunk is trading our findings for theirs, which is the
#     fixed-budget shape four decode arms already failed to find. Read it as capacity, not as data.
#
# HARNESS CHECK BEFORE READING ANY OF THAT. The checkpoint config must record `scheme: merged`, and
# the run's own log must say it loaded 5,143 studies and held out 909. Three runs have entered the
# ledger as facts about an architecture when they were facts about a flag that never landed.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
# control  lr4e-5_ws = 0.8525 (declared 2026-09-03 from this header own prose, knee-sgzgc)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

MERGED_LABELS="$PROCESSED/labels/merged_labels.csv"
require "$MERGED_LABELS"

run_train merged_corpus \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --scheme merged \
  --labels "$MERGED_LABELS" --out "$MODELS/merged_corpus_s2026.pt"
