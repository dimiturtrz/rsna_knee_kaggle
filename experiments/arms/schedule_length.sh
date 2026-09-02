#!/usr/bin/env bash
# schedule_length.sh [epochs] — is a 12-epoch run buying anything after epoch 6?
#
# THE EVIDENCE THAT PROMPTS IT, from the arms already on disk:
#   whole_stack       best epoch  6 = 0.8457   final 0.8394
#   resolution_392    best epoch  6 = 0.8523   final 0.8504
#   slice_attention   best epoch  6 = 0.8473   final 0.8454
#   intensity_392     best epoch  9 = 0.8459   final 0.8448
# Three of four peak at SIX of twelve and then decline. We save the best checkpoint, so nothing is lost
# in score - what is lost is half an hour of card time per arm, every arm, spent producing a model we
# then throw away.
#
# WHY NOT EARLY STOPPING, which is the obvious answer and the wrong one here. OneCycle ties the whole
# learning-rate trajectory to the epoch count: stopping at a plateau abandons the anneal that the last
# epochs exist to perform, so the run that stops early is NOT the run that trained for that many epochs.
# Shortening the SCHEDULE is a different model, and the only honest way to ask the question.
#
# THE PREDICTION, stated before the number: a shorter schedule anneals faster, so it should reach a
# comparable peak EARLIER rather than a higher one. If it lands within the 0.0062 noise floor of 0.8457
# it is free speed on every arm after it; if it lands clearly below, then those last six epochs are
# doing something the epoch-by-epoch holdout curve does not show, and 12 stays.
#
# control  wsfold0_e6 = 0.8457 - the same recipe at 12 epochs, whose per-epoch curve is above.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
# twin-of: whole_stack
# RUN BARE THIS ARM IS `whole_stack`. EPOCHS is an argument and its default reproduces that arm's
# flag line exactly, so a queue entry spelled `experiments/arms/schedule_length.sh` with no argument
# spends a card slot on a run already on disk. Queue it as `schedule_length.sh <n>` and check that
# spelling with `arm_check --as-queued`, which is the only spelling worth checking.
#
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
EPOCHS="${1:-8}"
require "$LABELS"

# COMPILED, unlike the control it is quoted against - and that is measured, not assumed: one epoch each
# way on this recipe gave 0.7600 eager against 0.7606 compiled, +0.0006 on a 0.0062 floor, while the
# epoch fell from 365 s to 272 s. Leaving it off would make the arm ask its question 25% slower for a
# difference the ledger cannot see.
run_train "schedule_${EPOCHS}" \
  $RECIPE --epochs "$EPOCHS" \
  --slice-pooling max --whole-stack \
  $THROUGHPUT \
  --labels "$LABELS" --out "$MODELS/schedule_${EPOCHS}_s2026.pt"
