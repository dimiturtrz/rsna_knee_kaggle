#!/usr/bin/env bash
# Replace the per-diagnosis slot attention with a uniform mean over present slots.
#
# THE DENOMINATOR NOBODY MEASURED. Every arm in the ledger was run on a recipe that already contains
# slot attention, so nothing says what the attention itself is worth. That number sets the bar for
# every future pooling idea: a scheme that beats mean-pooling by 0.01 is uninteresting if attention
# already beats it by 0.05, and it is the whole game if attention is worth 0.002.
#
# IT IS ALSO THE PROXY'S SECOND PAIR. `head_axes.jsonl` holds exactly ONE head-side pair the ruler
# resolves (the anatomical prior, -0.0076), so the frozen-encoder harness's correlation gate would run
# on n=1, which is not a correlation. This axis is pure head - the pixels and the encoder are
# untouched - so whatever it scores here can be rerun frozen in seconds and compared.
#
# control  `whole_stack` fold 0 = 0.8457. Single variable: MeanSlotHead only overrides `attention`;
#          the projection, slot embeddings, dropout, readout and parameter count are identical, and
#          `tests/unit/kneemri/slot_head.py` asserts the two compute the same function at constant
#          attention logits. So a delta here is the attention, not a different-sized head.
# risk     none to the recipe - this arm cannot become a member (it is the ablation, not a proposal).
#          The cost is one training slot for a number that prices a mechanism we already ship.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

# 12 epochs to MATCH the control. One-cycle is defined over total_steps, so a shorter budget would vary
# the rate trajectory alongside the pooling and the arm would price two things at once.
# `--slice-pooling max` is the CONTROL's value and is not optional here: $RECIPE sets no slice pooling,
# so the default 'channels' stacks slices before the encoder, which whole-stack refuses outright. Armed
# without it, this arm died on that ValueError in its first smoke — a queue slot to discover.
run_train slot_mean \
  $RECIPE --epochs 12 --slice-pooling max \
  --slot-pooling mean --whole-stack --recompute-activations \
  --labels "$LABELS" --out "$MODELS/slotmean_s2026.pt"
