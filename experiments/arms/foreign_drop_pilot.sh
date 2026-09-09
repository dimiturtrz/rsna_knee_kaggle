#!/usr/bin/env bash
# foreign_drop_pilot.sh — validate the shifted path before spending the card on eleven members.
#
# reading: logs/foreign_drop_pilot.log
# deliberate: two members through foreign_readout, to check slot_cost reads ~0.0000
# bead: knee-lm37v
#
# WHY A PILOT AT ALL. `foreign_readout`'s middle number is a plumbing self-check, not a result: with the
# full splice `kept` serves OUR pixels through the SHIFTED code path, so `slot_cost = kept - native` must
# read ~0.0000. Anything else means the shifted slab is broken and `decode_cost` is not about pixels at
# all — the one failure that is otherwise invisible, since a broken slab still yields a plausible score.
# The tool has never been run on this pool, so that assertion has never been observed to hold here.
#
# Two members, one from each side of the question: `aug_long` (an augmentation member, 0.8511) and
# `band_dense_hold` (a non-augmentation member above it, 0.8573). Costs a few minutes of card.
#
# PRE-REGISTERED: if slot_cost is outside +/-0.005 on EITHER member, the full run does not start and the
# defect is the finding. If it reads ~0, the eleven-member arm runs and `decode_cost` is a pixel number.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

python -m kneemri.evaluation.foreign_readout \
  --checkpoints "$MODELS/aug_long_s2026.pt" "$MODELS/band_dense_hold_s2026.pt" \
  --labels "$LABELS" --foreign D:/kaggle/input/raptor-corpus --device cuda
