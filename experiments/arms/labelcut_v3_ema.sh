#!/usr/bin/env bash
# bead: knee-j48lj
# reading: labelcut_v3_ema_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   `labelwise_ext_drop40_long_ema.sh`'s line with ONE byte changed: --labels points at the FP-only cut
#   `report_labels_v3.csv` instead of v2. The ruler stays v2 through --eval-labels, so the contrast is
#   the TRAINING table and nothing else. Its control is that arm's own 0.9073 ledger75, already on disk.
#
# WHY IT IS AN INSTRUMENT AND NOT AN IMPROVEMENT
#   The closed-set sweep left thirteen lever searches with zero hits, and exactly one cell in it is
#   UNEXPLAINED rather than measured-null: v2 over-calls one-sided 3.4:1, v3 is the cut that fixes it
#   (error 0.2047 -> 0.1696, false positives 108 -> 47), and the axis has never been read cleanly.
#   Five v3 members exist. Not one is a matched contrast -- `matched_axes --key labels` resolves only
#   v2 vs negpull (0.9053 vs 0.9044) and places no v3 member in any group, and two of the five v3 reads
#   are TRUNCATED floors (recal_labelwise_s2026 0.8392, recal_through_s2026 0.7712) that may not be
#   quoted at all. So the cell is open because nobody ever ran the single-variable arm, not because it
#   was run and read null.
#
# PRE-REGISTERED, against the twin's 0.9073 ledger75, fold 0, 909 studies
#   >= +0.0028   the cut clears the solo bar on its own and the label table becomes a lever, not a cell.
#                Still not a ship -- see the contamination line below.
#   +0.0010 to +0.0028   directional. Keep the cut where it is free, do not plan around it.
#   -0.0010 to +0.0010   the cell closes null. Fourteen lever searches, zero hits, and the label table
#                stops being a place to look.
#   <= -0.0010   the cut costs. v2's over-call is carrying signal the cut throws away, which is itself
#                a reading worth having about every future table.
#
# THE POSITIVE BRANCH IS CONTAMINATED AND MUST BE SAID OUT LOUD WHEREVER IT IS QUOTED.
#   v3 was FITTED on the 58 gold studies (experiments/README.md:14538). Those studies are inside the
#   fold-0 holdout, so a positive read here is partly the fit reading itself back. That is why even the
#   >= +0.0028 branch licenses a FOLLOW-UP on a fold v3 never saw, never a submission.
#
# ZERO SLOTS. Fold 0, ledger75, never the board.
#
# deliberate: every flag is `labelwise_ext_drop40_long_ema.sh:51-56` byte for byte except --labels.
#             That arm's own 0.9073 is the control, so a second moving flag would make it a similar
#             run rather than a matched one and there would be nothing left to compare against.
# deliberate: --eval-labels "$LABELS". Without it the run is SCORED on v3 too and measures agreement
#             with its own training table instead of with the readings -- the one way this arm returns
#             a number that means nothing. Same pin as transduce_dmlb.sh.
# deliberate: --vram-gb 11, against the twin's uncapped `peak memory 17.28 GB reserved`.
#             The first version of this arm carried NO cap, copying the twin, on the reasoning that a
#             binding ceiling changes allocator behaviour. That reasoning was right about BINDING and
#             wrong about ABSENT: with no cap the caching allocator grows into whatever is free and
#             never releases it, so an uncapped arm on a shared card claims the CARD rather than its
#             recipe's share -- and its "peak reserved" then records what was available, not what was
#             needed. Quoting 17.28 as the requirement was quoting that greed back as a spec.
#             `arm_check` prices the real figure off --slices at ~9.6 GB; 11 leaves ~1.4 GB of headroom
#             and is under a third of the card. Measured 2026-09-21: our uncapped 16.5 GB vanished on
#             kill and the device total did not move, because two peers absorbed it within seconds.
# deliberate: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Fragmentation headroom is the other
#             half of the same greed -- it is reserve the run holds and never uses. Numerics are
#             unchanged, so the contrast against the twin survives it.
#
# CARD. 24 epochs x 206 s measured on the twin's own log = ~1.4 h, plus ~2 min to price.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

CUT="$PROCESSED/labels/report_labels_v3.csv"
require "$CUT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_train labelcut_v3_ema \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT --vram-gb 11 \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --scheme extended \
  --calibrated-targets --hold-annotations --ema \
  --batch 4 --accumulate 3 --slot-dropout 0.40 \
  --labels "$CUT" --eval-labels "$LABELS" --out "$MODELS/labelcut_v3_ema_s2026.pt"

ARMPY="uv run --no-sync --project $REPO python"

$ARMPY -m kneemri.evaluation.ledger75 --holdouts "$MODELS" \
  --only labelcut_v3_ema_s2026 labelwise_ext_drop40_long_ema_s2026
