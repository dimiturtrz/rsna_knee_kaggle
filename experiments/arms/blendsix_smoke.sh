#!/usr/bin/env bash
# blendsix_smoke.sh — the two minutes of card that gate `blendsix.sh`'s seven hours.
#
# bead: knee-4i3m
#
# blendsix carries THREE flags on every one of its five members (`--scheme merged`, the blend label
# table, `--eval-labels`) and a fold flag that removes the holdout which would otherwise notice a
# mistake. This runs ONE member of it at sixty studies and one epoch, so a typo costs two minutes
# instead of a queue slot and seven hours of card.
#
# WHAT TO READ. Not a score — sixty studies score nothing. The printed recipe must name `merged` on
# "scheme", `merged_blend_v1.csv` on "labels", `336` on "image_size" and `-1` on "fold", and the run must
# reach `ARM DONE blendsix_smoke` with a non-zero throughput line. Anything else and blendsix does not
# get queued.
#
# THE OUTPUT NAME IS DELIBERATELY ITS OWN. `_smoke` is not a member and must never be seatable; writing
# to any name blendsix uses would leave a sixty-study checkpoint sitting where a twelve-epoch one is
# expected, which is exactly the failure `final_fit_six.sh` names in its NEW NAMES paragraph.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

MERGED_BLEND="$PROCESSED/labels/merged_blend_v1.csv"
require "$MERGED_BLEND"
require "$LABELS"

run_train blendsix_smoke \
  $RECIPE --fold -1 --ema --studies 60 --epochs 1 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --scheme merged --labels "$MERGED_BLEND" --eval-labels "$LABELS" \
  --out "$PROCESSED/blendsix_smoke.pt"
