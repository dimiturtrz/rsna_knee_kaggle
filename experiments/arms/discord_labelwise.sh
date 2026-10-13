#!/usr/bin/env bash
# bead: knee-5vilf
# reading: labelwise_dense_drop40_long_discord_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   `labelwise_dense_drop40_long`'s byte-identical line with ONE input changed: the training table is
#   `discordant_masked.csv` instead of `report_labels_v2.csv`. That table is v2 with the 13.94% of cells
#   (7369 of 52872) where four of our extractor tables disagree set to UNK 0.28 at conf 0.05, so they
#   train at ~1/19 of a confident cell's weight. Everything else, the eval ruler included, is held.
#
# WHY NOW
#   The leaders' stated lever (forum, 745861: keep only high-agreement LLM labels, per-class state
#   filtering) is label SELECTION. Ours is `discord_mask`, run once on the 0.85-era band_span recipe
#   (see its ledger section). It was never run on the labelwise line our seats use.
#
# PRE-REGISTRATION (delta = this arm minus labelwise_dense_drop40_long_s2026 0.9048, ledger75)
#   >= +0.0040   seat it in place of its twin in the next entry's soup share and spend a slot.
#   -0.0039..+0.0039
#                read gold58 (annotated_readings) before deciding; a gold lead with a flat ledger is
#                the discord_mask signature, since the ledger's own truth IS the v2 table.
#   <= -0.0040   selection by our four-way disagreement does not transfer to the shipping recipe.
#   peak         on its last epoch: no verdict, record it.
#
# deliberate: --eval-labels pinned to $LABELS — the masked table must not move the ruler.
# deliberate: every other flag inherited verbatim from labelwise_dense_drop40_long.
#
# control  labelwise_dense_drop40_long_s2026 = 0.9048 ledger75 (on disk). No control run is owed.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

MASKED="$PROCESSED/labels/discordant_masked.csv"
require "$MASKED"
require "$LABELS"

run_train labelwise_dense_drop40_long_discord \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --slot-dropout 0.40 \
  --labels "$MASKED" --eval-labels "$LABELS" \
  --out "$MODELS/labelwise_dense_drop40_long_discord_s2026.pt"
