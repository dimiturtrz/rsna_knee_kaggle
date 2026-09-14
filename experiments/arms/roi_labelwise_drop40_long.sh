#!/usr/bin/env bash
# bead: knee-vgz5b.5
# reading: roi_labelwise_drop40_long_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   `labelwise_drop_drop40_long`'s line with `--scheme roi` added: the six canonical views kept whole,
#   plus the three sagittal stacks again, cropped to the joint the SKM-TEA tissue locator drew
#   (`python -m kneemri.embeddings.roi_cache`, locator_sag.pt) and upsampled back to 336 px, ~1.5x zoom.
#   A blank box leaves its ROI slot absent in the mask. Coronal and axial get no ROI view.
#
# OFFLINE ONLY until knee-vgz5b.7 rules on SKM-TEA's licence: the crop placement is SKM-TEA-derived, so
# this member may not ship to Kaggle.
#
# PRE-REGISTRATION — solo, LEDGER75, peak-to-peak
#   control  labelwise_drop_drop40_long_s2026 = 0.9051 ledger75 (e18, 909 rows, on disk).
#   >= 0.9201  (+0.015) clears the bar: zoom on the located joint is a real view.
#   +0.005..+0.015  directional; worth a seat test on the ensemble, never a finding.
#   <= +0.005  more pixels of the same frame add nothing; record which labels moved before closing.
#   precedent  labelwise_ext_drop40_long_s2026, three EXTRA views on this recipe, read 0.9045 — -0.0006.
#   peak       if it peaks on its last epoch the budget binds at 24 and no verdict may be read.
#
# deliberate: --scheme roi — the variable under test.
# deliberate: --batch 4 --accumulate 3 — nine views, imported from labelwise_ext_drop40_long (same slot
#             count), keeping the effective batch at twelve.
# deliberate: every other flag inherited verbatim from labelwise_drop_drop40_long.
#
# PRICE and P(clears) live in the bead, as arm_check reads them.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train roi_labelwise_drop40_long \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --scheme roi \
  --calibrated-targets --hold-annotations \
  --stochastic-depth 0.05 --batch 4 --accumulate 3 --slot-dropout 0.40 \
  --labels "$LABELS" --out "$MODELS/roi_labelwise_drop40_long_s2026.pt"
