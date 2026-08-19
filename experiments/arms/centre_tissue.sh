#!/usr/bin/env bash
# Keep 130 mm, move where it is taken about: the tissue, not the image centre. One variable (knee-yrf).
#
# WHAT IT ISOLATES. crop80_tissue.sh changes the extent AND the centre together, because at 80 mm the
# image centre is not survivable. That coupling is deliberate but it costs interpretability: if that arm
# fails, the tight crop and the relocated centre are both suspects. This arm holds the extent at the
# value every existing member uses, so the ONLY difference against the 130 mm baseline is the centre.
#
# WHAT IT WOULD SHOW, and it is a smaller claim than the tight crop's. Measured on 60 studies, the
# tissue centroid sits a median 7 mm and p90 18 mm from the image centre, against ~15 mm of margin at
# this extent. So most studies are unaffected and only the worst tenth is cropped differently at all —
# a whole-corpus gain here would be surprising, and the honest expectation is roughly nothing.
#
# WHICH IS WHY IT IS WORTH RUNNING ANYWAY, cheaply: a NULL here is what licenses reading crop80_tissue's
# result as a resolution result rather than a centring one. A null that is measured is worth more than
# an assumption that is not, and this is the cheapest place to buy it.
#
# THE LOCATOR IS CRUDE AND THAT IS RECORDED. TissueCentre takes the centroid of an Otsu-thresholded
# slice, medianed over the stack. For a sagittal series the leg spans the whole superior-inferior
# extent, so that axis moves little and the anterior-posterior estimate carries most of the correction.
# A real joint locator would use the intercondylar geometry; this one bounds the question rather than
# answering it.
#
# control  the existing 130 mm image-centred member on the same fold, labels and epochs.
#
# NEEDS ITS OWN CACHE:
#   experiments/arms/build_cache.sh --image-size 336 --slices 12 --centre tissue
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train centre_tissue \
  $RECIPE --epochs 8 \
  --slice-pooling max --whole-stack --recompute-activations \
  --centre tissue \
  --labels "$LABELS" --out "$PROCESSED/centre_tissue_s2026.pt"
