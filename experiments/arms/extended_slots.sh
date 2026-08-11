#!/usr/bin/env bash
# Route the three sequences the canonical six slots cannot express. Nine slots instead of six.
#
# THE LOSS, measured over all 4,407 studies by kneemri.slot_ties --lost, one DICOM header per series
# and no pixels: 2,421 of 24,371 series (9.9%) match NO slot predicate, so nothing can ever route them.
# Three sequences are 96% of that:
#
#   Coronal | fluid | no fat-sat   1,139
#   Axial   | T1    | no fat-sat     621
#   Axial   | fluid | no fat-sat     575
#
# WHY THIS IS THE SAME FAMILY AS THE ONE THING THAT WORKED. Every large effect in this project has
# come from giving the model more of the data it was already handed - whole-stack reading was +0.0147
# and its mechanism is exactly that. Every knob INSIDE the design returned ~0.005. This is a data
# recovery, not a knob.
#
# AND IT POINTS AT THE WORST LABEL. The largest discarded group is CORONAL, and MCL is read coronally:
# it scores 0.755 against Baker's 0.896, the weakest of the twelve on an UNWEIGHTED mean. A coronal
# sequence that never reaches the model cannot be read for a medial ligament.
#
# PREDICTION, written before the run: the gain concentrates in MCL and LCL, the two collateral
# ligaments, and is small for the sagittal-read targets (ACL, PCL, menisci). A uniform lift instead
# would say the extra slots act as generic capacity or regularisation rather than as the missing view,
# which is a different finding and a weaker one.
#
# control  wsfold0_e6 = 0.8457, the same whole-stack recipe over six slots.
# cost     a 9-slot cache is 1.5x the 6-slot one (~54 GB) and a step encodes 1.5x the images.
#
# NOT INTEROPERABLE, on purpose: a nine-slot member has a wider slot embedding than the published
# six-slot ones, so it cannot join an ensemble with them. That is a cost this arm pays knowingly, and
# it is cheaper now than it was yesterday - fold and seed ensembling are out, and the entry we are
# moving toward is a single model.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

# The scheme is part of the cache's NAME and therefore part of this gate. The canonical mask already
# exists, so gating without the scheme would pass immediately and train on a cache that is not there.
wait_for_cache 336 12 extended

run_train extended_slots \
  --fold 0 --epochs 12 --one-cycle --weight-floor --rank-weight 1.0 \
  --slice-pooling max --whole-stack --scheme extended \
  --recompute-activations --vram-gb 26 \
  --labels "$LABELS" --out "$PROCESSED/extended_ws_s2026.pt"
