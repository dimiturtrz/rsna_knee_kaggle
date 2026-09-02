#!/usr/bin/env bash
# bead: knee-n3yx
#
# band_dense's recipe with --crop-mm 140 in place of the shipped 130, and NOTHING else. One variable.
#
# WHY THE CROP MOVES OUTWARD AND NOT INWARD. Every crop arm the project has run tightened the field:
# crop80_tissue (80 mm) cost -0.0112 on the mean against its control, and centre_tissue since separated
# that loss into the resolution change rather than the centring. So the axis has been measured in one
# direction only, and the direction it was measured in is the one that throws tissue away. Corpus-wide
# over 1822 series (core.data.corpus.crop_coverage --sample 400 --tight-mm 140), the shipped 130 mm window
# discards a MEDIAN 15% of imaged tissue -- p10 27%, worst 55% -- and the discarded ring is the
# periphery: the medial margin where MCL lives, the posterior where Baker's lives, the intercondylar
# where ACL crosses.
#
# WHY 140 AND NOT 150. SlotReader._crop does not pad: it returns the stack untouched when the window
# does not fit, so a crop wider than the series is honoured only on the subset wide enough and silently
# breaks the constant-mm-per-pixel invariant for the rest. Served share is 130mm 1822/1822 (100%),
# 140mm 1718/1822 (94.3%), 150mm 1489/1822 (81.7%), 170mm 358/1822 (19.6%). 140 buys +5.6pp median
# tissue for 5.7% of series falling back to native extent; 150 buys a little more and breaks 18%.
# Padding is what would make 150 readable, and it is a separate change (knee-7zw7).
#
# PREDICTION, written before the run. The mean is NOT where this arm's value is and the prediction says
# so: 0.8520-0.8620, i.e. band_dense's 0.8578 plus or minus rather more than the 0.0062 floor. The
# clause that decides the arm is PER-LABEL and it is the mirror of crop80_span's measured gradient --
# that arm tightened the crop and cost MCL -0.0265*, ACL -0.0263*, Baker's -0.0164*, each clearing its
# own interval. If field of view is what those labels were paying for, the same three go POSITIVE here
# against band_dense and the seat value against comp/champion.json is positive with a flat mean, which
# is what a member on an unoccupied decode group is for.
#
# THE REFUTATION CONDITION is that periphery group coming out flat. That would say crop80_span's losses
# were the RESOLUTION the tighter window spent on less anatomy and not the field it discarded -- the two
# being confounded in every crop arm run so far, because tightening the window changes both at 336 px.
# A flat periphery here separates them and retires the outward direction of this axis, which is worth
# the slot on its own.
#
# NO --vram-gb: DeviceBudget derives the cap from --slices (20 slices reserve 28.26 GB, which band_dense
# confirms). --crop-mm changes the physical extent the 336 px covers, not the activations.
#
# cost 12 epochs at ~250 s -- about 52 minutes. The cache is already built (34m59s, 2026-08-21), so the
# card time is the whole price.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train band140 \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --crop-mm 140 \
  --labels "$LABELS" --out "$MODELS/band140_s2026.pt"
