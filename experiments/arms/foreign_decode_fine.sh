#!/usr/bin/env bash
# foreign_decode_fine.sh — the coarse sweep placed a PEAK, not a plateau; this asks where it actually is.
#
# WHAT THE COARSE SWEEP MEASURED. Four bands, fold-matched pooled AUC on the 909 holdout studies:
#
#     0.08-0.92   0.8582
#     0.12-0.88   0.8812   <- the band their released kernel serves, and the one entry 56060097 ships
#     0.16-0.84   0.8601
#     0.20-0.80   0.8492   <- their own TRAINING band
#
# Moving the band 0.04 in either direction costs 0.02 AUC. Three points cannot place a maximum that
# steep, so 0.12-0.88 is the best of four rather than the argmax, and the arm that shipped is reading
# through a dial nobody has resolved.
#
# WHY THIS IS A SLICE-POSITION QUESTION AND NOT A COVERAGE ONE. Coverage is monotone in the band width
# and the readings are not: 0.84 of the stack (0.08-0.92) scores BELOW 0.68 of it (0.16-0.84). What is
# not monotone is where sixteen slices spread over the band actually LAND — a band shift moves every
# slice, so the axis being swept is the sampling grid, not the anatomy included. The count that would
# separate the two cannot be varied: `N_SLICE` is a constructor argument of their `Net` read from the
# checkpoint cfg, so a served count other than 16 builds a model their state dict does not fit. Band is
# the ONLY handle this family has on its own sampling grid, which is why the grid is refined instead.
#
# WHAT IS SWEPT. Four points at 0.01 either side of the peak: 0.10-0.90, 0.11-0.89, 0.13-0.87, 0.14-0.86.
# The centre (0.12-0.88) is on disk as `foreign_dinov3_matched` and is not re-served.
#
# HOW IT IS READ, written before the number. Pre-registered by mastermind at 18:18Z:
#   * any band >= 0.8862 fold-matched (+0.005 solo over the shipped one) is priced as a 6x seat on the
#     served base of 8; >= +0.005 seat delta there revives the decode branch for the 23:30Z slot, and its
#     kernel must be pushed by 21:00Z to make census by 22:30Z.
#   * nothing >= 0.8862 — the decode axis closes for good, the shipped band stands, and 23:30Z is C.
#
# COST. ~12 min of card, 909 studies x five folds x four bands. ZERO submission slots.
#
# deliberate: --band (the only moving variable; their lifted program is otherwise untouched)
# deliberate: four points at 0.01 (the coarse grid's 0.04 step is wider than the peak it found)
# reading: foreign_dinov3_band1090.holdout.npz
# reading: foreign_dinov3_band1189.holdout.npz
# reading: foreign_dinov3_band1387.holdout.npz
# reading: foreign_dinov3_band1486.holdout.npz
# control  foreign_dinov3_matched = 0.8812 fold-matched pooled, the shipped band.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

FOREIGN_KERNEL="$REPO/external/dinov3folds/bend_the_knee_to_dinov3_ensembled.py"
FOREIGN_WEIGHTS="D:/data/volumetric/mri/models/foreign/kaggle_foreign/mattiaangeli_dinov3"
RAW="$(dirname "$PROCESSED")/../raw/rsna_knee_abnormality"
# HOLDOUT is the arm's to set, not lib.sh's — `foreign_decode_sweep.sh:41` sets the same file, and this
# arm inherited its `require` line without it. Under `set -u` that is an unbound variable, and the arm dies
# at the require BEFORE the first band, on a line that reads like a missing file. It cost the whole fine
# sweep on 2026-09-06T19:23Z: the chain reported one line and went straight on to `foreign_dissent`.
HOLDOUT="$MODELS/band_dense_s2026.holdout.npz"

require "$LABELS"
require "$HOLDOUT"

source "$(dirname "${ARM_LIB:-$(dirname "$0")/../lib.sh}")/foreign_sweep_lib.sh"

sweep_band 0.10 0.90 band1090
sweep_band 0.11 0.89 band1189
sweep_band 0.13 0.87 band1387
sweep_band 0.14 0.86 band1486
echo "ARM DONE foreign_decode_fine $(date -u +%FT%TZ)"
