#!/usr/bin/env bash
# The slice band, 0.20-0.80 -> 0.10-0.90, at CONSTANT slice spacing.
#
# THE ONE READING RULE NEVER VARIED. `slot_reader.SLICE_BAND = (0.20, 0.80)` carried one line of
# justification and no number, and knee-nas5 found `band_low`/`band_high` in none of the six all-ours
# members' recorded options: every member on disk was fitted under it. band_coverage (2026-08-17, 400
# studies, 1848 series) priced what it removes — 20.8 mm cut from EACH end of a 103.8 mm sagittal span,
# against the 8-15 mm bar knee-1v0n wrote down before the measurement as "mechanism live".
#
# WHY 16 SLICES AND NOT 12. Widening the band 0.6 -> 0.8 of the series while holding the cached count
# would thin the spacing by a third, which changes two things at once. 12 -> 16 over a 1.33x wider span
# holds mm-per-slice where the control has it, so the ONLY variable is how much of the series is seen.
#
# WHY THE 16-SLICE COUNT IS NOT ITSELF THE CONFOUND. `slices16_ws` already priced the count under this
# recipe INSIDE the old band: 0.8452 against the control's 0.8457, -0.0005. Slice count alone is flat,
# so a move here belongs to the coverage.
#
# WHAT IT DOES NOT SETTLE. band_coverage could not separate discarded TISSUE from discarded JOINT — the
# outer slices hold thigh and calf as well as whatever compartment sits in the outer 20 mm. This arm is
# the only thing that can, because it asks the model rather than the pixels.
#
# control  rate_wholestack = 0.8525 (336 px, 12 slices, 0.20-0.80, whole-stack, 4e-5) — the best member
#          this project has produced, and the recipe every flag below matches except the band.
# cost     ~1.35x the control's step time (96 images a study against 72); 12 epochs.
#
# PREDICTION, written before the run: the band carries joint anatomy the model can use -> above 0.854,
# and the gain lands in the labels whose compartments sit off-centre (Lateral/Medial Meniscus, Lateral
# OA), not uniformly. A uniform lift instead would read as more augmentation-like regularisation than
# anatomy, and a flat result retires the last live version of "we discard data" in knee-jk3.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train band_wide \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 16 --band-low 0.1 --band-high 0.9 \
  --labels "$LABELS" --out "$MODELS/band_wide_s2026.pt"
