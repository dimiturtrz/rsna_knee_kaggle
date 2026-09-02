#!/usr/bin/env bash
# regions224.sh — native224's recipe with `--slice-regions 4` and nothing else moved. The third of four
# arms transporting the AGGREGATION family onto the 224 control, and the one I expect to LOSE amplitude.
#
# bead: knee-7d53
#
# WHY THE FAMILY. `experiments/README.md:5024-5075`: every arm that changed how slices are AGGREGATED
# gained on the six model-bottlenecked findings; `depth_regions` is the third of the four (+0.0016 focal,
# +0.0048 regional, +0.0024 mean). All four ran at 336 px and 20 slices and none has run at 224.
#
# THE MECHANISM. The control takes ONE elementwise maximum over the whole 0.20-0.80 band, so a study's
# vector is a single summary of twelve slices and the head cannot know whether two responses came from
# the same depth or from opposite ends of the knee. `--slice-regions 4` cuts the band into four contiguous
# depth quarters and takes one maximum per quarter, so anterior and posterior evidence arrive as separate
# terms. It costs no parameters and no extra encoder pass — the same twelve slices are encoded and the
# reduction is regrouped — and at one region it is the historic pooling bit for bit.
#
# THE RESOLUTION ARGUMENT, AND HERE I PREDICT LESS. The honest reason is that the axis this arm acts on is
# NOT the axis 224 changed. Regions partition depth; the downsample coarsens the in-plane grid. What
# actually moves between `depth_regions` and this arm is the SLICE BUDGET: 20 slices over a 0.1-0.9 band
# gave five slices per quarter, and native224's 12 slices over 0.2-0.8 give three. The gain regions
# recover is the dilution a whole-band max imposes, and that dilution scales with how many slices are
# pooled together — twelve dilute less than twenty, so there is LESS to recover here before the split even
# starts. Against that, three slices per region is a thin maximum: with the per-slice features already
# noisier at 224, a max over three draws is a noisier estimate than a max over five, and four such terms
# concatenated give the head four noisy numbers where it had one steadier one. Both readings point the
# same way, which is why this is the weakest-priced of the four transports and why it is still worth
# running: +0.0016 focal at 336 was the smallest positive in the table, and a flat 224 reading would say
# the region mechanism is a function of slice count rather than of resolution — a claim the ledger
# currently cannot make in either direction.
#
# NOT COMBINED WITH --slice-pooling attend, AND THE REASON IS A DEFECT. `model.py:322` returns the attend
# path and never reaches `over_depth`, so `--slice-regions` beside `attend` is silently ignored — no
# error, no log line, a recorded region count that was never used (`experiments/README.md:5075`). This arm
# keeps the control's `max` and `attend224.sh` keeps the default region count.
#
# control  native224 = 0.8502, best at epoch 7, converged not truncated. 12 epochs, 12 slices, 0.20-0.80
#          band, whole-stack, backbone 4e-5, 224 px, same labels, same 909-study holdout, one region.
#          The single variable is --slice-regions.
# cost     ~19 min for twelve epochs at 94 s. Same cache as native224; the regrouped reduction is free.
# price    19 min for P(clears the 0.0062 floor on the MEAN) = 0.10, the lowest of the four transports:
#          +0.0024 on the mean at 336 with five slices a region, and this arm has three. It is bought as
#          an INSTRUMENT — it separates "regions help because slices dilute" from "regions help at 336"
#          — and it should be read as one, not sold as an improvement.
#
# HARNESS CHECK BEFORE READING ANY NUMBER. The printed recipe must say "slice_regions": 4 and
# "image_size": "224", and the log must load 4,406 studies and hold out 909. Four regions widen the head's
# input by 4x; a parameter count equal to native224's means the flag did not land, and the delta is then a
# PLUMBING verdict rather than a null. 12 divides by 4 exactly, so no region is short.
#
# THE FALSIFIABLE PART, written before the run.
#   CONFIRMED   the mean peak clears 0.8564 (control + floor). Depth regions survive at 224 with THREE
#               slices a region, which would mean the mechanism is about ordering the depth axis rather
#               than about undoing dilution — and that reopens the region count as its own dose axis.
#   DIRECTIONAL positive under the floor. Kept as free and mechanism-sound; feeds `fuse_regions`.
#   FLAT, WHICH IS THE MODAL OUTCOME AND THE ONE THIS ARM IS PRICED FOR. A flat reading at 224/12 against
#               +0.0016 at 336/20 attributes the region gain to the slice budget rather than to the
#               pixels, and closes regions for every 12-slice member the roster ships.
#   REFUTED     the mean falls by more than the floor. Four maxima over three noisy slices each is a
#               worse summary than one over twelve; `depth_regions_span` already read −0.0052 when the
#               band widened, so this mechanism is measured to be sensitive to how the regions are cut.
#
# deliberate: --slice-regions 4 (the arm's thesis — one maximum per depth quarter, not one per band)
# deliberate: --image-size 224 (native224's control, per that row's standing instruction)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train regions224 \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 --slice-regions 4 \
  --labels "$LABELS" --out "$MODELS/regions224_s2026.pt"
