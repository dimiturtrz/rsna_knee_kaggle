#!/usr/bin/env bash
# warm224_brief.sh — native224's recipe with `--warmup 0.05` and nothing else moved. The first arm to
# move the one-cycle schedule's warmup off the value every ledger number was measured at.
#
# bead: knee-s9cr
#
# THE CONSTANT AND WHY IT IS SUSPECT. `ONE_CYCLE_WARMUP = 0.15` was a module constant in `trainer.py`
# with no flag; `one_cycle` hands it to OneCycleLR as `pct_start`, so every one-cycle run this project
# has ever done spent the first 15% of its steps climbing to the peak rate. Over twelve epochs that is
# 1.8 epochs below the rate the recipe was tuned for — on a run whose BEST EPOCH IS 7. Roughly a seventh
# of the budget is spent warming up, and a quarter of the distance to the peak epoch. The flag is new
# code, added with this arm, and it is attached before the schedule is built because `one_cycle` reads
# the fraction once at construction.
#
# THE MECHANISM. Warmup exists to stop a large early step wrecking a pretrained representation before
# the optimiser's second-moment estimates have settled — a real risk that is proportional to how far the
# encoder is being asked to move. Two things about this recipe bound that risk. The head and the trunk
# are on separate groups at a 125:1 ratio, so the trunk's rate is 4e-5 while the schedule's peak danger
# is the head's; and only the last blocks are unfrozen, so the layers a bad early step could damage most
# are not moving at all. Against that, twelve epochs is a SHORT budget: the anneal is what the one-cycle
# family buys, and a shorter warmup gives the anneal more of the run — the peak epoch moves earlier and
# more of the budget is spent at rates that actually fit.
#
# AND I PREDICT MORE EFFECT AT 224, WHICH IS WHY THIS IS A 224 ARM. The argument is about what there is
# to learn. At 224 each token averages ~2.25x more tissue, so there is less in-plane structure to fit
# and the run converges on what it CAN learn sooner — native224 peaks at 7 of 12 while the dense members
# generally hold on longer. A fixed 15% warmup is therefore a larger share of the useful part of a 224
# run than of a dense one, and shortening it recovers a bigger fraction of the budget here than it would
# at 336. The competing reading — that a coarser input gives noisier early gradients, so warmup matters
# MORE — predicts a loss, and it is the reading to take seriously if this arm regresses.
#
# WHY 0.05 AND NOT 0. Zero warmup is not the same experiment: OneCycleLR with `pct_start=0` starts at
# the peak and anneals monotonically, which is a different schedule family (cosine-from-peak), not a
# shorter version of this one. 0.05 is 0.6 epochs — long enough that the moment estimates settle,
# short enough to be a 3x cut and clear the noise floor. `warm224_long.sh` takes the other side at 0.30.
#
# control  native224 = 0.8502, best at epoch 7, converged not truncated. 12 epochs, 12 slices, 0.20-0.80
#          band, whole-stack, backbone 4e-5, 224 px, same labels, same 909-study holdout, warmup 0.15.
#          The single variable is --warmup.
# cost     ~19 min for twelve epochs at 94 s. Same cache as native224; a schedule shape is free.
# price    19 min for P(clears the 0.0062 floor on the MEAN) = 0.15. The mechanism is sound but the
#          quantity is small — it redistributes about one epoch of a twelve-epoch budget — and this
#          project's floor is set against effects of that size. Its second value is that the answer
#          transfers to every one-cycle arm in the ledger, which is all of them.
#
# HARNESS CHECK BEFORE READING ANY NUMBER. The printed recipe must say "warmup": "0.05", "one_cycle":
# "True" and "image_size": "224", 4,406 studies loaded and 909 held out. THE ONE-CYCLE FLAG IS PART OF
# THE CHECK: without a schedule the warmup is inert and the run is a byte-identical control, so a
# missing `--one-cycle` in the recipe line would make this arm unreadable rather than null. `$RECIPE`
# carries it; verify it survived. The landed check compares `one_cycle_warmup` against the flag.
#
# THE FALSIFIABLE PART, written before the run.
#   CONFIRMED   the mean peak clears 0.8564 (control + floor). The inherited warmup was costing a
#               twelve-epoch run real budget; every one-cycle arm in the ledger — which is every arm —
#               was measured under it, and the fraction becomes a recipe-level change rather than a
#               knob. Check the peak EPOCH moved earlier too; if the score rose while the peak stayed
#               at 7, the story above is wrong even though the number is right.
#   DIRECTIONAL positive under the floor. Kept as free and mechanism-sound; folded into the 224 recipe,
#               not planned around, not carried to 336 without its own reading.
#   FLAT        the schedule shape is insensitive here, which retires a suspicion cheaply and says the
#               twelve-epoch budget is not warmup-bound. Read with warm224_long before closing the axis.
#   REFUTED     the mean falls by more than the floor. Early large steps do damage a pretrained trunk
#               even at 4e-5 on the last blocks — and that would be the first evidence this project has
#               that the encoder is rate-sensitive at the START of a run rather than only in aggregate,
#               which points straight back at the rate sweep's own warmup assumptions.
#
# deliberate: --warmup 0.05 (the arm's thesis — 1.8 epochs of climbing on a run that peaks at 7)
# deliberate: --image-size 224 (native224's control, per that row's standing instruction)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train warm224_brief \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 --warmup 0.05 \
  --labels "$LABELS" --out "$MODELS/warm224_brief_s2026.pt"
