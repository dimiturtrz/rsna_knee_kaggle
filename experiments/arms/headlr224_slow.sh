#!/usr/bin/env bash
# headlr224_slow.sh — native224's recipe with `--head-lr 2e-4` and nothing else moved. The SLOW point of
# the dose pair whose fast point is `headlr224.sh` at 5e-3. Read the two together.
#
# bead: knee-s9cr
#
# WHY A PAIR AND NOT A BETTER-ARGUED SINGLE POINT. Every result in this ledger that died died as a single
# number inside the 0.0062 floor. The head/backbone ratio has NO prior measurement at all — see
# `headlr224.sh` for the audit trail (`switch_coverage.py:50-57` carries `head_lr` as a switch no arm
# selects; the one arm that names it, `headlr_dense.sh`, has never run) — so there is no way to guess
# which side of the default the operating point sits on, and a one-sided probe that reads flat cannot
# distinguish "the axis is flat" from "I moved the knob the wrong way". Two points spanning 25x, both
# against the same control, can: monotone response is signal neither point carries alone, and a flat
# pair CLOSES the ratio rather than deferring it. The pair costs 38 minutes at 224, which is less than
# the single 336 point `headlr_dense.sh` was written to spend.
#
# THE MECHANISM FOR THIS SIDE SPECIFICALLY. `Adaptation`'s docstring
# (`kneemri/training/trainer.py:55-69`) argues the head should be FAST because it is random at
# initialisation. That argument is about the first epoch or two. It says nothing about the remaining ten,
# and the failure it cannot see is the mirror of the one it prevents: a readout stepping 25x faster than
# the features under it converges onto whatever those features happen to say EARLY and then holds that
# fit while the trunk moves out from under it. The slot head is not tiny — `_HIDDEN` with a learned slot
# embedding, a per-diagnosis query and a report projection (`kneemri/models/slot_head.py:47-70`) — and it
# carries `_DROPOUT = 0.2` (`slot_head.py:24`, itself a never-varied hardcoded constant with no flag). A
# head that fits fast and is then regularised hard is exactly the configuration where slowing the readout
# lets the trunk lead. `whole-stack` at 4e-5 is the first recipe in this ledger where the trunk actually
# moves, which is what makes the question live now and not before.
#
# WHY THIS IS NOT COVERED BY THE 224 CLOSURE. Same argument as the fast point: `mixup224_long`
# (README:4178-4215) closed the axes that "perturb how the SAME supervision is presented to the SAME
# encoder". This changes neither, only the optimiser's path.
#
# control  native224 = 0.8502, best at epoch 7, converged not truncated. Identical to the fast point's
#          control and to the fast point itself in every flag but this one, so the two arms are directly
#          comparable to each other as well as to the control.
# cost     ~19 min for twelve epochs at 94 s, ~8124 MiB reserved.
# price    19 min for P(clears the floor) = 0.15 — half the fast point's, because the inherited design
#          argument points the other way and this side is the one being probed for completeness. It earns
#          its slot as the half that makes a FLAT reading conclusive; alone it would be a coin flip on a
#          knob with no prior, which is not worth a card.
#
# HARNESS CHECK BEFORE READING ANY NUMBER. The printed recipe must say "head_lr": 0.0002 and
# "image_size": "224", and the log must load 4,406 studies and hold out 909. A flag that never landed
# reproduces the control exactly; a delta of 0.0000 is a plumbing verdict, not a null.
#
# THE FALSIFIABLE PART, written before the run, READ AS A PAIR with headlr224.
#   CONFIRMED   the peak clears 0.8564 AND the fast point does not. That inverts the inherited design
#               argument at the settled trunk rate and is the more interesting of the two directions:
#               it would say the 125:1 ratio is an artefact of a frozen-trunk recipe and that every
#               whole-stack member in the roster is training its readout too fast.
#   DIRECTIONAL positive under the floor, with the fast point at or below the control. Kept, not planned
#               around, no confirmation seed.
#   REFUTED     the peak falls by more than the floor — a head that cannot keep up with its own trunk,
#               which is the failure the inherited default exists to prevent and would confirm it.
#   BOTH POINTS FLAT is the modal outcome and closes the ratio by measurement across a 25x span. That is
#               the reading the schedule wants: it retires an entire never-measured optimiser key for
#               38 minutes of a night that has ~33 arm slots at this resolution.
#
# deliberate: --head-lr 2e-4 (the arm's thesis — the slow half of the dose pair)
# deliberate: --image-size 224 (native224's control, per that row's standing instruction)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train headlr224_slow \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 --head-lr 2e-4 \
  --labels "$LABELS" --out "$PROCESSED/headlr224_slow_s2026.pt"
