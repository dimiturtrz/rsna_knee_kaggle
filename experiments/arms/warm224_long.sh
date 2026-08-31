#!/usr/bin/env bash
# warm224_long.sh — native224's recipe with `--warmup 0.30`. The other side of the schedule-shape axis,
# and the arm that turns a guess about direction into a bracket around the inherited 0.15.
#
# bead: knee-s9cr
#
# WHY THE OTHER SIDE. `ONE_CYCLE_WARMUP = 0.15` was a constant with no flag and every ledger number was
# measured under it (see `warm224_brief.sh` for its history; the flag is new code added with these
# arms). A single shorter point can only say "less is better/worse"; it cannot distinguish an inherited
# value that happens to be near-optimal from one whose optimum lies the other way. 0.05 and 0.30 sit
# roughly 3x either side of 0.15, so the three readings bracket it.
#
# THE MECHANISM FOR MORE. Warmup protects a pretrained representation from large steps taken before
# Adam's second-moment estimates have settled, and this recipe has two features that make that risk
# HIGHER than the reference recipe it inherited the constant from. The backbone rate is 4e-5, five times
# what the sweep started at — the encoder now actually moves, which is the whole point of the rate
# result and also the condition under which an early bad step can cost something. And `--whole-stack`
# puts twelve slices of one study through the trunk per sample, so an early batch's gradient is a much
# larger, more correlated push than the three-slice window the original recipe assumed. A longer climb
# is the standard answer to both.
#
# AND I PREDICT LESS EFFECT AT 224 THAN AT 336, which is the opposite of what warm224_brief predicts for
# its direction and is the honest reading of the same fact. At 224 there is less in-plane structure to
# fit, so the run reaches what it can learn sooner (native224 peaks at 7 of 12) and budget spent
# climbing is budget the anneal does not get. A longer warmup costs more here than it would on a dense
# run that holds its peak later. If this arm nevertheless WINS, the protective mechanism is real and it
# outweighs the budget arithmetic — which would be the more interesting outcome of the pair, because it
# says the encoder is fragile early rather than the schedule being roughly right.
#
# WHAT THE PAIR CAN SAY, written before either runs.
#   both flat         the schedule shape is insensitive between 0.05 and 0.30 at twelve epochs; the axis
#                     is closed and the inherited constant is exonerated for every arm in the ledger.
#   brief wins        the budget argument holds; shorten the warmup in the 224 recipe and read the peak
#                     epoch to confirm it moved earlier for the reason claimed.
#   long wins         early steps damage the trunk at 4e-5 under whole-stack reading; that reopens the
#                     rate sweep, whose arms all ran at 0.15, and it is a bigger finding than the number.
#   both positive     noise, or a curve flat inside the floor — not a dose result either way.
#
# control  native224 = 0.8502, best at epoch 7, converged not truncated. 12 epochs, 12 slices, 0.20-0.80
#          band, whole-stack, backbone 4e-5, 224 px, same labels, same 909-study holdout, warmup 0.15.
#          The single variable is --warmup. Read against the control first, then against warm224_brief.
# cost     ~19 min for twelve epochs at 94 s. Same cache as native224; a schedule shape is free.
# price    19 min for P(clears the 0.0062 floor on the MEAN) = 0.10, the lowest of the three schedule
#          and clipping arms: 3.6 epochs of a twelve-epoch budget spent climbing is a large cost for a
#          protective effect this recipe may not need. Bought for the BRACKET — a flat reading on both
#          sides closes an axis that sits under every number this project has, and that is worth one
#          19-minute slot in a 33-slot window.
#
# HARNESS CHECK BEFORE READING ANY NUMBER. The printed recipe must say "warmup": "0.3", "one_cycle":
# "True" and "image_size": "224", 4,406 studies loaded and 909 held out. Without `--one-cycle` there is
# no schedule and the fraction is inert, so the run would be a byte-identical control rather than a
# null; `$RECIPE` carries the flag, verify it survived. The landed check compares `one_cycle_warmup`.
#
# THE FALSIFIABLE PART, written before the run.
#   CONFIRMED   the mean peak clears 0.8564 (control + floor). Early-step damage is real at this rate,
#               and the finding is not about 224 — it is about every arm the rate sweep produced.
#   DIRECTIONAL positive under the floor. Kept only if warm224_brief is NOT also positive; opposite
#               directional readings on one axis are the floor talking, not a curve.
#   FLAT        with warm224_brief also flat, the axis is closed at twelve epochs and 224 px. That is
#               this arm's modal and most useful outcome.
#   REFUTED     the mean falls by more than the floor. The budget argument holds and it is quantified:
#               3.6 epochs of climbing costs more than the protection is worth, which makes
#               warm224_brief's direction the one to follow and makes a third point below 0.05 sensible.
#
# deliberate: --warmup 0.30 (the arm's thesis — a 5x rate and whole-stack batches may need a longer climb)
# deliberate: --image-size 224 (native224's control, per that row's standing instruction)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train warm224_long \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 --warmup 0.30 \
  --labels "$LABELS" --out "$PROCESSED/warm224_long_s2026.pt"
