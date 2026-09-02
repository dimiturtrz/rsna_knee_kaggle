#!/usr/bin/env bash
# focal224.sh — native224's recipe with `--pool cls_mean_focal` and nothing else moved. The fifth
# aggregation transport, and the only one whose 336 reading was NEGATIVE.
#
# bead: knee-7d53
#
# WHY RUN A MEASURED LOSER. `focal_pool` read 0.8432 at dense, −0.0144 on the focal six and −0.0094 on
# the mean (`experiments/README.md:5024-5075`) — inside the floor on the mean but clearly not a gain.
# It is here because it is the ONE aggregation switch whose mechanism is defined against the patch grid,
# which is exactly the thing 224 changes, and because the direction the resolution argument predicts is
# the opposite of the dense reading. An arm whose prior says "this loses" is worth a slot only when the
# mechanism says the resolution should reverse it; that is the case being made below, and if it fails
# the flag is closed at both resolutions rather than at one.
#
# THE MECHANISM. `cls_mean` is the class token beside the mean over all patch tokens, so a finding that
# occupies a small part of the field is averaged against everything that is not it. `cls_mean_focal`
# appends the mean of each channel's TOP EIGHTH over the patch grid — an order statistic, so it reads
# the strongest region per channel rather than the average one. The small findings (meniscus, ACL, MCL)
# are the ones that occupy a fraction of the field, and they are the six the ledger calls focal.
#
# THE RESOLUTION ARGUMENT, AND I PREDICT MORE — WHICH IS TO SAY, LESS BAD. The top eighth is a fraction
# of the token grid, so what it MEANS depends on how many tokens there are. At 336 px with 14 px patches
# the grid is 24x24 = 576 tokens and the top eighth is 72 of them; at 224 it is 16x16 = 256 and the top
# eighth is 32. A 1-3 mm finding covers a roughly fixed area of tissue, so at dense it is a small share
# of a 72-token pool — the focal statistic is itself diluted, which is a coherent explanation for why
# the dense arm lost: it added a noisy near-mean as a third term and paid for it in head width. At 224
# the same finding is a much larger share of 32 tokens, so the statistic is sharper relative to what it
# is trying to isolate. The competing reading predicts worse: at 224 a 1-3 mm structure may span barely
# one token, so the top eighth is 32 tokens of mostly surrounding tissue and the term is pure noise. The
# two make opposite predictions on the SAME arm, which is what makes this readable either way.
#
# WHY IT IS NOT A DUPLICATE OF sided224. Both move `--pool`, but to different values and on different
# axes: `sided` splits the grid by SIDE and keeps means, this one keeps the whole grid and adds an ORDER
# STATISTIC. They are single-variable against the same control and against each other only in the sense
# that any two values of one flag are; neither composes with the other by construction.
#
# control  native224 = 0.8502, best at epoch 7, converged not truncated. 12 epochs, 12 slices, 0.20-0.80
#          band, whole-stack, backbone 4e-5, 224 px, same labels, same 909-study holdout, `cls_mean`
#          token pooling. The single variable is --pool.
# cost     ~19 min for twelve epochs at 94 s. Same cache as native224; the extra term is a top-k mean
#          over 256 tokens per slice, negligible against the trunk, and it widens the head's input.
# price    19 min for P(clears the 0.0062 floor on the MEAN) = 0.08 — the lowest in this window, and
#          honestly so: the arm starts 0.0094 down at dense and needs the grid argument to be worth more
#          than that before it is worth anything at all. It is bought as an INSTRUMENT on the QUESTION
#          "does an order statistic over patches depend on the grid size", which no other arm asks, and
#          it must be reported as one rather than sold as an improvement.
#
# HARNESS CHECK BEFORE READING ANY NUMBER. The printed recipe must say "pool": "cls_mean_focal" and
# "image_size": "224", and the log must load 4,406 studies and hold out 909. The focal term widens the
# head's input, so a parameter count equal to native224's means the switch did not land and the delta is
# a PLUMBING verdict rather than a null.
#
# THE FALSIFIABLE PART, written before the run.
#   CONFIRMED   the mean peak clears 0.8564 (control + floor). An order statistic over patch tokens is
#               grid-dependent, the dense loss was about dilution rather than about the mechanism, and
#               every measurement this project has of a top-k pooling is resolution-conditional. That is
#               a claim about how to read the WHOLE aggregation table, not about one flag.
#   DIRECTIONAL positive under the floor. Read as a SIGN REVERSAL against dense's −0.0094, which is the
#               finding, and kept only for the focal six where the mechanism lives.
#   FLAT        the transport carried the dense result's sign but not its size; the grid argument is
#               worth something and not enough. Closes the flag at both resolutions.
#   REFUTED     the mean falls by more than the floor, i.e. dense's −0.0094 reproduced or deepened. The
#               top eighth of 256 tokens is mostly background at this resolution; `--pool cls_mean_focal`
#               is then closed everywhere and the focal findings need a spatial prior, not an order
#               statistic — which is what `--slice-regions` and the slot attention already are.
#
# deliberate: --pool cls_mean_focal (the arm's thesis — the top eighth is sharper on a smaller grid)
# deliberate: --image-size 224 (native224's control, per that row's standing instruction)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train focal224 \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 --pool cls_mean_focal \
  --labels "$LABELS" --out "$MODELS/focal224_s2026.pt"
