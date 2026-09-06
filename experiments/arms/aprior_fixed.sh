#!/usr/bin/env bash
# aprior_fixed.sh — the anatomical prior held as a CONSTRAINT rather than as a warm start.
#
# WHY THIS IS NOT A RE-RUN OF A REFUTED ARM. `aprior_dense_s2026` read 0.8557 against band_dense's
# 0.8578, −0.0021, and the ledger closed it. That refutation stands, and this arm does not dispute it.
# What it disputes is the SCOPE of it. The ledger's own explanation of the failure is:
#
#   "the warm start washed out rather than failed to land. Train loss tracks the control to the fourth
#    decimal at every epoch, so 72 numbers of initialisation on an nn.Parameter the data may overrule
#    are gone by the time the fit matters."
#
# That is a statement about the HOLDING, not about the anatomy. `SlotHead.hold_prior` made the prior an
# `nn.Parameter`; one gradient step may erase it, and the training loss says one did. The published
# foreign program holds the same construct as a `register_buffer` added to the attention logits on every
# forward pass (`bend_the_knee_to_dinov3_ensembled.py`, SLOT_PRIOR_STRENGTH = 0.55), which gradient
# descent cannot erase — it can only learn around it. So "does a good starting point help" was asked and
# answered no; "does a permanent constraint on where each finding may attend help" has never been asked
# here. `FixedPriorSlotHead` is that second question and nothing else: same projection, same slot
# embeddings, same dropout, same readout, same parameter count minus the 72 the optimiser no longer
# sees.
#
# control  band_dense_s2026 = 0.8578, and aprior_dense_s2026 = 0.8557 is the SECOND control — the same
#          prior, the same table, the same twelve epochs, differing only in the holding. Two controls on
#          one axis is what makes this readable: a gain over both is the constraint; a landing between
#          them is noise; a landing at aprior_dense says the holding was never the mechanism.
# cost     about 73 min. The flag adds no compute and REMOVES 72 parameters from the optimiser.
#
# PRE-REGISTERED, three outcomes:
#   PREDICTION — the peak mean clears 0.8578 by more than the 0.0062 holdout floor, and the gain
#     concentrates in the findings whose prior ROW IS PEAKED rather than flat (MCL, PF OA, Synovitis,
#     Medial and Lateral OA). Flat-row findings — ACL, both menisci — should barely move, exactly as
#     pre-registered for aprior_dense, because a constraint that says nothing constrains nothing.
#   REFUTATION — the peak mean sits at or below 0.8578. Then the head recovers the mapping from
#     study-level supervision alone whichever way the prior is held, the anatomical-prior family closes
#     on a MECHANISM rather than on a tally of four knobs, and the localisation argument has to buy a
#     real localisation signal (knee-mxj) instead of a table.
#   THIRD OUTCOME — the mean is flat but the peaked/flat per-label split moves. Then the constraint does
#     what it claims and the ceiling is bounded elsewhere; the member stops being a mean lever and is
#     priced as a DIVERSITY generator — errors shaped by a different attention geometry than the rest of
#     the pool — by seat value against the champion, never by its solo score.
#
# HONEST PRICE, stated before it runs: +0.000 to +0.005 expected, P(clears +0.010) ~ 0.15. This is not
# sold as an improvement. It is the instrument that says whether "the anatomical prior does not help
# this head" is true of the IDEA or only of the warm-start form we happened to implement.
#
# deliberate: --slot-pooling attend_fixed (the single variable against aprior_dense, which is otherwise
#             this exact command line; band_dense is the same again with --anatomical-prior dropped)
# bead: knee-nin6h
# reading: aprior_fixed_s2026.holdout.npz
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train aprior_fixed \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --anatomical-prior --slot-pooling attend_fixed \
  --labels "$LABELS" --out "$MODELS/aprior_fixed_s2026.pt"
