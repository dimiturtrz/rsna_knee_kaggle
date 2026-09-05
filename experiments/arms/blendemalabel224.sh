#!/usr/bin/env bash
# blendemalabel224.sh — the THIRD term, and the one that turns two points into a slope.
#
# bead: knee-y8ies
# reading: logs/blendemalabel224_ledger.log
#
# GATED. DO NOT RUN until blendema224 has landed in its ADDITIVE branch (0.8555-0.8679, see that arm's
# header). Every other branch cancels this arm rather than modifying it: SUPER-ADDITIVE demands a second
# seed of the two-term result before a third term is added to something not yet believed; PARTIAL
# SUBSTITUTION says the terms already overlap and a third would be priced at a discount nothing has
# measured; SUBSTITUTES closes composition at this base and there is no chain to extend. Running this
# arm on a two-term result that has not cleared its own gate is the exact mistake knee-orjq was closed
# for - stacking a term whose predecessor was never confirmed.
#
# WHY A THIRD TERM AND NOT A SECOND SEED. Two points cannot fit a composition rule; that limit is
# written into blendema224's own header and into knee-orjq's close. The question the 0.953 plan actually
# needs answered is not "do these two add" but "does a CHAIN add", because +0.015 is not reachable from
# any single term this project owns and is only reachable as a sum. Three terms is the shortest chain
# that can distinguish a sum from a coincidence: if term three lands additive on top of terms one and
# two, the increments are behaving independently over a range, which is what licenses pricing a fourth
# and fifth by arithmetic instead of by card time.
#
# THE THIRD TERM. --label-pooling labelwise, measured at this exact corner by labelwise224_s2026 at
# 0.8517 against native224_s2026 at 0.8502: +0.0015. It is the SMALLEST of the three terms and that is
# deliberate. A chain is tested by whether its small increments survive, not by whether its large one
# does - a third term big enough to dominate would hide a substitution, and this one cannot.
# Mechanically it is a third independent axis: term one changes what the targets SAY (blended labels),
# term two changes which weights are KEPT (EMA), term three changes how the head READS the pooled
# features. Nothing about one predicts another.
#
# ONE VARIABLE against blendema224: --label-pooling labelwise. Every other flag is that arm's.
#
# THE ADDITIVE PREDICTION, and it inherits blendema224's measured value rather than its prediction -
# fill in the observed number, do not use 0.8617:
#     predicted = blendema224_observed + (labelwise224 - native224) = blendema224_observed + 0.0015
#
# PRE-REGISTERED, BEFORE THE NUMBER, against the 0.0062 paired floor. Note that the term itself
# (+0.0015) is a quarter of that floor, so the branches are NOT about resolving this term - they are
# about whether the chain has started to bend:
#   >= predicted+0.0062   Super-additive again, at a second link. Suspect it hard; two super-additive
#                         links in a row at a corner where the dense analogue was 7/7 negative is more
#                         likely a harness fault than a mechanism. Check --label-pooling fired.
#   within 0.0062         ADDITIVE AT THE THIRD LINK. This is the result that licenses arithmetic: the
#                         project may then price a fourth and fifth term from their singles without
#                         spending card on each combination, and the 0.953 route becomes a search over
#                         which singles to chain rather than a hope that a single arm clears +0.015.
#   <= predicted-0.0062   THE CHAIN BENDS. Two terms add and three do not, which is the more interesting
#                         negative: it puts a LENGTH on composition at this base and says a stack is
#                         worth about one extra term, not five. Record the length; it is a number the
#                         plan can use even though it kills the plan as written.
#
# COST. ~20 min of card. Zero submission slots. Same corner, same holdout, same gold ruler via
# --hold-annotations.
#
# control  blendema224_s2026, byte-identical recipe without --label-pooling labelwise.
#
# deliberate: --label-pooling labelwise (the arm's thesis, and the third link in the chain)
# deliberate: --ema (term two, held from blendema224 and not under test)
# deliberate: --labels blend_three_v1 (term one, held from blendhold and not under test)
# deliberate: --hold-annotations (matched to both controls; buys the gold ruler in the same run)
# deliberate: --eval-labels report_labels_v2.csv (the proxy must not move with the arm)
# deliberate: every other flag — they are blendhold's and may not be touched in this arm.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
BLEND="$PROCESSED/labels/blend_three_v1.csv"
require "$BLEND"
require "$LABELS"

run_train blendemalabel224 \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 --hold-annotations --ema --label-pooling labelwise \
  --labels "$BLEND" --eval-labels "$LABELS" --out "$MODELS/blendemalabel224_s2026.pt"
