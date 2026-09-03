#!/usr/bin/env bash
# v3large_stack.sh — the matched control for `ortho_stack`, and the only thing that makes it readable.
#
# WHY THIS ARM EXISTS. `ortho_stack` moves ONE variable against `ortho_dense` (the recipe, 12 slices at
# 224 px -> 20 at 336) and TWO against its stated control `band_dense` (the recipe AND the trunk, a
# 303M ViT-L against an 86M ViT-B). So `ortho_stack` alone cannot separate the two readings its own
# middle branch collapses together:
#
#   (a) the knee-continuation gain is REAL but resolution-bound — it was worth +0.0194 at 224/12 and
#       shrinks at 336/20 because the fine-tune recovers from data what the continuation supplied;
#   (b) the continuation gain is intact and the ViT-L simply LOSES the recipe step the ViT-B gained
#       (+0.0069, 0.8509 -> 0.8578), which is `members.declines:195`'s trunk-size reading.
#
# Under (a) the in-domain trunk lane closes. Under (b) it does not — it says the OPERATING POINT is
# wrong for this trunk size, which is a different and still-open question. `ortho_stack` returns the
# same number under both.
#
# THE SEPARATION. This is `ortho_stack`'s exact flag set with `--variant v3large` — the weights
# `ortholarge` was continued FROM (`facebook/dinov3-vitl16-pretrain-lvd1689m`, added 2026-09-02 as the
# control for `ortholarge` and only that, backbone.py:198). Trunk size, recipe, slices, band, rate,
# epochs, batch and accumulation are all HELD. The single free variable is the 1.25M-knee continuation
# corpus, exactly as it was in the 224/12 pair.
#
# That gives the project the SAME single-variable contrast at two operating points:
#   224 px / 12 slices  ortho_dense 0.8452  v3large_dense 0.8258   +0.0194  (measured)
#   336 px / 20 slices  ortho_stack ?       v3large_stack ?        ?        (this pair)
#
# and the delta-of-deltas is the answer to Q-CW: does a domain-continued SSL gain shrink as the
# downstream token budget grows (2352 -> 8820 tokens a study)? No prior in this project's ledger
# measures that, and I could not find one asserted with evidence rather than intuition.
#
# BRANCHES, written before the run, read as a PAIR with `ortho_stack`:
#   ortho_stack - v3large_stack >= +0.0062   the continuation carries to the shipped operating point.
#                 Reading (b) is dead. The trunk lane is OPEN and ortholarge is the incumbent's rival.
#   within the 0.0062 floor                   the continuation gain is RESOLUTION-BOUND. Reading (a).
#                 This closes the in-domain trunk lane on the fitted ruler at the point we actually
#                 ship, which is the closure the frozen probe was never entitled to make.
#   <= -0.0062                                the continuation HURTS at high resolution. Unexpected;
#                 would say the knee corpus overwrote something the natural-image prior needed, and
#                 the next question is the continuation's own recipe, not ours.
#
# Note the pair is readable even if BOTH arms land under `band_dense` = 0.8578. That outcome is
# reading (b) plus a trunk-size cost, and it is informative; `ortho_stack` on its own would have
# reported it as "the in-domain lane is empty", which would have been wrong.
#
# COST, priced before queueing. Identical geometry to `ortho_stack`: ~1950 s an epoch at batch 1,
# ~6.5 h for 12, ZERO submission slots. This arm's own plausible gain as a MEMBER is low — v3large_dense
# read 0.8258, the worst non-CNN trunk in the corpus, and nothing suggests the recipe step rescues it
# past the 0.8459 seat bar. It is priced as an INSTRUMENT, not an improvement: it costs 6.5 h and it
# converts `ortho_stack`'s three-way ambiguous read into a single-variable delta on the one axis the
# project has not closed. Without it, 6.5 h of `ortho_stack` buys a number nobody can interpret, which
# is the worse ratio of the two.
#
# bead: knee-b7a50
# deliberate: --batch --accumulate --variant
# control  ortho_stack = the paired arm, identical flags with the CONTINUED trunk.
#          v3large_dense = 0.8258, identical trunk at 224 px / 12 slices.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train v3large_stack \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant v3large --batch 1 --accumulate 12 \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --labels "$LABELS" --out "$MODELS/v3large_stack_s2026.pt"
