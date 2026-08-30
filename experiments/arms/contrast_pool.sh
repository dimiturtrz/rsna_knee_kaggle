#!/usr/bin/env bash
# contrast_pool.sh — the fix `contrast_dense` pre-registered against its own reading, finally wired.
#
# WHAT IT ANSWERS. `contrast_dense` read 0.8547 against control `calib_dense_hold_s2026` 0.8634 — down
# 0.0087, eleven of twelve labels below. Its own header named the suspect BEFORE the run: "a batch of
# twelve sees a 7%-prevalence finding roughly once", so for most of the twelve findings the positives a
# supervised-contrastive term needs are simply absent, the anchor forms no term, and the label drops out
# of the loss entirely. What was measured is therefore "supervised contrastive AT BATCH 12", and the
# named precondition for reading it any wider was a remembered pool of past readouts.
#
# That pool existed the whole time. `PairMemory` is proven by `RememberedRanking`; it had never been
# pointed at this term, because `run_setup.objective` was a single if-chain whose contrast branch
# returned before reaching the memory one. `RememberedContrast` is the composition, and `contrastive`
# dispatches on the pair (knee-agrj).
#
# THE ONE VARIABLE. `--pair-memory 8` against contrast_dense's flags, nothing else moved. At capacity 8
# the population each anchor is contrasted against grows from 12 to 96, so a 7% finding is expected in
# seven keys rather than in one. The gradient still flows only through the live step; the remembered
# contexts are fixed targets, exactly as in a contrastive memory queue — the same trade `RememberedRanking`
# makes, and it is not free: a remembered context came from weights that have since moved.
#
# control  contrast_dense_s2026, holdout 0.8547 — the arm this one widens, and the honest single-variable
#          comparison. calib_dense_hold_s2026 at 0.8634 is the SECOND reading: whether the widened term
#          gets back to the no-contrast recipe at all.
# cost     one 12-epoch fit at ~200 s an epoch, ~40 min. Zero submission slots.
#
# IT IS AN INSTRUMENT, NOT A BOARD BET, and it is armed as one. The house bar is 1.5% absolute and no
# member-quality gain of that size is available here — calibsix's measured ~23% board pass-through
# ceilings it. What a slot of card buys is which of three things is true, all three written down first:
#
#   flat, inside +/-0.0062 of 0.8634  -> supervision was not the binding constraint. The representation
#                                        lane closes ON PRICE, and the topology question goes to the
#                                        strong form (masks, knee-2sbk) rather than to a weight sweep.
#   above +0.0062 over contrast_dense -> the batch-size artefact is CONFIRMED. The -0.0087 was a fact
#                                        about a batch of twelve, not about shaping the representation,
#                                        and the axis re-opens with a mechanism behind it.
#   below contrast_dense              -> harmful independent of batch size. The lane closes on MECHANISM,
#                                        which is a different and stronger closure than price.
#
# AND PRICE THE SEAT IN ALL THREE. `contrast_dense` LOST 0.0087 solo and still seats fifth under
# `record_mix --largest-pool --vote logit`. A solo number is not this member's whole value, so run
# `seat_value` whatever the holdout says.
#
# VERIFY BOTH FLAGS LANDED before believing any of the three. `--pair-memory` was REFUSED beside
# `--contrast-weight` until today, so the failure to rule out is not a typo but a stale binary: read the
# recipe line in the log and assert it carries the contrast weight AND the pool size. `RememberedContrast`
# reports the pool by asking the memory itself, so a capacity that silently fell back to the default
# would show as 8-that-is-really-PAIR_MEMORY — which is the same number, and is why the WEIGHT is the
# half to check hardest.
#
# bead: knee-agrj
#
# NO --vram-gb: DeviceBudget derives the cap from --slices — 20 slices reserve 28.26 GB. The memory adds
# 8 steps of `b o h` contexts, ~5 MB at this trunk, which is three orders below the activations.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train contrast_pool \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --contrast-weight 1.0 --pair-memory 8 \
  --labels "$LABELS" --out "$PROCESSED/contrast_pool_s2026.pt"
