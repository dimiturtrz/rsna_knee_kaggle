#!/usr/bin/env bash
# bead: knee-9grwy
# reading: labelwise_drop_drop40_long_ff.pt
#
# WHAT THIS ARM IS
#   `labelwise_drop_drop40_long`'s byte-identical line refit at full fold with an EMA. Nothing else
#   moves. It is a MEMBER BUILDER, not a contrast: its product is a single solo checkpoint.
#
# WHY THIS MEMBER, AND WHY THE EDGE IS NOT THE ARGUMENT
#   `labelwise_drop_drop40_long_s2026` reads 0.9051 ledger75, fourth in the top-6 band, and is the only
#   member there with no `_ff` on disk — nineteen `_ff` checkpoints exist and none is this one. Its
#   +0.0006 over the shipping member's 0.9045 is a tenth of the 0.0062 seed spread, so it starts
#   statistically TIED with what we ship. That tie is the premise, not the case for running it.
#
# THE MECHANISM IS THE VARIANCE OF THE OPERATION, MEASURED ON THE BOARD
#     extsolo_v1    0.937 -> extffsolo_v1  0.945   (+0.008)
#     headlrsolo_v1 0.937 -> headlrff_v1   0.938   (+0.001)
#   Same operation, same starting board score, eightfold different payoff. So refitting a tied member is
#   a draw from a distribution with a measured +0.008 upper tail, and this is the LAST such draw the top
#   band holds: every other top-6 member is already shipped, or is `soup_anneal2` at -0.0029 below ship.
#
# THAT MECHANISM IS RETIRED, AFTER LAUNCH. `noemasolo_v1` (board 0.944, 2026-09-20) showed the +0.008
#   pair CROSSES THE EMA AXIS — `extffsolo_v1` is ema ON, `extsolo_v1` is ema OFF — so it prices refit
#   plus ema, not refit. The matched ema-ON twin prices at board 0.9413, leaving +0.004. The term across
#   four bases is +0.001 / +0.003 / +0.004 / +0.007; there is no +0.008 upper tail to draw from. This
#   arm's base reads 0.9051 offline75, which the solo transfer prices at 0.9387, so +0.004 lands it at
#   0.9427 and beating the 0.946 champion needs 1.6 sd of the residual: P ~0.06. It runs to completion
#   because its card is sunk, and it SHIPS AS AN INSTRUMENT OR NOT AT ALL. See the ledger, 2026-09-20.
#
# deliberate: --fold -1 — the point of the arm. A full fit holds nothing back.
# deliberate: --ema — knee-0ed8f landed 2026-09-20T09:22Z, the first matched single-variable ema
#             contrast on this corner: ema-ON ahead by +0.0028 ledger75 (0.9073 vs 0.9045, same fold,
#             same 909 studies, convergence clean). Under the 0.0053 paired floor and directional only,
#             but the OPPOSITE SIGN to the cross-member +0.0049 ema-OFF in `matched_axes`, and every
#             shipping `_ff` is already ema-on.
# deliberate: --stochastic-depth 0.05 --slot-dropout 0.40 --epochs 24 --slices 20 --band-low 0.1
#             --band-high 0.9 --backbone-lr 4e-5 --calibrated-targets --hold-annotations — inherited
#             verbatim from the fold-0 twin. Two things move, the fold and the EMA.
#
# deliberate: --vram-gb 22 — the owner's 50-70% band (16.3-22.8 GB), matched to every sibling _ff arm.
#             The first launch carried NO bound at all and stamped `vram_gb 0.0`. The second tried 18,
#             the bottom of the band, on the reasoning that a neighbour was resident (3.7 GB, 0% util):
#             it OOMed at 14.31 GB allocated plus 3.09 GB reserved-but-unallocated, i.e. 17.4 of 18
#             allowed, killed by fragmentation rather than by true demand. The arm's real need is
#             ~17.5 GB, so 18 left no fragmentation headroom at all. 22 is the proven sibling value and
#             still leaves ~6 GB for the neighbour at its measured 4.2 GB footprint.
# deliberate: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — the allocator fix the OOM itself named,
#             aimed at exactly the 3.09 GB of reserved-but-unallocated that caused it. It lowers the
#             headroom the bound has to carry, which is what makes 22 comfortable rather than merely
#             larger.
#
# CARD. The twin measured ~0.8 h over 24 epochs; a full fold adds back the held-out quarter, so ~1.1-1.4 h.
#   Nothing else of ours may share the card (`card_check`).
#
# NO OFFLINE READ IS OWED OR POSSIBLE. The full-fold member has seen the 909 rows its twin is scored on,
#   so no ledger row can price it — the read is the BOARD, on a later slot, and only if a cheaper signal
#   does not refuse it first.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FIT="--fold -1 --ema"

run_train labelwise_drop_drop40_long_ff \
  $RECIPE $FIT --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --vram-gb 22 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations \
  --stochastic-depth 0.05 --slot-dropout 0.40 \
  --labels "$LABELS" --out "$MODELS/labelwise_drop_drop40_long_ff.pt"
