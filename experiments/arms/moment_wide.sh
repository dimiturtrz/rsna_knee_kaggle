#!/usr/bin/env bash
# moment_wide.sh — the two moments handed to the head CONCATENATED, which is what was actually measured.
#
# bead: knee-2rmdc
# reading: moment_wide_s2026.holdout.npz
#
# WHY THIS RUNS AT ALL, AND IT IS A MEASUREMENT NOT A RETRY. `moment_dense` ran the mixed-down form of this
# statistic at this exact corner and read **0.8516 against calib_dense_hold 0.8634, -0.0118** — the bottom
# band, whose pre-registered instruction was "suspect the MIX before the mechanism". That instruction was
# followed before any verdict was written, and it retired BOTH suspects the band named:
#
#   shape (768, 1536)
#   mean|first| 0.01349  mean|spread| 0.01068  ratio 0.7921
#   spread nonzero: True  max|spread| 0.14384
#   first drift from identity: mean|first-I| 0.01223
#
# The spread half is ALIVE at 0.79x the first moment's magnitude, so it neither stayed at its zero
# initialisation nor was starved by the 0.26x scale the arm measured in advance. What the dump DID show is
# the THIRD difference `moment_dense`'s own header named before it ran, and it is now the only one left:
# the identity initialisation on the mean half is **gone** (`mean|first - I| = 0.0122` against a
# `mean|first|` of 0.0135, while `torch.eye`'s own mean magnitude at 768 wide is 1/768 = 0.0013). The head
# was never fed the mean beside its spread. It was fed a fully learned 768x768 projection of the pair, in
# place of the max pool, fitted from scratch inside the same 12-epoch budget.
#
# So `moment_dense` measured a WIDTH COLLAPSE, not the statistic. `feature_probe --slice-readout` ordered
# `moments > mean > max` on eight of nine trunks by +0.0116 to +0.0179 and read the two moments
# CONCATENATED, at double width, never mixed back down. This arm is that reading: `WideMomentPooling` has
# **no parameters at all**, `SlicePooling.out_multiplier` widens the head that receives it, and the only
# thing the 12 epochs fit is the head — exactly as under `max`.
#
# ONE VARIABLE. Byte-identical to `moment_dense` except `--slice-pooling moments` -> `moments_wide`, which
# makes it byte-identical to `calib_dense_hold` (ledger 0.8634) except for the pooling name. Same recipe,
# epochs, backbone rate, slices, band, calibrated targets, held annotations, label table and seed. Both
# controls are measured; neither is re-run.
#
# PRE-REGISTERED BANDS, against calib_dense_hold 0.8634 and the 0.0062 paired floor:
#   >= 0.8696  the spread carries signal the first moment discards, AND the width was the whole story of
#              moment_dense's -0.0118. Follow-up is a uniform re-fit of the calibrated seats at double
#              head width, not a seventh seat.
#   0.8666-0.8695  directional, the right size for the frozen delta, and it still convicts the mix — but
#              it is not a finding on one arm. Honest next step is the same swap at the SPAN corner.
#   0.8573-0.8665  FLAT. A learned head over a max pool recovers what the standard deviation carries; the
#              frozen-probe gap does not survive fine-tuning. THE READOUT AXIS THEN CLOSES WITH THE
#              AGGREGATION AXIS, and the -0.0118 is priced as the cost of the mix alone. Still a diversity
#              seat: genuinely different internal structure at the same quality.
#   <= 0.8572  the statistic itself is worse than the maximum under fine-tuning, at this corner, at double
#              width and with nothing left to blame on a projection. The axis closes AGAINST the probe,
#              and what is then owed is an explanation of why a frozen ridge and a fine-tuned head
#              disagree by this much — not a third pooling.
#
# HONEST CAVEAT, CARRIED FORWARD FROM moment_dense'S HEADER: `dinov2-small`, the trunk this recipe uses,
# carries the SMALLEST gap in the probe table (+0.0129), so the frozen number is an UPPER BOUND on what
# this corner can show.
#
# HARNESS CHECK, BEFORE THE VERDICT AND NOT AFTER. The printed recipe must name "slice_pooling":
# "moments_wide". Then assert the head was BUILT WIDE: the checkpoint's first head Linear must have
# in_features = 2 x pooled width (1536, not 768). This module has no tensors of its own, so there is no
# dead-mix check to run and no optimiser group it can fall out of (knee-7gyw) — the width IS the plumbing
# assertion, and it is the one thing that would silently make this arm a re-run of something else.
#
# PRICE: ~56 min at 20 slices (279 s/epoch MEASURED x 12, not the ~2.5 h this line guessed before any
#        dense arm had been timed), no submission slot — the reading is offline against a ledger control.
# EXPECTED GAIN: the frozen probe's +0.0129 at this trunk is the ceiling; the mix's -0.0118 is the floor it
# has to climb back over first. So the honest expectation is a FLAT, and this is an INSTRUMENT: it retires
# the width confound, and it is what licenses closing (or re-opening) the whole readout axis.
# VRAM: 20 slices reserve 28.26 GB by DeviceBudget's derivation; the head's first Linear doubles its input
# width and nothing else grows, so the change does not land on activations.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train moment_wide \
  $RECIPE --epochs 12 --slice-pooling moments_wide --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --labels "$LABELS" --out "$MODELS/moment_wide_s2026.pt"
