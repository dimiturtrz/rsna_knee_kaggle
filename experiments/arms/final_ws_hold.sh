#!/usr/bin/env bash
# final_ws_hold — the fold-0 twin of `final_ws_cf`, run as an INSTRUMENT on `board_transfer`.
#
# reading: final_ws_hold_s2026.holdout.npz
# bead: C163 §3 (approved), D155 §1/§4 (the coverage hole)
#
# WHAT IT UNBLOCKS, AND IT IS NOT A SEAT. D155 §1 censused every row of `comp/board.jsonl` and found the
# transfer constant is fitted on board 0.904-0.937 while every decision it prices is taken at 0.944. The
# reason is one member: **all 16 unpriced rows at board >= 0.940 name a `final_ws` checkpoint, 16 of 16,
# and `final_ws` has five checkpoints on disk and ZERO holdout records in any variant** — every one of them
# is `--fold -1`, so nothing can be rebuilt on the L75 ruler. This arm produces the missing record and
# nothing else has to change: the 16 rows then rebuild, and the constant is fitted where it is used.
#
# WHY IT COMES BEFORE THE GEOMETRY ARM. The geometry arm costs 2.5-3.5 h and its whole justification is an
# offline gain MULTIPLIED BY THIS CONSTANT. Running the expensive commit before the instrument that prices
# its units inverts the cost-tier rule. Cell D prices that arm's SIZE (correlation); this one prices its
# UNITS. Both are cheap and both come first (C163 §3(b)).
#
# PRE-REGISTERED, before it starts. D155 §2 refits `board_transfer` on the rebuilt L75mid column at
# **+1.135, CI [+1.02, +1.25]**, n=36, R2 0.962, resid SD 0.0016 — over rows at board 0.904-0.937.
#
#   - Slope above 0.940 materially BELOW 1.135: the board compresses at the top, which this project has
#     suspected since `board_transfer` was first quoted and has never been able to test for want of
#     coverage. Every offline lever is then worth less than quoted, and the seat programme is deader than
#     C163 §2 already says. That is the outcome that redirects the plan toward model-level work.
#   - At or above 1.135: the seats are worth what we think and the geometry arm's price is confirmed.
#
# Both outcomes change what is run next, which is the test the `deep32`-shaped arms failed.
#
# IT IS AN INSTRUMENT, NOT A MEMBER. It holds fold 0 out, so it leaves a scoreable member behind and a
# ledger row that prices it. That is a BONUS and never the reading — the reading is the refitted slope.
#
# WHAT IS HELD FIXED. The recipe is `final_ws_cf`'s verbatim (`experiments/arms/calibsix.sh:112-115`):
# same trunk, 12 epochs, max slice-pooling, whole-stack, backbone-lr 4e-5, the dense band, calibrated
# targets, `$LABELS`. The SINGLE moving variable is `--fold -1 --ema` becoming `--fold 0`, which is what
# buys the holdout. `--ema` goes with the fold-0 line because with a holdout there IS best-epoch selection,
# and calibsix's own `# deliberate:` line says ema was its only variance control precisely BECAUSE there
# was none.
set -u
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

FIT="--fold 0"
DENSE="--slices 20 --band-low 0.1 --band-high 0.9"
CALIB="--calibrated-targets"

run_train final_ws_hold \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $DENSE $CALIB \
  --labels "$LABELS" --out "$MODELS/final_ws_hold_s2026.pt"
