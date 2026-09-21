#!/usr/bin/env bash
# bead: knee-nchjt
# reading: transduce_dmlb_c41*.pt
#
# WHAT THIS ARM IS
#   Four more seeds of `transduce_dmlb.sh`, unchanged in every other flag, taking the treated arm to
#   n=6 against the control's n=6. Same parent, same pseudo table, same --eval-labels pin.
#
# WHY A SECOND ARM ON A KNOB-SIZED DELTA, WHICH IS NORMALLY FORBIDDEN
#   The first two seeds read 0.9066 and 0.9080 against a control mean of 0.9063 (n=6, sd 0.00017):
#   both above ALL SIX controls, mean +0.0010, which is one fifth of the 0.0053 paired floor. On its
#   own that is a knob and the rule says do not spend a confirmation seed on it. The exception is that
#   it GATES something bigger, and here it gates the only lever left whose offline number is a LOWER
#   BOUND BY CONSTRUCTION: this instrument holds out same-corpus studies, so its shift is ~0, and the
#   real test set's shift is not. A transductive-inference kernel is a multi-day build; whether it is
#   worth opening at all turns on whether this effect is real at zero shift.
#
# THE FIRST ARM'S PRE-REGISTRATION WAS ARITHMETIC ON THE WRONG DENOMINATOR, and this arm exists partly
#   to repair it. It said: 'six control seeds at sd 0.00017 make +/-0.0010 a ~6 sd band, so two seeds
#   resolve it.' That assumed the TREATED arm inherits the CONTROL's variance. It does not -- the two
#   treated seeds spread 0.0014, eight times the control sd, so the band was never ~6 sd of anything
#   the treated arm could produce. Six treated seeds put the standard error near 0.0004, which is the
#   first design here that can actually separate +0.0010 from zero.
#
# PRE-REGISTERED, on the six-seed treated mean against the control mean 0.9063, ledger75, fold 0
#   >= +0.0010   the effect survives at zero shift. Transductive inference becomes a build worth
#                pricing, because every real-test version of it starts from a bound, not a guess.
#   -0.0010 to +0.0010   the two-seed read was the treated arm's own scatter. Suspect (1) retires and
#                the axis closes; the first arm's +0.0010 was a mean of two draws from a wide spread.
#   <= -0.0010   closure, harder.
#
# ZERO SLOTS. Fold 0, ledger75, never the board.
#
# deliberate: seeds 409-412, fresh, so no child here is any earlier run under a new name.
# deliberate: the pseudo table is REQUIRED, not rebuilt. transduce_dmlb.sh wrote it from the teacher's
#             record; rebuilding it here would let a changed teacher silently redefine the treatment
#             halfway through one measurement, which is the one thing that would make n=6 a lie.
# deliberate: every other flag is transduce_dmlb.sh byte for byte, because these six seeds are ONE
#             sample and a flag that moves between them is a second variable inside the denominator.
#
# CARD. 4 children x 4 epochs x ~350 s = ~100 min. Reserve measured at 16.57 GB on the first arm.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

PARENT="$MODELS/dml_b_s2026.pt"
require "$PARENT"

PSEUDO="$PROCESSED/labels/pseudo_dmlb_fold0.csv"
require "$PSEUDO"

ARMPY="uv run --no-sync --project $REPO python"

for SEED in 409 410 411 412; do
  run_train "transduce_dmlb_c$SEED" \
    $RECIPE --epochs 4 --slice-pooling labelwise --whole-stack $THROUGHPUT \
    --backbone-lr 2e-5 --slices 24 --vram-gb 22 --band-low 0.1 --band-high 0.9 \
    --calibrated-targets --hold-annotations --ema \
    --batch 4 --accumulate 3 --slot-dropout 0.40 \
    --init "$PARENT" --init-scope all --seed "$SEED" --pseudo-fold 0 \
    --labels "$PSEUDO" --eval-labels "$LABELS" --out "$MODELS/transduce_dmlb_c$SEED.pt"
done

$ARMPY -m kneemri.evaluation.ledger75 --holdouts "$MODELS" \
  --only transduce_dmlb_c407 transduce_dmlb_c408 transduce_dmlb_c409 transduce_dmlb_c410 \
         transduce_dmlb_c411 transduce_dmlb_c412 \
         basinsoup_dmlb_c401 basinsoup_dmlb_c402 basinsoup_dmlb_c403 \
         basinsoup_dmlb_c404 basinsoup_dmlb_c405 basinsoup_dmlb_c406 dml_b_s2026
