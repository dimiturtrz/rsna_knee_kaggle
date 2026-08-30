#!/usr/bin/env bash
# speed_combo.sh — compile AND bf16, which are orthogonal and were measured only separately.
#
# THE ERROR THIS CORRECTS: the throughput arm ran plain, compiled and bf16 as three alternatives, and
# the summary read as though bf16 were an alternative TO compile. It is not. compile fuses kernels;
# bf16 changes the autocast dtype. Nothing about one excludes the other, and their measured gains
# (+48% and +6% over plain) have no reason to be exclusive either.
#
# THERE IS A MECHANISM FOR MORE THAN THE SUM, worth stating so the result can refute it: bf16 removes
# the GradScaler entirely - its exponent range is fp32's, so there is nothing to scale - and that
# removes the per-step inf/nan check and unscale that inductor otherwise has to leave standing between
# fused regions. Whether that is visible at this size is exactly the question.
#
# EXPECT LITTLE. The honest prior is that this lands at compiled's 1206 img/s plus bf16's few percent,
# not above it, because the two act on different costs and the card is already at 96% of its power
# budget. A one-epoch run is cheap enough that "probably additive" is not worth assuming.
#
# controls, both one epoch of the same recipe on an idle card:
#   compiled  0.7606 holdout, 272 s, 1206 img/s
#   bf16      0.7589 holdout, 322 s,  865 img/s
#   plain     0.7600 holdout, 365 s,  817 img/s
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
# twin-of: throughput_bf16
# twin-of: throughput_compiled
# Same flag line as either half of `throughput.sh`: compile and bf16 are ENVIRONMENT, not flags, and
# this arm runs both at once where that shell ran each alone. The identity is what makes the three
# wall-clocks comparable.
#
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train speed_combo \
  $RECIPE --epochs 1 \
  --slice-pooling max --whole-stack \
  --recompute-activations \
  --labels "$LABELS" --out "$PROCESSED/speed_combo.pt"

echo "=== steady-state img/s, later steps only ==="
grep -oE "[0-9]+ img/s" "$LOGS/speed_combo.full.log" | tail -5 | tr '\n' ' '
echo
