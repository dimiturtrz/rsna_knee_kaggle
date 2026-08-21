#!/usr/bin/env bash
# encoder_adapt_thawed.sh — the cheapest of the three fixes encoder_adapt_lower's own row prescribes.
#
# WHERE THIS COMES FROM. `encoder_adapt_lower` imported ONLY the stem and blocks 0-5 from a drift-gated
# adaptation and still lost 0.1269 against `band_dense` 0.8578. That refuted the diagnosis it was built
# on — the rewritten TOP was not the collapse, because the top was left stock and the collapse survived.
# Its row names what is left: "converge the MAE first, on the arm's own cache, and either do not freeze
# the adapted blocks or unfreeze deeper." The first two cost a fresh adaptation run. The third is free,
# and it is the one that tests the mechanism the row actually established.
#
# THE MECHANISM IT TESTS, and the instrument is the TRAINING LOSS rather than the AUC. Control
# `band_dense` runs 1.2162 -> 0.5780 over twelve epochs; all three encoder arms floor near the control's
# epoch-ZERO loss (1.3452 -> 1.1978, 1.3393 -> 1.1960, 1.2874 -> 1.1073, the last still falling at epoch
# 11). Those arms are not fitting worse, they are barely fitting. The stated reason is that the head sits
# on FROZEN blocks 0-5 rewritten for spectral reconstruction, and the six blocks supervision can write
# cannot re-encode what the frozen half no longer represents. If that is true, giving supervision a
# gradient into blocks 0-5 has to move the loss curve back toward the control's, and the loss is what
# this arm is read on first.
#
# ONE VARIABLE against `encoder_adapt_lower` (0.7309): `--unfreeze-last` 6 -> 12, same checkpoint, same
# depth, same cache, same rate, same schedule. It is TWO variables against `band_dense`, which is why
# the primary control here is the adaptation arm and not the shipped recipe.
#
# THE INTERPRETIVE TRAP, named before the run because after it it is a rationalisation. Unfreezing every
# block also lets supervision OVERWRITE the imported weights, so a recovery to roughly control does NOT
# say the adaptation helped — it says the import was neutralised, which is the null. The three outcomes
# are therefore separated by WHERE the number lands, not by whether it improves:
#   * ABOVE control by more than the 0.0062 floor: the only outcome that supports on-domain SSL. The
#     early blocks carried something supervision wanted and could not have learned from 4,407 studies.
#   * ROUGHLY CONTROL (within the floor): the frozen-bottom diagnosis is CONFIRMED as the cause of the
#     0.127, and the adaptation itself is worth nothing — supervision simply erased it. The axis closes
#     with the collapse finally explained, which is the outcome `encoder_adapt_lower` could not reach.
#   * STILL NEAR 0.73: the frozen bottom was not the cause either, and that is the third diagnosis this
#     axis has refuted. At that point the remaining suspects are the adaptation's own two faults (six
#     unconverged epochs, the DEFAULT 12-slice cache against an arm that trains 20 slices at 0.10-0.90),
#     and no further fine-tune arm should run until a converged, on-cache adaptation exists.
#
# COST NOTED HONESTLY: twelve trainable blocks instead of six raises activation and optimizer memory
# over `band_dense`'s 26.92 GB peak, and the card is shared. Smoked before it is queued.
#
# control  encoder_adapt_lower = 0.7309 (best epoch 11), byte-identical but --unfreeze-last 6.
# reference band_dense = 0.8578 (best epoch 8), the shipped recipe with no adaptation at all.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
ADAPTED="$PROCESSED/adapt_small_e6_lr1e5.pt"
require "$LABELS" "$ADAPTED"

run_train encoder_adapt_thawed \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --unfreeze-last 12 \
  --encoder-weights "$ADAPTED" --encoder-adapt-depth 6 \
  --labels "$LABELS" --out "$PROCESSED/encoder_adapt_thawed_s2026.pt"
