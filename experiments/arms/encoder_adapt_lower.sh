#!/usr/bin/env bash
# encoder_adapt_lower.sh — import the half of the adaptation the argument was ever about.
#
# WHY THE FIRST TWO ARMS DID NOT TEST THIS AXIS. `encoder_adapt` (lr 1e-4) scored 0.6902 and
# `encoder_adapt_gentle` (lr 1e-5) scored 0.6875, against a 0.8578 control. Two adaptations whose early
# blocks differ FOURFOLD in drift landing 0.0027 apart is not a reading about encoder quality — if the
# quality of the initialisation were the variable, those two numbers would not agree. Something they
# SHARE carried the collapse.
#
# WHAT THEY SHARE, measured with `kneemri.evaluation.encoder_drift`: a rewritten top. Block 11 moved
# 0.5510 and 0.2144 of its own norm; supervision has never moved that block more than 0.0080. The head
# reads block 11 and the final norm, so both arms handed the classifier a representation reshaped for
# reconstruction, and a 4e-5 fine-tune over twelve epochs cannot walk 21-55% back.
#
# AND THE ORIGINAL ARGUMENT NEVER ASKED FOR THAT TOP. It was about the BOTTOM: `--unfreeze-depth 6`
# freezes blocks 0-5, supervision's drift there is exactly 0.0000 in every member ever shipped here, so
# whatever self-supervision writes into them is information the fine-tune has no gradient to overwrite —
# and, symmetrically, no gradient to repair. `--encoder-adapt-depth 6` states that argument exactly:
# take the patch embedding and blocks 0-5 from the adaptation, leave blocks 6-11 and the final norm
# stock. The final norm is left with the blocks it is calibrated to, deliberately.
#
# THE ADAPTATION IT IMPORTS ALREADY CLEARED THE DRIFT GATE: adapt_small_e6_lr1e5.pt reads 0.0046-0.0179
# across blocks 0-5, inside supervision's own 0.0000-0.0225 range, while its spectral term still fell
# 1.4482 -> 0.6196. So the weights being imported here moved, and moved gently.
#
# ONE VARIABLE against the control (`--encoder-weights` plus the depth that scopes it), and one variable
# against `encoder_adapt_gentle` (the depth alone, same checkpoint).
#
# THE FALSIFIABLE PART. Three outcomes, separated before the run:
#   * A LIFT CONCENTRATED IN THE FOCAL FINDINGS (Medial Meniscus, Lateral Meniscus, ACL, MCL, PF OA)
#     supports the stated mechanism: a high-pass reconstruction objective bought early-layer features
#     for small structure. The diffuse four (Effusion, Baker's, Contusion, Synovitis) should barely move.
#   * A RECOVERY TO ROUGHLY THE CONTROL, with no focal concentration, confirms the DIAGNOSIS above — the
#     top blocks were the whole collapse — and says the frozen half carries nothing either way. That
#     closes the axis honestly rather than leaving 0.17 unexplained.
#   * A COLLAPSE AGAIN, anywhere near 0.687, refutes the diagnosis outright: the damage would not be in
#     the top blocks, and the six frozen ones would be doing something no drift number predicts. That
#     outcome sends this back to the harness, not to the hypothesis.
#
# control  band_dense = 0.8578 (best epoch 8), byte-identical recipe with the flag off.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
ADAPTED="$PROCESSED/adapt_small_e6_lr1e5.pt"
require "$LABELS" "$ADAPTED"

run_train encoder_adapt_lower \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --encoder-weights "$ADAPTED" --encoder-adapt-depth 6 \
  --labels "$LABELS" --out "$PROCESSED/encoder_adapt_lower_s2026.pt"
