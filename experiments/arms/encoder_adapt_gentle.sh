#!/usr/bin/env bash
# encoder_adapt_gentle.sh — the same adaptation, at a rate that does not destroy what it adapts.
#
# WHY THIS ARM EXISTS. `encoder_adapt` lost 0.1676 uniformly (focal -0.1680, regional -0.1669, all
# twelve labels down between -0.11 and -0.24). A uniform loss is an INITIALISATION reading, not a
# mechanism reading: by that arm's own pre-registered clause it never tested the spectral term at all.
# `kneemri.evaluation.encoder_drift` says what happened instead — six epochs at lr 1e-4 moved encoder
# blocks 0-5 by 3.4-8.3% of their own norm, where supervision moves those six by EXACTLY 0.0000 and has
# never moved any block anywhere by more than 2.1%. Six epochs cannot rebuild a representation; it can
# only damage one. And `--unfreeze-depth 6` freezes blocks 0-5, so the fine-tune that could not overwrite
# the damage could not repair it either — the precondition this axis was built on cuts both ways.
#
# ONE VARIABLE AGAINST THAT ARM. Objective, data, masking, schedule shape and epoch count are unchanged;
# only the pretraining RATE moves, 1e-4 -> 1e-5. The fine-tune below is byte-identical to `band_dense`
# except `--encoder-weights`, so it is also one variable against the control.
#
# THE GATE WAS PRE-REGISTERED AND IS MEASURED, not asserted. A checkpoint may spend a card slot only if
# its early-block drift lands inside supervision's own range (<= ~0.02). This one:
#   stem 0.0174 | b0 0.0179  b1 0.0083  b2 0.0136  b3 0.0046  b4 0.0093  b5 0.0109
#                 b6 0.0104  b7 0.0110  b8 0.0154  b9 0.0425  b10 0.1216  b11 0.2152
# Largest early block 0.0179, and 0.821 of all drift sits in the top four blocks. The adaptation still
# HAPPENED — the spectral term fell 1.4482 -> 0.6196 over the six epochs — it just spent itself where a
# representation can afford to move. A one-minute CPU read now guards seventy-five minutes of card.
#
# THE FALSIFIABLE PART, unchanged from `encoder_adapt` because the hypothesis is unchanged and only the
# harness was broken: the gain must CONCENTRATE IN THE FOCAL FINDINGS — Medial Meniscus, Lateral
# Meniscus, ACL, MCL, PF OA. Effusion, Baker's, Contusion and Synovitis are large, diffuse, already our
# best labels, and are the frequencies the high-pass term discards; they should barely move. A UNIFORM
# LIFT credits a better initialisation, which is a real result but a DIFFERENT one. A LIFT IN THE
# DIFFUSE FOUR refutes the stated mechanism outright.
#
# AND THE THIRD OUTCOME THIS ARM CAN NOW PRODUCE, which the first could not: a result INDISTINGUISHABLE
# from the control. That would mean the rate is low enough to be safe and too low to matter, and it
# brackets the axis rather than killing it — 1e-4 destroys, 1e-5 does nothing, and the answer if any is
# between them.
#
# control  band_dense = 0.8578 (best epoch 8), byte-identical recipe with the flag off.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
ADAPTED="$MODELS/adapt_small_e6_lr1e5.pt"
require "$LABELS" "$ADAPTED"

run_train encoder_adapt_gentle \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --encoder-weights "$ADAPTED" \
  --labels "$LABELS" --out "$MODELS/encoder_adapt_gentle_s2026.pt"
