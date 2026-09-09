#!/usr/bin/env bash
# v3small_control.sh — the plain v3small dense run that two spent arms and one running arm are read against.
#
# reading: v3small_control_s2026.holdout.npz
# deliberate: the 12-epoch dense recipe on v3small with NO encoder-weights and NO label-errors
# bead: knee-j2ovn
#
# WHAT IS MISSING. `latent_register_dense` finished at 0.8385 and there is nothing on disk that prices it.
# Its own header said so before it ran: `band_dense`'s 0.8578 is a dinov2-small number and cannot price a
# v3small encoder. The only v3small run at this exact supervised recipe is `errors_v3small_s2026` (0.8280),
# and it differs in TWO ways at once — it carries `--label-errors` and it carries no adaptation. On dinov2
# that same flag reads `errors_dense` 0.8420 against `band_dense` 0.8578, so it costs -0.0158 there. Carry
# that across and a plain v3small lands near 0.844, which puts 0.8385 INSIDE [0.828, 0.844]: the latent
# adaptation at mask 0.30 is neither shown to help nor shown to hurt. Two arms have now been spent on an
# axis whose zero point does not exist, and a third (`latent_register_deep`, mask 0.60) is on the card.
#
# WHY A TRANSFERRED CONSTANT IS NOT A CONTROL. -0.0158 is a knob-sized delta measured on a different trunk
# and substituted into an estimate for this one. This repo has priced that move wrong before. The number
# this arm produces is measured on the trunk it describes, at the recipe it describes, and it is one flag
# removed from each of the three runs it prices — which is what makes each of those a one-variable read.
#
# PRICE: ~2.0 h of card, no pretrain stage. It is an INSTRUMENT, not a member candidate: its own score is
# not the deliverable and no branch here promotes anything. P(clears 1.5%) ~ 0.05 as a member — it is a
# stock trunk at a recipe we have run a dozen times. What it buys is that three existing holdout numbers
# stop being unreadable, including one still being written. A member that happens to seat is a bonus and
# is priced by `seat_value` afterwards like any other.
#
# PRE-REGISTERED, before the run:
#   1. Lands near 0.844. Then the latent adaptation at 0.30 is flat-to-negative and `latent_register_deep`
#      is the last rung this axis gets; the family closes on a measured zero rather than on an estimate.
#   2. Lands near 0.828. Then the adaptation is worth +0.010 directional, `--label-errors` is not what
#      cost `errors_v3small` its ground, and the mask ladder is a live axis.
#   3. Lands outside [0.820, 0.850]. Then the recipe itself behaves differently on this trunk than the
#      transfer assumed, and every v3small number in the ledger read against a dinov2 control is suspect.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train v3small_control \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant v3small --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --labels "$LABELS" --out "$MODELS/v3small_control_s2026.pt"
