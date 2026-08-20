#!/usr/bin/env bash
# v3base_band.sh — the settled DATA recipe carried onto a DIFFERENT ENCODER FAMILY.
#
# THE QUESTION IS NOT "IS DINOv3 BETTER". That one is answered and the answer is no: every v3 run we
# hold loses to DINOv2 (best v3 0.8268 base-under-LoRA at 224, against DINOv2's 0.8587). This arm asks
# the thing that question hid — whether the +0.012 of recipe our ledger has bought since those runs is
# a property of the DATA or a property of DINOv2. Every v3 number was taken under the OLD operating
# point (8e-6 or LoRA's 5e-4, band 0.20-0.80, 12 slices) and none of them got the band widening
# (+0.0039), the density (+0.0014) or the rate work that followed. Nobody has run a v3 at the recipe we
# actually ship.
#
# WHY THAT MATTERS MORE THAN THE MEMBER. Every decode finding in this ledger — span is an interior
# optimum at 0.10-0.90, density buys focal labels, crop and span are antagonists — was measured on ONE
# encoder. If those are facts about knee MRI they must reproduce on an encoder distilled from a 7B
# teacher with a different patch grid, RoPE position and 4 registers. If they are facts about DINOv2's
# feature geometry, the ledger's whole decode chapter is narrower than it reads, and that is worth
# knowing before another arm is planned against it.
#
# WHY IT IS ALSO THE ROSTER'S BAREST AXIS. Seventeen mounted members and sixteen are DINOv2. The pool's
# diversity is RECIPE diversity almost entirely, and the two exceptions are the evidence that the other
# kind pays: deep24 is the WORST member in the file (0.8213, ranking term off) and greedy forward
# selection takes it anyway, because its errors are not shaped by the objective the other sixteen share.
# An encoder swap is that same argument with a stronger prior — a different pretraining corpus, teacher
# and patch grid shape errors more thoroughly than a dropped loss term does.
#
# SINGLE VARIABLE, and it is the DATA flags, not the encoder. This is dinov3_base_lora.sh's invocation
# with --slices 20 --band-low 0.1 --band-high 0.9 substituted for the defaults it inherited. Variant,
# LoRA rank, rate, image size and train size are all held at the values its 0.8268 was measured under.
#
# 224 px TRAIN SIZE STAYS. Base is 768-wide against small's 384; base_lora peaked at 16.1 GB with 12
# slices, so 20 slices lands near 27 GB against band_dense's measured 26.9 GB on the same card. That
# fits and it is the reason the pixel rungs are not touched here — one variable, and the VRAM budget
# only has room for that one.
#
# control  dinov3_base_lora = 0.8268 (v3base, LoRA r8, 5e-4, 336/224 px, 12 slices, 0.20-0.80, 12 ep).
# seat     comp/champion.json greedy4_v1 = 0.8679 over the 909-study fold-0 holdout.
# cost     324 s/epoch at 12 slices scales to about 540 s at 20, so roughly 1.8 h for 12 epochs.
#
# PRE-REGISTERED, three outcomes:
#   PREDICTION — solo clears 0.838, i.e. the band+density recipe carries about its DINOv2 size (+0.012)
#     across the encoder family, AND the member earns a seat against greedy4_v1 despite a solo score
#     ~0.02 under every incumbent, on the deep24 mechanism (different errors, not better ones).
#   REFUTATION — solo at or below its own 0.8268 control. The recipe does not transfer, which demotes
#     every decode number in this ledger from "a fact about the data" to "a fact about DINOv2", and the
#     span/density/crop chapter has to be re-read with that scope written on it.
#   THIRD OUTCOME — solo clears 0.838 and seat value is at or below zero. The recipe transfers and
#     encoder diversity still does not pay in a four-member RankAverage vote, which retires the axis
#     and says the roster's recipe-diversity is not a poverty but a finding.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

HF_TOKEN="$(grep -oE 'hf_[A-Za-z0-9_-]+' /d/life/creds/machine/hfpat.txt | head -1)"
export HF_TOKEN
[ -n "$HF_TOKEN" ] || { echo "MISSING: no Hugging Face token, and DINOv3 is a gated repository"; exit 1; }

run_train "v3base_band" \
  $RECIPE --epochs 12 $PATIENCE \
  --slice-pooling max --whole-stack --lora-rank 8 --backbone-lr 5e-4 \
  --variant v3base --image-size 336 --train-size 224 \
  --slices 20 --band-low 0.1 --band-high 0.9 \
  $THROUGHPUT \
  --labels "$LABELS" --out "$PROCESSED/v3base_band_s2026.pt"
