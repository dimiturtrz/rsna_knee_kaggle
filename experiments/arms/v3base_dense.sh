#!/usr/bin/env bash
# v3base_dense.sh — the champion's newest member, with the ENCODER FAMILY as the only variable.
#
# WHY THIS ARM AND NOT `v3base_band.sh`. Both ask about DINOv3. That one holds LoRA r8 over a frozen
# trunk at 5e-4, because its control is `dinov3_base_lora` (0.8268) and single-variable discipline said
# to move only the data flags. What happened on 2026-08-20 makes that the wrong control to be tied to:
# `base_dense` — DINOv2-base, FULLY adapted at 4e-5 on this exact recipe — scored 0.8512 solo, LOST its
# matched control by 0.0066 (outside the 0.0062 floor), and then took the board from 0.930 to 0.931 as
# the fifth member of `basedense_v1`. Every mounted DINOv3 member is a LoRA adapter over a frozen trunk
# and both are the pool's weakest (0.8215, 0.8268, taken 16th and 17th by greedy over eighteen). So the
# pool has never seen a fully-adapted non-DINOv2 encoder, and "DINOv3 is weak" is a claim measured only
# under the adaptation regime that also produces this ledger's worst members.
#
# SINGLE VARIABLE, and it is the encoder. This is `base_dense.sh`'s invocation with `--variant v3base`
# substituted for `--variant base`. Rate, slices, band, pooling, whole-stack, epochs, schedule and batch
# are all held at the values `base_dense`'s 0.8512 was measured under. That makes `base_dense` the
# matched control and the comparison a clean family swap.
#
# BATCH 6 IS INHERITED, NOT CHOSEN, and it is no longer a confound. `base_dense` ran at 6 because 336 px
# x 20 slices x 768 wide does not fit in 32 GB at 12, and it carried that as a named suspect. That
# suspect is now DEAD: `batch_teardown` arm 1 (`small_batch6`) ran the control's own recipe at batch 6
# on the encoder where 12 fits and peaked at 0.8598 against 0.8578 — +0.0020, inside the floor, on the
# positive side. Holding 6 here therefore matches the control without importing a bias.
#
# VRAM SHOULD BE EASIER THAN THE CONTROL'S, NOT HARDER. DINOv3-base is 768-wide like DINOv2-base but
# tiles at patch 16 rather than 14, so 336 px is 441 tokens against DINOv2's 576 — about 23% fewer, and
# attention is quadratic in them. `base_dense` peaked at 27.85 GB reserved. SMOKE IT ANYWAY before
# queueing: the four register tokens and RoPE are not free, and a smoke costs two minutes against an arm
# that costs two hours.
#
# WHAT THE ANSWER IS WORTH BEYOND THE MEMBER. If a fully-adapted v3base clears its LoRA sibling by the
# ~0.02 that separates adapted from adapted-frozen elsewhere in this ledger, then "DINOv3 is weak" was
# never an encoder finding — it was an adaptation finding wearing an encoder's name, and TWO mounted
# members have been mis-attributed since 2026-08-16.
#
# control  base_dense = 0.8512 (best epoch 8), byte-identical recipe on DINOv2-base.
# seat     comp/champion.json basedense_v1 = 0.8690 over the 909-study fold-0 holdout, five members.
# cost     base_dense ran ~920 s/epoch; ~23% fewer tokens suggests ~750-850 s, so roughly 2.5-2.9 h.
#
# PRE-REGISTERED, four outcomes. Floor 0.0062.
#   PREDICTION — solo lands between 0.84 and 0.86, i.e. within a floor or so of `base_dense`, AND earns
#     a seat against `basedense_v1` on the mechanism that member just demonstrated: a different
#     pretraining corpus, teacher and patch grid shape errors the five incumbents do not have. Seat
#     value is the number that matters and it is NOT predicted by the solo score — `base_dense` is the
#     proof, having lost its control outside the floor and won the board anyway.
#   REFUTATION OF THE ADAPTATION REREAD — solo at or below its LoRA sibling's 0.8268. Full adaptation
#     does not rescue v3 here, "DINOv3 is weak on this task" survives as an encoder finding, and the
#     two mounted LoRA members keep their current attribution.
#   THE INTERESTING NEGATIVE — solo clears 0.84 and seat value is at or below zero against the
#     five-member champion. Encoder diversity paid ONCE (base_dense) and does not pay TWICE, which
#     would say the fifth seat was bought by DINOv2-base specifically and not by family diversity as
#     such — and would make the next arm a sixth member from a genuinely different INDUCTIVE BIAS (a
#     CNN trunk), not a fourth ViT.
#   VRAM REFUSAL — if it OOMs at batch 6 despite the token count, do NOT hand-tune the batch down to 4.
#     That reintroduces the variable `small_batch6` just removed at 6 and not at 4. Take it to
#     `--accumulate 2` at batch 3 only after `small_accum` has reported.
#
# THE VRAM REFUSAL FIRED ON THE FIRST RUN AND IT WAS THE CAP, NOT THE RECIPE (2026-08-20 22:55Z):
# `Tried to allocate 946.00 MiB ... 9.17 GiB is free; 21.68 GiB allowed`. Nine gigabytes of the card sat
# unused while the run was refused — the same shape knee-kwdh fixed for `v3base_band`, one term further
# along. The derived cap is 1.335 x 20 slices x (441/576 tokens at patch 16) x (768/384 width) x (6/12
# batch) + 0.22, which is 21.7. Every factor there is right except the intercept: `RESIDENT_GB` is 0.22,
# fitted on DINOv2-SMALL, and it stands for weights plus gradients plus Adam state — quantities that
# scale with PARAMETER COUNT (22M for small, 86M for either base, so ~1.4 GB rather than 0.22). The
# slice term learned to read the encoder; the intercept did not.
#
# So the batch is NOT hand-tuned and the refusal clause above is honoured: the recipe is untouched and
# `--vram-gb 27` states the run's own need explicitly, which is the escape the cap is documented to
# provide for exactly this case. 27 is chosen against a MEASURED sibling rather than a formula —
# `base_dense` peaked at 25.38 GB allocated under a 26.92 derived cap, and this run holds the same
# weights at ~23% fewer tokens. The encoder-blind intercept is filed as its own fix; an arm is not the
# place to repair a shared constant.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

HF_TOKEN="$(grep -oE 'hf_[A-Za-z0-9_-]+' /d/life/creds/machine/hfpat.txt | head -1)"
export HF_TOKEN
[ -n "$HF_TOKEN" ] || { echo "MISSING: no Hugging Face token, and DINOv3 is a gated repository"; exit 1; }

run_train v3base_dense \
  $RECIPE --epochs 12 --batch 6 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant v3base --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --vram-gb 27 \
  --labels "$LABELS" --out "$PROCESSED/v3base_dense_s2026.pt"
