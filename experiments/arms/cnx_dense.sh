#!/usr/bin/env bash
# cnx_dense.sh — the first member that is not a vision transformer.
#
# WHY, AND IT IS A CLOSED AXIS RATHER THAN A HUNCH. `base_dense` (DINOv2-base among seven DINOv2-small
# members) took the board from 0.930 to 0.931 while LOSING its own matched control by 0.0066, so what it
# sold was error structure and not accuracy. The obvious next question — does a second family sell the
# same thing — was asked and answered: `v3base_dense` (DINOv3-base, a different corpus, a different
# teacher, a different patch grid, same recipe) is worth -0.0007 as a sixth seat and -0.0005 as a swap.
# Encoder diversity paid ONCE and does not pay twice WITHIN the transformer. Meanwhile the greedy ceiling
# over all eighteen mounted members is 0.8692 against the champion's 0.8690, so subsets are spent and the
# only lever left is a member that is genuinely new.
#
# ConvNeXtV2-base is that: local convolution with a stagewise hierarchy against global attention over a
# flat grid, which is a different INDUCTIVE BIAS rather than different weights. Capacity is held at the
# member that worked — 87.7M parameters against DINOv2-base's 86M — so a result here is about the bias.
# FCMAE pretraining keeps it self-supervised like the DINOs, then IN-22k finetuning, which is itself a
# difference: every incumbent is SSL-only.
#
# THREE DEVIATIONS FROM `base_dense`, and each is forced rather than chosen. State them before the run so
# neither can be reached for afterwards as an explanation.
#
#   1. 352 px, not 336. Four stages halving twice over is stride 32, and 336 does not tile it. 352 is the
#      nearest legal size and gives an 11x11 grid of 121 tokens — a fifth of a ViT's 576 at 336, which is
#      why a convolutional trunk of the same parameter count is cheaper per token and dearer per pixel.
#      It is reached with `--train-size` off the control's OWN 336 px cache, not a second decode: the
#      resample is 4.8% and adds no information, and the alternative is a 45-minute build to hand the
#      model pixels it would have interpolated anyway. So the cache, the band and the slice count are
#      byte-identical to the control's — only the grid the trunk tiles is different.
#   2. NO `--recompute-activations`. `ConvNextV2Model` does not implement gradient checkpointing in this
#      transformers version (`supports_gradient_checkpointing` is False, and asking raises). This is NOT
#      a learning variable: checkpointing recomputes activations and returns bit-identical gradients, so
#      dropping it changes the memory bill and nothing the model learns.
#   3. `--batch 2 --accumulate 3` for an effective 6, matching the control's batch. Without recompute the
#      full activation stack is resident, and 120 images a step at 352 px does not fit. Accumulation is
#      the substitution this project has ALREADY MEASURED rather than assumed: `small_accum` ran the
#      control recipe at batch 6 with `--accumulate 2` and peaked at 0.8579 against `small_batch6`'s
#      0.8598 and the control's 0.8578 — three points inside a 0.0020 spread, all peaking at epoch 8.
#
# `--vram-gb 16` IS NOT A TUNING KNOB, it is the documented escape from a cap that cannot price this
# family. `DeviceBudget.needed()` derives its ceiling from the FINAL token count, and at patch 32 a
# ConvNeXt's 121 tokens buy a 2.84 GB cap while 28 GB of the card sits free — the memory of a
# convolutional trunk lives in its early high-resolution stages, which that formula cannot see. The
# smoke MEASURED the real bill at 10.49 GB reserved for this exact shape, so 16 is that plus half
# again, and it deliberately leaves the rest of the card for the co-tenant.
#
# Everything else is `base_dense`'s: 4e-5, 20 slices, band 0.10-0.90, whole-stack, max slice pooling,
# 12 epochs one-cycle, the standing $RECIPE.
#
# control  base_dense = 0.8512 solo, and it is the RIGHT control because it is the member whose seat this
#          one is competing for. band_dense = 0.8578 is the recipe's own reference point.
# seat     comp/champion.json basedense_v1 = 0.8690 over the 909-study fold-0 holdout, five members.
# cost     unknown, and that is the point of smoking it: fewer tokens per slice but no recompute.
#
# SMOKE IT FIRST AND IT IS NOT OPTIONAL HERE. 352 px is a read size nothing has decoded, so the pixel
# cache for it does not exist; `python -m kneemri.experiments.arm_check --smoke` writes the truncated
# sixty-study one. It is also the only thing that will price the memory before forty minutes are spent
# discovering it, and this is the first trunk whose memory the derived cap was never fitted against.
#
# PRE-REGISTERED, four outcomes. Floor 0.0062, board resolution 0.0015.
#   PREDICTION — solo lands below the ViTs, somewhere in 0.82-0.85, AND earns a seat worth at least
#     +0.0015 against `basedense_v1`. That combination is the whole thesis: this arm is not trying to be
#     the best member, it is trying to be the most DIFFERENT one, and `base_dense` is the standing proof
#     that those come apart.
#   REFUTATION — seat value at or below zero, like `v3base_dense`. Then the fifth seat was bought by
#     DINOv2-base's capacity and not by diversity of any kind, "encoder diversity" stops being an axis
#     this project has evidence for, and the next member has to come from the DATA side (through-plane
#     structure, knee-ec0p) rather than the encoder side.
#   THE EXPENSIVE NEGATIVE — solo below 0.80. The head, the band and the slice pooling were all settled
#     on token sequences from a patch-14 ViT; a 121-token grid may simply be too coarse for the slot
#     head's spatial pooling, in which case the reading is about RESOLUTION and the retry is 448 px
#     (a 14x14 grid) before the trunk is judged.
#   VRAM — if it will not fit at batch 2, take accumulation up rather than slices or resolution down.
#     Slices and band are load-bearing (whole-stack reading is the +0.0147 lever); the batch is measured
#     not to be.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train cnx_dense \
  $RECIPE --epochs 12 --batch 2 --accumulate 3 --slice-pooling max --whole-stack \
  --variant cnxbase --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --train-size 352 \
  --vram-gb 16 \
  --labels "$LABELS" --out "$PROCESSED/cnx_dense_s2026.pt"
