#!/usr/bin/env bash
# bead: knee-67rgi
# reading: resnet_stack_224_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   `resnet_224_short`'s trunk under OUR recipe stack instead of the foreign baseline one. The trunk,
#   the resolution and the epoch budget are held; the head, the targets and the regularisation move.
#
# WHY. THE CNN IS TIED WITH THE TRANSFORMER AND HAS NEVER BEEN GIVEN THE STACK.
#   At matched 224 px on the same 909 holdout:
#     resnet_224_short_s2026   resnet50d   0.8527 midpoint / 0.8842 ledger75
#     native224                DINOv2-S    0.8502
#     base224                  DINOv2-B    0.8509
#   +0.0025 is inside the 0.0062 seed spread, so that is a TIE, not a win. It is a tie reached by a
#   family with zero recipe tuning against a family tuned for two months. The stack that did the tuning
#   — labelwise pooling, whole-stack window, calibrated targets, held annotations, slot dropout,
#   stochastic depth — took DINOv2 from 0.8502 to 0.867 midpoint / 0.9073 ledger75. It has never once
#   been run on a convolutional trunk. `cnnseat_v1`'s own note states what the gap needs: ONE member at
#   ~0.874 offline against our best honest ~0.866.
#
# THIS IS NOT THE CLOSED CNN AXIS. That closure (README:7796) is about `radnet_family` 0.6716,
#   `radnet_pinned` 0.7505, `cnx_dense` 0.7941, `coatnet_hybrid` 0.8259 — foreign-corpus and hybrid
#   trunks a full point below the seat line, priced as ENSEMBLE DIVERSITY and found worthless there.
#   It names no member above 0.83 and says nothing about an ImageNet CNN at 0.8842 ledger75 under a
#   recipe it never saw. Two different findings wearing one word.
#
# deliberate: --epochs 12 — `resnet_224_short`'s exact budget, so the contrast against 0.8527 isolates
#             the STACK and not the schedule. `resnetlong_v1` already closed 48 epochs on both rulers.
# deliberate: --backbone-lr 2e-4 --head-lr 2e-4 — the CNN's OWN rate, not the 4e-5 tuned on DINOv2.
#             A ViT rate on a conv trunk under-drives it, which is the `raddino_dense` failure: a
#             shallow slope there was as consistent with an under-driven backbone as with a trunk that
#             does not transfer, and it cost that arm its verdict.
# deliberate: --unfreeze-last 16 — resnet50d's own block count (3+4+6+3). `--whole-stack` selects the
#             SLICE WINDOW and does NOT open blocks; with it alone the log reads `pins 36 frozen norm
#             layer(s)` and the arm fits a frozen trunk. That flag is worth +0.0134 of the gap between
#             `resnet50d_short` 0.8393 and `resnet_224_short` 0.8527.
# deliberate: --slices 12 --band-low 0.2 --band-high 0.8 — `images_224px_20sl_0.1-0.9band.npy` is
#             DELETED and only the plain 12-slice cache survives, and it is banded 0.20-0.80. The ledger
#             prices the slice-count axis at -0.0001, flat, so the slice term is close to free. THESE ARE
#             THE TWO TERMS NOT MATCHED to `resnet_224_short`'s 20 slices at 0.1-0.9, they are stated
#             rather than inherited in silence (knee-aixw), and both narrow the window this arm sees —
#             so they bias AGAINST it, and a number that clears 0.8527 clears it despite them.
# deliberate: --slice-pooling labelwise --calibrated-targets --hold-annotations --slot-dropout 0.40
#             --stochastic-depth 0.05 — the stack itself, the variable under test.
# PRECISION AND COMPILE ARE BOTH LEFT AT THEIR DEFAULTS, AND THAT IS THE MEASURED CHOICE, not an
#   omission. bf16 is the default (`config_tree.py:131`; `--fp16` is the escape) and is what this fit
#   wants — bf16 carries fp32's exponent range and needs no loss scaler, which is the corner a 2e-4
#   whole-stack rate is worst at under fp16. torch.compile is also the default (`--eager-encoder` is the
#   escape) and is measured here at -21% per step against a 2.2 s one-off compile, score-neutral
#   (0.7604 plain / 0.7606 compiled). Under a co-tenant the instinct is to compile OFF, on the theory
#   that fused kernels queue the neighbour's bursts behind ours; that is reasoning against a
#   measurement. Finishing sooner is less total contention, so it stays ON.
#
# CO-TENANCY. THE CARD IS SHARED BY THE OWNER'S ARRANGEMENT AND MUST NOT BE THRASHED.
#   biohub is resident at 9.6 GB, SM 14-58% (mean ~36), memory controller 5-29% (mean ~16) — bursty and
#   NOT saturating, so the headroom is real but lives in the gaps between its bursts. The design rule
#   that follows is the inverse of normal throughput advice: MANY SHORT KERNELS, NOT FEW LONG ONES, so
#   batch stays at 2 and `--accumulate 4` restores the effective 8 rather than a larger micro-batch.
#   `vram_need` derives 5319 MiB for these flags; `--vram-gb 6` is that with no slack, because
#   `device_budget` records 2026-08-17 — an over-commit paged over PCIe, drew three nvlddmkm engine
#   resets, and took BOTH projects' processes down. Slack is not free when a co-tenant holds the card.
#   BatchNorm still sees ~144 images per micro-batch, an order past the 32 where Wu & He's curve flattens.
#   torch.compile is deliberately NOT used: it fuses into larger kernels, which is precisely what would
#   queue biohub's bursts behind ours. `channels_last` is absent from this codebase and is not taken.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train resnet_stack_224 \
  $RECIPE --epochs 12 --variant resnet50d --image-size 224 \
  --through-plane --slice-pooling labelwise --slices 12 --band-low 0.2 --band-high 0.8 \
  --whole-stack --unfreeze-last 16 --backbone-lr 2e-4 --head-lr 2e-4 --weight-decay 0.05 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 --stochastic-depth 0.05 \
  --batch 2 --accumulate 4 --vram-gb 6 \
  --labels "$LABELS" --out "$MODELS/resnet_stack_224_s2026.pt"
