#!/usr/bin/env bash
set -euo pipefail
#
# mricore_corpus - does the pretraining corpus pay once it is MRI, and once it is 6.1M images?
#
# WHAT THIS CHANGES, and it is one variable off two MEASURED points. `radiodino_corpus` is this exact
# script with `--variant radiodinob16`; `v3base_dense` is this exact recipe on the natural-image twin.
# Nothing else moves - same ViT-B/16, same DINO-family objective, same 336 px, same 20 slices, same
# band, same max slice pooling, same batch 2 accumulate 3, same 4e-5, same 12 epochs. The registry row
# for `mricore` carries the mount argument; the arm carries only the question.
#
# THE THIRD POINT ON A LINE THAT CURRENTLY HAS TWO. The corpus axis reads: natural LVD-1689M 0.8455,
# radiology RadImageNet 1.35M 0.8214. That is a 2.4-point COST for medical pretraining at fixed
# architecture and fixed objective, and it is the cleanest single-variable reading this project owns on
# `knee-m69u`. But both medical points so far are PROJECTION imaging - radiographs - and both are about
# 1.3M images. MRI-CORE is 6.1M MRI slices across eighteen body locations, the knee among them. So it
# moves the two things the existing cost cannot separate: the modality (projection to volumetric, which
# is the contrast mechanism our task actually reads) and the scale (4.5x the images).
#
# THE COUNTER-EVIDENCE, stated before the run rather than discovered after it. Every medical-corpus
# trunk this project has fine-tuned has cost: `radnet_family` 0.6716, `radnet_pinned` 0.7505,
# `radiodino_s16` 0.8108, `radiodino_corpus` 0.8214, `raddino_dense` 0.8137. Five arms, one direction.
# `knee-y59c` measured solo strength anti-correlating with distance from DINOv2 across seven trunks and
# `radiodino_corpus` showed that anti-correlation surviving when the distance is CORPUS ALONE - which is
# the strongest form the prior against this arm can take, and it is a measurement rather than a hunch.
#
# What survives that counter-evidence is the modality, and it is why this is worth a card rather than a
# probe. Nothing in the five arms above was pretrained on MRI. A radiograph and a knee MRI share almost
# no low-level statistics: one is a projection through the whole limb, the other a sectional read whose
# findings ARE the tissue contrast. If "medical corpus costs" is really "projection corpus costs", this
# arm separates them; if it is "any corpus that is not LVD-1689M costs", this arm confirms it on the
# last modality the axis had left and the corpus axis closes for good rather than by exhaustion.
#
# THE PRICE, before it runs. ~2.1 h of card at ~640 s an epoch x 12, measured on `radiodino_corpus`
# which is the same architecture at the same read. The plausible gain is bounded by the natural twin:
# reaching 0.8455 ties `v3base_dense` and reaching `band_dense`'s 0.8578 would be the first foreign
# trunk ever to lead here, which nothing predicts. So this arm CANNOT deliver 1.5% on its own and is
# not sold as one. It is priced as the instrument that CLOSES OR RE-OPENS THE CORPUS AXIS, and that
# matters beyond the row because `radiodino_corpus`'s seat priced at -0.0008 and the argument for
# declining every future medical trunk currently rests on two projection-imaging points.
#
# control  radiodino_corpus = 0.8214 (best epoch 8), byte-identical flags except the variant.
#          Second anchor v3base_dense = 0.8455, the same recipe on the natural-image twin.
#
# PRE-REGISTERED against radiodino_corpus = 0.8214 and v3base_dense = 0.8455, written before the run,
# with the 0.0062 holdout floor setting every boundary:
#
#   s >= 0.8455            IN-DOMAIN MRI PRETRAINING CLOSES THE CORPUS GAP. The medical-corpus cost was
#              a MODALITY cost, not a corpus-kind cost, and the axis re-opens: the next arms are this
#              trunk at the incumbent recipes and a seat price against honest12, and every medical-trunk
#              decline taken on `radiodino_corpus`'s argument is re-opened by argument rather than by
#              re-quoting its number.
#   0.8455 > s >= 0.8276   BEATS THE RADIOGRAPH TWIN, TRAILS THE NATURAL ONE. MRI or scale buys part of
#              the gap back. The trunk is a seat candidate on diversity - it would sit inside the ~5-point
#              NVIDIA IJCAI-ECAI 2022 boundary that `radiodino_corpus` also cleared and still returned
#              zero on, so the seat is PRICED against honest12 before it is claimed, never assumed.
#   0.8276 > s >= 0.8152   TIES THE RADIOGRAPH TWIN. 6.1M MRI slices buy nothing over 1.35M radiographs:
#              the medical-corpus cost is about the corpus KIND and is invariant to both modality and
#              4.5x scale. THE CORPUS AXIS CLOSES on a measurement rather than on exhaustion, and that
#              closure is what licenses spending no further card on a pretraining corpus.
#   s < 0.8152             Worse than the radiograph twin by more than the floor. SUSPECT THE MOUNT
#              BEFORE THE TRUNK, in this order: the normalisation (this row declares UNIT, [0,1] with no
#              mean or std, against every other ViT here; a wrong scale is what cost `raddino_dense`),
#              the position-table interpolation from the pretrained 14x14 to this read's 21x21, and the
#              block-chunk flattening. Only after `mricore` reproduces its published behaviour does this
#              branch say anything about the corpus.
#
# VERIFY THE FLAG LANDED BEFORE READING ANY SCORE. The recipe line must carry `"variant": "mricore"`
# with `"image_size": "336"` and `"backbone_lr": "4e-05"`. The load itself is already held: the
# checkpoint goes into a stated timm architecture under `strict=True`, so a mismatch refuses at build
# rather than training a random trunk - which is exactly the failure `radiodino_corpus` was nearly
# written off by, and the reason `TimmChunkedWeights` exists at all.
#
# deliberate: --variant radiodinob16 -> mricore is the ONE variable this arm moves off radiodino_corpus.
# Every other flag is byte-identical on purpose, because the question is whether the pretraining corpus
# pays once it is MRI at scale. A second difference would make the answer unreadable, and the axis
# already carries one arm (`radnet_family`) that moved three things at once and could conclude nothing.
#
# reading: logs/mricore_corpus_ledger.log
# bead: knee-m69u
# also: knee-p7it8, knee-y59c, knee-cj40f
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train mricore_corpus \
  $RECIPE --epochs 12 --batch 2 --accumulate 3 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant mricore --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --labels "$LABELS" --out "$MODELS/mricore_corpus_s2026.pt"
