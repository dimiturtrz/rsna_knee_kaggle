#!/usr/bin/env bash
# bead: knee-vp8q
# raddino_dense.sh — does a RADIOLOGY-pretrained initialisation close any of the generalisation gap?
#
# THE MECHANISM. Every encoder this project has fitted was pretrained on natural images or natural
# video, and they share a failure mode: `base_dense` reaches 0.8371 holdout at train loss 0.8456 while
# `cnx_dense` reaches 0.7927 at a BETTER train loss of 0.8332. A trunk that fits our pixels harder
# generalises 0.044 worse on them, which is not a capacity problem and not a recipe problem — it is
# what a mismatched initialisation looks like. knee-ombx prices attacking it at 1.5-3%, the only
# mechanism still above the bar, and its MRNet route is blocked on a licence decision that is not mine.
# RAD-DINO reaches the same mechanism unblocked: MIT, 882,775 chest radiographs.
#
# WHY THIS ARM READS CLEAN WHERE NO PREVIOUS ENCODER ARM DID. RAD-DINO is finetuned FROM
# `facebook/dinov2-base` — same ViT-B/14, same 768 width, same 12 blocks. So this file is
# `base_dense.sh` with `--variant base` changed to `--variant raddino` and NOTHING else, and its
# control is `base_dense_s2026` at 0.8512, already fitted. The initialisation weights are the only
# variable in the comparison. Every earlier family change (DINOv3, ConvNeXtV2, V-JEPA2) moved
# architecture and pretraining corpus together and could not attribute its own result to either.
#
# THE BATCH OF 6 IS INHERITED, NOT CHOSEN, and it is inherited from the control. base at 336 px and 20
# slices needs 56.3 GB at the control-of-the-control's batch of 12 against a 32 GB card; 6 derives
# 28.25 GB and `base_dense` measured 27.14 GB peak. Holding it here is what keeps `base_dense_s2026` a
# valid control — matching the confound is the point, since both arms carry it identically.
#
# PRE-REGISTERED, BEFORE THE NUMBER.
#   >= 0.866   (+0.015 on the control) — radiology initialisation is the lever. That licenses the
#              RadImageNet conversion (better match: 1.35M images, 11 anatomies, its own evaluation
#              includes ACL and meniscus tear on MR) and strengthens the MRNet case rather than
#              replacing it, since MRNet is knee MRI and this is chest projection radiography.
#   0.856-0.865 — directional and knob-sized, below the 1.5% bar. Keep the member as an ensemble seat
#              if a seat is free; do NOT plan around it and do NOT call it a finding. It would still
#              license RadImageNet, because a PARTIAL domain match returning a partial gain is exactly
#              what the mechanism predicts, and a closer match is the way to test that.
#   0.846-0.855 — a tie with the control. Domain-matched initialisation does not pay AT THIS DOMAIN
#              DISTANCE. That kills the chest-radiograph route and prices RadImageNet DOWN but does not
#              kill it: chest projection radiography shares neither the acquisition physics nor the
#              body part, and MRNet/RadImageNet both do. It says nothing about MRNet.
#   <= 0.845   SUSPECT THE HARNESS FIRST. The weights are a DINOv2-base checkpoint loaded by the same
#              `AutoModel` call as the control, so a large deficit is more likely a load that silently
#              fell back or a position-embedding interpolation than a real refutation.
#
# THE LOAD WAS VERIFIED BEFORE THE ARM WAS QUEUED, so the band above cannot be answered with "the
# weights never landed". `ENCODERS['raddino'].load()` returns hidden 768, patch 14, depth 12, zero
# register tokens — the declared values exactly — 86.6M parameters, and a 336 px read emits 577 tokens
# (1 + 24x24), the correct grid. Against `facebook/dinov2-base`: ZERO of 223 tensors are identical, so
# every weight moved during radiology pretraining and this is genuinely a different initialisation.
#
# NORMALISATION IS SHARED, NOT MATCHED, and that is stated because it is the one preprocessing
# difference worth naming. This repository applies no ImageNet mean/std anywhere; slices are scaled
# per study. So RAD-DINO does not see the intensity statistics it was pretrained under — and NEITHER
# DOES THE CONTROL, which is DINOv2-base under the same pipeline. The mismatch is common to both arms
# rather than differential, so it cannot explain a gap in either direction.
#
# NOT SUBMITTABLE YET. `raddino` has no Kaggle mount, so this member trains here and cannot enter a
# kit until the weights are mirrored as a dataset. MIT permits a public mirror; it is deferred until
# this arm says a seat is worth one.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train raddino_dense \
  $RECIPE --epochs 12 --batch 6 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant raddino --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --labels "$LABELS" --out "$PROCESSED/raddino_dense_s2026.pt"
