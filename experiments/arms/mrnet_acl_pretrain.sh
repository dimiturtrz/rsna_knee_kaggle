#!/usr/bin/env bash
set -euo pipefail
#
# STAGE ONE of the cross-corpus transfer pair. Can a knee finding be learned from a DIFFERENT knee
# corpus and carried back into ours?
#
# This arm never earns a seat and is not meant to. It exists to produce a trunk that has seen 1248
# MRNet exams supervised on ACL and nothing else, so that `mrnet_warm` can start from it. Its own
# holdout number is against MRNet's split and is not comparable to anything in the ledger.
#
# WHY ACL AND NOT FRACTURE. Fracture was the label the question was asked about, and it is the wrong
# ruler: only fastMRI carries it, its fastMRI+ category `Bone-Fracture/Contusion/dislocation` is a
# union of three findings we score separately, and our own holdout has 68 fracture positives in 909.
# ACL is in all three external corpora, has 248 positives here, and the best member on it reads
# 0.8976 — a ruler with room to move in both directions.
#
# WHY THIS IS NOT THE MERGED-CORPUS AXIS, which closed flat at +0.006. Every merged member CONCATENATED
# MRNet studies into the fine-tune cache: one optimisation, one label space, the foreign studies diluted
# among ours. This is SEQUENTIAL — a full optimisation on the foreign corpus alone, then a second one on
# ours. knee-ombx prices that mechanism at 1.5-3% and records that it has never been run in 107 arms.
# The two are different experiments and the flat result of the first says nothing about the second.
#
# WHAT THE LITERATURE SAYS IT SHOULD DO (A-BV, `research/deep_dives/2026-09-02_qbv-layer-transfer-partial-label-overlap.md`):
# the transferable part is the REPRESENTATION in the early and middle blocks, not the head, so a gain
# is not capped at ACL — it can reach labels MRNet has never seen. The same answer bounds the size:
# MRNet is far too small for the deep-feature-reuse regime, and the DINOv2 init is already strong.
# Against that stands `orthofoundation-L`, where knee-specific supervised continuation on 1.25M images
# SUBTRACTED (frozen probe 0.7387 < 0.7446, fitted member 0.8452 < 0.8502). This arm is the cheap read
# on which of those two the sequential route lands nearer.
#
# deliberate: --scheme mrnet (the foreign corpus alone; no study of ours is in this optimisation)
# deliberate: --labels mrnet_acl.csv (1248 exams, the only label table that corpus has)
# deliberate: --focus-label ACL (zeroes the other eleven loss weights; the head stays twelve wide so
#             stage two can --init from this checkpoint without a shape mismatch)
# deliberate: --epochs 8 (1248 exams against our 909; the budget is a fifth of a full arm, and this
#             checkpoint is an input, not a result)
# deliberate: every other flag is `base224`'s, unchanged, because stage two IS `base224` plus --init
#             and a trunk that was fitted under a different recipe would confound the read
#
# bead: knee-xlqgm
#
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

MRNET_LABELS="$PROCESSED/labels/mrnet_acl.csv"
require "$MRNET_LABELS"

run_train mrnet_acl_pretrain \
  $RECIPE --epochs 8 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant base --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 --image-size 224 \
  --scheme mrnet --focus-label "ACL" \
  --labels "$MRNET_LABELS" --out "$MODELS/mrnet_acl_pretrain_s2026.pt"
