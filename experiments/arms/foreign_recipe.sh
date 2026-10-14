#!/usr/bin/env bash
# foreign_recipe.sh — mattiaangeli's DINOv3 configuration, trained on OUR split.
#
# WHY THIS AND NOT A SEATING. Their fold-0 solo reads 0.8874 on our 909 studies. The best DINOv3 member
# this project owns reads 0.8268 (`dinov3_base_lora`), and our v3small runs read 0.8057 (224 px) and
# 0.8215 (LoRA). That is a SIX-POINT gap on the same trunk family, which is four times the 1.5% bar and
# is not a seating question — something in their recipe carries score that ours does not. Replicating it
# is the SOTA-replicate step, and it is the largest single lever the pool has open.
#
# THE ADMISSIBILITY QUESTION DISAPPEARS. Their checkpoint is only shippable if their fold-0 split can be
# shown disjoint from our holdout, and it cannot: neither of their two public kernels
# (`bend-the-knee-to-dinov3-ensembled`, `bend-the-knee-to-the-dinosaurs`) contains split code — both are
# inference kernels, `KFold`/`GroupKFold`/`train_test_split` appear zero times in either. A twin trained
# on OUR fold-0 train set with OUR 909 held out has no split question at all, is roster-able, and prices
# the recipe honestly.
#
# WHAT THE cfg GIVES, read from `m_f0.pt` and not from memory:
#   backbone vit_small_patch16_dinov3.lvd1689m · img 336 · n_slice 16 · stem native, stem_lr 0.0
#   pool xcodex · cond token (n_sites 109) · norm none · pe_init tiled · triplet none
#
# WHAT IT DOES NOT GIVE, and this is the one line of honesty the arm owes: augmentation, loss, optimiser
# and schedule are absent from the cfg, and three architecture terms have no equivalent here — `xcodex`
# pooling, `token` site-conditioning over 109 sites, and `tiled` positional-embedding init. So this is
# the CLOSEST recipe we own on a DINOv3 trunk at their geometry, not a reproduction. If it lands near
# 0.887 the geometry was the carrier; if it lands near our own 0.82 the carrier is one of the three terms
# we cannot express, and THAT is the finding — it names exactly what to build next.
#
# THE GEOMETRY IS THE UNTESTED PART. `dinov3_v3small_336_s2026.pt` is on disk with no holdout record and
# no metrics file, so 336 px on this trunk has never been priced here; every DINOv3 number in the ledger
# is a 224 px number. Their cfg reads 336 with 16 slices. The rate is 8e-6, the value `dinov3.sh` argues
# for a trunk placed with RoPE and distilled from a 7B teacher, and the one our 224 runs used.
#
# PRE-REGISTERED, before the run:
#   * solo >= 0.875 on the 909 — the geometry carries it, keep, roster, and the blend is priced next.
#   * 0.83-0.87 — partial: the geometry carries some of it and one of the three unexpressed terms carries
#     the rest. Build the cheapest of them (site-token conditioning is a head change, not a trunk one).
#   * <= 0.83 — the geometry is not the carrier. Their score lives in the terms we cannot express, and the
#     foreign member stays a board-only question.
#
# SIXTEEN SLICES IS NOT ON DISK, AND WAITING FOR IT IS THE WRONG TRADE. The decode cache is keyed by
# `<size>px_<slices>sl`, and `images_336px_16sl.npy` does not exist — the first launch died on exactly
# that, `FileNotFoundError`, thirty seconds in. `images_336px_20sl.npy` DOES exist at the same 336 px
# and the same default 0.2-0.8 band. So the arm runs at TWENTY slices by default, which holds the axis
# that has never been priced here (v3small, 336 px, whole-stack) and moves only the axis that is a knob:
# 16 against 20 slices is a sampling-density difference within one band, not a different geometry. The
# faithful 16-slice cache builds on CPU beside this run, and `foreign_recipe.sh 16` is the follow-up the
# moment it lands — a follow-up that only matters if 20 slices already reads near their number.
#
# COST: 12 epochs at 336 px, 20 slices, whole-stack — about 1.5-2 h.
#
# deliberate: --image-size 336 (their cfg, the whole point of the arm)
# deliberate: --slices 20 (their cfg says 16; that cache does not exist — see SIXTEEN SLICES above)
# deliberate: --backbone-lr 8e-6 (dinov3.sh's argued rate; their cfg does not carry one)
# reading: foreign_recipe_v3small_336_*sl_s2026.holdout.npz
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
SLICES="${1:-20}"
require "$LABELS"


run_train "foreign_recipe_v3small_336_${SLICES}sl" \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant v3small --image-size 336 --train-size 336 --backbone-lr 8e-6 --slices "$SLICES" \
  --labels "$LABELS" --out "$MODELS/foreign_recipe_v3small_336_${SLICES}sl_s2026.pt"
