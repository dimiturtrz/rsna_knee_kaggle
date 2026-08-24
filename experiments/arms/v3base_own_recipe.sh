#!/usr/bin/env bash
# v3base_own_recipe.sh — DINOv3-base at a rate chosen FOR DINOv3, which has never happened either.
#
# CONTINGENT ON `cnx_own_recipe` (knee-y59c). Do not launch this until that arm reads. It exists now
# because its argument is that arm's argument, applied to the candidate that is already nearest to the
# cell we are trying to fill, and writing it after the result would let the result choose the reasoning.
#
# THE CELL THIS AIMS AT. Nineteen displacement tests have passed zero candidates, and seven of those
# candidates came from outside the DINOv2 pretrain corpus (video-JEPA, RadDINO, DINOv3). Every one of
# the seven is 0.013-0.054 BELOW the seated band, and the only parity-quality candidate ever tested
# (`grade_ladder`, 0.8565) is inside the space. So NO candidate has ever been both outside the corpus
# AND at seated quality: the deciding cell is empty, and "outside-corpus members do not seat" is a claim
# measured only on members that were too weak to seat for a reason that has nothing to do with corpus.
#
# WHY v3base AND NOT ANOTHER FAMILY. `v3base_dense` reads solo 0.8455 — the best outside-DINOv2 member
# this project owns, 0.0011 under `crop80_span` which HOLDS A SEAT. It is the smallest quality gap any
# outside family has to close, so it is the cheapest way to put something in that cell at all.
#
# THE SINGLE VARIABLE, and it is the rate. `v3base_dense` ran at `--backbone-lr 4e-5`. That number was
# fitted across ~189 DINOv2-family runs; it is a DINOv2-small adapter rate handed to a different
# encoder unexamined, which is exactly the defect `cnx_own_recipe` was written to correct on the
# convolutional trunk. DINOv3's own post-training uses a markedly higher trunk rate than DINOv2's
# recommended linear-probe regime, and this project has never varied LR on any non-DINOv2 family.
# `--backbone-lr 1.5e-4` is under four times the inherited rate — deliberately below `cnx_own_recipe`'s
# 4e-4, because a ViT trunk at 768 wide is far more rate-sensitive than a ConvNeXt and because the
# control it must beat is only 0.0011 away, so overshooting costs the whole comparison.
#
# EVERYTHING ELSE IS HELD AT `v3base_dense`'s VALUES. Epochs, batch, slices, band, pooling, whole-stack
# and throughput are byte-identical to the control, so `v3base_dense` (0.8455) is the matched control
# and the single moved flag carries the whole reading.
#
# PRE-REGISTRATION, written before the number, against `v3base_dense`'s 0.8455.
#
#   FILLS THE CELL — solo at or above 0.8466 (`crop80_span`, the weakest seated member). For the first
#     time a member is both outside the DINOv2 corpus and at seated quality. Run the DISPLACEMENT test
#     immediately: full pool plus it, greedy from scratch, keep only if the new peak beats 0.8731. Even
#     a failed displacement is then INFORMATIVE for the first time, because the confound is gone.
#   RATE IS THE DEFICIT BUT NOT ALL OF IT — 0.8455 to 0.8465. The inherited rate was costing this family
#     real points and the direction is confirmed, but the cell stays empty and the next knob is
#     augmentation strength, never moved on any non-DINOv2 family.
#   RATE IS NOT THE DEFICIT — at or below 0.8454. The correction that (on the contingency) worked for a
#     convolutional trunk does not transfer to a different ViT corpus, which localises the cnx result to
#     the ARCHITECTURE rather than to "our recipe is DINOv2-shaped" in general. That is a narrower and
#     more useful claim than the one it replaces, and it is worth the arm on its own.
#
# The bands are 0.001 wide against a 0.0062 holdout noise floor, and that is deliberate: this arm is
# not trying to clear the significance bar by itself. It is trying to move a member ACROSS A THRESHOLD
# (0.8466) that decides whether a whole axis can be measured at all. The threshold is a gate, not a gain.
#
# bead: knee-y59c
#
# THE ONE MOVED FLAG, DECLARED.
# deliberate: --backbone-lr
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

HF_TOKEN="$(grep -oE 'hf_[A-Za-z0-9_-]+' /d/life/creds/machine/hfpat.txt | head -1)"
export HF_TOKEN
[ -n "$HF_TOKEN" ] || { echo "MISSING: no Hugging Face token, and DINOv3 is a gated repository"; exit 1; }

run_train v3base_own_recipe \
  $RECIPE --epochs 12 --batch 6 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant v3base --backbone-lr 1.5e-4 --slices 20 --band-low 0.1 --band-high 0.9 \
  --labels "$LABELS" --out "$PROCESSED/v3base_own_recipe_s2026.pt"
