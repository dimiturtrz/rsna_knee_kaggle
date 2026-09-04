#!/usr/bin/env bash
# radiodino_s8.sh — PATCH SIZE, with the trunk width, the corpus, the objective and the read all held.
#
# bead: knee-n9apo
# deliberate: --variant radiodinos8 (the arm's thesis: the same ViT-S on the same corpus at patch 8)
# deliberate: --image-size 224 (held with `radiodino_s16`, which is the control this is subtracted from)
#
# GATED ON `radiodino_s16` CLEARING 0.80. Do not queue this until that arm has a ledger row. A patch
# reading anchored on a trunk that cannot do the task is a comparison between two ways of being bad.
#
# THE ONE VARIABLE. `Snarcy/RadioDino-s8` and `Snarcy/RadioDino-s16` are the same release, the same DINO
# objective, the same RadImageNet corpus, the same ViT-S depth and the same 384 width, differing in patch
# stride alone. The recipe here is byte-identical to `radiodino_s16.sh` except for `--variant`. No other
# pair in this registry isolates patch size, and none can: every other trunk family publishes one stride.
#
# THE MECHANISM, and it is not a knob. Dehghan et al. (arXiv:2602.18614) measured smaller patches beating
# 14 and 28 across twelve medical datasets — up to +12.78% balanced accuracy in 2D and +23.78% in 3D. The
# claim is about WHERE the information sits: a medical finding is often a thin, locally-supported
# structure, and a 16 px patch at 224 spans a large fraction of it, so the first projection averages the
# finding away before any block sees it. Our worst finding is Lateral Meniscus at 0.7981 — focal,
# small-structure, exactly the class that argument is about.
#
# WHY IT IS WORTH MORE THAN ITS OWN NUMBER. Patch size is a property of every trunk in the table, not of
# this one. A real gain here says the operating point has been reading at the wrong granularity all along,
# which reopens the whole trunk row rather than adding one member — and that is the only class of finding
# that could plausibly clear the 1.5% bar, since the board gap at the top now prices at +0.0163 offline
# (knee-j4jog). A flat here closes the axis cheaply, which is worth the card on its own.
#
# COST IS A SEQUENCE-LENGTH COST, not a parameter one. 21.4M parameters against s16's 21.7M — patch 8's
# embedding conv is SMALLER. What grows is the token count: at 224 px, 28x28 = 784 tokens a slice against
# 196, so 4x the tokens, ~4x the MLP and up to 16x the attention. The honest estimate is 300-400 s/epoch
# against s16's 80-90, so 60-80 minutes for 12 epochs, and peak memory around 3x s16's. SMOKE IT FIRST:
# `python -m kneemri.experiments.arm_check --only radiodino_s8 --smoke <throwaway dir> --smoke-studies 60`
# prices the epoch and the peak in under two minutes, and 3x an unmeasured baseline is exactly the kind of
# derivation that has run a card out of memory here before.
#
# THE PACKAGING FAULT IS ALREADY FIXED, same as the sibling: `config.json` names `ViTModel` over
# timm-keyed weights, the registry row loads through `TimmHubWeights` with a strict `load_state_dict`, and
# `load_check` reads 384 loaded against 384 declared.
#
# HARNESS CHECK. The printed recipe must name "variant": "radiodinos8" and "image_size": "224". A silent
# fall back to s16 would read as a flat result and close the patch axis on a null that never fired.
#
# control  radiodino_s16, same recipe, same trunk at patch 16. `native224` = 0.8502 is the external anchor
#          and is NOT the control for the patch reading — it moves the corpus and the objective too.
# cost     ~60-80 min at 12 epochs, 784 tokens a slice. Estimate, not a measurement. Smoke first.
# seat     none pre-registered. If it wins, the seat is not the point — the axis is, and the follow-on is
#          patch size on the champion's own trunks, not a member built from this one.
#
# RESULT 2026-09-04: 0.7971, -0.0137. Lands in the UNNAMED GAP between the flat and mis-fit branches.
# The axis is closed by Q-EI on the PREMISE: Dehghan's images are 28x28, so their patch 2 IS our 196 tokens
# and their patch 1 IS our 784 -- same token counts, 64x the pixel area per token. Not a replication, a
# guess that the trend survives the resolution change. Their schedule is uniform (80 epochs, one LR), so
# the mis-fit branch's rate follow-on is dead and the epoch-6 peak is the mismatch showing up as
# memorisation. Do not re-run this at a shorter budget. The honest successor is matched TOKEN PHYSICAL
# SIZE -- smaller patches at a larger width -- which is a different arm and a dearer one.
#
# PRE-REGISTERED against `radiodino_s16`, floor 0.0062.
#   >= +0.015 — patch size is the axis. Dehghan transfers to volumetric knee MRI, the operating point has
#     been reading too coarsely, and the next arms are smaller patches on the trunks that already carry
#     the pool. This is the branch worth the card.
#   +0.006 to +0.015 — directional. Keep it as a member on the granularity axis (it would be the only
#     patch-8 trunk in a pool that is patch 14 and 16 throughout, which is an error-structure argument),
#     but do NOT plan the trunk row around it on this evidence.
#   within +/- 0.006 — flat. Patch size does not move this task at this width, the axis closes for the
#     cost of one arm, and Dehghan's result is about small-image datasets rather than about 224 px slices.
#     That is a clean close and it is the modal outcome.
#   <= -0.015 — patch 8 is being MIS-FIT rather than being worse: 4x the tokens at an unchanged rate and
#     an unchanged unfreeze depth is a schedule change nobody registered. The suspects in order are the
#     backbone rate and `--unfreeze-last`, and the follow-on is the rate, not the abandonment of the axis.
#
# DO NOT ARM THIS WHILE ANOTHER OF OUR JOBS HOLDS THE CARD — `card_check` is the gate.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train radiodino_s8 \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant radiodinos8 --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 --image-size 224 \
  --labels "$LABELS" --out "$MODELS/radiodino_s8_s2026.pt"
