#!/usr/bin/env bash
# cnx_top_rate.sh — the fourth rung of a rate ladder that has never turned over.
#
# THE LADDER, three finished 12-epoch runs on the same trunk (ConvNeXtV2-base-22k), all on disk:
#
#   backbone-lr   read   best holdout   run
#   4e-5          352    0.7927         cnx_dense        (a ViT ADAPTER rate, inherited unexamined)
#   1.2e-4        448    0.7993         cnx_mid_rate
#   4e-4          448    0.8186         cnx_own_recipe
#
# +0.0259 across the ladder and THE LARGEST STEP IS THE LAST ONE (+0.0193). A curve whose final step is
# its biggest has not found its maximum. This arm is the next 3x rung and nothing else moves.
#
# WHY THIS IS NOT A DISCARDED PRE-REGISTRATION, which is the objection that has to be answered before
# any further ConvNeXt card is spent. `cnx_own_recipe` scored 0.8186, landing in its own PARTIAL band
# (0.80-0.83), whose text reads: "the deficit is real but not all recipe, and the next knob is
# augmentation strength (never moved on this family) rather than another resolution step." Ignoring a
# fired branch is exactly how `cnx_dense` earned a verdict that had to be retracted, so the branch was
# checked before it was set aside — and the knob it names is not there.
#
#   1. `AugmentConfig` defaults to `geometric_only` (augmentation.py:60-68): rotation 8 deg, scale
#      jitter 0.08, shift 0.05, gain 0.10, and the intensity block ZEROED — no gamma, no blur, no
#      contrast, no noise.
#   2. `RECIPE` is `--fold 0 --one-cycle --weight-floor --rank-weight 1.0` (recipe.sh:23). It names no
#      augmentation flag at all, so `cnx_own_recipe` ran under that default.
#   3. `--mixup` defaults to OFF (options.py:74).
#
# So ConvNeXt was never handed an over-augmented ViT recipe. It got the public baseline's mild rigid
# jitter, byte-identical to what all ~189 DINOv2-family runs got. The branch assumed a ViT-tuned
# AugReg pipeline that this project does not have, and the published mechanism it would have been
# reaching for (Steiner et al., TMLR 2022 / arXiv 2106.10270 — ViTs need MORE augmentation than CNNs,
# and the gap widens as data shrinks) prescribes REDUCING RandAugment magnitude and dropping
# Mixup/Cutmix. Both are already at zero here. The branch is void on its own terms, not overruled.
#
# WHAT THE LITERATURE DOES AND DOES NOT SUPPORT, recorded because the ledger has been citing a folk
# rule as though it were established. The claim in `cnx_dense`'s entry that "convolutional trunks
# fine-tune roughly an order of magnitude higher" has NO published support: the most controlled
# comparison available (Zhou et al., arXiv 2108.05305 — ConvNets vs Transformers at matched ImageNet
# accuracy across 15 downstream tasks) deliberately gives both families the SAME rate, Adam 1e-4 for
# classification, and reports no CNN premium. Our own two families nevertheless diverge by an order of
# magnitude on this corpus: DINOv2's rate was swept and settled at 4e-5, and ConvNeXt is still climbing
# at 4e-4, with Zhou's family-neutral 1e-4 sitting between our second and third rungs. That divergence
# is unexplained, and these three points are the only prior anyone has for it. This arm adds a fourth.
#
# One further external reading, which is why the family is worth a fourth rung at all rather than being
# set aside at 0.8186: Taher et al. (MedIA 2025, PMID 40117988) find ConvNets MORE annotation-efficient
# than vision transformers when fine-tuning for medical tasks. On a ~1000-study corpus that predicts a
# CNN should be easier to train well, not harder — so 0.8186 reads as under-trained rather than as a
# correctly-priced weak family.
#
# ONE FLAG MOVES. Read size, batch, accumulate, slices, band, pooling, whole-stack, schedule, epochs
# and vram cap are all `cnx_own_recipe`'s, so the comparison isolates the rate exactly. This is a
# single-knob arm on purpose, against that arm's own argument for moving two together: the two-knob
# move was justified when the family had one data point and needed a coarse probe, and it worked — it
# is why there is a curve to extend. A curve is extended one point at a time or it is not a curve.
#
# control  cnx_own_recipe = 0.8186, the same trunk and the same recipe one rung down. The ONLY
#          comparison that isolates what this arm changes.
#          The non-DINOv2 seat bar is 0.8521; the best non-SSL family member this project owns is
#          swin_supervised at 0.8331.
# cost     ~2.5 h at 12 epochs, priced from cnx_own_recipe which is byte-identical but for the rate.
# seat     card-free with record_mix afterwards, and ONLY if the solo number earns it.
#
# PRE-REGISTERED, and these branches bind on this arm the way the last one's did on it. Floor 0.0062,
# board resolution 0.0015.
#   THE LADDER CONTINUES — solo clears 0.8340 (0.8186 + the floor + a margin, and past swin_supervised).
#     Then the maximum is above 1.2e-3, every ConvNeXt number this project owns is under-read, and the
#     family earns a fifth rung plus a re-reading of the four runs already on disk. It also makes the
#     DINOv2-vs-ConvNeXt rate divergence a real finding worth writing up rather than an artifact.
#   THE LADDER TURNS OVER — solo lands 0.8186 +/- the floor, or below. Then 4e-4 is the family's
#     maximum, the rate axis is CLOSED with four points and a named optimum, and 0.8186 is ConvNeXt's
#     honest quality on this corpus. That is 0.033 under the seat bar, and the family is then set aside
#     on a cost argument WITH the curve printed beside it — which is what `cnx_dense`'s write-up
#     reached for a run too early and three runs too soon.
#   IT DIVERGES — solo under 0.78, or the run destabilises. 1.2e-3 is past the edge of the trunk's
#     stability at this batch, which is itself the answer: the maximum is between 4e-4 and 1.2e-3 and
#     the next probe is 7e-4, not another 3x step.
#
# NO SEAT IS PRE-REGISTERED HERE. Even the best branch lands ~0.834 against a 0.8521 bar, so this arm
# is an INSTRUMENT about whether our CNN numbers are under-read, not a member bet. It buys a submission
# slot only if it clears the bar outright, which no branch above predicts.
#
# bead: knee-nj4l — the champion is an architecture monoculture, and this arm asks whether the one
# non-transformer family we own is weak or merely under-trained. Also answers knee-w102's augmentation
# branch (void, see above) and feeds knee-nz34's encoder-family pricing.
#
# deliberate: --backbone-lr
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train cnx_top_rate \
  $RECIPE --epochs 12 --batch 1 --accumulate 6 --slice-pooling max --whole-stack \
  --variant cnxbase --backbone-lr 1.2e-3 --slices 20 --band-low 0.1 --band-high 0.9 --train-size 448 \
  --vram-gb 16 \
  --labels "$LABELS" --out "$MODELS/cnx_top_rate_s2026.pt"
