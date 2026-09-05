#!/usr/bin/env bash
# coatnet_hybrid.sh — the first hierarchical trunk whose TOP blocks --unfreeze-last actually reaches.
#
# WHY THIS ARM EXISTS, AND WHY IT IS NOT A FOURTH "TRY ANOTHER FAMILY". Three hierarchical trunks have
# been fitted here and two of them are closed on a priced argument. Swin holds 18 of its 24 blocks in
# stage 2 and ConvNeXtV2 27 of its 36, so `EncoderSurgery.blocks` counted a run that lives ENTIRELY below
# the widest stage, and `--unfreeze-last 6` opened stage-2 blocks while the 1024-wide top stayed frozen
# for every run in both families. `swin_wholetop` then measured what opening it is worth — byte-identical
# to `swin_supervised` but for the fixed discovery, 37.9M trainable instead of 19M — and came back 0.8274
# against 0.8331, i.e. -0.0057, with the deficit visible from epoch 1. That is what closed knee-uk4f: the
# thaw correction is worth <= 0 on a staged trunk, so neither family is re-opened by a re-quote.
#
# CoAtNet's stages are 2/6/14/2 and they are NOT one block type — the first two are MBConv, the last two
# are `TransformerBlock2d`. Grouping by element type therefore counts the 16 TRANSFORMER blocks across
# stages 2 AND 3, so `--unfreeze-last 6` opens the last four of stage 2 plus BOTH stage-3 blocks: 35.5M
# trainable parameters, measured through this registry entry, with the convolutional stem left frozen.
# This is the first hierarchical arm here whose trainable suffix includes the trunk's actual top, and it
# gets there by the trunk's own shape rather than by the correction that was already priced at <= 0.
#
# THE MIXING RULE IS THE DIVERSITY CLAIM. IN-12k supervised then IN-1k finetuned, so the pretraining
# OBJECTIVE is Swin's rather than a new axis. What differs is how the trunk mixes: depthwise convolution
# with squeeze-excite for stages 0-1, then GLOBAL relative attention over the whole grid for stages 2-3.
# Not ConvNeXtV2's purely local convolution, not Swin's attention confined inside a 12x12 window, and not
# DINOv2's global attention over a flat grid from the first block. A convolutional prior feeding a
# globally-attending top is the one arrangement this registry has never held.
#
# 144 FINAL TOKENS IS NOT THE HANDICAP IT LOOKS LIKE, and this was checked before the arm was written.
# Every non-DINO trunk we own is stride-32 and every DINO trunk is patch-14/16, so "family" and "few final
# tokens" are perfectly confounded in our data and the 0.03 deficit could have been a resolution reading
# all along. The ledger already answers it: the 224 corner reads 256 tokens for native224 0.8502,
# tissue224 0.8491, blendlabels 0.8516 against the 336 corner's 0.851 band, and the 336 -> 392 move was
# attributed to the RATE and not the pixels. Dropping 576 tokens to 256 costs this pipeline almost
# nothing, so 144 is not what would sink this run, and the non-DINO deficit stays a fact about trunks.
#
# 384 IS THE ONLY LEGAL READ AND --train-size MAY NOT BE SWEPT. Unlike Swin's case it is not the window
# that pins it: `rmlp` builds a relative-position MLP against the stage-3 grid at construction, and a 448
# px read raises `The size of tensor a (784) must match the size of tensor b (576)` on the first forward.
# Measured here through the registry entry, not inferred. `require_tiling` cannot express it because 448
# tiles stride 32 perfectly well.
#
# AND 384 BUYS GEOMETRY, NOT PIXELS — the decode cache is 336px_20sl_0.1-0.9band and `run_setup.resample_to`
# upsamples inside the trainer, exactly as swin_supervised did. No cache has to be built for this arm.
#
# THE BACKBONE RATE IS NOT A GUESS THIS TIME. 1e-4 is what swin_supervised ran and it is the only measured
# point for a SUPERVISED hierarchical trunk under this recipe: it beat both ConvNeXtV2 arms (0.8331 against
# 0.8186 and 0.7927) from the same recipe family. Reusing it makes this arm a one-variable comparison
# against swin_supervised, which is what a first arm on a new trunk should be. It is not swept.
#
# control  swin_supervised = 0.8331 — same objective, same 384 read, same rate, same recipe. The one
#          number this arm is a controlled comparison against. The 0.855-0.863 dense band is the BAR,
#          quoted in the branches, and 0.8459 is the merit level a seat has had to clear.
# seat     priced afterwards with record_mix against a FULLY SCOREABLE entry (honest12), never against an
#          entry whose members are --fold -1 and carry no holdout row.
# cost     12 epochs at 384 px on a 72.9M trunk whose stage 2 runs global attention over 576 tokens for 14
#          blocks. Priced off swin_supervised's 90 min at the same read, uplifted for that attention: ~2.2 h.
#
# PRICE: ~2.2 h of card. P(clears the 0.0062 report floor as a priced seat into honest12) = 0.15.
# P(solo at or above swin_supervised's 0.8331) = 0.35. This is priced as an INSTRUMENT and a seat
# candidate, not as an improvement: the member axis is closed on seven board flats and nothing here
# claims to re-open it. What it buys is the last untried arrangement on the trunk axis, on a card that
# is otherwise idle, and a reading on whether the frozen-top confound was ever the story.
#
# PRE-REGISTERED.
#   LANDS — solo at or above 0.8460. A supervised hybrid on its first arm at the merit level. Mirror the
#     weights to a Kaggle dataset (Apache-2.0, no licence question), add the mount, price the seat, and
#     spend a slot on the entry if the seat delta is at or above +0.002 offline.
#   SEAT-ONLY — solo 0.8331 to 0.8459. Beats the controlled comparison but not the bar. Mount it only if
#     the priced seat delta into honest12 is at or above +0.002 offline, which transfers one-for-one.
#   FLAT — solo 0.8100 to 0.8330. At or below swin_supervised despite a trainable top. The frozen-top
#     confound was never the story, the non-DINO deficit is about the trunks, and the hierarchical
#     row of the registry closes with three families measured rather than two.
#   REFUTES — solo below 0.8100, i.e. at or under cnx_own_recipe's 0.8186 band with the top open. The
#     reading is then about this RECIPE on hybrid trunks and the follow-up is nothing: the axis is spent.
#
# ATTEMPT 1 IS VOID ON A NORMALIZATION DEFECT, and the defect is why the registry has that test. The
# entry declared no `normalization`, so it took the field's default, IMAGENET — while timm publishes
# (0.5, 0.5, 0.5) for this tag inside `config.json`'s `pretrained_cfg`, which is the AugReg convention
# and is `HALF` here. 25 minutes of card ran the trunk on inputs centred a third of a standard deviation
# off in every channel, and it would have produced a number attributable to the HYBRID ARRANGEMENT that
# was really a fact about the preprocessing. `test_every_encoder_states_its_own_constants` refused it:
# timm ships no `preprocessor_config.json`, so an entry on a timm repository has to carry its evidence
# in prose, and this one carried none. Attempt 1's artefacts are renamed `_voidnorm` rather than deleted,
# because a stitched series is the corpus defect knee-gnzx was filed for.
#
# bead: knee-llcgj
# reading: logs/coatnet_hybrid_seat.log
#
# deliberate: --variant       (coatnet2 — the whole point of the arm)
# deliberate: --train-size    (384, pinned by the rmlp position table — see above, do not sweep)
# deliberate: --backbone-lr   (1e-4, swin_supervised's, so this is a one-variable comparison)
# deliberate: --batch
# deliberate: --accumulate
# deliberate: --pair-memory  (15 = C(6,2), the pairs an effective batch of 6 forms, by band_pairmem's own
#                             rule that 66 = C(12,2) matched --accumulate 12. NOT a free choice: at
#                             --batch 1 the RECIPE's --rank-weight 1.0 forms no pairs at all, so without
#                             this flag the rank term contributes exactly zero. swin_supervised ran that
#                             way, which is a second difference between it and this arm, stated here
#                             rather than hidden — the trainer now refuses the combination outright.)
# deliberate: --recompute-activations  (matches swin_supervised; trades compute for memory, cannot move a
#                                       score, so it is declared rather than dropped)
# deliberate: --vram-gb
# NO --vram-gb, deliberately. swin_supervised names 8 and a smoke at 8 died here with `Tried to allocate
# 4.22 GiB ... 8.00 GiB allowed`: a hand-named cap is taken EXACTLY as typed, and this trunk runs 14
# blocks of global attention over 576 tokens where Swin's are confined to a 12x12 window. Saying nothing
# hands the cap to `DeviceBudget.capped`, which gives the co-tenant share of the card (~25.5 GB) rather
# than a ceiling fitted to a different trunk. knee-ylp0 says the same thing from the other side: no arm
# needs the flag.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train coatnet_hybrid \
  $RECIPE --epochs 12 --batch 1 --accumulate 6 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant coatnet2 --backbone-lr 1e-4 --slices 20 --band-low 0.1 --band-high 0.9 --train-size 384 \
  --pair-memory 15 \
  --labels "$LABELS" --out "$MODELS/coatnet_hybrid_s2026.pt"
