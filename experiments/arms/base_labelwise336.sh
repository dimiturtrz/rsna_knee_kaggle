#!/usr/bin/env bash
# base_labelwise336.sh — carry the validated trunk up to the corner we ship at.
#
# bead: knee-mm6v0
# reading: base_labelwise336_s2026.holdout.npz
#
# DO NOT RE-LAUNCH THIS ARM. It has now OOMed five times and the reason is none of the three things that
# were tried. The record, because each attempt cost a card slot and the wrong conclusion was written down
# twice:
#   * 20 GB cap, 18.67 allocated. Read as a cap that was too low.
#   * 27 GB cap, 23.21 allocated + 2.38 requested. Read as the 336 corner being barred for a base trunk.
#   * `base_labelwise_dense336` at 20 slices, 26 GB cap: 21.06 + 3.57, 6.17 free of 31.84.
#   * this arm at 12 slices, 26 GB cap: 23.21 + 2.38 — BYTE-IDENTICAL to its own 27 GB death.
#   * this arm again with `--eager-encoder`: 23.21 + 2.38 again, 3.38 free of 31.84.
#
# WHAT IS NOT THE CAUSE, each ruled out by measurement rather than by argument.
#   * NOT missing gradient checkpointing. `experiments/recipe.sh:31` defines THROUGHPUT as exactly
#     `--recompute-activations`, and every one of these invocations passes `$THROUGHPUT`. Checkpointing
#     was ON for all five. A commit on 2026-09-22 claimed the opposite and is wrong.
#   * NOT the slice count. 12 slices died at a LARGER allocated figure than 20 did. The footprint is not
#     monotone in slices, so the dominant term is not what the slice arithmetic models.
#   * NOT `torch.compile`. `--eager-encoder` (`recipe.sh:28-30`, compile is on by default) reproduced the
#     request to the byte.
#   * NOT the `--vram-gb` cap. The 27 GB attempt died at 25.59 GB of demand, under its own cap.
#
# WHAT THE NUMBERS ACTUALLY SAY. At the crash PyTorch holds 23.21 GiB allocated and the CARD has 3.38 GiB
# free of 31.84 — roughly 28.5 GiB is reserved against 23.21 in use. The binder is allocator fragmentation,
# and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a confirmed no-op on this Windows machine. The
# remaining levers all change the experiment: batch may not be cut without changing the 66 within-batch
# pairs the `--rank-weight` term scores, and `--unfreeze-last` cuts the very capacity the arm is testing.
#
# So the base trunk cannot reach 336 on this machine, and the capacity axis has to be spent somewhere it
# fits. Base at 224/12sl reads 0.8996 (`base_labelwise224_dense_drop40_long36`), 0.0034 under the best solo
# the project owns, which is inside the 0.0062 paired floor — the trunk is not the thing that is missing.
#
# WHAT LICENSES THIS ARM. `base_labelwise224` landed 0.8899 ledger75 against `labelwise224`'s 0.8832 —
# +0.0067, above the 0.0062 paired floor, converged over 12 epochs on 909 studies. That is the band-1
# outcome its own pre-registration named: capacity is NOT inert once the readout can spend the width, so
# the three `rank24` rows in `backbone.py:177-206` are disowned rather than suspect, and `adaptation.py:121`
# ("capacity is the only axis that has ever cleared our noise floor") is the sentence that survives.
# The pre-registration also said what came next in that band: base becomes the trunk to re-fit the 336
# family on. This is that arm.
#
# ONE VARIABLE against `base_labelwise224`: `--image-size 336`. Byte-identical otherwise — same trunk,
# same 12 slices, same 0.2-0.8 band, same batch 12, so the pairwise `--rank-weight` term sees the same 66
# pairs per step. The control is free and on disk at 0.8899.
#
# WHY 336 AND NOT THE FULL SHIPPING RECIPE. The shipping family (labelwise_dense 0.8964,
# labelwise_dense_drop40_long36 0.9030) is 336 at TWENTY slices, and small measured 24.93 GB allocated
# there. base is 1.89x small's footprint on the one matched pair this project has measured (6.86 -> 12.99
# GB at 224/12sl), which puts base at 336/20sl near 47 GB — over the whole card, at any batch that keeps
# the rank term intact. At TWELVE slices the same scaling lands near 28 GB, which fits. So the slice count
# is held at the control's 12 and only the pixels move; the 20-slice corner stays owed, and what it is
# owed is a VRAM answer, not another guess.
#
# MECHANISM. 336 is 2.25x the pixels of 224. A 1-3 mm meniscal tear spans ~2-5 px at 224 and ~3-8 at 336:
# the claim is that the finding becomes closer to REPRESENTABLE, a SAMPLING change. `focal_labelwise224`
# (0.8796, inert) is correlated evidence that this head is not token-STARVED, but that was a SELECTION
# mechanism — re-weighting tokens the model already held — so it constrains P here without refuting it.
#
# PRICE, out loud. `base_labelwise224` ran 171 s/epoch; 2.25x the pixels puts this near 385 s/epoch, so
# ~1.3 h of card for 12 epochs. Zero decode (the 336px/12sl array is the default build and is on disk),
# zero submission slots, control free. P(clears 0.8961) ~ 0.35. P(clears the 1.5% bar) ~ 0.10.
#
# BANDS, pre-registered, ledger75, against base_labelwise224 0.8899.
#   * >= 0.8961  resolution and capacity COMPOSE. The next arm is the 20-slice corner and the question it
#                carries to the owner is VRAM, not value.
#   * 0.8899-0.8961  directional. Keep nothing, compose nothing.
#   * <  0.8899  the trunk gain does not survive the resolution step. Capacity stays real at 224 and the
#                336 family's lead over it is then about slices or the dense recipe, not pixels — which
#                is a verdict on where the remaining headroom in the shipping family actually sits.
#
# READ THE FOCAL LABELS, NOT ONLY THE MEAN. Lateral Meniscus -0.1925, ACL -0.1225, MCL -0.0964 lead the
# model-side error census and are the labels a sampling change has to move. If the mean moves and those do
# not, the delta is not this mechanism. Say so.
#
# HARNESS CHECK. The printed recipe must name "variant": "base" AND "image_size": "336" AND
# "slice_pooling": "labelwise", and the trunk must load `facebook/dinov2-base`.
#
# control  base_labelwise224_s2026 = 0.8899 ledger75 @ epoch 8, 909 studies. No control run is owed.
#
# deliberate: --image-size 336 (the arm's thesis)
# deliberate: --vram-gb 27 — named rather than derived, because the derived figure reads the slice count
#             and the image size and is blind to --variant, so it returns 36 GB for a corner a small trunk
#             measured at 9.30 GB allocated (`mergedk_labelwise_drop40_long_336`). Scaling that by the
#             1.89x base/small ratio measured at 224/12sl predicted 17.6 GB, and 20 was typed on it; the
#             run OOMed against its own cap at `20.00 GiB allowed, 18.67 GiB allocated`, so that ratio
#             understates at this corner. 27 is set on the crash rather than on the scaling. It is 85% of
#             the card — inside the owner's 2026-09-22 grant of the whole card, NOT inside the 50% share.
# deliberate: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — the OOM named 1.21 GB reserved and
#             unallocated, which is the fragmentation this setting exists to remove.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train base_labelwise336 \
  $RECIPE --epochs 12 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 336 --variant base --vram-gb 26 \
  --labels "$LABELS" --out "$MODELS/base_labelwise336_s2026.pt"
