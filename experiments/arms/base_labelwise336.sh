#!/usr/bin/env bash
# base_labelwise336.sh — carry the validated trunk up to the corner we ship at.
#
# bead: knee-mm6v0
# reading: base_labelwise336_s2026.holdout.npz
#
# WHY IT OOMED TWICE AND WHAT CHANGED. Both deaths (20 GB cap / 18.67 allocated, then 27 / 23.21 + 2.38)
# happened with gradient checkpointing OFF: `--recompute-activations` (`adaptation.py:248`, default False)
# was simply absent from the line, and twenty-five other arms pass it. It is now passed. The flag is the
# ENABLING one, not a second variable — `encoder_surgery.py:470-495` measures it on this workload at 798
# img/s against 946 (a 19% tax) for nearly all of the encoder's activation memory, same batch, same loss,
# same 66 within-batch pairs the `--rank-weight` term scores, which is why that docstring chose it over
# splitting the batch.
#
# AND WHY 12 SLICES RATHER THAN 20. The 20-slice version of this corner (`base_labelwise_dense336`) was
# run WITH checkpointing on 2026-09-22 and still OOMed: 21.06 GiB allocated, 3.57 more requested, 26.00
# allowed, 6.17 free of 31.84 on the card. That corner is machine-barred for a base trunk, and raising the
# cap toward 30 does not fix it — it PERMITS PCIe paging, which thrashes instead of crashing. Slices are
# the only free dimension, because batch may not be cut without changing the pairwise term. 12/20 of that
# footprint is ~14.8 GiB, well inside the 26 cap.
#
# PRICE, restated. ~385 s/epoch at this corner x the measured 1.23 checkpointing tax is ~474 s, so ~1.6 h
# for 12 epochs. Zero decode, zero submission slots, control free. P(clears 0.8961) ~ 0.35.
# P(clears the 1.5% bar) ~ 0.10.
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
  --recompute-activations \
  --image-size 336 --variant base --vram-gb 26 \
  --labels "$LABELS" --out "$MODELS/base_labelwise336_s2026.pt"
