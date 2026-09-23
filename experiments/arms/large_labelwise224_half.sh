#!/usr/bin/env bash
# large_labelwise224_half.sh — the capacity rung, with MY OWN confound taken out of it.
#
# bead: knee-jzho6
# reading: large_labelwise224_half_s2026.holdout.npz
#
# WHY THIS ARM EXISTS. `large_labelwise224_dense` was launched as "one variable against
# `base_labelwise224_dense`: --variant large". It is not one variable, and the defect is mine.
# `--unfreeze-last 6` is a constant every recipe here inherited, fitted on a 12-block trunk where it
# adapts HALF. DINOv2-large has 24 blocks, so the same flag adapts a QUARTER. That arm therefore moved
# WIDTH and the ADAPTED FRACTION together, and gave the wider trunk half the relative adaptation.
#
# The trace said so before it finished. At matched epochs, midpoint, large sat behind base at every
# epoch but one (e0 -0.0084, e2 -0.0057, e3 -0.0023, e4 -0.0088, e5 -0.0068) with a HIGHER train loss
# at the matched epoch (0.7015 against 0.6824 at e5). Higher train loss is the signature of a model
# learning SLOWER, not one overfitting — which is what a quarter-adapted trunk looks like and is NOT
# what a capacity ceiling looks like. So whatever that arm scores is evidence about the harness, and it
# may not be used to close the capacity axis in either direction.
#
# ONE CHANGE against `large_labelwise224_dense`: `--unfreeze-last 12`. Twelve of twenty-four is half the
# trunk, which is exactly the fraction `base_labelwise224_dense` got from its inherited 6 of 12. That
# makes the comparison against the base control a WIDTH comparison at matched adaptation, which is the
# thing the capacity axis was supposed to be asking. Everything else is byte-identical: 224 px, 12
# slices, 0.2-0.8 band, batch 12 (so the pairwise `--rank-weight` term keeps its 66 pairs per step),
# `--calibrated-targets --hold-annotations`, 12 epochs, same array on disk.
#
# NOT THE SAME AS THE DEPTH ARMS THAT DIED. `labelwise_dense_deep12` and `_deep9` also raised
# `--unfreeze-last` and both OOMed — but those were at 336/20sl on the SMALL trunk, the corner where
# seven crashes have now been traced to allocator fragmentation (`docs/AXES.md`). This is 224/12sl,
# where `large_labelwise224_dense` measured 23712 MiB under a 26 GB cap and did not fragment. They are
# the same flag at different corners, and the corner is what decided those deaths.
#
# MECHANISM. Large is 304M parameters and 1024-wide against base's 86.6M and 768. The claim the 224
# corner has twice sustained is that a LABELWISE readout can spend width — it holds a separate query per
# label, so a wider feature gives each of the twelve its own subspace instead of forcing them to share.
# The competing outcome is that 909 studies cannot support 150M trainable parameters and the run
# overfits, which now WOULD show as a low train loss with a holdout peak well before epoch 12 — the
# opposite signature from the quarter-adapted arm, which is why this arm can tell them apart and that
# one could not.
#
# PRICE, out loud. `large_labelwise224_dense` ran 386 s/epoch at 753 img/s (against the base control's
# 1642). Twice the trainable blocks costs a little more per step, so ~420 s/epoch and ~1.4 h for 12
# epochs. Zero decode, zero submission slots, control free on disk. P(clears the 0.8950 base control)
# ~ 0.35. P(clears 0.9078, the best solo) ~ 0.15. P(clears the 1.5% bar) ~ 0.08.
#
# VRAM, and the risk named before it is paid. The quarter-adapted arm sits at 23712 MiB. Six more
# 1024-wide blocks is ~76M extra trainable parameters, and at ~16 bytes each of optimizer state that is
# ~1.2 GB; activations stay cheap because `--recompute-activations` is on (`recipe.sh:31`, and it was on
# for all seven of the 336 crashes). So ~25 GiB against a 26 GB cap — tight, and P(OOM) ~ 0.3. An OOM
# costs ~3 minutes under a named cap, not the 1.4 h, which is what makes this worth launching rather
# than modelling further.
#
# BANDS, pre-registered, ledger75, against base_labelwise224_dense 0.8950 (the matched-fraction control).
#   * >= 0.9012  width pays once adaptation is matched, and the quarter-adapted arm was measuring my own
#                flag rather than the trunk. The next arm is this recipe at
#                `--epochs 36 --slot-dropout 0.40` and that is the solo deliverable candidate. Note it
#                would still be short of the shelf: the best solo is `dml_b_s2026` 0.9078, and
#                `basinsoup_dmlb_k4` 0.9092 (a weight soup is a single averaged checkpoint, so it ships
#                as a solo).
#   * 0.8950-0.9012  directional. Keep nothing, compose nothing. Read the epoch of the peak: still
#                improving at 12 means the budget bound it and the 36-epoch version is the re-test.
#   * <  0.8950  capacity has a ceiling between base and large on 909 studies, and it is now a clean
#                statement because the fraction is matched. Read the TRAIN loss to say which ceiling:
#                below base's 0.5663 at a worse holdout is overfitting (the data is the constraint, and
#                the axis is DATA-closed, which retires every larger trunk); above it is still
#                under-adaptation and the constraint is the rate schedule, not the capacity.
#
# READ THE FOCAL LABELS, NOT ONLY THE MEAN. Lateral Meniscus -0.1925, ACL -0.1225, MCL -0.0964 lead the
# model-side error census. Width's claim is per-label subspace, so it should move the labels currently
# sharing a representation worst — the three focal ones. A mean that moves without them is not this
# mechanism. Say so.
#
# HARNESS CHECK. The printed recipe must name "variant": "large" AND "unfreeze_last": "12" AND
# "image_size": "224" AND "slices": "12" AND "slice_pooling": "labelwise" AND
# "calibrated_targets": "True", and the trunk must load `facebook/dinov2-large`. Stamped values are
# STRINGS; a grep for lowercase true returns empty and reads like a missing flag. If "unfreeze_last"
# reads "6" this is the confounded arm re-run and the card is being spent on nothing.
#
# control  base_labelwise224_dense_s2026 = 0.8950 ledger75 @ epoch 8, 909 studies. No control run is owed.
#
# deliberate: --unfreeze-last 12 — matched ADAPTED FRACTION against the base control's 6 of 12, which is
#             the arm's whole point and not a capacity knob in its own right.
# deliberate: --variant large (the axis under test)
# deliberate: --vram-gb 28 — RAISED from 26 after this arm OOMed at 26 in two minutes, 2026-09-23T01:57Z.
#             The ceiling was the binder, not fragmentation: PyTorch held 21.94 GiB allocated against a
#             26.00 GiB cap with 3.48 GiB reserved-unallocated, and the CARD still had 3.83 GiB free of
#             31.84 when the 916 MiB request failed. So the memory existed and the cap forbade it. 28 GB
#             leaves the driver ~2.8 GB and covers the ~1.2 GB the six extra trainable 1024-wide blocks
#             cost over the quarter-adapted arm's measured 23712 MiB. If it OOMs AGAIN at 28 the binder
#             is fragmentation after all and this corner joins 336 as machine-closed for a wide trunk.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train large_labelwise224_half \
  $RECIPE --epochs 12 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 --unfreeze-last 12 \
  --image-size 224 --variant large --vram-gb 28 --calibrated-targets --hold-annotations \
  --labels "$LABELS" --out "$MODELS/large_labelwise224_half_s2026.pt"
