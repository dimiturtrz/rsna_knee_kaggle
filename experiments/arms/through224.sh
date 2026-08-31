#!/usr/bin/env bash
# through224.sh — native224's recipe with `--through-plane` and nothing else moved. The fourth of four
# arms transporting the AGGREGATION family onto the 224 control, and the only one that acts BEFORE the
# encoder sees a slice.
#
# bead: knee-7d53
#
# WHY THE FAMILY. `experiments/README.md:5024-5075`: fourteen arms, a total split by kind — more input
# lost on the focal six, changed aggregation gained. `through_dense` is the fourth positive (+0.0005
# focal, +0.0065 regional, +0.0033 mean) and the only one whose mechanism is upstream of the feature.
# All four were measured at 336 px and 20 slices.
#
# THE MECHANISM. The trunk takes three input channels because it was pretrained on RGB, and today each
# slice is fed as three copies of itself — two thirds of the input bandwidth carries no information.
# `DepthWindow.channels` (`kneemri/models/depth_window.py`) fills them with (i-1, i, i+1) instead: the
# centre channel still holds the slice being encoded, so whatever the first convolution learned about the
# middle channel of an RGB image applies to the slice the feature vector is about. The gather runs along
# the slice axis WITHIN a slot and never across one, and the edges CLAMP rather than wrap, so the first
# and last slices see exactly what they see today. Slice count, band, pooling, parameter count and FLOP
# count are all unchanged; this is the same encoder reading the same pixels with its dead channels used.
#
# THE RESOLUTION ARGUMENT, AND I PREDICT MORE, WITH A CAVEAT THE CONTROL CANNOT SETTLE. At 224 each patch
# is 14 px over a 1.5x wider field, so each token averages a fixed-mm structure with ~2.25x more
# surrounding tissue: the in-plane detail is precisely what the native read gives away. The through-plane
# offset is NOT downsampled — a neighbour slice is a whole slice regardless of the in-plane grid — so the
# cheapest remaining detail in this configuration lies along z, and the channel that carries it is free.
# That is the argument for MORE: the coarser the in-plane read, the larger the share of the model's
# remaining discriminative budget that depth context can supply. THE CAVEAT, and it cuts the other way:
# the neighbours are taken on the SAMPLED axis, not the source series, so the gap between i and i±1 is
# 0.60/12 = 0.050 of the stack here against 0.80/20 = 0.040 at the dense read — a 1.25x wider jump, less
# registered anatomy per channel and more parallax. And 2 of 12 slices are edge-clamped (17%) against 2 of
# 20 (10%), so the arm differs from its control on ten of twelve slices rather than eighteen of twenty.
# Net I still predict MORE, because the wider gap costs redundancy and buys coverage, while the in-plane
# loss is unambiguous. If it reads flat, the honest suspect is the spacing and not the mechanism, and the
# follow-up is `--spacing adjacent` (already armed at 224 as `adjacent224.sh`) crossed with this flag —
# not a kill.
#
# control  native224 = 0.8502, best at epoch 7, converged not truncated. 12 epochs, 12 slices, 0.20-0.80
#          band, whole-stack, backbone 4e-5, 224 px, same labels, same 909-study holdout, three copies of
#          each slice in the encoder's channels. The single variable is --through-plane.
# cost     ~19 min for twelve epochs at 94 s. Same cache as native224 — the window is a gather over
#          slices already decoded, so no re-read and no extra encoder pass.
# price    19 min for P(clears the 0.0062 floor on the MEAN) = 0.15. +0.0033 mean at 336 is half the
#          floor, so a clear needs the resolution argument to roughly double the effect; the mechanism is
#          the most plausible of the four for that, since it is the only one whose value is defined
#          against what the in-plane read discards.
#
# HARNESS CHECK BEFORE READING ANY NUMBER. The printed recipe must say "through_plane": true and
# "image_size": "224", and the log must load 4,406 studies and hold out 909. Parameter count is
# UNCHANGED by this flag — the three channels were always there — so the manifest stamp is the only
# witness; `--through-plane` defaults false, and a flag that never reached `StudyDesign` produces a run
# byte-identical to the control, which is a PLUMBING verdict rather than a null.
#
# THE FALSIFIABLE PART, written before the run.
#   CONFIRMED   the mean peak clears 0.8564 (control + floor). Free depth context at the native read; it
#               becomes default for every 224 member and it composes with the other three, since it acts
#               at a different point than any of them.
#   DIRECTIONAL positive under the floor. Kept — zero parameters, zero FLOPs, zero decode — and stacked
#               into `fuse_attend`; not a finding on its own.
#   TRANSPORT LOSS  focal six flat and regional positive, i.e. the 336 shape at lower amplitude. Reads as
#               the slice-gap caveat above rather than as a verdict on depth context.
#   REFUTED     the mean falls by more than the floor. A 1.25x wider inter-slice gap with the in-plane
#               grid also coarsened can put three mutually misregistered images in the channels of one
#               token, and a trunk pretrained on registered RGB has no reason to tolerate that. This
#               closes through-plane AT 12 SLICES, not at 20, and the ledger must record which.
#
# deliberate: --through-plane (the arm's thesis — the encoder's dead channels carry depth, not copies)
# deliberate: --image-size 224 (native224's control, per that row's standing instruction)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train through224 \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 --through-plane \
  --labels "$LABELS" --out "$PROCESSED/through224_s2026.pt"
