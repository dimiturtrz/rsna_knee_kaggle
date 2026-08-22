#!/usr/bin/env bash
# V-JEPA2-large as a FITTED member, on the depth axis the frozen probe could not reach.
#
# WHAT THE PROBE SETTLED AND WHAT IT DID NOT. vjepa_probe read 0.7620 against a DINOv2-large control's
# 0.7669 — a delta of -0.0049, inside the 0.0062 holdout floor, and the arm's own pre-registration calls
# that branch a TIE rather than a kill. Capacity was removed by construction (both trunks are ViT-L,
# 1024 wide, 24 blocks), so what remained under test was pretraining DOMAIN: video volume against
# ImageNet plane. The tie blocks the CHEAP path only. V-JEPA's objective is prediction in LATENT space,
# not linear separability, so a flat frozen probe is a known property of that objective and says little
# about what the trunk does once its blocks are opened.
#
# WHAT THE PROBE COULD NOT USE, AND WHY THAT IS THE POINT. The probe read this trunk through a window of
# THREE slices. A tubelet transformer tokenises TWO slices per token; three slices is one and a half
# tubelets, so the one thing this encoder has that a 2D ViT-L does not — a learned depth axis — was
# spent on a window too narrow to carry it. That was not a choice: `Trainer.whole_stack` is the only
# path to a window wider than GROUP = 3, and its guard demanded per-slice pooling, which a volumetric
# read is not by definition. The path was unarmable from any flag combination until `check_whole_stack`
# moved the guard onto the model. This arm is the first read of the trunk with its depth axis lit.
#
# WHY THESE NUMBERS AND NOT OTHERS.
#   --train-size 256   NOT a knob. `VolumeRead.check_tokens` asserts the trunk's NATIVE crop, because
#                      the spatial grid is where the fitted weights live (native // patch). 336 px was
#                      never a runnable option here, and the earlier audit that priced it was pricing a
#                      size the code refuses. Depth is genuinely free — 3D RoPE derives position ids
#                      from the run-time frame count, with no learned position table to interpolate.
#   --slices 20        Even, so tubelet 2 divides it, and the same depth budget band_dense fits at.
#   --slice-pooling channels   The only non-per-slice pooling, which is what `VolumeRead.of` requires:
#                      the volumetric read consumes the stack as FRAMES, so there is no per-slice
#                      embedding to pool over afterwards.
#   no --vram-gb       DELIBERATELY. `DeviceBudget.needed()` derives the reservation from --slices, and
#                      arm_check reports a hand-typed 26 as OVERSTATED by 4.30 GB — i.e. the derivation
#                      lands at 21.7, which is the corrected number to four figures. knee-2nxh had the
#                      tubelet direction BACKWARDS: `units` is a PER-SLICE quantity, so a token spanning
#                      two slices HALVES it. Uncorrected this read priced at 38.3 GB against a 31.8 GB
#                      card and would have parked the run in `await_headroom` forever on memory it never
#                      intended to allocate — knee-bj90's failure mode, not an OOM.
#
# control  band_dense = 0.8578 (DINOv2-base, 336 px, 20 slices, max pooling, whole-stack), the best
#          single member on disk. NOT a matched control — the trunk, the crop and the pooling all
#          differ, and they have to: this trunk refuses the control's reading. So a raw comparison of
#          means is NOT what this arm buys.
# cost     ~2 h wall for 12 epochs (ViT-L at 20 slices a study), one cache build already on disk.
#
# WHAT IT BUYS, PRICED HONESTLY. This is an INSTRUMENT, not an improvement. Its job is to retire or
# license a whole axis — "does volume pretraining pay once the depth axis is usable" — which every
# 3D-trunk idea in the backlog is downstream of. As a member it is priced by DISTANCE, not by solo
# score: it is the only candidate whose pretraining domain differs from all six seated members, and the
# board has already read a seventh seated on OBJECTIVE distance (sevengrade, 0.931, tie) and on PIXEL
# distance (sevenplane, pending). Pretraining domain is the third axis and the largest of the three.
#
# PREDICTION, written before the run. A mean at or above 0.845 says the depth axis carries something
# the frozen probe could not see, and the member is worth a seat priced by `seat_value` against the
# champion five. 0.83-0.845 is the tie again, now taken with the trunk's advantage actually spent —
# which downgrades volume pretraining from "untested here" to "does not pay at this data scale", and
# retires the axis rather than the trunk. Below 0.82, suspect the HARNESS before the idea, in this
# order: did `--whole-stack` land (the trainer logs its window — 20, not 3), did the volumetric read
# build at all (`VolumeRead.of` returns None silently when the config lacks `tubelet_size`, and the
# model then reads as an ordinary 2D ViT-L at 256 px), and did six blocks actually open. Only with all
# three clean does the number belong to the encoder.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train vjepa_member \
  $RECIPE --epochs 12 --variant vjepa2l --pool mean --slice-pooling channels \
  --whole-stack $THROUGHPUT --slices 20 --train-size 256 --batch 6 \
  --backbone-lr 4e-5 \
  --labels "$LABELS" --out "$PROCESSED/vjepa_member_s2026.pt"
