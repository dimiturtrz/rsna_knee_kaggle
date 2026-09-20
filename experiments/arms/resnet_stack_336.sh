#!/usr/bin/env bash
# bead: knee-mc23l
# reading: resnet_stack_336_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   resnet_stack_224 at the resolution the stack was tuned for. ONE term moves: 224 -> 336. The trunk,
#   the head, the targets, the regularisation, the slice count, the batch and the epoch budget are held.
#
# WHY. THE ONLY UNTESTED OBJECT SIZED LIKE THE GAP IS A DIFFERENT REPRESENTATION.
#   Live board 2026-09-20T12:08:11Z: rank 70 of 4037 at 0.946, leader 0.957, gap 0.011, ours
#   +0.00016/day against a frontier +0.00030/day -- losing 0.00014/day. Everything that composes into
#   0.946 is spent: best solo ever shipped 0.945, a three-seat ensemble buys ONE tick over it, the
#   refit term is ADDITIVE and fixed at +0.005 (knee-uh1j5), and the best honest base we own
#   (dml_b_s2026 0.9078) is already refit AND shipped at 0.943. First place needs an honest base near
#   0.916 ledger75; the top eight honest members span 0.0033, so it cannot come from inside the
#   DINOv2-S family. knee-fvzvm closed test-time adaptation with the word REPRESENTATION-BOUND.
#
# deliberate: --image-size 336 --slices 12 --band-low 0.1 --band-high 0.9 — the cache is
#             images_336px_12sl_0.1-0.9band.npy, on disk. The BAND is the champion's. The SLICE COUNT
#             is 12 rather than the champion's 20 because the ledger prices the slice-count axis at
#             -0.0001, flat, so the term is free, and because holding 12 keeps the contrast against
#             resnet_stack_224 to resolution ALONE. This is the opposite of knee-67rgi's position,
#             where 12sl/0.2-0.8 was a FALLBACK forced by a deleted cache and was carried as a stated
#             bias against that arm; here both terms are chosen and one of them is now matched.
# deliberate: --batch 2 --accumulate 4 — held from the 224 arm, not re-tuned. At 336 the activations
#             are 2.25x, so this lands near 13 GB against the owner's 22 GB cap, which leaves real
#             slack; device_budget records an over-commit that paged over PCIe, drew three nvlddmkm
#             engine resets and took BOTH projects' processes down. Batch is also held because
#             BatchNorm statistics are a CONFOUND: dropping to batch 1 to buy headroom would move the
#             trunk comparison and the normalisation together, and the result would not be readable.
# deliberate: --variant resnet50d --unfreeze-last 16 --backbone-lr 2e-4 --head-lr 2e-4 — the CNN's own
#             trunk, block count and rate, all inherited unchanged from resnet_stack_224. A ViT rate on
#             a conv trunk under-drives it; that ambiguity cost raddino_dense its verdict.
# deliberate: --epochs 12 — the same budget, so the contrast isolates resolution and not the schedule.
#
# CO-TENANCY. The card is shared by the owner's arrangement. resnet_stack_224 measured the cost of
#   ignoring that: biohub 10.0 it/s pre-arm -> 6.7 with our arm resident -> 9.1 within 90 s of the kill,
#   -33% on the neighbour, and the recovery is what makes it a mechanism rather than a coincidence.
#   card_check before arming. A co-tenancy sample taken inside the first ~2 min measures WARMUP and
#   lies -- that one read +25% FASTER before settling at -33%.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train resnet_stack_336 \
  $RECIPE --epochs 12 --variant resnet50d --image-size 336 \
  --through-plane --slice-pooling labelwise --slices 12 --band-low 0.1 --band-high 0.9 \
  --whole-stack --unfreeze-last 16 --backbone-lr 2e-4 --head-lr 2e-4 --weight-decay 0.05 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 --stochastic-depth 0.05 \
  --batch 2 --accumulate 4 --vram-gb 14 \
  --labels "$LABELS" --out "$MODELS/resnet_stack_336_s2026.pt"
