#!/usr/bin/env bash
# The rung above 392, which was behind a memory wall that no longer exists.
#
# WHY THIS FILE EXISTS AT ALL, given `resolution.sh` already takes the size as $1: arm_check and the
# smoke run every arm with NO arguments, so a rung reachable only by an argument is a rung nothing
# checks, nothing smokes and nothing can queue. A parameter that no committed invocation passes is not
# an arm — it is a comment. This file is the invocation; the recipe stays in one place.
#
# WHAT CHANGED (knee-d2fc, measured 2026-08-17, one epoch on 400 studies, fold 0, whole-stack +
# compile + activation checkpointing + bf16, --vram-gb 28 on a co-tenanted card):
#
#   392 px, batch 12 (shipping):  20.46 GB peak,  733 img/s,  55 s/epoch
#   420 px, batch 12:             23.42 GB peak,  618 img/s,  89 s/epoch
#   448 px, batch  9:             20.00 GB peak,  560 img/s,  88 s/epoch
#   448 px, batch 12:             OOM in six seconds
#
# experiments/README.md:370 records that 392 already held 28.4 GB and that buying the next rung would
# mean a smaller batch — which changes the ranking term's within-batch pair count (66 pairs at 12, 28
# at 8) and stops it being one variable. Activation checkpointing took 392/batch-12 to 20.5 GB, and the
# headroom it freed buys 420 at the SAME batch 12. That confound applies to 448 and no longer to 420.
#
# WHY IT IS WORTH A SLOT. Resolution across the image plane is the ONE cause of the 0.054 frontier gap
# (knee-jk3) that survived testing: breadth across sequences was refuted at +0.0002, the 12-of-36 slice
# cache at no cost. 336 -> 392 is also the largest single measured jump in the ledger (+0.0066, and
# confounded with the encoder rate). 0.33 mm/px -> 0.31 mm.
#
# CONTROL: res392_ws_s2026 = 0.8523, same recipe, same fold, same holdout — so the rate stays at
# resolution.sh's 4e-5 default rather than being argued down to 8e-6, because that is what the control
# was trained at and the point here is ONE variable.
#
# COST: 1.6x the 392 arm's wall-clock. No decode — the cache is already 448 px and slot_features
# resamples down from whatever it is handed.
#
# 448/batch-9 is the rung after this one, and it DOES move the batch. Read it only if 420 pays.
#
# Delegating rather than copying: every guard resolution.sh carries — the labels requirement, the
# cache wait, the VRAM cap argued from a measured OOM — applies unchanged, and a second copy of them
# is a second thing to keep true.
#
# Backslashes folded to forward ones before the directory is taken: arm_check hands the arm a Windows
# path (`experiments\arms\resolution_420.sh`), and `%/*` on a string with no forward slash returns the
# string unchanged — so the delegate resolved to `…resolution_420.sh/resolution.sh`, printed nothing,
# and arm_check reported the arm as not existing. Silent, and exactly the clean-pass-over-nothing shape
# that module carries two scars from.
# control  res392_ws_s2026 = 0.8523 (declared 2026-09-03 from this header own prose, knee-sgzgc)
HERE="${BASH_SOURCE[0]//\\//}"
exec bash "${HERE%/*}/resolution.sh" 420 448
