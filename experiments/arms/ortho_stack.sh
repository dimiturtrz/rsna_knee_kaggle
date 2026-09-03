#!/usr/bin/env bash
# ortho_stack.sh — does the knee-continuation gain survive the SHIPPED operating point?
#
# THE READING THIS ARM IS BUILT ON, AND WHY IT IS NOT THE ONE THE DECLINE FILE MADE. `ortho_dense`
# (ortholarge) and `v3large_dense` (dinov3-vitl16, the weights ortholarge continues FROM) are the same
# recipe with `--variant` the only difference: 12 slices, 224 px, band 0.2-0.8, 4e-5, 12 epochs, both
# converged at 13. Trunk size is HELD (303M against 303M) and the 1.25M-knee continuation corpus is the
# single free variable. Fitted holdout: v3large 0.8258, ortholarge 0.8452 — +0.0194, three times the
# 0.0062 floor. That is the cleanest single-variable trunk read this project owns and nothing has been
# built on it.
#
# THE FROZEN PROBE AGREES, ONCE IT IS READ CORRECTLY. `comp/members.declines:199` closed this lane on a
# probe that ranked ortholarge 0.7387 under v3large 0.7446. That reading is void: knee-x4yx found
# feature_probe compared every trunk at a SHARED ridge penalty while trunk feature magnitudes differ
# ~17x, so one penalty cannot be fair to two trunks. Re-read at the penalty each trunk is entitled to
# (nested split inside its own training positions, n=909): ortholarge 0.8249, v3large 0.8046 — +0.0203
# at focal, +0.0202 at mean. The sign reverses, and the corrected probe now agrees with the fitted arm
# to 0.001 on a +0.020 continuation gain. Two independent instruments, same sign, same size.
#
# THE FROZEN-VERSUS-FITTED CAUTION STILL BINDS, and it is why this arm exists rather than a probe.
# radnet121 probes 0.8197 at its OWN penalty — above the incumbent — and `radnet_family` fitted that
# trunk on band_dense's recipe for 0.6716. So the probe is sound for one trunk here and inverted for
# another, and no probe result licenses a seat. Only a fit does.
#
# The decline's other leg reads `v3large`'s GREEDY SEAT VALUE (-0.0012) as the subtraction. Seat value
# and solo quality are different questions and the seat one is measured inside a pool that is 100%
# DINOv2, where any ViT-L is redundant with nothing. This arm asks the solo question only.
#
# NOT A RE-OPEN ON TRUNK SIZE, which is what that note forbids. Size is fixed at 303M on both sides of
# the +0.0194. The free variable is the continuation corpus and it stays the free variable here.
#
# ONE VARIABLE AGAINST `ortho_dense`: the RECIPE, moved as a unit. This is band_dense's flag set with
# `--variant ortholarge`, which is the same unit step `base224` -> `band_dense` took on the ViT-B
# (0.8509 -> 0.8578, +0.0069). If the +0.0194 is a real property of the initialisation it should be
# visible on top of that step; if it was an artifact of 224 px and 12 slices it will not be.
#
# THE MEMORY CLAIM IN `ortho_dense.sh` WAS TRUE ONLY AT THE DEFAULT BATCH. That header judged a
# 1024-wide 24-block ViT-L could not hold 20 slices at 336 px in 32 GB, and at `--batch 12` it cannot:
# `vram_need` prices it at 106144 MiB. Accumulation costs no activation memory (options.py:72 — the
# effective batch is `batch * accumulate` at the memory of `batch` alone), so `--batch 1 --accumulate
# 12` holds the effective batch at the 12 EVERY arm in the ledger ran, and prices at 13274 MiB. The
# recipe was never out of reach; only one flag combination was.
#
# BRANCHES, written before the run, against `band_dense` = 0.8578:
#   >= +0.0062   the knee continuation pays at the operating point we ship. Roster it, and the
#                OrthoDiffusion licence question (Q-CE / knee-nxthh) stops being speculative — the
#                weights are MIT (arXiv:2601.18250) and what is actually open is whether the 300k
#                fastMRI slices in its corpus reach us through the NYU DSA. Answer it then, not now.
#   within the   the +0.0194 does not survive the strong recipe: it was a low-resolution advantage, and
#   0.0062 floor a natural-image ViT-B catches a knee-continued ViT-L once both get the full stack.
#                THAT closes the in-domain trunk lane on the FITTED ruler, which is the closure the
#                frozen probe was not entitled to make.
#   <= -0.0062   the ViT-L loses the recipe step the ViT-B gained, which is the trunk-SIZE reading
#                `members.declines:195` already argued. It would confirm that note by the right
#                measurement and close the lane for a different reason than the one on file.
#
# THE MIDDLE BRANCH IS CONDITIONAL, ANNOTATED 2026-09-04 WHILE THIS ARM WAS STILL RUNNING. It reads
# "closes the in-domain trunk lane", and it says that on the assumption that OFFLINE ORDER TRANSFERS TO
# THE BOARD. For this trunk family that assumption is now the thing in question. Fixing a scope defect
# in comp/board.jsonl put `ortho_solo_v1` into board_transfer's fit for the first time — ortho_dense
# served alone, the only non-DINOv2 board point the project owns — and it produced the record's first
# two RESOLVED inversions in 342 pairs, both on its own side: it carries the LOWEST offline of all 37
# entries (0.8452) and read 0.907, about +0.02 board above where the DINOv2-fitted slope puts it.
#
# So an `ortho_stack` that merely MATCHES band_dense offline may be a member that ties the incumbent on
# our ruler and beats it on the board. The threshold is NOT moved — moving a pre-registered number after
# seeing a related measurement is how a result becomes an artefact — but the middle branch now closes
# the lane only if knee-l67zk's two-solo instrument comes back null. That instrument is what separates
# "ortholarge transfers better than its holdout says" from "the current kit transfers better than the
# August kits did"; the board record cannot, because offline level and date are collinear across it.
# See interpretations/kneemri/2026-09-04_the-ruler-had-four-rows-outside-it.md.
#
# WHAT IT CANNOT SETTLE. Nothing about SEATING — the pool is DINOv2-monoculture and a seat read here
# would be confounded exactly as `v3large`'s was. Solo holdout only. If it clears, seat value is the
# next question and a separate one.
#
# COST, priced before queueing. ViT-L at 20x(336/16)^2 = 8820 tokens a study against the 2352 it ran at
# 224/12, so ~3.75x the compute of a measured 401 s epoch, at batch 1: ~1950 s an epoch, ~6.5 h for 12.
# MEASURED, from the trainer's own row rather than a progress line: epoch 0 took 1583 s (`seconds` in
# ortho_stack_s2026.metrics.jsonl), so ~5.3 h for 12 — the 1950 s estimate was slightly PESSIMISTIC.
# An earlier note here read ~3200 s an epoch and ~11 h; that was derived from the 309 img/s progress
# line and was wrong, and it briefly made the control arm conditional on a price that did not exist.
# Read the epoch time from the metrics row, never from img/s.
# Peak VRAM measured 6.51 GB against the 13274 MiB `vram_need` priced, so the card has room beside it.
# ZERO submission slots. Plausible gain +0.019 offline if the term carries, which lands ~0.877 against
# the best solo on disk (`stack_increments_s2026` 0.8642) — +0.013 over it, and at board transfer
# +0.97-1.25 that is +0.013-0.016 board against a 0.008 gap to the leader. Odds it carries: even. The
# mechanism is measured with size and recipe matched, so it is not a guess; the risk is entirely that
# resolution erases it. 6.5 h ÷ an even shot at covering the whole gap is the best ratio open — every
# other lane is closed, and the card is otherwise idle.
#
# --pair-memory 66, AND WHY THE FIRST LAUNCH OF THIS ARM WAS VOID. The first run reached epoch 2 at
# 0.8010 against band_dense's 0.8197 and was killed there, because `--batch 1` had silently switched
# the ranking term OFF. `RankedObjective.ranking` forms pairs WITHIN the micro-batch (`pairs(batch,
# batch)`) and self-pairs drop out, so a batch of one forms no pairs at all; `--accumulate 12` cannot
# repair that, since it sums the gradients of micro-batch losses that were each computed alone. The
# control ran at batch 12 with the term ON. So the arm moved TWO variables against band_dense — the
# trunk and the objective — while `settings()` recorded `rank_weight: 1.0` on both sides, which is
# exactly the failure objective.py:159 warns about ("a --rank-weight that never reached the
# constructor trains plain cross-entropy under a ranking arm's name"), arriving by batch size rather
# than by dispatch. The middle branch below CLOSES the in-domain trunk lane, and closing the only open
# lever against a control that trained a different objective is the worst outcome available here.
#
# CAPACITY, DERIVED, NOT PICKED. band_dense at batch 12 forms C(12,2) = 66 unordered pairs a label. At
# batch 1 a memory of M steps supplies M partners, so M = 66 matches the pair mass rather than
# approximating it. Staleness moves the RIGHT way: 66 micro-batches is 5.5 optimiser steps at
# accumulate 12, against the remembered default of 8, and 66/3492 = 1.9% of an epoch against
# 8/291 = 2.7%. The pool is wider AND fresher than the control's.
#
# bead: knee-b7a50
# deliberate: --batch --accumulate --variant --pair-memory
# control  band_dense = 0.8578 (best epoch 8), identical flags with the incumbent trunk.
#          ortho_dense = 0.8452, identical trunk at 224 px / 12 slices.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train ortho_stack \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant ortholarge --batch 1 --accumulate 12 \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --pair-memory 66 \
  --labels "$LABELS" --out "$MODELS/ortho_stack_s2026.pt"
