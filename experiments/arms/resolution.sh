#!/usr/bin/env bash
# 336 -> 392 px read from the 448 px cache. A mechanism that could plausibly clear the 1.5% bar,
# unlike any knob measured so far. (448 px was the original target and does not fit — see below.)
#
# We read a 130 mm crop at 336 px, so a pixel is ~0.39 mm. A meniscal tear, a cruciate fibre
# discontinuity and a cartilage defect are millimetre-scale. Five of the twelve targets are meniscus
# or ligament.
#
# THE EVIDENCE THAT WE ARE NOT RESOLVING THEM: occlusion says centre-only 0.8021 and border-only
# 0.8006 against a whole-image 0.8448. A model reading fine joint structure would collapse when the
# joint is removed. Ours does not - which is what leaning on coarse global context looks like.
#
# THE TARGET IS NOW NAMED, which it was not when this arm was filed. Per-label scores from the
# whole-stack member (mean 0.8457) put MCL at 0.755 and Fracture at 0.784 against Baker's 0.896 and
# Medial Meniscus 0.893. The metric is an UNWEIGHTED mean of twelve AUCs, so lifting MCL alone to a
# level five other labels already reach is worth +0.0098 on the mean - twice the encoder-rate result.
#
# And MCL is where a resolution limit would bite: a thin ligament at the MEDIAL MARGIN of the field,
# read coronally, at 0.39 mm per pixel. Rarity and label noise were both TESTED as explanations and
# both refuted - Synovitis is rarer (580 positives vs 767) with the corpus's lowest reader confidence
# (0.189) and still scores 0.873. So MCL is hard to SEE, which is an imaging problem.
#
# CONTROL: wsfold0_e6 = 0.8457, the same whole-stack recipe at 336 px on the same fold. Single
# variable. (This arm originally read a windowed 336 px control, lr5x_d6_s2026 = 0.8323; see the
# note above the run for why the control moved.)
#
# THE PREDICTION, written before the run so it cannot be decided afterwards:
#   the gain must CONCENTRATE in ACL, MCL, Medial Meniscus, Lateral Meniscus
#   and be near zero for Effusion and Baker's, which are large fluid collections a coarse model
#   already sees.
# A uniform lift across all twelve REFUTES the resolution story and points at capacity or
# optimisation instead. Read the per-label table, not the mean - kneemri.evaluation.scores already reports it.
#
# control  the same recipe at 336 px on the same fold.
# cost     bytes and step time both grow with the SQUARE of the side: ~1.8x.#
# VRAM BUDGET, measured 2026-08-11 rather than guessed. Uncapped, this arm took all 32 GB and ran at
# 145 W of 575 with no epoch in fifteen minutes - Windows pages GPU memory over PCIe instead of
# raising OOM, so a run that does not fit crawls while reporting 100% utilisation. Capped at 18 GB it
# holds 19.2 GB total, 99% utilisation and 476-514 W, i.e. roughly 3.4x the throughput in 40% less
# memory. The cap is what keeps it inside real VRAM, and it leaves ~12 GB for the other project.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

# 392, NOT 448, and the reason is a hard ceiling rather than caution. Tokens go as the square of the
# side: 336 px is 576 tokens, 392 is 784 (1.36x), 448 is 1024 (1.78x). Whole-stack at 336 already
# holds 19.2 GB WITH fp16 autocast and gradient checkpointing, so 448 is ~34 GB on a 32 GB card - it
# OOMed at 20:48 UTC after ~90 seconds, and no VRAM cap could have helped.
#
# The alternatives were rejected for the same reason across the board: batch 8 fits but the ranking
# term scores pairs WITHIN a batch (66 at 12, 28 at 8), and fewer slices at 448 collides with the
# slice-count arm. Both trade the single variable away to buy pixels. 392 keeps batch, slices and
# reading fixed and moves only the sampling grid: 0.39 mm/px -> 0.33 mm.
#
# Costs no decode: slot_features interpolates from whatever cache it is handed, so this reads the
# 448 px cache that already exists and resamples down.
SIZE="${1:-392}"
CACHE_PX="${2:-448}"

# THE RATE IS AN ARGUMENT NOW, because this arm changed TWO things and the ledger credited one. It ran
# at 4e-5 against a control at 8e-6, so its +0.0066 is resolution AND five times the encoder rate - and
# the rate alone measured +0.0044 in the windowed era (knee-3om). Run this at 8e-6 to find out whether
# resolution survives on its own; the default stays 4e-5 so re-running reproduces the member on disk.
#
# THE CAP IS 26, not 28: --vram-gb bounds PyTorch's allocator and not the device, so a number close to
# physical leaves the driver no room and Windows pages over PCIe instead of raising OOM - 40 img/s
# against ~1000, measured 2026-08-13.
#
# --patience 3 saves about 9% of wall-clock and cannot change the recorded best (measured by replaying
# every arm in the ledger).
RATE="${3:-4e-5}"

# THE BATCH IS AN ARGUMENT NOW, because the rung this arm's own examples name does not fit without it.
# `python -m kneemri.experiments.arm_check --as-queued "experiments/arms/resolution.sh 448 448" --smoke`
# on 2026-08-17 died with `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.35 GiB` -
# the queued 448 rung was carrying batch 12, which knee-d2fc measured as an OOM in six seconds while
# batch 9 holds 20.00 GB at 560 img/s. The default stays 12, so every committed invocation and the
# member on disk are unchanged; 448 is queued as `resolution.sh 448 448 4e-5 9`.
#
# It costs the single variable, and the note at the run says so: 66 within-batch pairs at 12, 36 at 9.
# --pair-memory forms pairs ACROSS batches and is the lever that would buy it back, which makes 448 a
# TWO-variable arm until that is run at 392 first. 420 at batch 12 is the rung that stays single.
BATCH="${4:-12}"

# The batch goes in the NAME whenever it is not the default, because two rungs at one size would
# otherwise write one checkpoint and one ledger row - the exact shape of the dinov3 scar, where a
# rename that matched nothing had a second probe overwriting the run it was being compared against.
# Empty at batch 12, so every existing name and `res392_ws_s2026.pt` are byte-identical.
TAG=""
[ "$BATCH" = "12" ] || TAG="_b${BATCH}"
require "$LABELS"

# Fragmentation, not capacity, wrote the last 2.54 GB of that OOM: 1.17 GB sat reserved-but-unallocated
# inside the cap. Expandable segments WOULD be the standing fix and is a NO-OP HERE - torch prints
# 'expandable_segments not supported on this platform' on Windows, so this arm fitted for another
# reason and the fragmentation was never reclaimed. Measured 2026-08-13; the line is removed.

# NOT `require` on the images .npy: build_cache allocates that file at build START and fills it over
# hours, so its existence proves the build began and nothing more. Training on it early would not
# crash - it would fit a mostly-zero array and return a plausible, wrong answer to the resolution
# question. The MASK is written by a single np.save after the decode loop returns, so it exists if and
# only if the build finished.
wait_for_cache "$CACHE_PX"

# READ UNDER WHOLE-STACK, and the control moves with it: fold 0 whole-stack = 0.8457, not the
# windowed lr5x_d6_s2026 = 0.8323 this arm was written against. Resolution and slice reading are
# separate variables, so either control is single-variable; the whole-stack one is the better
# question because it is the recipe we now ship. Asking it under the windowed reading would price
# 448 px inside a design we have already left, and a win there would still have to be re-measured
# before it could be used.
#
# 12 epochs, not 24: both whole-stack folds peaked at epoch 6, and one-cycle is defined over
# total_steps, so an inflated budget leaves the rate far from annealed where the model converges
# (knee-1je). The budget belongs to the recipe.
#
# BATCH STAYS 12 even though 448 px is ~1.8x the activations. The ranking term scores pairs WITHIN a
# batch - 66 at batch 12, 28 at batch 8 - so shrinking the batch changes the objective and this stops
# being a single-variable arm. If 22 GB does not hold it, Windows pages over PCIe rather than raising
# OOM: watch img/s in the first minutes, not for a crash.
run_train "resolution_${SIZE}${TAG}" \
  $RECIPE --epochs 12 \
  --slice-pooling max --whole-stack \
  --image-size "$CACHE_PX" --train-size "$SIZE" \
  --backbone-lr "$RATE" --batch "$BATCH" --patience 3 $THROUGHPUT --vram-gb 26 \
  --labels "$LABELS" --out "$PROCESSED/res${SIZE}${TAG}_ws_s2026.pt"
