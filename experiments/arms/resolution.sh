#!/usr/bin/env bash
# 336 -> 448 px. A mechanism that could plausibly clear the 1.5% bar, unlike any knob measured so far.
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
# CONTROL: lr5x_d6_s2026 = 0.8323, the same recipe at 336 px (channel stacking, rate 4e-5). Single
# variable. NOT compared against the whole-stack members - those differ in the slice read as well.
#
# THE PREDICTION, written before the run so it cannot be decided afterwards:
#   the gain must CONCENTRATE in ACL, MCL, Medial Meniscus, Lateral Meniscus
#   and be near zero for Effusion and Baker's, which are large fluid collections a coarse model
#   already sees.
# A uniform lift across all twelve REFUTES the resolution story and points at capacity or
# optimisation instead. Read the per-label table, not the mean - kneemri.scores already reports it.
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
SIZE="${1:-448}"
require "$LABELS"

# NOT `require` on the images .npy: build_cache allocates that file at build START and fills it over
# hours, so its existence proves the build began and nothing more. Training on it early would not
# crash - it would fit a mostly-zero array and return a plausible, wrong answer to the resolution
# question. The MASK is written by a single np.save after the decode loop returns, so it exists if and
# only if the build finished.
wait_for_cache "$SIZE"

run_train "resolution_${SIZE}" \
  --fold 0 --epochs 24 --one-cycle --weight-floor --rank-weight 1.0 \
  --image-size "$SIZE" --backbone-lr 4e-5 --recompute-activations --vram-gb 20 \
  --labels "$LABELS" --out "$PROCESSED/res${SIZE}_s2026.pt"
