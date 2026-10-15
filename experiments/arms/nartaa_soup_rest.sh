#!/usr/bin/env bash
# bead: knee-p844l
# reading: nartaa_soup_k4_s2026.pt
# unmeasurable: fold -1, read on the BOARD against basinsoupff_v1 (0.946), like basinsoup_ff.
#
# nartaa_soup's remainder, ONE child at a time. The pair held 31.6 GB of a 32.6 GB card: Windows paged
# and froze the owner's desktop, so c802 was killed at epoch 2 step 427 (2026-10-15 11:58Z). It resumes
# from the .resume its epoch 1 left (same schedule, same streams), and c803 follows cold. Its `after`
# soups the four, as in nartaa_soup.
# deliberate: --resume on c802 only: recovery of the killed run, not a recipe change.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
run="$REPO/experiments/run.sh"
ALLOW_OVERWRITE=1 bash "$run" experiments/recipes/distillation/nartaa/nartaa_soup_c802.toml --resume "$@" || exit
exec bash "$run" experiments/recipes/distillation/nartaa/nartaa_soup_c803.toml "$@"
