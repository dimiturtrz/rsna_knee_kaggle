#!/usr/bin/env bash
# bead: knee-p844l
# reading: pfteach_soup_k4_s2026.pt
# unmeasurable: fold -1, read on the BOARD against nartaa_soup_k4 (solo 0.948) and basinsoupff_v1 (0.946).
#
# nartaa_soup with ONE change: the teacher file is the pf line's (CoAtNet-384 + heliosli's R6 reader at
# heliosli's per-finding shares) instead of the CoAtNet alone. Children run ONE at a time: a pair held
# 31.6 GB of the 32.6 GB card on 2026-10-15 and froze the desktop (nartaa_soup.sh). c803's `after` soups.
#
# PRE-REGISTERED ON THE BOARD (solo, one seat):
#   >= 0.950   the second teacher family transfers; the solo pick moves.
#   0.948-0.949 inside the board's resolution of nartaa_soup; R6 adds nothing the CoAtNet taught.
#   <= 0.947   the wider teacher spends nartaa_soup's gain; this teacher closes.
# CAVEAT: both teachers fitted every labelled study, so their train readings are partly memorised, as
# nartaa_soup's were. Only the board reads it.
#
# deliberate: seeds 801-803, nartaa_soup's own, so the teacher file is the only difference.
# requires: pf_teacher.npz — `kneemri-r6teacher` on Kaggle, then `python -m kneemri.kaggle.pf_teacher`.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
run="$REPO/experiments/run.sh"
bash "$run" experiments/recipes/distillation/pfteach/pfteach_soup_c801.toml "$@" || exit
bash "$run" experiments/recipes/distillation/pfteach/pfteach_soup_c802.toml "$@" || exit
exec bash "$run" experiments/recipes/distillation/pfteach/pfteach_soup_c803.toml "$@"
