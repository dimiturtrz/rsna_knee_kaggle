#!/usr/bin/env bash
# bead: knee-p844l
# reading: nartaa_soup_k4_s2026.pt
# unmeasurable: fold -1, read on the BOARD against basinsoupff_v1 (0.946), like basinsoup_ff.
#
# basinsoup_ff with ONE change: each 4-epoch child also fits the nartaa CoAtNet-384 teacher's readings
# (OAI + JEV-blend supervision, board 0.949 solo) at distill_weight 0.5. Parent, seeds' role, epochs,
# rate and the K=4 average are basinsoup_ff's, so basinsoup_ff_k4_s2026 (board 0.946) is the control.
#
# PRE-REGISTERED ON THE BOARD (solo, one seat):
#   >= 0.948   the teacher's labels transfer into our trunk; the next general-pick candidate.
#   0.946-0.947 inside the board's resolution of the control; the teacher adds nothing a soup lacks.
#   <= 0.945   distillation spends the soup's gain; the pseudo-label axis closes for this teacher.
# CAVEAT: the teacher fitted on all labelled studies, so its readings on our training set are partly
# memorised labels. That is the point (they are its blended labels) and why only the board can read it.
#
# deliberate: --fold -1, --init the 0.945 parent, --epochs 4, --backbone-lr 2e-5 (basinsoup_ff's op).
# deliberate: --distill-weight 0.5 = nartaa's 1/2 share in the champion blend, in label space. Not 0.7:
# tblendsolo used 0.7 under a teacher BELOW its student and read -0.017 on the board.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
# Two children share the card: one peaks at 11.7 GB reserved (basinsoup_ff_c301.full.log), so a pair
# fits a 32.6 GB card and the third follows. The second takes its own lock dir; the free-memory gate
# admits it after two minutes, since its derived need (vram_gb 22) is the cap, not the 11.7 GB used.
# lib.sh first: it re-runs this arm from ONE snapshot and exports ARM_LIB, so the children below share it.
# A bare `run.sh` started from the repo would re-exec itself and drop its recipe argument, and two of them
# at once race on the same snapshot dir. The K=4 soup is c803's `after`, as in basinsoup_ff.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
run="$REPO/experiments/run.sh"
bash "$run" experiments/recipes/distillation/nartaa/nartaa_soup_c801.toml "$@" &
first=$!
GPU_LOCK_DIR="$HOME/.gpu.lock.pair" FOREIGN_WAITS=2 bash "$run" experiments/recipes/distillation/nartaa/nartaa_soup_c802.toml "$@" &
second=$!
wait "$first" || exit
wait "$second" || exit
exec bash "$run" experiments/recipes/distillation/nartaa/nartaa_soup_c803.toml "$@"
