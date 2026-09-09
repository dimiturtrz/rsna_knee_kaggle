#!/usr/bin/env bash
# foreign_drop_pool.sh — every member's generalisation gap under a second decode of the same studies.
#
# reading: logs/foreign_drop_pool.log
# deliberate: eleven members through foreign_readout, to see whether the drop ranks them differently
# bead: knee-lm37v
#
# WHY. Every offline number this project owns is read on 909 studies through OUR decode under OUR labels,
# which is an IN-DISTRIBUTION redundancy measure. Sixteen candidates spanning every axis we own have
# priced negative on it while both seats that ever moved the board were out-of-distribution members.
# `raptor-corpus` holds all 4,407 of our studies at (44, 336, 336) — a different pixel path over the SAME
# studies with the SAME labels — so a drop measured here moves the decode alone and costs no training.
#
# THE THREE DELIVERABLES, in order:
#   1. the drop per member;
#   2. whether the drop ranks members DIFFERENTLY from solo score. If the ordering is identical this
#      readout adds nothing and that is reported immediately, not buried;
#   3. whether the augmentation members drop LESS.
#
# PRE-REGISTERED: if the augmentation members do not drop less, augmentation is not the carrier of
# out-of-distribution robustness and no card is spent on that axis.
#
# The solo scores this is read against, measured 2026-09-09 off each member's own holdout npz (midpoint
# cut, n=909): stackfit .8631 · band_dense_hold .8573 · small_feat_fit .8571 · crop160_span .8564 ·
# mixup_dense .8552 · mixup224_strong .8516 · warm224_long .8512 · aug_long .8511 · native224 .8502 ·
# tissue224 .8491 · latent_register_dense .8385. Note the augmentation members are NOT the bottom of that
# list — crop160_span and mixup_dense sit ABOVE native224 and warm224_long — so deliverable 3 is an
# ordering question, not a losers-versus-winners one.
#
# `intensity392_ws_s2026` (0.8459) is DELIBERATELY ABSENT: its source cache `images_448px_12sl.npy` was
# deleted, so its native side is uncomputable and its failure would take the other ten down with it.
#
# `arm_check` CANNOT CHECK THIS ARM: it dry-runs an arm and captures the invocations
# `run_train`/`run_pretrain` print, and this one calls `python -m` directly and prints none, so
# `--only foreign_drop_pool` exits on "no arm printed an invocation named ...". The same blind spot
# the `foreign_run` arms sit in. Flags verified against the tool's argparse
# (kneemri/evaluation/foreign_readout.py:170-177); `bash -n` covers the syntax.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

# THE INTERPRETER IS PINNED, not resolved — bare `python` on PATH is neither `.venv` nor `.venv-arm`,
# and it killed this arm's first pilot on a missing `jaxtyping` that both venvs actually have.
"$REPO/.venv/Scripts/python.exe" -m kneemri.evaluation.foreign_readout \
  --checkpoints \
    "$MODELS/aug_long_s2026.pt" \
    "$MODELS/mixup_dense_s2026.pt" \
    "$MODELS/mixup224_strong_s2026.pt" \
    "$MODELS/tissue224_s2026.pt" \
    "$MODELS/crop160_span_s2026.pt" \
    "$MODELS/band_dense_hold_s2026.pt" \
    "$MODELS/stackfit_s2026.pt" \
    "$MODELS/small_feat_fit_s2026.pt" \
    "$MODELS/warm224_long_s2026.pt" \
    "$MODELS/native224_s2026.pt" \
    "$MODELS/latent_register_dense_s2026.pt" \
  --labels "$LABELS" --foreign D:/kaggle/input/raptor-corpus --device cuda
