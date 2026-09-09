#!/usr/bin/env bash
# foreign_drop_pilot.sh — validate the shifted path before spending the card on eleven members.
#
# reading: logs/foreign_drop_pilot.log
# deliberate: two members through foreign_readout, to check slot_cost reads ~0.0000
# bead: knee-lm37v
#
# WHY A PILOT AT ALL. `foreign_readout`'s middle number is a plumbing self-check, not a result: with the
# full splice `kept` serves OUR pixels through the SHIFTED code path, so `slot_cost = kept - native` must
# read ~0.0000. Anything else means the shifted slab is broken and `decode_cost` is not about pixels at
# all — the one failure that is otherwise invisible, since a broken slab still yields a plausible score.
# The tool has never been run on this pool, so that assertion has never been observed to hold here.
#
# Two members, one from each side of the question: `aug_long` (an augmentation member, 0.8511) and
# `band_dense_hold` (a non-augmentation member above it, 0.8573). Costs a few minutes of card.
#
# PRE-REGISTERED: if slot_cost is outside +/-0.005 on EITHER member, the full run does not start and the
# defect is the finding. If it reads ~0, the eleven-member arm runs and `decode_cost` is a pixel number.
#
# `arm_check` CANNOT CHECK THIS ARM, and that is structural rather than an oversight. It works by
# dry-running an arm and capturing the invocations `run_train`/`run_pretrain` print; this one calls
# `python -m` directly and prints none, so `--only foreign_drop_pilot` exits on "no arm printed an
# invocation named ...". The same blind spot the `foreign_run` arms sit in, reached from the other
# direction. The flags were verified against the tool's own argparse instead
# (kneemri/evaluation/foreign_readout.py:170-177), and `bash -n` covers the syntax.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

# THE INTERPRETER IS PINNED, not resolved. The first run of this arm died on `ModuleNotFoundError: No
# module named 'jaxtyping'` — and NOT because an environment was missing it: `.venv-arm` imports it fine
# and so does `.venv`. Bare `python` on PATH is a third interpreter that is neither, and an arm sourcing
# lib.sh only sets UV_PROJECT_ENVIRONMENT, which steers `uv run` and says nothing about a bare `python`.
# Naming the interpreter is the only form of this that cannot be wrong somewhere else.
# THROUGH `uv`: `arm_check` RUNS every arm file to read its flags, and the only thing making that a
# lint is a stub shadowing `uv` alone — a direct interpreter walks past it and the lint becomes a card
# job (measured 2026-09-09). Verified the module imports under `.venv-arm`, which is what
# `experiments/lib.sh` points `uv` at, before the switch.
UV_NO_SYNC=1 uv run python -m kneemri.evaluation.foreign_readout \
  --checkpoints "$MODELS/aug_long_s2026.pt" "$MODELS/band_dense_hold_s2026.pt" \
  --labels "$LABELS" --foreign D:/kaggle/input/raptor-corpus --device cuda
