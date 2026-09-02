#!/usr/bin/env bash
# Occlusion diagnostics on a trained member: what is the model actually reading?
#
# Not a training arm - it spends no gradient and produces three measured deltas rather than a picture.
# See kneemri/attribution.py for why saliency is deliberately absent.
#
# 2026-08-11, on base + ranking (0.8283):
#   slots   AX_FLUID_FS +0.0193, the other five +0.0042..+0.0094 (each inside the 0.006 noise floor)
#   region  whole 0.8448 | centre-only 0.8021 | border-only 0.8006
#   depth   best window at slice 7 = 0.8332, range 0.7976..0.8332 over ten windows
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
CHECKPOINT="${1:-$MODELS/rankbase24_s2026.pt}"
require "$LABELS" "$CHECKPOINT"

# THE LOG IS NAMED AFTER THE MEMBER IT READ. It was `attribution.full.log` whatever checkpoint was
# passed, so asking the same question of a second member silently replaced the first member's answer —
# and this arm produces ONLY a log, so that is the whole result gone. The three deltas at the top of
# this file were measured on rankbase24 and nothing in the file said so.
MEMBER="$(basename "$CHECKPOINT" .pt)"

# THE CARD IS TAKEN HERE, because this arm does not cross run_train and run_train is where every other
# GPU path takes it. lib.sh states that run_train is "the ONE point every path that spends GPU crosses";
# this arm is the counterexample, and it has been one since it was written on 2026-08-15. It spends no
# gradient, but 400 studies of occlusion is 400 forward passes over the same encoder a trainer holds
# 15 GB for, so a chain starting it under a running arm is the 31.9-GB PCIe-paging collapse the lock
# exists to prevent — the serialiser simply could not see it. Queue order made this safe so far, not
# design. Held around the reading only, and released before the exit checks, so a chain behind us
# starts the moment the GPU work ends rather than when the shell does.
hold_card
uv run --no-sync --project "$REPO" python -u -m kneemri.evaluation.attribution \
  --checkpoint "$CHECKPOINT" --labels "$LABELS" --studies 400 2>&1 | tee "$LOGS/attribution.${MEMBER}.full.log"
status="${PIPESTATUS[0]}"
release_card
[ "$status" -eq 0 ] || { echo "ARM FAILED attribution (exit $status)"; exit "$status"; }
echo "ARM DONE attribution"
