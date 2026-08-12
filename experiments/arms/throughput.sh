#!/usr/bin/env bash
# What does torch.compile actually buy, in images per second? One epoch each, nothing else varied.
#
# WHY THIS IS A MEASUREMENT AND NOT AN ARM. It changes no recipe and produces no member worth keeping;
# the only output that matters is the heartbeat's img/s. Speed is close to priority zero here — a
# whole-stack epoch is ~485 s at 16 slices, so a 25% cut is two hours a day of extra experiments — and
# `--compile-encoder` has been available and unmeasured for a day.
#
# ONE EPOCH IS ENOUGH AND TWO WOULD BE MISLEADING. Compilation cost is paid ONCE per process, so an
# epoch-1 comparison charges the whole graph build to the compiled arm and understates it; the number
# to read is the STEADY-STATE img/s from the heartbeat's later steps, not the epoch's wall clock.
# That is also why the epoch count is 1: everything after the first epoch is paying for a graph that
# is already built.
#
# WHAT WOULD MAKE IT NOT WORTH IT: our batches are RAGGED. The last batch of an epoch is partial and
# the slot count varies per study, so a static graph would recompile constantly — which is why the
# flag compiles the encoder with dynamic shapes and compiles the ENCODER only, since training calls
# supervised() rather than forward(). If dynamic compilation still recompiles, img/s will be WORSE
# than the control and the recompile churn will show as a stall in the heartbeat rather than a crash.
#
# control  the same command without --compile-encoder, run first in this same script.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

BASE=(--fold 0 --epochs 1 --one-cycle --weight-floor --rank-weight 1.0
      --slice-pooling max --whole-stack --recompute-activations --vram-gb 22
      --labels "$LABELS")

run_train throughput_plain "${BASE[@]}" --out "$PROCESSED/throughput_plain.pt"
run_train throughput_compiled "${BASE[@]}" --compile-encoder --out "$PROCESSED/throughput_compiled.pt"

echo "=== steady-state img/s, later steps only (the first steps pay for the graph) ==="
for name in throughput_plain throughput_compiled; do
  printf "%-22s " "$name"
  grep -oE "[0-9]+ img/s" "$LOGS/$name.full.log" | tail -5 | tr '\n' ' '
  echo
done
