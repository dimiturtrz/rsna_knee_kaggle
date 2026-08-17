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
# control  the same command under --eager-encoder, run first in this same script. THE CONTROL IS THE
#          FLAGGED ONE NOW: compiling became the default on 2026-08-17, so the arm that measures what
#          compiling is worth is the one that has to ask for eager.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

BASE=($RECIPE --epochs 1
      --slice-pooling max --whole-stack --recompute-activations --vram-gb 22
      --labels "$LABELS")

run_train throughput_plain "${BASE[@]}" --eager-encoder --out "$PROCESSED/throughput_plain.pt"
run_train throughput_compiled "${BASE[@]}" --out "$PROCESSED/throughput_compiled.pt"
# bf16 rides along because the expensive part - a whole-stack epoch on an idle card - is already being
# paid twice, and a third run answers a question that would otherwise need its own hour. THE EXPECTED
# ANSWER IS 'no faster': bf16 and fp16 have the same matmul throughput on Blackwell, so this is here to
# CONFIRM that, and to show the epoch-0 holdout is not moved by the precision change. What bf16 removes
# is the loss scaler, and with it scale collapse - insurance, not speed.
run_train throughput_bf16 "${BASE[@]}" --out "$PROCESSED/throughput_bf16.pt"

# THE CHECKPOINTING TAX, which we have paid on every arm since 2026-08-11 and never priced. The
# docstring claims 'about 30% compute' from the literature; that is a guess about our workload, and 30%
# of a 320 s epoch is 96 s per epoch on every arm we will ever run. It cannot be measured at batch 12,
# because that is exactly the batch that does not fit without checkpointing - so BOTH sides run at
# batch 6, where both fit. The pair is comparable to each other, not to the rows above: a smaller batch
# changes the ranking term's within-batch pairs, and img/s is per IMAGE, so the tax reads clean.
HALF=($RECIPE --epochs 1
      --slice-pooling max --whole-stack --batch 6 --vram-gb 28 --labels "$LABELS")
run_train throughput_ckpt_on "${HALF[@]}" --recompute-activations --out "$PROCESSED/throughput_ckpt_on.pt"
run_train throughput_ckpt_off "${HALF[@]}" --out "$PROCESSED/throughput_ckpt_off.pt"

# HOW MUCH OF THE BACKWARD IS THE ENCODER'S TOP SIX BLOCKS. We unfreeze six of twelve by default and
# have never asked what four would cost - and our own numbers say capacity is NOT the constraint here
# (base over small was +0.0011 against a 0.0053 floor), which is precisely the condition under which
# opening fewer blocks should be close to free. Speed compounds across every future arm, so a 15%
# saving here is worth more than most single-arm results. ONE EPOCH prices the throughput; whether the
# SCORE holds needs a full arm, and that is a separate decision this measurement informs.
run_train throughput_unfreeze4 "${BASE[@]}" --unfreeze-last 4 --out "$PROCESSED/throughput_unfreeze4.pt"

echo "=== epoch-0 holdout, which precision must not move ==="
for name in throughput_plain throughput_compiled throughput_bf16 throughput_ckpt_on throughput_ckpt_off throughput_unfreeze4; do
  printf "%-22s " "$name"; grep -oE "holdout mean AUC [0-9.]+" "$LOGS/$name.full.log" | tail -1
done

echo "=== steady-state img/s, later steps only (the first steps pay for the graph) ==="
for name in throughput_plain throughput_compiled throughput_bf16 throughput_ckpt_on throughput_ckpt_off throughput_unfreeze4; do
  printf "%-22s " "$name"
  grep -oE "[0-9]+ img/s" "$LOGS/$name.full.log" | tail -5 | tr '\n' ' '
  echo
done
