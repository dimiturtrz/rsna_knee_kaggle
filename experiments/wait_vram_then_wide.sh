#!/usr/bin/env bash
# Hold basinsoup_wide until the shared card has room for it.
#
# The peer biohub_kaggle job holds this card by arrangement and is not ours to kill or to wait out with
# a fixed sleep. Launching --vram-gb 22 into a full card does not OOM on Windows WDDM; the allocator
# spills to host memory and the arm runs at ~1/10 throughput while reporting 100% utilisation, which
# reads as a compute cost rather than the memory fault it is. So gate on FREE memory, not on time.
#
# The gate is the peer's ABSENCE, not our own footprint. 13000 MiB was the measured reservation
# (basinsoup_soft: 11618 MiB, and arm_check calls --vram-gb 22 OVERSTATED by 10.65 GB, knee-ylp0) plus
# backfill headroom -- and it fired at 02:56Z while the peer still held 14.6 GB. Our arm then reserved
# 17.84 GB into a card with 14.6 GB already spoken for, so WDDM spilled: epoch 0 took 1789 s against
# round 1's ~255 s on the identical recipe, a 7x slowdown reported as 100% utilisation. Sizing the gate
# to what WE need is the error; two jobs that each fit still do not both fit.
NEED_MIB=20000
REPO="$(cd "$(dirname "$0")/.." && pwd)"

for i in $(seq 1 720); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ -n "$FREE" ] && [ "$FREE" -ge "$NEED_MIB" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) free ${FREE} MiB >= ${NEED_MIB}, launching basinsoup_wide"
    exec bash "$REPO/experiments/arms/basinsoup_wide.sh"
  fi
  [ $((i % 10)) -eq 1 ] && echo "$(date -u +%H:%MZ) free ${FREE} MiB, waiting for ${NEED_MIB}"
  sleep 60
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) gave up after 12 h; card never freed"
exit 1
