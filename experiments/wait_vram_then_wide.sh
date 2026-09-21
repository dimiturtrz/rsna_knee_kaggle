#!/usr/bin/env bash
# Hold basinsoup_wide until the shared card has room for it.
#
# The peer biohub_kaggle job holds this card by arrangement and is not ours to kill or to wait out with
# a fixed sleep. Launching --vram-gb 22 into a full card does not OOM on Windows WDDM; the allocator
# spills to host memory and the arm runs at ~1/10 throughput while reporting 100% utilisation, which
# reads as a compute cost rather than the memory fault it is. So gate on FREE memory, not on time.
#
# NOT --vram-gb 22. That flag is a ceiling, not the reservation: arm_check reports the recipe
# OVERSTATED by 10.65 GB against what its own --slices reserve (knee-ylp0), and basinsoup_soft measured
# 11618 MiB actually reserved at the identical shape. Gate on the true figure plus backfill headroom, so
# the arm starts as soon as the peer drops to ~19 GB instead of waiting for a nearly empty card.
NEED_MIB=13000
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
