#!/usr/bin/env bash
# after_resnetshort.sh — smoke `resnet_224_short`, then run it. Queued 2026-09-09 19:25Z, card idle since
# ~17:35Z when `ortho_lora` finished at 0.8349 (BELOW branch: the adaptation lever closes on every trunk
# we own, and its own text sends the search back to the data and architecture axes).
#
# WHY THIS ARM AND NOT ANOTHER. It is the only unrun arm that changes the ARCHITECTURE FAMILY. Twelve of
# thirteen registry encoders are transformers; the one convolutional row is RadImageNet weights, not
# ImageNet. Every seat we own is a ViT, and the seven board flats were all reseatings inside that family.
# The smoke's verdict is the arm's own line, never arm_check's exit code — that reads repo-wide findings.
source "$(dirname "$0")/lib.sh"

SMOKE_LOG="$MODELS/smoke_resnet_224_short.check.log"
UV_NO_SYNC=1 uv run python -m kneemri.experiments.arm_check \
  --only resnet_224_short --smoke "$MODELS/smoke_resnet_224_short" --smoke-studies 60 2>&1 | tee "$SMOKE_LOG"
if grep -qaE "^(BROKEN|NO ROOM) resnet_224_short" "$SMOKE_LOG" || ! grep -qa "^ok resnet_224_short" "$SMOKE_LOG"; then
  echo "=== SMOKE FAILED - resnet_224_short NOT launched ==="
  exit 1
fi
echo "=== SMOKE PASSED ==="
bash "$(dirname "$0")/arms/resnet_224_short.sh"
