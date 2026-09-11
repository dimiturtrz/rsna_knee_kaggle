set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="D:/models/rsna_knee_abnormality/experiments/v2delonly_span_s2026.full.log"

while true; do
  if grep -q "ARM DONE v2delonly_span" "$LOG" 2>/dev/null; then break; fi
  if grep -q "ARM FAILED" "$LOG" 2>/dev/null; then break; fi
  sleep 30
done

bash "$HERE/arms/v8named_span.sh"
