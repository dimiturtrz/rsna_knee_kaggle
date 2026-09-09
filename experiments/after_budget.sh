#!/usr/bin/env bash
# after_budget.sh <windows-pid-of-the-perstudy-read>
#
# Gates the CARVE budget sweep behind the running `fov_perstudy_read`, then runs the padding split that
# costs no card while the sweep has the card.
#
# THE GATE IS AN OR, and that is not belt-and-braces. The artefact alone hangs forever on a silent exit-0
# death -- which is exactly how this read failed once already today, writing PERSTUDY DONE with no `.npz`
# on disk. The pid alone fires the next arm the instant a crash frees the card, on a reading nobody has.
# `ps -W` because the MSYS pid is a different number from the Windows one and `kill -0` reads the wrong
# process table.
source "$(dirname "$0")/lib.sh"
PID="$1"

until [ -f "$MODELS/fov_perstudy.npz" ] || ! ps -W | grep -q "[^0-9]$PID[^0-9]"; do sleep 20; done
echo "=== PERSTUDY GATE OPEN ==="

if [ -f "$MODELS/fov_perstudy.npz" ]; then
  UV_NO_SYNC=1 uv run python -m kneemri.evaluation.padding_split \
    --per-study "$MODELS/fov_perstudy.npz" \
    --metadata D:/data/volumetric/mri/processed/rsna_knee_abnormality/metadata/canonical_metadata.csv
  echo "=== PADDING SPLIT DONE ==="
else
  echo "=== PERSTUDY DIED WITHOUT ITS ARTEFACT - padding split not run ==="
fi

bash "$(dirname "$0")/arms/carve_budget_sweep.sh"
