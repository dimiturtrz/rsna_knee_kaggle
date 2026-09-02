#!/usr/bin/env bash
# Waits for the running card arm to exit, then encodes the pixel cache through OrthoDiffusion's
# three plane denoisers. The wait is on the WINDOWS pid card_check reported, tested with tasklist:
# an MSYS pid tool reads a Windows python as dead and would start a second job on the card.
set -u
cd "$(dirname "$0")/../.."
until ! tasklist //FI "PID eq ${1}" 2>/dev/null | grep -q "${1}"; do sleep 60; done
echo "ARM DONE through_dense_ramp — starting ortho encode $(date -u +%FT%TZ)"
timeout 21600 .venv/Scripts/python.exe -m kneemri.embeddings.build_ortho_embeddings --batch 4
echo "ARM DONE ortho_encode $(date -u +%FT%TZ)"
