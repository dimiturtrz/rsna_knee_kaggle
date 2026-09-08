#!/usr/bin/env bash
# C191 s2's gate arm, behind the arm that is currently on the card.
#
# WHY. The v3 label table measured +0.0106 without calibration and -0.0007 with it, on ONE pair each.
# `recal_through` is the faithful one-token replication at a second geometry: `--labels` moves to
# report_labels_v3.csv, `--eval-labels` is PINNED to v2 so the ruler cannot move with the arm (that pin
# is the `recal_holdannot` incident), and every other flag is `through_dense`'s. Control 0.8867 at
# ledger75. Pre-registered: >= +0.005 -> the axis is real and the remaining seats follow; < +0.002 ->
# the +0.0106 is one pair's property and the axis closes with a number; between buys ONE more arm.
#
# THE MARKER IS SPELLED WITHOUT THE SEED and the LOG IS SPELLED WITH IT — `_mark` writes the run_train
# name, `_arm_log` names the file for the checkpoint basename, and that file lives in
# $MODELS/experiments/, not under logs/.
set -u
cd "$(dirname "$0")/.."

MARKER="ARM DONE calib_dense"
ARM_LOG="D:/models/rsna_knee_abnormality/experiments/calib_dense_s2026.full.log"

until grep -q "$MARKER" "$ARM_LOG" 2>/dev/null; do sleep 60; done
echo "$MARKER seen — starting recal_through $(date -u +%FT%TZ)"

bash experiments/arms/recal_through.sh
echo "QUEUE recal_through exited $? $(date -u +%FT%TZ)"
