#!/usr/bin/env bash
# bead: knee-il96
# normcount224.sh — the LAST unmeasured recipe key, run to close the knob space rather than to win.
#
# PRICE: ~19 min of card at 224 px (8124 MiB), zero slots. P(clears the 0.0062 floor) ~0.10.
#
# WHY A KEY WITH A ~0.10 PRIOR GETS CARD TIME. knee-s9cr's survey walked all 61 keys of a recorded
# recipe stamp against `experiments/README.md` and against the 110 files in `experiments/arms/`, and
# found exactly ONE key that is neither in the ledger nor behind an arm: `--normalise-by-count`, which
# takes the loss `.mean()` instead of dividing by the weight sum. Running it converts "the knob space is
# enumerated" from a SURVEY claim into a MEASURED one — and that claim is load-bearing right now, because
# it is why knee-s9cr redirected away from a 224 px knob sweep and why the schedule points at members.
#
# THE PRIOR IS WEAK AND STATED BEFORE THE NUMBER. The nearest relative, `--per-label-loss`, is refuted at
# −0.0043 with its falsifiable clause resolving the wrong way. Expected score movement ~0. This is an
# INSTRUMENT; it must not be sold as an improvement if it happens to land up.
#
# THE SINGLE VARIABLE. `native224`'s recipe verbatim — 12 slices, band 0.2–0.8, --image-size 224, 12
# epochs, backbone-lr 4e-5, max pooling, whole-stack, fold 0 — plus --normalise-by-count.
#
# control  native224 = 0.8502, best epoch 7, convergence_check clean. On the ledger, so this costs ONE run.
#
# THE FALSIFIABLE PART, against the 0.0062 three-seed paired floor:
#   * >= 0.8564 — the weight-sum denominator was costing us, and the weighted-loss family reopens.
#   * 0.8440–0.8564 — within the floor. The knob space is CLOSED, measured rather than surveyed.
#   * <= 0.8440 — the denominator is load-bearing, so the UNK down-weighting it interacts with
#     (--weight-floor, 0.2875 on a UNK cell) is doing real work, which reads with confident224's answer.
#
# HARNESS CHECK BEFORE READING ANY OF THAT. The printed recipe must say "normalise_by_count": "True" and
# "image_size": "224", and the log must load 4,406 studies and hold out 909.
#
# deliberate: --normalise-by-count (the arm's thesis — the single key it exists to measure)
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

run_train normcount224 \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 --normalise-by-count \
  --labels "$LABELS" --out "$PROCESSED/normcount224_s2026.pt"
