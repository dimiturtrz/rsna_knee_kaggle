#!/usr/bin/env bash
# finding_calib_dense.sh — the pooled level map's own harness suspect, run as an arm.
#
# THE THING THIS TESTS. `--calibrated-targets` replaces each of the reader's five levels with the rate
# the radiologist was measured to agree at. That map is POOLED over all twelve findings: it says a 0.68
# on Contusion and a 0.68 on Synovitis are worth the same 0.378. `ReaderAgreement.by_label` says they
# are not. Over the reader's positive calls the radiologist agrees 44.4% of the time on Contusion and
# 81.2% on Synovitis — 37 points across a pooled 66.7% — and the two fail in OPPOSITE directions
# (Contusion over-called 20 times against 2 misses; Synovitis under-called 14 against 3 over-calls).
# `--finding-calibration` adds ONE empirical-Bayes-shrunk logit offset per finding on top of that map.
# An offset of zero reproduces today's table exactly, so this is strictly an addition to a calibration
# that already works, and the null is a no-op rather than a different member.
#
# WHY IT IS WORTH A CARD HOUR. `noise_binding` (2026-08-27, zero slots) measured what the members
# actually did with the reader's confident errors: over-called tracking +0.683 / +0.700 on the two
# uncalibrated corners, +0.538 / +0.550 with `--calibrated-targets`. The label lane is LIVE — the fit
# did NOT average the reader's invented positives away — and calibration is the one axis demonstrably
# decoupling members from them, at -0.145 (span) and -0.150 (dense), the same size at two independent
# corners. It removes about a fifth of a binding that starts near 0.7, which is where the remaining
# headroom is. This arm asks whether un-pooling the map takes another bite out of the same binding.
#
# control  calib_dense_hold (0.8634 proxy, tracking +0.550). SINGLE VARIABLE against it:
#          --finding-calibration and nothing else. Same corner, same recipe, same hold.
# cost     one 12-epoch fit, ~40 min. Zero submission slots — the control already exists.
#
# HOW IT IS SCORED, AND WHY NOT ON THE HOLDOUT. The offline holdout is scored against the REPORT
# labels, which carry the very defect these offsets correct, so the proxy is BIASED AGAINST any table
# that stops agreeing with the reader — the knee-bs4l sign flip (+0.0072 proxy against an unheld
# -0.0020) is that bias already measured once on this same axis. PRIMARY reading is therefore
# `kneemri.evaluation.annotated_readings` and `kneemri.evaluation.noise_binding` against
# calib_dense_hold's +0.550; the holdout mean is a FLOOR, not the verdict.
#
# PRE-REGISTERED LADDER, on tracking against calib_dense_hold's +0.550:
#   <= +0.475   the per-finding spread is real and composes — un-pooling is a lever, refit unheld and seat it
#   +0.475..+0.625  FLAT. The pooled map was already carrying what the spread had to give at 684 cells
#   >= +0.625   WORSE. The offsets are fitting sampling noise the shrinkage failed to remove; the tau
#               estimate is the suspect, and the arm retires per-finding calibration at this corner
# The band is +-0.075, roughly the CI half-width `noise_binding` reports on 58 studies. A ratio on that
# denominator can tell 1 from 0 and cannot resolve 0.6 from 0.7 — anything inside the band is FLAT.
#
# THE HARNESS SUSPECT, NAMED IN ADVANCE. The offsets are fitted on the SAME 684 cells the annotated
# read scores against, so a positive annotated delta is a consistency check between two rulers sharing
# an origin, never an independent confirmation. `--hold-annotations` keeps those cells out of the
# training TABLE, but the offsets are hardcoded CONSTANTS carrying them, which is the same walk-through
# calib_dense_hold's header bounded for `MEASURED_AGREEMENT` — twelve scalars over ~57 cells each are a
# population estimate and cannot encode which study has an effusion. `noise_binding` is the reading
# least exposed to it: its poles are cells where reader and radiologist AGREE, which no offset moves.
#
# bead: knee-ep74
#
# NO --vram-gb: DeviceBudget derives the cap from --slices — 20 slices reserve 28.26 GB.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train finding_calib_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --finding-calibration \
  --hold-annotations \
  --labels "$LABELS" --out "$MODELS/finding_calib_dense_s2026.pt"
