#!/usr/bin/env bash
# band224_hold.sh — the 224 px control, so the cheap recipe can carry the label axis.
#
# bead: knee-ogzq1
#
# WHY THIS ARM EXISTS AND WHAT IT IS NOT. It is not a member and it is not a question. It is the CONTROL
# every subsequent 224 px arm will be read against, and it is armed because `band224` measured the thing
# that makes 224 px worth using: 142 seconds an epoch against 450 for the same recipe at 336, on this
# card, for 0.0049 of solo AUC (0.8529 against `band_dense`'s 0.8578) — under the 0.0062 holdout noise
# floor. A twelve-epoch arm drops from ninety minutes to twenty-eight.
#
# WHY IT CANNOT BE `base224_s2026`. That checkpoint exists and is NOT a `--hold-annotations` fit, so
# `annotated_readings` refuses it — and the gold 58 is the ruler the whole label axis now turns on, since
# our report holdout reads the same weights 0.039 lower than the radiologists do. A 224 arm scored on
# gold needs a 224 control scored on gold. `band224_s2026` itself is also not held, which is the other
# half of the reason this arm is not redundant with it.
#
# ONE VARIABLE against `band224_s2026`: `--hold-annotations`. Everything else is `band224`'s read.
#
# WHAT IT LICENSES. The supervision axis has one refuted point (`errors_dense`, the confident-joint mask
# at 5,360 cells) and two running (`relabel_hold`, `rank966_hold`), each costing ninety minutes because
# each is matched to `band_dense_hold` at 336. With this control on disk the DOSE curve — how gain scales
# with the number of corrected cells — is 28 minutes a point instead of 90, which is the difference
# between pricing the axis and sampling it.
#
# NO PRE-REGISTERED BRANCH, DELIBERATELY. A control is not a hypothesis. The number it produces is the
# baseline the next arms are read against, and there is no score it could return that would be a finding
# on its own. The one thing that would invalidate it is a gold read that lands far from `band224`'s
# proxy-to-gold relationship for no reason the recipe explains — if `band224_hold` reads gold ABOVE
# `band_dense_hold`'s 0.8956 despite being 0.0049 under it on the proxy, that is a harness question and
# not a resolution win, and it gets asked before any dose arm is queued behind it.
#
# COST. ~28 minutes for twelve epochs at 224 px and 20 slices.
#
# reading: band224_hold_annotated*.json
#
# deliberate: --hold-annotations (the arm's whole purpose — the gold ruler will not score an unheld fit)
# deliberate: --image-size 224 (band224's read exactly — the recipe this control serves)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train band224_hold \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --image-size 224 --hold-annotations \
  --labels "$LABELS" --out "$MODELS/band224_hold_s2026.pt"
