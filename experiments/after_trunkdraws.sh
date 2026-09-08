#!/usr/bin/env bash
# Wait for `deep32_batch6` to release the card, then run the two trunk-axis draws the re-bucket licenses.
#
# THIS FILE REPLACES `after_instruments.sh`, WHICH RAN `resnet50d_lr` THEN `vjepa_cold`. D128 §2 takes
# `vjepa_cold` off the queue and it is not a hold this time — it is the ratio. The `options`-read
# re-bucket prices every family against the expected max of the same number of draws from the default
# family's own 154-arm distribution, and the volumetric family reads **n=2, best 0.8693, E[max|null]
# 0.8834, −0.0141, p=0.055** — the worse half of the non-default rows. 3.2 h buys a fourth draw of a
# family reading below its own null. The same hours buy a draw of the only family reading above it.
# The unfair-draw argument (D118 §3) is not refuted and the checkpoint-recomputable comparison still
# stands; what changed is that something better than it now exists for the same card-hour.
#
# WHY THESE TWO, AND WHY IN THIS ORDER.
#   `resnet50d_lr`  17 min, n=1 at −0.0006 — AT its expectation, which is the status that licensed it,
#                   and it names `--variant resnet50d` explicitly so the re-bucket does not touch its
#                   own number. Cheapest verdict on the queue; it goes first so the expensive arm never
#                   holds a cheap read hostage.
#   `base_lr2e5`    ~4.5 h, the fourth draw of `base` — the ONE family above E[max|null] (n=3, 0.8886
#                   against 0.8867, +0.0019, p~0.60). Unrefuted is not supported; this buys the draw.
#
# D128 §2 ASKED FOR TWO BASE ARMS AT "~1.4 h each". That price is not this family's:
# `experiments/arms/base_clean.sh:59` prices the line at **~4.5 h of card**, roughly four times small's
# arithmetic at 336 px / 20 slices, batch 6 accumulate 2. Two draws are 9 h — a third of the UTC day —
# so ONE is queued here and the second waits on this one's read. Raised in the same wake.
#
# D128 §4 GATES THE BASE ARMS ON A CARD-FREE SEAT PRICE of `base_clean_s2026` as a 13th seat on
# `honest12`, against same-trunk controls of matched solo. That commission decides whether base is a
# SCORE lever or an instrument — it does not decide whether the draw runs, since base is the only
# family above its null in either branch. If the seat read lands negative before the gate clears, this
# file is REWRITTEN, never edited live.
#
# WHY NO SMOKE. Both arms reuse a reading that ran to completion — `resnet50d_short` at 224 px / 20
# slices, `base_clean` at 336 px / 20 slices in the 0.1-0.9 band — so the caches are on disk and a
# smoke would pay for them twice.
#
# WHY THIS GATE PATH. A `run_train` arm re-runs from a snapshot whose `tee` writes
# `$MODELS/experiments/<checkpoint>.full.log`; `logs/` holds the `foreign_run` arms, a different
# launcher. The FILE carries the checkpoint spelling (`deep32_batch6_s2026`) and the MARKER TEXT
# carries the `run_train` name (`deep32_batch6`) -- two different variables, and this greps the marker
# as an existing `ARM DONE` line spells it.
set -u

cd /d/personal_projects/rsna_knee_kaggle || exit 1

MODELS="D:/models/rsna_knee_abnormality"
GATE="$MODELS/experiments/deep32_batch6_s2026.full.log"
LOG="logs/after_trunkdraws.log"

echo "waiting for 'ARM DONE deep32_batch6' in $GATE" >> "$LOG"
until grep -q "ARM DONE deep32_batch6" "$GATE" 2>/dev/null; do sleep 30; done
echo "gate cleared at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

bash experiments/arms/resnet50d_lr.sh >> "$LOG" 2>&1
echo "resnet50d_lr finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

# base_lr2e5 IS HELD (D129) AND DELIBERATELY NOT CHAINED HERE. On the NATIVE fold-0 909 ruler — the one
# every 12-epoch arm was fitted and selected against — `base` reads n=4, best 0.8570, E[max|null] 0.8605,
# **-0.0036 BELOW** its null (p=0.274), where the 896/`L75mid` frame read +0.0019 above it. Same family,
# two rulers, opposite signs. `small`'s best is the same strength on both frames (0.8964 / 0.8959) while
# `base`'s moves +0.0316, so the disagreement is not n and not member strength: either the 896 rows are
# different members than the native ones, or that frame genuinely favours base — which would be a bigger
# finding than the arms. Neither is worth 4.5 h before it is named. `experiments/arms/base_lr2e5.sh`
# stays on disk, priced and pre-registered, and gets chained the moment the reconciliation lands.
#
# ALSO CLOSED CARD-FREE WHILE THIS WAS BEING DECIDED (D129 §4): "a bigger trunk wants a longer schedule"
# is false. Peak epoch over 12-epoch arms is base n=4 median e7, small n=141 median e7, and base peaks
# frac(>=e9) 0.20 against small's 0.34 — base saturates NO LATER than small. Pool-wide, 256 runs, 196
# converged and 14 undertrained, of which thirteen asked for 4 or 8 epochs. The epoch-budget lever is
# not a lever for either trunk.
