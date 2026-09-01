#!/usr/bin/env bash
# bead: knee-9707
# focus_mcl.sh — is MCL's ceiling the price of JOINTNESS, or something the encoder never sees?
#
# IT RAN, AND IT LANDED ITS OWN INFORMATIVE NULL. This header said CANCELLED BEFORE IT RAN until
# 2026-09-01; `focus_mcl_s2026.metrics.jsonl` holds twelve epochs. MCL peaks at 0.7861 (e7) against the
# control's 0.7931 — inside the pre-registered 0.783-0.802 band, JOINTNESS COSTS MCL NOTHING, at maximum
# tilt. The ledger had recorded it as refuted at a solo of 0.5332, which is the MEAN this file warns
# below must not be read. Do NOT re-run it. Its follow-up (mm/px on the medial margin) is answered too,
# by crop80 and band140 already on disk, and in the other direction. See
# `interpretations/kneemri/2026-09-01_mcl_falls_to_every_axis_we_own.md`.
#
# THE PRICE ARGUMENT BELOW STANDS AND IS NOW MOOT, ON MECHANISM AS WELL. Read
# `interpretations/kneemri/2026-08-23_the_per_label_axis_is_worth_0.004.md` before running this.
# `kneemri/evaluation/weights_gap.py` compared our weights against the published members' on the SAME
# 909 studies with the same labels: ours wins ten of twelve labels and the mean (+0.0198 against their
# single member, and it beats their four-seed pool too). MCL and Fracture are the only losses. Closing
# BOTH deficits completely, to weights that provably exist, is (0.0346 + 0.0167) / 12 = +0.0043 on the
# mean — a quarter of the bar, and that is the CEILING of the whole weak-column programme rather than an
# estimate of what this arm gets. MCL alone is +0.0029. This arm costs 3.1 h of card for that.
# The mechanism below is sound and still untested; the prize is what fails, whatever the answer.
# The file stays so a future reason re-tests it instead of assuming it dead.
#
# WHY MCL AND NOT LATERAL MENISCUS. `focus_latmen.sh` aimed this same instrument at Lateral Meniscus
# because published SPECIALISTS score it 0.20 above us. That is the wrong denominator: the metric is the
# unweighted mean of twelve of OUR OWN columns, so what a label is worth is its distance from OUR mean,
# not from someone else's model. Against `band_dense_llrd`'s own mean of 0.8481 at e11, Lateral Meniscus
# is -0.0058 — the middle of our pack — and ACL is +0.0301, one of our best. The columns that are
# actually cheap are MCL -0.1059 (0.7422) and Lateral OA -0.0391 (0.8089). Lifting those two to ~0.85
# is +0.0124 of the mean: under the 1.5% bar alone, but the largest clean piece of it, and increments
# compose. MCL is 0.0090 of that 0.0124 — the biggest single column on the sheet.
#
# FRACTURE IS DELIBERATELY NOT IN THAT ARITHMETIC, and the ledger is why. Our fold-0 column reads 0.7830,
# which looks like the second-cheapest label — but `label_power` predicted from first principles that an
# AUC over ~60 positives carries a standard error near 0.04, and Fracture read 0.784 / 0.903 / 0.864
# across three folds of ONE recipe. On the corpus ruler it is 0.8923, our STRONGEST label. A 909-study
# fold cannot measure a 313-positive label, so its apparent deficit is sampling, not headroom. Targeting
# it is the mistake this ledger already made once (`+0.0173 from MCL and Fracture`, retired 2026-08-14).
#
# THE TWO RULERS DISAGREE ABOUT MCL, AND THE DISAGREEMENT IS THE POINT. The corpus ruler (4,407 studies,
# floor 0.0024) is built from the twenty published CC0 members' out-of-fold predictions — THEIR weights,
# not ours, as that document states plainly. On their weights MCL reads 0.8166 +/- 0.0155 and Lateral
# Meniscus 0.7981 is the weakest label. On OURS, MCL is the weakest by a wide margin (0.7422-0.79 across
# 83 runs, best ever 0.8042) and Lateral Meniscus is mid-pack. Both readings are correct about the model
# they measured. The one that governs our score is OURS, because what ships is five all-ours members —
# so `focus_latmen`, which was built off the corpus ruler's weakest label, was aimed at a deficit that
# belongs to somebody else's encoder, and is retired.
#
# AND THAT COMPARISON IS AN EXISTENCE PROOF, which is what makes this arm worth a slot. Another model
# fitted on this same corpus reaches 0.8166 on MCL, 0.04 above our typical and above our best-ever
# 0.8042. So MCL is NOT an intrinsic ceiling of the data or of the label — the signal is in there and
# somebody's weights extract it. The ceiling is ours, and something we do is what imposes it.
#
# MCL IS A FAMILY CEILING, NOT A KNOB — 83 recorded runs say so, at zero card cost:
#   crop band     (0.0,1.0) 0.7823 n=1 | (0.1,0.9) 0.7793 n=26 | (0.2,0.8) 0.7662 n=17  — 0.013 span
#   slices        12 -> 0.7643 n=34 | 16 -> 0.7745 n=2 | 20 -> 0.7777 n=21
#   resolution    336 -> 0.7695 n=54 | 448 -> 0.7711 n=3
#   encoder       dinov2-small 0.7732 n=48 | base 0.7812 | vjepa2l 0.7803 | dinov3-small 0.7257
#   slot scheme   extended (9 slots, +83,297 slices of imaging) 0.7604 vs canonical 0.7931 — WORSE
#   best ever     0.8042 (band_wide_s2026)
# Nothing in the recipe axis moves it. And it is not rarity or label noise either: MCL's prevalence at
# grade >= 0.5 is 0.174 with label confidence 0.789 (HIGH), while Synovitis — confidence 0.189,
# prevalence 0.132 — reads 0.8719. The label is clean and the model still cannot fit it.
#
# WHAT IS LEFT, AND WHY THIS ARM. After recipe, data volume, resolution and encoder family are all
# refuted on measurement, the untested axis is the OBJECTIVE: eleven other findings competing for one
# encoder's representation. `--focus-label` zeroes the other eleven loss weights, which drops them out
# of the cross-entropy, the ranking term and BOTH normalisers, so MCL descends at its natural scale with
# the whole encoder behind it. Maximum tilt is deliberate: if the extreme does not move a ceiling that
# nothing else moved, no partial reweight will, and the null is worth having.
#
# THE CONFOUND, STATED BEFORE THE NUMBER. A focused run sees ONE supervisory signal per study where the
# control sees twelve, so it is data-starved relative to its control — `half_data` measured data volume
# as the binding constraint here (-0.0234 for half the studies, 3.8x the noise floor). A WIN is therefore
# strong, because it won DESPITE the starvation; a LOSS is ambiguous between "jointness costs nothing"
# and "one column is too little supervision", and its follow-up is an UPWEIGHT rather than a mask.
#
# THE CONTROL IS `band_dense`, ALREADY FITTED, and this file is its invocation with `--focus-label`
# added and nothing else changed — same recipe, same 20 slices, same band, same default dinov2-small.
# Read the MCL COLUMN, not the mean: eleven of twelve heads carry no gradient, so the mean is noise by
# construction. `band_dense_s2026` reads MCL 0.7931 at its best epoch (mean 0.8578, epoch 8).
#
# PRE-REGISTERED, BEFORE THE NUMBER. All bands are on the MCL column against 0.7931.
#   >= 0.843   (+0.05) — CONTENTION IS THE MECHANISM, and it breaks the 0.8042 all-runs ceiling as well.
#              Licenses a specialist member per weak column (MCL, Fracture, Lateral OA), assembled under
#              the twelve-column contract the head already emits. That is the +0.018 path.
#   0.803-0.842 (+0.01 to +0.05) — clears the recorded ceiling but under the bar as a finding.
#              Directional: says specialisation is real and knob-sized. Keep the column if a seat is
#              free; do not plan the roster around it.
#   0.783-0.802 (within +/-0.01) — JOINTNESS COSTS MCL NOTHING. The informative null: it retires the
#              objective as an explanation and points the whole MCL deficit at what the INPUT carries —
#              a superficial medial-collateral ligament that 20 slices of a fluid-sensitive stack at
#              336 px may simply not resolve. Next axis is acquisition/representation, not loss — and
#              `band140` already named the specific form: MCL is the most SAMPLING-DENSITY-sensitive
#              label on the sheet, -0.016/-0.021/-0.017 at e6/e7/e8 when 140 mm of span was pushed
#              through the same pixel grid, the largest margins in the twelve-label table while the
#              thick labels held flat. Combined with the existence proof above, the null here points
#              at mm/px on the medial margin, which is a CROP-AND-RESOLUTION arm, not a loss arm.
#   <= 0.782   AMBIGUOUS ON THE STARVATION CONFOUND, not a refutation. Follow up with an upweight (the
#              same mask at a weight rather than at zero), which keeps all twelve signals.
#
# NOT A CANDIDATE MEMBER IN ITS OWN RIGHT, whatever it scores. Eleven of its columns are untrained, so
# it cannot be averaged into a roster as-is; it enters a kit only as one column of a per-finding
# assembly, and only if the top band lands.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train focus_mcl \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --focus-label "MCL" \
  --labels "$LABELS" --out "$PROCESSED/focus_mcl_s2026.pt"
