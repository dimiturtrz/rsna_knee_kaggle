#!/usr/bin/env bash
# bead: knee-nyqc
# disagreed_sampler.sh — does spending the epoch where the FAMILY SPLITS beat spending it uniformly?
#
# THE ONE PART OF THE RECIPE NO ARM HAS EVER MOVED. Every run in the ledger draws its epoch as a uniform
# permutation of the training positions: each study once, order reshuffled, nothing else. That is a
# policy and not a law, and after recipe, data volume, encoder family, objective, crop, span and
# resolution have all been measured, the draw is the largest untouched axis left in the loop.
#
# WHY THERE IS ANYTHING THERE. `residual_split` partitioned the holdout's errors: ~24.3% the members all
# get wrong (unanimous), ~5.4% a combiner could recover, and ~70% in the band that more than half but
# not all of them get wrong. The unanimous block is out of reach of any draw — five board flats already
# say no reseating touches it. The middle band is exactly the set of studies the family is still
# undecided about, and a draw is the only instrument that can spend more gradient there.
#
# WHY THE WEIGHT IS DISAGREEMENT AND NOT LOSS, which is the conventional choice and is wrong here.
# `uncertainty_split` puts ~87% of the residual's mass on the aleatoric side, so the highest-loss
# studies are disproportionately the mislabelled ones and hard-mining them amplifies label noise —
# the known way this family of method fails. Across-member spread is EPISTEMIC: high where the family
# is undecided, near zero where it agrees, INCLUDING where it agrees on being wrong. It separates the
# 70% band from the 24.3% block by construction, which is the one thing loss cannot do.
#
# THE WEIGHT IS IN-SAMPLE AND THE ARM KNOWS IT. `kneemri.evaluation.train_disagreement` scored twelve
# honest12 members over the studies they were all trained on, so the spread is smaller than the same
# members would show on unseen studies and a memorised study reads as agreement. What survives that
# bias is the ORDERING, and the ordering is all a sampler consumes. Nothing here may be quoted as a
# held-out measurement.
#
# control  disagreed_uniform = 0.8507 (declared 2026-09-03, knee-sgzgc; this arm's OWN alpha-zero run,
# not a ledger row — it is the second of the two run_train calls below, and the note under this line is
# the argument for why the ledger's band_dense would be the wrong denominator)
# THE CONTROL IS THE SECOND RUN AND NOT THE LEDGER, which is the sharp part of this design. At alpha
# zero the weights go flat but the draw stays WITH REPLACEMENT, so an epoch still holds duplicates and
# still misses about a third of the pool — a bootstrap resample, not a permutation. Scoring the tilted
# arm against `band_dense_s2026` would therefore confound the tilt with the resampling, and the
# resampling alone is a plausible loss (less unique data per epoch) or a plausible gain (implicit
# bagging). Two runs, identical but for alpha, is what makes the tilt the only difference.
#
# THE TILT IS SET BY THE MEASURED SPREAD AND NOT BY TASTE, and the measurement moved it. The report on
# `train_spread_honest12.npz` reads mean 0.1244, median 0.1267, p90 0.1571, max 0.2128, and bands the
# corpus as: spread < 0.05 ZERO studies · 0.05-0.10 699 · 0.10-0.15 2175 · >= 0.15 624. Two things follow.
# First, twelve separately-trained members never unanimously agree ANYWHERE — the empty quiet band says
# the 24.3% the family all gets wrong is unanimous in VERDICT, not in confidence, which is a fact about
# the family and not about this arm. Second, and the reason alpha is 1.0 here and not the 0.5 this arm
# was first written with: at 0.5 the split band drew 1.145 against the quiet band's 0.840, a 1.36x
# contrast, and twelve epochs of that is a null BY CONSTRUCTION — and the uninformative null, since a
# flat result would only say "1.36x does nothing" and could not close the axis. At 1.0 the same bands
# draw 1.29 against 0.68, a 1.9x contrast, for the same runtime and no code change. That is still modest;
# it is the most contrast this weight HAS, and the arm is priced against it below.
#
# PRICE, BEFORE THE NUMBER. 2 x 3.1 h of card = 6.2 h for one axis. The mechanism is aimed at the 70%
# band, which is by far the biggest block on the residual sheet, so a mechanism that works at all has
# room for well over the 1.5% bar; P(clears +0.015) ~ 0.15, P(clears the noise floor in either
# direction) ~ 0.6. The null is worth having on its own: it would close the DRAW axis the way the
# 83-run table closed the recipe axis, and it is the last cheap axis in the loop.
#
# PRE-REGISTERED, BEFORE THE NUMBER. All bands on the twelve-label holdout mean, tilted arm minus its
# own alpha-zero control.
#   >= +0.015  THE DRAW IS A REAL LEVER. Licenses a tilt sweep (0.25 / 0.75) and a re-scored weight from
#              held-out members rather than in-sample ones, which should only sharpen it.
#   +0.005 to +0.014  DIRECTIONAL, under the bar. Keep the flag, keep the member if a seat is free, do
#              not plan the roster around it and do not spend a confirmation seed on it.
#   -0.004 to +0.004  THE DRAW DOES NOT MATTER AT THIS TILT. The informative null: it says the loop's
#              uniformity is not what is holding the 70% band, and points the band at the INPUT or the
#              LABEL rather than at how often the model sees a study. Closes the axis AT THE LARGEST
#              CONTRAST THIS WEIGHT CAN PRODUCE (1.9x, alpha 1.0) — which is what makes it a closure and
#              not a knob reading. A sharper weight (a power on spread, a rank tilt) would re-open it.
#   <= -0.005  THE TILT COSTS. Read it with the spread report before concluding: if the split band is
#              thin, the tilt concentrated the epoch on a handful of studies and the follow-up is a
#              lower alpha, not a refutation of the mechanism.
#
# BOTH CHECKPOINTS ARE CANDIDATE MEMBERS whatever the comparison says. All twelve heads carry gradient
# in both runs, so unlike `focus_mcl` these are ordinary members — and a member trained on a different
# DRAW is a diversity axis no other member on the roster has, which is worth a seat test on its own.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

SPREAD="$PROCESSED/train_spread_honest12.npz"
require "$SPREAD"

run_train disagreed_uniform \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --sample-spread "$SPREAD" --sample-alpha 0.0 \
  --labels "$LABELS" --out "$MODELS/disagreed_uniform_s2026.pt"

run_train disagreed_full \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --sample-spread "$SPREAD" --sample-alpha 1.0 \
  --labels "$LABELS" --out "$MODELS/disagreed_full_s2026.pt"
