#!/usr/bin/env bash
# weakdraw.sh — the THIRD form of the weak-column fix: spend the epoch there. Tilt + its own alpha-zero twin.
#
# THE AXIS, AND WHY THIS IS THE LAST FORM OF IT. `column_headroom` puts the block's remaining macro in the
# weak columns. The DATA form (foreign tables carrying that column) and the LOSS form (`--focus-label`,
# per-label weighting) have both been priced and neither moved the macro. The DRAW is the third and last
# form, and the only one that changes what the optimiser SEES rather than what it is told about what it
# saw. A null here closes the weak-column axis on all three forms.
#
# THE COLUMN IS FRACTURE, AND THAT IS A MEASUREMENT, NOT A PREFERENCE. `column_spread` over
# `report_labels_v2.csv` (n=4406) reads verdict YES at MCL 0.1741 and Lateral OA 0.2467 — union 0.3768, so
# those positives are already better than a third of every epoch and a tilt cannot make them less scarce
# than they are not. `Disagreed` normalises by the mean, so a ratio r puts their mass at
# r*.3768/(r*.3768+.6232): 0.547 at r=2, 0.645 at r=3, against a uniform 0.377 — 1.45x and 1.71x. The
# floor for THIS instrument is already in the ledger: `disagreed_sampler` found a 1.36x contrast is a null
# by construction and had to be re-pitched at 1.9x. Fracture reads YES 0.0713, so at r=6 its positives
# hold 0.315 of the epoch against a uniform 0.071 — a 4.4x contrast on the same machinery and the same
# runtime. It is also the one column D77's per-column read has NEGATIVE on the clean eight (Frx -0.006),
# so a lift lands on the seat table's only losing column rather than on a flat one.
#
# WHAT THIS ARM CANNOT FIX, said here so the null is not over-read. 56.4% of studies are UNK on Fracture
# and 33.1% on Lateral OA. For those columns the binding constraint is annotation COVERAGE, and no draw
# policy reaches it. A flat result closes the DRAW form of the weak-column fix; it says nothing about the
# coverage form.
#
# THE CONTROL IS THE SECOND RUN, NEVER THE LEDGER. At alpha zero the weights go flat but the draw stays
# WITH REPLACEMENT, so an epoch still holds duplicates and still misses about a third of the pool — a
# bootstrap resample, not a permutation. D79 measured that resampling alone at -0.0071 (`disagreed_uniform`),
# which is larger than the effect being looked for, so scoring the tilt against a ledger row would price
# the resample and call it the tilt. Two runs, identical but for alpha.
#
# PRE-REGISTERED, BEFORE THE NUMBER. TWO rulers, both on the 896 L75mid frame, both read as the tilted
# arm minus its OWN alpha-zero twin — never against a ledger row.
#
# The FRACTURE column, which is the one the draw is aimed at:
#   >= +0.015  the draw reached the column. DIRECTIONAL even so: the per-label ruler is +-0.040 single-seed,
#              so this becomes a finding only if the macro moves with it.
#
# The MACRO, which is what decides the axis:
#   >= +0.002  THE DRAW IS A LEVER ON A WEAK COLUMN. Licenses the clean pair at 12 epochs with r swept
#              {3, 6, 10}, about 2.7 h.
#   -0.002 to +0.002  FLAT. The sampling form closes WITH the loss form and the weak-column axis is closed
#              on all three forms (data, loss, sampling). This run has to exist before that sentence can be
#              written: `per_label_loss` read Frx -0.022 on the LOSS side, and reweighting the loss changes
#              the gradient over a fixed draw while r=6 changes WHICH studies the epoch sees. Same column,
#              different mechanism, and the draw is the form the axis was never priced on.
#   <= -0.002  THE TILT STARVES THE REST. Record the give-back PER COLUMN across the other ten: a tilt that
#              spends a third of the epoch on 7% of the corpus trades eleven columns for one, and the macro
#              averages exactly that away.
#
# deliberate: --sample-spread (the draw is the moving axis; everything else is band_dense's recipe)
# deliberate: --sample-alpha (1.0 on the arm, 0.0 on its twin — the ONLY difference between the two runs)
# reading: weakdraw_frx_s2026.holdout.npz
# bead: knee-vv9d9
# control  weakdraw_flat_s2026 = this arm's own alpha-zero run, the first of the two run_train calls below
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

COLUMN="Fracture"
RATIO=6.0
SPREAD="$PROCESSED/labels/spread_weak_frx_r6.npz"

require "$LABELS"
python -m kneemri.evaluation.column_spread \
  --labels "$LABELS" --column "$COLUMN" --ratio "$RATIO" --out "$SPREAD"
require "$SPREAD"

run_train weakdraw_flat \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --sample-spread "$SPREAD" --sample-alpha 0.0 \
  --labels "$LABELS" --out "$MODELS/weakdraw_flat_s2026.pt"

run_train weakdraw_frx \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --sample-spread "$SPREAD" --sample-alpha 1.0 \
  --labels "$LABELS" --out "$MODELS/weakdraw_frx_s2026.pt"
