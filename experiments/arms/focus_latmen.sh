#!/usr/bin/env bash
# bead: knee-9707
# focus_latmen.sh — is the per-finding gap the price of JOINTNESS, or a defect in the encoder?
#
# THE MECHANISM, AND WHY NOTHING HERE HAS EVER MEASURED IT. All 107 recorded runs fit twelve findings
# through one encoder. Every external number this project prices its deficit against comes from a
# SPECIALIST: MRNet, ELNet and OrthoDiffusion train one model for one-to-three findings and reach
# 0.965-0.970 on ACL and 0.85-0.90 on meniscus. The only published model doing what we do — a twelve
# finding panel, one encoder, 1,748 subjects across five centres (Nat Commun 2024) — averages 0.812,
# which is BELOW our own 0.8695. So the Lateral Meniscus deficit that knee-9707 prices at +0.017 of
# the twelve-label mean may not be an encoder that cannot see a 1-3 mm tear; it may be eleven other
# findings competing for the same representation. That is an untested axis, not a knob.
#
# WHY LATERAL MENISCUS AND NOT ANOTHER FINDING. It is the largest model-side gap in the corpus
# (knee-dtk, -0.1932) and it is the one whose deficit is attested on THREE independent rulers that
# could not have produced each other: the report/corpus ruler, the annotation ruler against
# radiologist truth (0.6497 achieved against a 0.8429 REFERENCE — the report label used as a
# PREDICTOR of that truth), and the published external best of 0.850. The label carries the signal
# and the model does not extract it. Every cheap data-side explanation of that is already refuted on
# its own measurement (knee-yrg): laterality flip (there are no flips, augmentation.py:39), the slice
# band, prevalence (matched counts move the gap by 0.0001), and the in-plane crop — which clips
# MEDIAL three times harder (0.0173 vs 0.0060) and therefore points the wrong way.
#
# WHY THE EXTREME AND NOT A REWEIGHT. `--focus-label` zeroes the loss weight of the other eleven
# columns, which puts the WHOLE encoder behind one finding. That is the strongest tilt the mechanism
# allows: if maximum tilt does not recover the gap, no partial reweighting will either. Testing the
# ceiling first is what makes a null here worth something.
#
# THE CONFOUND, STATED BEFORE THE NUMBER. A focused run sees ONE supervisory signal per study where
# the control sees twelve, so it is data-starved relative to its control — and `half_data` already
# measured that data volume is the binding constraint on this corpus (-0.0234 for half the studies,
# 3.8x the noise floor). That asymmetry is the point of the design rather than a flaw in it: a WIN is
# strong evidence, because it won DESPITE the starvation, and a LOSS is ambiguous between "jointness
# costs nothing" and "one finding is too little supervision to fit an encoder on". The follow-up for
# an ambiguous loss is an UPWEIGHT, which starves nothing.
#
# THE CONTROL IS `band_dense`, ALREADY FITTED, and this file is its invocation with `--focus-label`
# added and nothing else changed. Read the LATERAL MENISCUS COLUMN, not the mean: the mean of a
# focused run is meaningless, since eleven of its twelve heads carry no gradient at all.
# `band_dense` reads Lateral Meniscus 0.842 at its best epoch (mean 0.8578, epoch 8).
#
# PRE-REGISTERED, BEFORE THE NUMBER. All bands are on the LATERAL MENISCUS column against 0.842.
#   >= 0.892   (+0.05) — CONTENTION IS THE MECHANISM. Twelve heads competing for one encoder is what
#              costs the finding, and per-label specialisation becomes the lever on the only
#              externally-attested +0.025 this project has. Licenses a specialist member per weak
#              finding, ensembled under the twelve-column contract the head already emits.
#   0.852-0.891 (+0.01 to +0.05) — directional, below the 1.5% bar as a finding. Says the panel is
#              near its per-label ceiling and specialisation buys a real but knob-sized amount. Keep
#              the member if a seat is free; do not plan around it.
#   0.832-0.851 (within +/-0.01) — JOINTNESS COSTS NOTHING. Retire contention as an explanation of
#              the per-finding gaps. That is the informative null: it points the whole deficit at the
#              encoder's pretraining rather than at the objective, which is knee-ombx's axis.
#   <= 0.831   AMBIGUOUS ON THE STARVATION CONFOUND, not a refutation. Follow up with an upweight
#              (the same mask, at a weight rather than at zero), which keeps all twelve signals.
#
# NOT A CANDIDATE MEMBER IN ITS OWN RIGHT, whatever it scores. Eleven of its columns are untrained,
# so it cannot be averaged into a roster as-is; it enters a kit only as one column of a per-finding
# assembly, and only if the top band lands.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train focus_latmen \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --focus-label "Lateral Meniscus" \
  --labels "$LABELS" --out "$PROCESSED/focus_latmen_s2026.pt"
