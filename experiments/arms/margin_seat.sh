#!/usr/bin/env bash
# margin_seat.sh — descend the metric itself instead of a proxy that is not it.
#
# WHAT THE METRIC IS. Twelve per-label ROC AUCs, unweighted mean. AUC is rank-based: it reads only the
# ORDER of the twelve scores within each label, never their values. Cross-entropy reads values. Every
# member this project has fitted descends cross-entropy (plus a ranking term at 1.0) and is then scored
# by something that cannot see most of what that loss spent its capacity on — a confidently-correct
# positive pushed from 0.9 to 0.99 costs the loss and buys the metric exactly nothing.
#
# PRIOR ART, NOT AN INVENTION. Yuan et al., "Large-scale Robust Deep AUC Maximization" (arXiv 2012.03173,
# ICCV 2021); LibAUC. It took 1st place on CheXpert 2020 — a multi-label medical-imaging leaderboard
# scored on mean per-label AUC, which is this competition's metric with a different organ. That is the
# single closest published match to this task that exists, and this repo has never run it.
#
# WHY NO NEW OPTIMISER AND NO VENDORED DEPENDENCY. The published objective is a min-max over (a, b)
# minimised and a dual alpha maximised, solved with PESG. All three inner variables have closed forms —
# a and b are the class-conditional means, alpha is the margin shortfall — and by the envelope theorem
# the gradient w.r.t. the network is unchanged when they are substituted rather than tracked. The whole
# thing collapses to
#     var[p|pos] + var[p|neg] + relu(margin - (mean_pos - mean_neg))^2
# which is the pairwise squared-hinge AUC surrogate, decomposed. No stored parameter, no second
# optimiser, no untested interaction with this project's one-cycle AdamW. `MarginObjective.auc_margin`
# is that expression; the derivation is written into its docstring so the substitution is auditable.
#
# THE CUT MUST BE THE SCORER'S CUT. Targets here are five graded levels (0.08/0.28/0.68/0.82/0.94) and
# the scorer binarises at `MIDPOINT = 0.5`. A loss that cuts anywhere else silently optimises a
# different partition than the one being measured, and the arm would be unreadable.
# `test_the_margin_cuts_where_the_scorer_cuts` imports both constants and asserts they are equal.
#
# ONE VARIABLE against the control. This objective REPLACES cross-entropy; it does not add to it. The
# shared recipe carries `--rank-weight 1.0`, so `--rank-weight 0` is spelled here — `run_setup` refuses
# the composition outright (`_check_margin_alone`), because a run holding both would descend two
# orderings at two strengths with no number afterwards saying which moved it. Everything else is
# `band_dense` unchanged.
#
# THE WEAKNESS, PRE-REGISTERED. A batch holds twelve studies and Fracture is positive in ~7% of them, so
# no positive-negative pair can be formed for that label in roughly half of all steps and its term is
# skipped there — while cross-entropy is skipped in none. A rare-label regression is therefore a
# BATCH-SIZE artefact before it is a refutation of the objective, and `RememberedRanking` already proves
# the cross-batch fix inside this repo. Adding it here would move two variables at once, so it is
# deliberately not in this arm. READ THE PER-LABEL TABLE BEFORE THE MEAN.
#
# THE FALSIFIABLE PART, written before the run.
#   * mean at or above the control, common labels up, Fracture down — the objective works and the
#     batch is what is small. Licenses the pair-memory variant, not a verdict on the objective.
#   * mean down by more than the 0.0062 floor with the profile FLAT across labels — the surrogate is
#     genuinely worse here than cross-entropy at this operating point (margin width is what is open).
#   * mean up by more than the floor — the metric was being left on the table by every member fitted so
#     far, and the objective belongs in the recipe, not in one seat.
#
# THE FLAG SET IS DIFFED AGAINST THE CONTROL BY HAND, per knee-f8j9: on 2026-08-24 an arm declared
# `band_dense` byte-for-byte, omitted `--backbone-lr`, took the trainer's 8e-6 default against a control
# at 4e-5, and burned 75 minutes on a number nobody could read. `--backbone-lr 4e-5` below is that flag.
#
# COST. ~80 min of card (12 epochs at ~400 s), zero submission slots, no cache to build.
# SMOKE IT BEFORE QUEUEING: `--studies 60 --epochs 1`.
#
# bead: knee-bdhb
# deliberate: --auc-margin
# deliberate: --rank-weight
# control  band_dense = 0.8578 (best epoch 8), identical recipe descending cross-entropy + ranking.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train margin_seat \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --rank-weight 0 --auc-margin 1.0 \
  --labels "$LABELS" --out "$PROCESSED/margin_seat_s2026.pt"
