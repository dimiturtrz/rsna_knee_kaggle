#!/usr/bin/env bash
# The CONTROL for `ortho_lpft`, and the arm that makes the LP-FT question readable at all. `knee-6hfi`.
#
# `ortho_lpft` fits a head over a frozen `ortholarge`, then unfreezes the whole stack at 1e-5 starting
# from that head. `ortho_dense` fine-tunes the same trunk end to end at 4e-5 from a random head. Read
# against each other, those two move TWO variables - the initialisation and the rate - so a gain from
# the pair alone could be either. This arm is the third point that separates them: `ortho_lpft`'s stage
# two exactly, at 1e-5, whole stack, with a RANDOM head instead of a fitted one.
#
#   ortho_dense        random head, 4e-5   -> the rate point
#   ortho_fresh_lowlr  random head, 1e-5   -> THIS ARM, the initialisation control
#   ortho_lpft         fitted head, 1e-5   -> the hypothesis
#
# Two differences, one per pair, and every other flag identical across all three.
#
# WHY IT IS NOT OPTIONAL. Kimi's Q-BN reply (2026-09-02) reports that no published work runs LP-FT with
# a NONLINEAR head: Kumar et al. (arXiv:2202.10054) prove the feature-distortion tradeoff for two-layer
# linear networks and list non-linear activations as an open question; Tomihari & Sato
# (arXiv:2405.16747) carry it to Transformers through the growth of the LINEAR head's norm; Davila et
# al. (arXiv:2406.10050) replicate on medical CNNs with a standard linear classifier. Our head is a
# `LayerNorm -> Linear -> GELU` projection, per-diagnosis attention over slots, then a per-row readout.
# One live failure mode is precisely that a head with its own depth ABSORBS the large first gradients,
# so a fresh-head fine-tune at a low rate is already doing what the two stages do. That failure mode is
# invisible without this arm and would otherwise be read as "LP-FT works".
#
# PRE-REGISTERED, and this arm's own branches are about the RATE rather than about the warm start:
#
#   s >= ortho_dense + 0.015   The rate was carrying it, not the initialisation, and 4e-5 was already
#               too hot for a trunk this size on 4,400 studies. That re-opens `radnet121`'s written
#               decline (0.6716 fine-tuned at 4e-4 against 0.8197 frozen) as a RATE finding, which is
#               the cheaper of the two explanations and the one that generalises to every external
#               trunk this project has priced.
#   |s - ortho_dense| < 0.006   The rate is inert between 1e-5 and 4e-5 here, so whatever `ortho_lpft`
#               reads above this is the warm start alone - which is the clean version of the answer.
#   s <= ortho_dense - 0.006   1e-5 UNDERTRAINS this trunk in twelve epochs. Then `ortho_lpft` must be
#               read against THIS number and not against `ortho_dense`, and a flat between the two 1e-5
#               arms says the warm start did not rescue an undertrained schedule rather than that it
#               does nothing. Check the run's last epoch against its best with `convergence_check`
#               before writing either sentence.
#
# deliberate: NO --init (the whole point of the control)
# deliberate: --backbone-lr 1e-5 (ortho_lpft's stage two exactly)
# deliberate: --variant ortholarge --slices 12 --image-size 224 --band-low 0.2 --band-high 0.8
#             (base224's read, held identical across the triple)
#
# bead: knee-6hfi
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train ortho_fresh_lowlr \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant ortholarge --backbone-lr 1e-5 --slices 12 --image-size 224 \
  --band-low 0.2 --band-high 0.8 \
  --labels "$LABELS" --out "$MODELS/ortho_fresh_lowlr_s2026.pt"
