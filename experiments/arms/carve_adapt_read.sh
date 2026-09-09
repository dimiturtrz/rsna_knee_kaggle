#!/usr/bin/env bash
# carve_adapt_read.sh — the same three members, the same corpus, the same ruler, ONE flag added.
#
# WHAT THIS PRICES. The 11-member generalisation readout of 2026-09-09 measured the cost of reading the
# foreign corpus instead of ours: toll mean -0.0336, sd 0.0038, range 0.0131. It is FLAT across members
# (rho(native, decode_cost) 0.1727, p 0.61) and tighter than the score spread it is drawn from, which
# makes it a property of the corpus boundary rather than of any member. 0.0336 is more than three times
# the board gap of 0.010, and no member we own has ever been shown to pay less than -0.0265.
#
# THE MECHANISM. CARVE (arXiv:2607.15556) tunes a member's LayerNorm affines to the scan in front of it
# for two Adam steps at 1e-5, sharpening its own top-k labels and suppressing the rest, then undoes it.
# Published: AUROC +0.034 to +0.036 absolute, episodic per scan, on a multi-label head. It is the only
# published multi-label medical-classification TTA number and it is lever-sized against our gap. The
# same paper states the ceiling: depth/protocol shift costs -0.125 BEFORE adaptation, and adaptation is
# bounded by the transferred representation. So this tunes within what the trunk already serves; it is
# not a repair for a pipeline mismatch, and if the toll is mostly mismatch this will read near zero.
#
# WHY IT IS A CONTROLLED PAIR AND NOT A NUMBER. `fovjitter160_read.sh` runs the SAME three checkpoints
# over the SAME foreign directory on the SAME device minutes earlier. This arm differs from it in the
# single flag `--adapt`. The comparison is therefore against a control taken on the same card in the
# same hour, not against a remembered figure.
#
# WHAT IS PROVISIONAL, STATED SO IT CANNOT BE FORGOTTEN. CARVE's Eq. 8-9 were built here from a prose
# description rather than from the equations (M265 asks the researcher for them, still unanswered). The
# reduction, the sharpening term's exact form, and whether the label mean is taken before or after the
# backward are THIS repo's reading of the recipe. Any number this produces is a first read, and a NULL
# does not refute CARVE -- it refutes this reading of CARVE. That distinction is the whole reason the
# objective is a strategy object with the departure written into its docstring.
#
# OUR ONE DELIBERATE DEPARTURE. CARVE's k_min is at least one because a CT findings task assumes
# something is there to find. Over a tenth of our studies read clean (p10 = 0 over all 4406 studies of
# report_labels_v2 at the 0.5 midpoint), so k_min 1 would manufacture a positive on every healthy knee.
# Our bounds are (0, 8), our own p5 and p95.
#
# cost   three members over the foreign holdout, plus two graded forward/backward passes per study per
#        member. The adapt pass is ONE view at ONE start, not the test-time average, so it adds two
#        forwards and two backwards against a read that is already views x starts forwards. Call it
#        under 2x the control read: minutes, not hours. This is the cheapest instrument on the board.
# price  minutes of card for a mechanism whose published effect (+0.035) is 3.5x the board gap, against
#        a toll (-0.0336) that no member-side lever has ever touched. Even a NULL is worth the minutes:
#        it would say the toll is representation-bound rather than calibration-bound, which is the
#        finding that decides whether the next arm is a data arm or a model arm.
#
# PRE-REGISTERED ON THE MEAN DECODE COST OVER THE THREE MEMBERS, read from the same readout the control
# takes. Not on any single member: the toll's whole finding is that it is flat, so a per-member reading
# would be three samples of one number.
#   CARRIES   — mean decode_cost improves by 0.0150 or more. Half the published effect, still 1.5x the
#               board gap, and far outside the pool's 0.0038 sd. Adaptation is then a real lever on the
#               corpus boundary, it goes into the submission path behind witness_check, and the next
#               question is whether it composes with the extent axis fovjitter160 is testing.
#   PARTIAL   — 0.0050 to 0.0149. Directional and mechanism-sound but under the bar. KEEP the code, do
#               not re-cut the operating point on it, and the follow-up is CARVE's actual equations
#               (M265) before a second reading, because a mis-read objective is the likeliest cause of
#               a shortfall this size.
#   NULL      — -0.0050 to 0.0049. Adaptation does not touch this toll. That is a FINDING and it is the
#               one CARVE's own ceiling predicts: the toll would then be representation-bound, the
#               model-side lever is closed, and the whole weight moves to the data axis (M266).
#   REGRESSION— worse than -0.0050. The objective is actively harming the reading. The suspect is THIS
#               repo's reading of Eq. 8-9, not the recipe; the re-test is blocked on M265 and the code
#               stays behind its default-off flag until then.
#
# WHAT THIS ARM CANNOT SETTLE. It reads three members, all of ours, all on one foreign corpus. A gain
# here is evidence about THAT boundary, not about the Kaggle private set, which is a third corpus
# neither of these is. And the members are not independent samples of the toll -- they share a trunk
# family and two of them share a cache.
#
# bead: knee-18c5r
# reading: foreign_readout stdout, captured to this arm's own log
#
# deliberate: --adapt (THE ONE MOVED FLAG, against fovjitter160_read.sh which is otherwise identical)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

# THROUGH `uv`, NOT THROUGH THE INTERPRETER PATH. `arm_check` RUNS every arm file to read its flags,
# and the only thing that makes that safe is a stub that shadows `uv` (arm_check.py:208-223). An arm
# naming `.venv/Scripts/python.exe` walks straight past the stub, so a lint launches a card job:
# measured 2026-09-09, `arm_check --only carve_adapt_read` put 32 GB of `foreign_readout` on the card
# beside a live training arm sixteen seconds after it started. Eight arms still do it (knee-364re's
# sibling bead); this one does not.
UV_NO_SYNC=1 uv run python -m kneemri.evaluation.foreign_readout \
  --checkpoints \
    "$MODELS/fovjitter160_s2026.pt" \
    "$MODELS/crop160_span_s2026.pt" \
    "$MODELS/band_dense_hold_s2026.pt" \
  --labels "$LABELS" --foreign D:/kaggle/input/raptor-corpus --device cuda --adapt
