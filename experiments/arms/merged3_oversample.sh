#!/usr/bin/env bash
# bead: knee-eb68g
# reading: merged3_oversample_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   merged3_corpus.sh's exact line plus one flag: --corpus-oversample 2.22.
#
# WHY IT IS NOT A REPEAT
#   D188 §4 recomputed the flag census off every arm on disk: `--corpus-oversample` is set by ZERO of
#   291 arms. It is fully plumbed (options.py:138-140 -> run_setup.py:190-221 -> MixedCorpus, landed-
#   checked at landed.py:291-292/362-363, unit-tested at tests/unit/kneemri/training/run_setup.py:493)
#   and has never once been on a command line. This is the only never-set flag in the training surface.
#
# THE FACTOR IS DERIVED, NOT PICKED
#   sampling.py:231-236 states the policy's own goal: "roughly half the batch from external corpora".
#   MixedCorpus weights every NON-target position by `factor` and the target at 1.0, so parity needs
#   factor = canonical / foreign. The joined table reads 4406 canonical, 1248 mrnet, 736 rijeka:
#   4406 / 1984 = 2.221. Target stays the default, `canonical`.
#
# THE LABELS TABLE, AND THE CONFOUND THAT WOULD HAVE BEEN INVISIBLE
#   run_setup.py:214 refuses without a `corpus` column and merged3_labels.csv carries none. The table
#   that does — merged3_tagged_labels.csv — is RECALIBRATED (ACL 0.2995 -> 0.2233, Effusion 0.5009 ->
#   0.3387, five columns measured), so training on it would confound the draw policy with label
#   calibration, the one axis positive at BOTH corners (+0.0129 / +0.0161). This arm trains on
#   merged3_labels_tagged_raw.csv: merged3_labels.csv with the corpus column JOINED on
#   StudyInstanceUID by `python -m core.data.corpus.corpus_column`, every finding value byte-identical
#   (asserted: a.equals(b[a.columns]) is True). The two source tables are in different row order, so a
#   positional assignment would have mislabelled most rows and looked fine.
#
# CONTROL, STATISTIC, POPULATION, CUT
#   merged3_corpus_s2026 = 0.8521 — per-column ROC-AUC, unweighted 12-column macro, population
#   holdout909, cut midpoint 0.5. Same recipe, same epochs, same slices, same eval labels, one flag.
#
# PRE-REGISTRATION (D188 §4, written before the run)
#   >= +0.015   the foreign rows were being drowned; dilution is binding and per-corpus head
#               conditioning is worth scoping as a build.
#   +0.006 .. +0.015   directional only; keep the member, re-open nothing.
#   within +-0.006     READING B, and the expected outcome: the foreign rows are already fitted and
#               weighting them harder buys nothing. The naive-concatenation ceiling is not a sampling
#               artefact, which retires the cheapest explanation for why the corpus axis saturated.
#   <= -0.015   dose-response B: foreign rows are actively worse per-sample and the merged corpus wins
#               on count alone.
#
# PRICE
#   ~40 min of card (12 slices, 336 px, 12 epochs, 6390 studies). P(clears the +0.015 bar) ~ 0.10.
#   IT IS AN INSTRUMENT, not a member bid: its value is that it decides whether a head-conditioning
#   build gets scoped at all, and it is the cheapest arm that can decide it. Sold that way, not as a
#   score.
#
# WHAT IT DOES NOT TEST
#   Per-site FiLM/embedding conditioning (Q-POOL mechanism 1) — that is code to write, and this arm is
#   what prices whether writing it is worth the week.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

MERGED3_TAGGED="$PROCESSED/labels/merged3_labels_tagged_raw.csv"
require "$MERGED3_TAGGED"
require "$LABELS"

run_train merged3_oversample \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --scheme merged3 --corpus-oversample 2.22 \
  --labels "$MERGED3_TAGGED" --eval-labels "$LABELS" --out "$MODELS/merged3_oversample_s2026.pt"
