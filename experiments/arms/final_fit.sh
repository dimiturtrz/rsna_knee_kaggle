#!/usr/bin/env bash
# final_fit.sh [epochs] — fit ONE model on EVERY study, which is what we submit now that fold and seed
# ensembling are out.
#
# knee-5895 IS DECIDED, 2026-08-22, AND THE DECISION IS OPTION (a): this arm is re-based onto
# `band_dense` exactly — 336 px, 20 slices, band 0.10-0.90, --backbone-lr 4e-5. Audited 2026-08-20 by
# resolving the argv rather than reading the comments, which found two defects:
#   - it named neither --slices nor --band-low/--band-high, so it took the PARSER DEFAULTS of 12 slices
#     and band 0.20-0.80 against a settled recipe of 20 and 0.10-0.90. The +0.0039 band widening is
#     described elsewhere as "in the recipe every arm inherits"; this arm did not inherit it, because
#     those are per-arm flags and not part of $RECIPE.
#   - the 448/392 argument rested on res392 = 0.8523 being our best single member. It is not, any more:
#     all four members of the 0.930 champion are 336 px, res392 is in none of them, and band_dense at
#     336 scores 0.8578 — +0.0055 ABOVE the member that geometry was built around.
#
# WHY (a) AND NOT (b), WHICH WAS THE STANDING RECOMMENDATION'S ONLY RIVAL. Option (b) was to measure
# 392 px at the settled slices and band on fold 0 first and ship whichever won, because naming the
# missing flags at 392 would otherwise ship a configuration nobody has run. That option was priced on
# 2026-08-22 and it is not worth its card: 448 px exists on disk at TWELVE slices only
# (`images_448px_12sl.npy`), so the arm needs a fresh ~106 GB cache built before its first step, which
# turns a forty-minute measurement into something over an hour — spent deciding a question the
# resolution axis has already answered flat twice at 336, for a member gain that would be under the bar
# even if res392's +0.0066 transferred whole. Option (a) costs no card, ships the configuration that
# MEASURED best and is seated in the champion, and changes nothing unmeasured. The coverage lever 392 px
# was kept for is not discarded — it is left re-testable, and it becomes worth a slot again only if
# something makes a 448 cache cheap or makes resolution look live at 336.
#
# WHAT CHANGED AND WHY. Our leaderboard entries have been rank-averaged fold members: five at 0.888,
# then two whole-stack ones. That is ensembling-by-k-fold, and it is banned here for a good reason -
# it buys variance reduction rather than signal, and it costs one full run per member. It also has a
# quieter cost that matters more: every member is fitted on 80% of the corpus, so the ensemble never
# contains a model that has seen the fifth fold. This arm trades N models on 80% for ONE model on 100%.
#
# THE THING THAT MAKES IT DIFFERENT FROM EVERY OTHER ARM: with --fold -1 no study is held out, so
# there is NO SCORE. Nothing here can be compared to 0.8457, the run cannot rank its own epochs, and
# train.py keeps the LAST epoch by construction rather than the best. This arm produces a submission,
# not a measurement - and it must therefore only ever be run on a recipe that a fold-0 arm has already
# settled.
#
# WHICH IS WHY THE EPOCH COUNT IS AN ARGUMENT AND NOT A DEFAULT DECISION. It is imported from the arm
# that settled the recipe: whole-stack peaked at epoch 6 of a 12-epoch one-cycle on both folds tried.
# Seven is the honest reading of that, and it is a mild EXTRAPOLATION rather than a measurement: a
# 7-epoch one-cycle is not the schedule whose epoch 6 scored 0.8457, because one-cycle is defined over
# total_steps and a shorter budget anneals faster. knee-1je measured that direction as favourable
# (12 epochs beat 24, 0.8457 vs 0.8413) but not at this length.
#
# EMA IS ON, and this is the case it was argued for: with no holdout there is no best-epoch selection
# to fall back on, so the answer would otherwise depend entirely on where the last step landed.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
# control  res392 = 0.8523 (declared 2026-09-03 from this header own prose, knee-sgzgc)
# unmeasurable: final_fit fits ONE model on EVERY study — that is its definition, and it is why no offline number can exist for it. Its ruler is the board row `finalfit_v1` in comp/board.jsonl.
#
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
# EVERY CHOICE HERE IS NOW THE ONE THAT SURVIVED, and four of them were not on 2026-08-11.
#
# 336 px, 20 SLICES, BAND 0.10-0.90 — `band_dense`, COPIED RATHER THAN REASSEMBLED. This is the best
# single member the project has (0.8578) and it is seated in the champion. The rule this arm's own
# header states is that the deliverable may only ever run a recipe a fold-0 arm has already settled, and
# copying a settled arm's flags verbatim is the only way to obey it: every reassembly, however sensible
# each flag looks alone, is a configuration nobody ran.
#
# --backbone-lr 4e-5 IS PART OF THAT MEMBER. The default is 8e-6 (trainer.py) and `band_dense` was
# fitted at 4e-5 — five times the rate — so dropping it here would build one of those unrun
# configurations in the one place with no holdout to catch it.
#
# TWELVE EPOCHS, NOT SEVEN. OneCycle is defined over total_steps, so seven is a COMPRESSED schedule
# rather than a truncated one - measured, schedule_8 scored 0.8429 against the twelve-epoch 0.8457, so
# compressing costs 0.0028. Every member in the ledger was fitted under the twelve-epoch shape and the
# deliverable has to be fitted under the same one.
#
# NO --patience, and that is by construction rather than an oversight: --fold -1 leaves no holdout, so
# the epoch INDEX stands in for the score, it increases every epoch, and nothing ever looks stale. The
# last epoch is kept, which is what a final fit wants.
#
# bf16 AND compile, both measured and both free: -21% per step for 2.2 s of compilation, and bf16
# carries fp32's exponent range so the loss scaler goes with it.
#
# --ema STAYS. It is the sanctioned variance reduction now that folds and seeds are banned: it averages
# within a run rather than across runs, so it buys what a seed ensemble bought without a second fit.
#
# THE CAP IS NOT NAMED HERE and does not need to be at 336 px: `band_dense` reserves 28934 MiB at these
# exact flags (its own gate line) and ran twelve epochs at 368 s each without stalling. The history is
# still worth keeping, because it is the failure this cap exists for: while this arm was at 392 px the
# cap read 22 and the arm did not run at all — a --studies 60 smoke died in the attention backward with
# `Tried to allocate 1.03 GiB ... 22.00 GiB allowed`, and the corpus size does not enter it, so the
# forty-minute deliverable would have died in the same place. DeviceBudget
# derives the cap from --slices — but the reason a cap exists at all is worth keeping: --vram-gb bounds
# PyTorch's allocator and not the device, so set close to physical, Windows pages over PCIe instead of
# raising OOM and the run crawls at a twenty-fifth of the throughput while reporting 100% utilisation.
EPOCHS="${1:-12}"
require "$LABELS"
# $RECIPE ends with `--fold 0`; the `--fold -1` after it wins, which is argparse's documented last-one-
# wins and was confirmed by parsing this exact argv. Sourcing the recipe rather than repeating its flags
# is not cosmetic here: this is the arm that SHIPS, and it was the last one still holding a private copy
# of the shared fitting flags — the drift recipe.sh exists to remove, and the drift its own header cites
# knee-7tq needing four corrections for. The live case is `--rank-weight`, which rank0_dense is measuring
# now: if the term is refuted the recipe drops it, and a private copy would keep fitting the deliverable
# with a term the project had just removed, silently and with no holdout to notice.
run_train "final_fit_e${EPOCHS}" $RECIPE --fold -1 --epochs "$EPOCHS" --slice-pooling max --whole-stack --ema --slices 20 --band-low 0.1 --band-high 0.9 --backbone-lr 4e-5 $THROUGHPUT --labels "$LABELS" --out "$MODELS/final_ws_e${EPOCHS}.pt"
