#!/usr/bin/env bash
# raptor_features.sh — the teacher export that stage one of knee-8009 (b) requires.
#
# WHY THIS IS AN ARM AND NOT A REMEMBERED STEP. `raptor_feat_pre.sh` opens with
# `require "$PROCESSED/raptor_features.npz"` and dies if it is absent, so the export is a hard
# precondition of a four-hour two-stage chain. A precondition that lives only in a human's head is the
# defect knee-61ez names: a hand-rolled chain is invisible to `queue_check`, which globs
# `logs/after_*.log` for the waiting line. Shaped as an arm, the export is a link `after.sh` can hold
# and `queue_check` can see.
#
# WHY IT IS NOT INLINED INTO `raptor_feat_pre.sh` instead, which is where `raptor_feat_fit.sh` puts ITS
# conversion. The fit arm's `fitted_encoder` call costs seconds. This costs ~35-40 min of card, and
# `arm_check --only raptor_feat_pre --smoke` runs the arm at 60 studies to check its flags in about two
# minutes. A generate-if-missing export inlined there would fire the FULL-corpus export inside every
# smoke, which turns a two-minute flag check into a forty-minute one and would be discovered by waiting.
#
# NOT PRE-REGISTERED, and it has no control: it produces a file, not a member and not a number. The one
# thing to check on its log is that the written npz joins on the studies the next arm trains on — a
# teacher file that joins on nothing trains a plain labelled run and hands stage two a control wearing an
# arm's name, which is the fourth suspect `raptor_feat_fit.sh` pre-registers.
#
# THE TEACHER IS IN-SAMPLE. `foreign_overlap` puts raptor at 98.6% in-sample on this corpus. That caveat
# is argued where the branches are, on `raptor_feat_fit.sh`; it is named here so nobody reads this file
# as evidence the export is clean.
#
# cost   ~35-40 min of card, zero slots.
#
# bead: knee-8009
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
# THE TEACHER CHECKPOINT IS THE CHAMPION'S OWN FOREIGN SEAT, deliberately. `raptor_v4_swa` is what
# `comp/champion.json` mounts at 1/7, it is one of the four DISTINCT members the fleet decode found among
# thirteen foreign checkpoints (the other nine are copies), and using any other one would make the trunk
# imitate a representation this project has never shipped. `RAPTOR_WEIGHTS` points at the unzipped
# `dreaddevelopment/raptor-knee-widedense` dataset; it is an env var with no default because the download
# root is machine-specific and no machine path belongs in git (see `kneemri/evaluation/foreign_member.py`
# for the two fetch commands that populate it).
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

RAPTOR_WEIGHTS="${RAPTOR_WEIGHTS:?set RAPTOR_WEIGHTS to the unzipped raptor-knee-widedense directory}"

require "$LABELS"
require "$RAPTOR_WEIGHTS/raptor_ft_coatnet_v4_full_swa.pt"

uv run --no-sync --project "$REPO" python -m kneemri.kaggle.raptor_features \
  --weights "$RAPTOR_WEIGHTS/raptor_ft_coatnet_v4_full_swa.pt" \
  --out "$PROCESSED/raptor_features.npz" || exit 1
