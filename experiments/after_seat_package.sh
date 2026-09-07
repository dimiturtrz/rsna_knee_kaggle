#!/usr/bin/env bash
# Packages each `retrain_six_distill` seat the MINUTE it lands, instead of at the end of 8.5 h
# (mastermind 03:35Z). The arm is blind — five of six seats fit at `--fold -1` and hold nothing out —
# so the board is the only ruler, and the 17:50Z entry is "the champion with whatever k seats have
# finished". That entry is only buildable in ten minutes if the package is already current.
#
# IT APPENDS A ROSTER LINE ONLY AFTER THE CHECKPOINT EXISTS, and that ordering is the point:
# `package_members` reads a COMMITTED roster and a line naming a file that is not there yet is a
# packaging run that dies at whatever hour the watcher happened to fire. Waiting for the `.pt` makes
# the roster true at every instant rather than eventually.
#
# It does NOT upload. Pushing a Kaggle dataset version runs from PowerShell with `.venv\Scripts\kaggle.exe`
# and is a deliberate act priced against a mark; this script leaves `members_2/` current so that push is
# one command whenever a mark is due.
set -u
cd "$(dirname "$0")/.."

MODELS=D:/models/rsna_knee_abnormality
ROSTER=comp/members2.roster

# Weakest-first, the order `retrain_six_distill` runs them in. `merged_blend_df` is first because it is
# the arm's only offline-readable seat, not because it is the weakest.
SEATS="merged_blend_df final_ws_df calib_span_df grade_ladder_df through_dense_df lr4e-5_ws_df"

for seat in $SEATS; do
  ckpt="$MODELS/$seat.pt"
  until [ -f "$ckpt" ]; do sleep 60; done
  sleep 30   # the trainer writes then closes; do not read a half-flushed checkpoint

  if grep -q "^$seat\.pt" "$ROSTER"; then
    echo "$seat already on the roster $(date -u +%FT%TZ)"
  else
    printf '\n# retrain_six_distill seat under teacher_oof5 (knee-ygi2k). Every seat but merged_blend_df fits\n# at --fold -1 and holds nothing out, so for those NO offline number exists by construction and the\n# board is the ruler; merged_blend_df is fold 0 like its original and its number is in the ledger.\n# Packaged on landing so a partial-swap entry is buildable at any mark.\n%s.pt\n' "$seat" >> "$ROSTER"
    echo "rostered $seat $(date -u +%FT%TZ)"
  fi

  .venv/Scripts/python.exe -m kneemri.kaggle.package_members --roster "$ROSTER"
  echo "PACKAGED $seat exit $? $(date -u +%FT%TZ)"
done

echo "ALL SEATS PACKAGED $(date -u +%FT%TZ)"
