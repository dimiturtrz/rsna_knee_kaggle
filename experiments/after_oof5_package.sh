#!/usr/bin/env bash
# Rosters and packages `seat_oof5_distill_s2026` the minute it lands, so the 12:10Z solo entry
# (`comp/kernels/kneemri-oof5solo/`, already written) needs no work between the member existing and
# the mark. Separate from `after_seat_package.sh` because that one is LIVE and a watcher is never
# edited in place — a running loop re-reads nothing, so the edit lands only for a process that has
# already exited.
#
# It does NOT upload. A Kaggle dataset push runs from PowerShell and is a deliberate act priced against
# a mark; this only keeps `members_2/` current.
set -u
cd "$(dirname "$0")/.."

CKPT=D:/models/rsna_knee_abnormality/seat_oof5_distill_s2026.pt
ROSTER=comp/members2.roster

until [ -f "$CKPT" ]; do sleep 60; done
sleep 30   # the trainer writes then closes; do not read a half-flushed checkpoint

if grep -q "^seat_oof5_distill_s2026\.pt" "$ROSTER"; then
  echo "already rostered $(date -u +%FT%TZ)"
else
  cat >> "$ROSTER" <<'EOF'

# The leak removed by dropping the HOLDOUT rather than by weakening the teachers: fitted at --fold -1
# against teacher_oof5.npz, five REAL out-of-fold blocks covering all 4406 studies, each predicting only
# the fold it held out. No evaluation set is left for a memorised label to leak into, by construction —
# and by the same construction this member has NO offline number and can never have one. Its ruler is
# the board (kneemri-oof5solo, pre-registered against holdsolo's 0.928). Its held-out twin
# seat_clean_distill read 0.8634, matching the leaky 0.8633 out of teachers -0.0067 weaker, which is
# what says the term being shipped here is a treatment effect. knee-zwcyk
seat_oof5_distill_s2026.pt           # no offline number by construction, --fold -1
EOF
  echo "rostered $(date -u +%FT%TZ)"
fi

.venv/Scripts/python.exe -m kneemri.kaggle.package_members --roster "$ROSTER"
echo "PACKAGED seat_oof5_distill exit $? $(date -u +%FT%TZ)"
