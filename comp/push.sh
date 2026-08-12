#!/usr/bin/env bash
# push.sh dataset|kernel <path> [message] — the ONE way anything reaches Kaggle from here.
#
# WHY THIS EXISTS. There were two publish paths and only one of them had a gate. `publish.py` refuses
# to push while the kit's bundled source differs from the repo (`_refuse_if_stale`), which is the
# README's "rebuild the kit whenever kneemri/ changes" turned into an actual check. The official CLI —
# the path we have used for every real push since the bearer token expired on 2026-07-23 — had no such
# check at all. So the guarded path was the one nobody ran.
#
# That is not hypothetical: a kernel run was lost to `ModuleNotFoundError: No module named
# 'kneemri.submission'` because the members dataset was pushed while the kit lagged the repo. The gate
# was written after that, onto the transport that had already stopped being used.
#
# The gate covers EVERY subcommand rather than only the kit's own push, and for the same reason
# publish.py did it in `main`: a kernel imports the kit, and a members version is loaded by code from
# the kit, so anything reaching Kaggle while the kit lags the repo ships a code path nobody is looking
# at.
set -u

WHAT="${1:?usage: push.sh dataset|kernel <path> [message]}"
TARGET="${2:?path}"
MESSAGE="${3:-}"
cd "$(dirname "${BASH_SOURCE[0]}")"

# BEFORE authentication and before any upload: a stale kit then fails in a second, without a token,
# rather than after a credential check and half a dataset.
uv run python verify_kit.py || {
  echo "REFUSING to push: the kit differs from the repo. Run build_kit.py --wheels <dir> first."
  exit 1
}

case "$WHAT" in
  dataset)
    # --dir-mode zip, and from INSIDE comp/: `-p comp/<dir>` mangles the temp path on Windows into
    # `comp/<dir>_manifest.json.json`.
    uv run --with kaggle kaggle datasets version -p "$TARGET" -m "${MESSAGE:?a dataset version needs a message}" --dir-mode zip
    ;;
  kernel)
    uv run --with kaggle kaggle kernels push -p "$TARGET"
    ;;
  *)
    echo "unknown target '$WHAT' — expected dataset or kernel"
    exit 2
    ;;
esac
