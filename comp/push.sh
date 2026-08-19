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
# 'kneemri.kaggle.submission'` because the members dataset was pushed while the kit lagged the repo. The gate
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

# THE TARGET IS RESOLVED BEFORE THE cd, because the cd below moves to comp/ and a caller naturally
# spells the path from the repo root. `comp/kernels/kneemri-crop` then reached the CLI as
# comp/comp/kernels/… -> `Invalid folder` (2026-08-17). The dataset branch had survived the same bug by
# accident: it takes `dirname` of the target, so the failed `cd comp` left it in comp/ and the basename
# happened to resolve. An absolute path is unchanged by this.
TARGET="$(cd "$(dirname "$TARGET")" 2>/dev/null && pwd)/$(basename "$TARGET")"
cd "$(dirname "${BASH_SOURCE[0]}")"

# NOT UNDER WSL, and this is a destructive failure rather than a wrong answer. `bash` on the Windows
# PATH is WSL's bash, not Git Bash, so `bash comp/push.sh ...` from PowerShell runs the LINUX uv against
# the Windows checkout mounted at /mnt/d — and `uv run` then rebuilt .venv with manylinux wheels,
# deleting the Windows environment on the way (2026-08-17: torch-2.11.0+cu128-cp314-manylinux failed
# half-installed and left .venv holding lib64, share and pyvenv.cfg; a full `uv sync` rebuilt it).
# Refuse before uv is resolved, because resolving it is already the damage.
case "$(pwd)" in
  /mnt/*)
    echo "REFUSING to push: running under WSL against a Windows checkout. Use Git Bash:"
    echo "  '/c/Program Files/Git/usr/bin/bash.exe' comp/push.sh ..."
    exit 5
    ;;
esac

# uv IS RESOLVED, NOT ASSUMED. Git Bash builds its own PATH from a login profile, so `uv` is absent
# there even though PowerShell finds it — and `uv run` then exits 127, which the stale-kit check below
# read as "the kit differs from the repo". That is a WRONG DIAGNOSIS on a missing binary, the same
# shape as bare `nox` and as the CLI returning 0 on a failed push: whoever hit it would have rebuilt a
# kit that was already correct. Resolve it once, and say which failure it actually is.
UV="$(command -v uv || true)"
[ -n "$UV" ] || for candidate in "$HOME/.local/bin/uv" "$HOME/.local/bin/uv.exe" "${LOCALAPPDATA:-}/Programs/uv/uv.exe"; do
  [ -x "$candidate" ] && UV="$candidate" && break
done
[ -n "$UV" ] || {
  echo "REFUSING to push: uv is not on PATH and is not where it installs by default."
  exit 4
}

# BEFORE authentication and before any upload: a stale kit then fails in a second, without a token,
# rather than after a credential check and half a dataset.
"$UV" run python verify_kit.py
case "$?" in
  0) ;;
  127)
    echo "REFUSING to push: '$UV run' could not start. This is NOT a stale kit — nothing was checked."
    exit 4
    ;;
  *)
    echo "REFUSING to push: the kit differs from the repo. Run build_kit.py --wheels <dir> first."
    exit 1
    ;;
esac

# THE CLI RETURNS 0 ON A FAILED PUSH. Measured 2026-08-17: `kaggle kernels push` printed `Kernel push
# error: Maximum batch GPU session count of 2 reached.` and exited 0. A caller that trusts the exit
# code then submits a STALE kernel version and spends one of five daily slots on code Kaggle never
# received. The CLI's own source (kaggle_api_extended.py:6467-6499, 5399-5408) prints the outcome and
# swallows it, so the outcome is what has to be read — this is the same shape as bare `nox` reporting
# the echo's status, and it is the third time in this repo that an exit code has lied.
confirmed() {
  local marker="$1"
  shift
  local output
  output="$("$@" 2>&1)"
  echo "$output"
  case "$output" in
    *"$marker"*) ;;
    *)
      echo "REFUSING to report success: Kaggle never said '$marker'. The push did NOT land — do not submit."
      exit 3
      ;;
  esac
}

case "$WHAT" in
  dataset)
    # --dir-mode zip, and from the target's PARENT with a bare directory name: the CLI pastes whatever
    # `-p` says into its upload-temp path, so any path with a separator in it — `comp/kit`,
    # `../../../data/.../kit`, or an absolute `D:/…` — becomes a temp filename with directories that do
    # not exist, and every file fails with `[Errno 2] ... .kaggle/uploads\D_/data/...json`.
    PARENT="$(cd "$(dirname "$TARGET")" && pwd)"
    # --project, because `uv run` resolves its environment from the CWD and the CWD is now the kit's
    # parent — out of the repo, where there is no pyproject.toml to resolve.
    REPO="$(cd .. && pwd)"
    (cd "$PARENT" && confirmed "is being created" "$UV" run --project "$REPO" --with kaggle kaggle datasets version -p "$(basename "$TARGET")" -m "${MESSAGE:?a dataset version needs a message}" --dir-mode zip)
    ;;
  kernel)
    # THE SAME PARENT-AND-BASENAME SHAPE AS THE DATASET BRANCH, for a different reason. The resolution
    # above makes TARGET a POSIX absolute (`/d/personal_projects/...`) because that is what Git Bash's
    # `pwd` prints, and the Kaggle CLI is a WINDOWS python: it cannot open `/d/...` and answers `Invalid
    # folder` for a directory that exists (2026-08-19, kneemri-band, three spellings refused — a repo
    # relative path, a POSIX absolute and a `D:\` absolute all normalise to the same POSIX string here).
    # A bare directory name has no drive letter to disagree about.
    PARENT="$(cd "$(dirname "$TARGET")" && pwd)"
    REPO="$(cd .. && pwd)"
    (cd "$PARENT" && confirmed "successfully pushed" "$UV" run --project "$REPO" --with kaggle kaggle kernels push -p "$(basename "$TARGET")")
    ;;
  *)
    echo "unknown target '$WHAT' — expected dataset or kernel"
    exit 2
    ;;
esac
