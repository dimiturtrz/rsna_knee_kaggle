#!/usr/bin/env bash
# Everything a submission needs, up to but NOT including anything outward-facing.
#
# Packaging and verification are reversible and get automated. Pushing a dataset, pushing a kernel and
# spending a submission are not, so this prints those commands rather than running them — the last
# step before a slot is spent should be a human reading what is about to happen.
#
#   bash comp/prepare_submission.sh 'wholestack_fold*.pt'
#
# It fails rather than warns if the kit has drifted, because a kernel imports the kit and a stale kit
# has already cost one run to ModuleNotFoundError.
set -u

PATTERN="${1:?usage: prepare_submission.sh '<checkpoint glob>'}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

echo "=== 1. package the members ==="
uv run python -m kneemri.package_members --out comp/members --pattern "$PATTERN"

echo
echo "=== 2. verify every member loads and fingerprints under its OWN reading ==="
# The check that matters: a member fitted with --slice-pooling max and rebuilt with the default
# channel stacking differs from its stored fingerprint by 2.25 against a 2e-3 tolerance. That is a
# different function on different pixels, not numerical drift.
uv run python comp/verify_members.py --root comp/members || {
  echo "MEMBERS FAILED VERIFICATION - do not submit"
  exit 1
}

echo
echo "=== 3. kit must match the repo (a kernel imports it) ==="
uv run python comp/build_kit.py --wheels "${WHEELS:-$TEMP/kneewheels}"
uv run python comp/verify_kit.py || { echo "KIT STALE - do not submit"; exit 1; }

echo
echo "=== READY. The outward-facing steps are yours to run: ==="
echo "  bash comp/push.sh dataset members '<what changed>'"
echo "  bash comp/push.sh kernel  kernels/<kernel>"
echo "  then run the kernel on Kaggle and submit its output"
