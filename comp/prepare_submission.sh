#!/usr/bin/env bash
# Everything a submission needs, up to but NOT including anything outward-facing.
#
# Packaging and verification are reversible and get automated. Pushing a dataset, pushing a kernel and
# spending a submission are not, so this prints those commands rather than running them — the last
# step before a slot is spent should be a human reading what is about to happen.
#
#   bash comp/prepare_submission.sh [roster]
#
# WHICH members it packages comes from `comp/members.roster` — a committed file — rather than from a
# glob typed here. The glob used to live in shell history while the package it produced did not, and
# two local member directories ended up stamping one live dataset id with three members and four while
# the live package held six.
#
# It fails rather than warns if the kit has drifted, because a kernel imports the kit and a stale kit
# has already cost one run to ModuleNotFoundError.
set -u

ROSTER="${1:-comp/members.roster}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ONE HOME, RESOLVED ONCE. This script used to spell `--out comp/members` and hand the same path to
# both checks, which is how the repo grew a second directory stamping the live dataset id: the package
# that ships lives under the processed root, and the in-repo copy aged behind it until `mount_check`'s
# default invocation refused a valid kernel (knee-n0w0). Every path below comes from this one line.
PACKAGE="$(uv run python -c "from kneemri.inference.members import MemberPackage; print(MemberPackage.shipped_root())")"

echo "=== 1. package the members named by $ROSTER into $PACKAGE ==="
uv run python -m kneemri.kaggle.package_members --roster "$ROSTER"

echo
echo "=== 2. verify every member loads and fingerprints under its OWN reading ==="
# The check that matters: a member fitted with --slice-pooling max and rebuilt with the default
# channel stacking differs from its stored fingerprint by 2.25 against a 2e-3 tolerance. That is a
# different function on different pixels, not numerical drift.
uv run python comp/verify_members.py || {
  echo "MEMBERS FAILED VERIFICATION - do not submit"
  exit 1
}

echo
echo "=== 2b. does INFERENCE read each member the way training did? ==="
# The fingerprint pins image size and group only, so it cannot see a consumer that reads a default
# instead of the member's own config - which is how whole-stack members were served three slices at a
# time all the way onto the leaderboard. This re-derives each member's witness through the inference
# path. UNVERIFIED is not a failure: members trained before the witness existed have nothing to check.
# --device cpu NAMED, not defaulted. The check has two tolerance regimes and this line used to pick
# one by accident: CUDA compares under the member's own autocast dtype against a flat 2e-3, CPU runs
# float32 and compares against 4x the WITNESS dtype's eps. The flat constant is 2 eps for a float16
# member and 0.26 eps for a bfloat16 one, so on 2026-08-19 it refused 7 of 12 members and every
# refusal was bf16 - two independent bf16 roundings (the witness's, and a re-derivation batched
# differently) cannot land 0.26 eps apart. The same twelve on CPU: 10 ok, 2 unverified, 0 refused,
# the widest at half its bound. So the gate names the regime whose bound is derived from the
# arithmetic rather than the one whose bound was written for float16 and applied to both. Calibrating
# the CUDA constant is knee-0nm1; it stays a constant until then, because a bound raised to admit an
# unexplained 0.019 is the gate switching itself off.
uv run python comp/witness_check.py --device cpu || {
  echo "A MEMBER IS READ DIFFERENTLY BY INFERENCE THAN BY TRAINING - do not submit"
  exit 1
}

echo
echo "=== 3. kit must match the repo (a kernel imports it) ==="
# NO --wheels: build_kit's own default is the processed root's wheels/, which is the only directory
# that holds all four. This line used to name a scratch dir under $TEMP that held two of them, and a
# kit built from it loses transformers and safetensors — the exact failure build_kit's wheel check was
# written for on 2026-08-17, re-introduced here by the flag that was supposed to make it configurable.
uv run python comp/build_kit.py
uv run python comp/verify_kit.py || { echo "KIT STALE - do not submit"; exit 1; }

echo
echo "=== READY. The outward-facing steps are yours to run: ==="
echo "  bash comp/push.sh dataset '$PACKAGE' '<what changed>'"
echo "  bash comp/push.sh kernel  kernels/<kernel>"
echo "  then run the kernel on Kaggle and submit its output"
