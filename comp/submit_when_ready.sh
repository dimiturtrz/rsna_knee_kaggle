#!/usr/bin/env bash
# submit_when_ready.sh <kernel-slug> <version> "<message>" — submit a code entry once its kernel run
# finishes, instead of watching it.
#
# A code submission is not a file upload: Kaggle re-runs the notebook against the hidden test set, so
# the gap between pushing a kernel and having something to submit is tens of minutes. That is dead
# time to a human and none to a loop.
#
# TWO THINGS THIS ENCODES, both learned by paying for them:
#   - the submit call needs an explicit `-v <version>`. Without it, CreateCodeSubmission answers 403
#     rather than naming the missing field, which reads exactly like a permissions problem.
#   - Kaggle runs at most TWO GPU sessions at once, so a third kernel cannot even be pushed while two
#     are live. `--push-first` waits for a free slot and pushes, then waits for that run.
set -u

SLUG="${1:?usage: submit_when_ready.sh <owner/kernel> <version> <message> [--push-first <path>]}"
VERSION="${2:?version}"
MESSAGE="${3:?message}"
PUSH_PATH="${4:-}"
COMP=rsna-knee-abnormality-detection
cd "$(dirname "${BASH_SOURCE[0]}")"

kaggle() { uv run --with kaggle kaggle "$@"; }

if [ -n "$PUSH_PATH" ]; then
  echo "waiting for a free GPU session before pushing $PUSH_PATH"
  until kaggle kernels push -p "$PUSH_PATH" 2>&1 | tee /dev/stderr | grep -q "successfully pushed"; do
    sleep 120
  done
fi

echo "waiting for $SLUG to finish"
until kaggle kernels status "$SLUG" 2>&1 | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 120; done
status="$(kaggle kernels status "$SLUG" 2>&1)"
echo "$status"
case "$status" in
  *COMPLETE*) ;;
  *) echo "REFUSING to submit: $SLUG did not complete"; exit 1 ;;
esac

kaggle competitions submit "$COMP" -k "$SLUG" -v "$VERSION" -f submission.csv -m "$MESSAGE"
