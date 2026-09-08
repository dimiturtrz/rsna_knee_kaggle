#!/usr/bin/env bash
# Waits out latent_mask_ladder.sh, then puts the REGISTER arm on the card in its place.
#
# Replaces after_ratelad.sh, whose successor (latent_rate_ladder.sh) lost its main argument the moment the
# mask ladder's epoch 1 produced both collapse shapes at one rate: the sink-versus-uniform split was the
# reason to walk the rate, and it turned out not to be a rate signature. latent_register.sh asks a bigger
# question for the same card time — the trunk every run in this family adapted has no register tokens at all.
#
# The gate is the ARM's OWN interpreter, not a marker line: `latent_mask_ladder.sh` ends either by its rung
# finishing the fine-tune or by `require` failing, and only one of those writes anything. `.venv-arm` is used
# by no other launcher on this machine, so its absence from the process table is the one signal that means
# "the ladder is off the card" under both. `ps -W` is required: MSYS pids are a different number space and
# `kill -0` reads a live arm as dead.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1

# EITHER SLASH, and the character class is the whole point. `ps -W` prints WINDOWS argv, so a job shows
# as `D:\...\.venv-arm\Scripts\python.exe` — but an arm calling `"$REPO/.venv-arm/Scripts/python.exe"`
# itself shows FORWARD slashes, because $REPO is spelled that way. So the two halves of one arm are
# spelled differently in the process table: the pretrain matches a forward-slash pattern and the
# `run_train` fine-tune, launched from a snapshot, does not.
#
# Measured 2026-09-08: this loop waited out all eight pretrain epochs correctly and then released the
# instant the 12-epoch fine-tune started, because that half is backslashed. It read as "the ladder is off
# the card" with the ladder's second half just beginning. Nothing was lost only because `run_pretrain`
# now takes the card lock, so the register arm parked behind it instead of stacking on it — the lock is
# what makes a premature arm survivable, not this gate.
while ps -W 2>/dev/null | grep -qE 'venv-arm[\\/]Scripts[\\/]python'; do
  sleep 30
done

echo "mask ladder is off the card at $(date -u +%Y-%m-%dT%H:%M:%SZ); arming the register arm"

UV_NO_SYNC=1 uv run python -m kneemri.experiments.card_check || {
  echo "card_check refused — not arming"
  exit 1
}

nohup bash experiments/arms/latent_register.sh > logs/latent_register.log 2>&1 &
disown
echo "register arm armed as pid $!"
