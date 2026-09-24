"""Average the WEIGHTS of runs that share an architecture: `python -m kneemri.models.weight_soup`.

Eighty-four fold-0 checkpoints on disk carry byte-identical architecture options — `small` / `cls_mean` /
`max` / `attend` / 336 px / 20 slices / whole stack / six unfrozen blocks / band 0.1-0.9 — the corner
`calib_dense_hold` sits at, and they all descend from the one `facebook/dinov2-small` initialisation.
That is Wortsman's model-soup setting exactly: runs that share an init and differ only in recipe tend to
land in one basin, so their weight-space average is a valid network rather than noise, and the published
effect is around +0.5 to +1.0 absolute over the best single member.

WHAT IS NOT THIS. `experiments/arms/weight_average.sh` averaged the weights of ONE run across ITS OWN
epochs (an EMA) and read +0.0003. Every "soup" in the ledger is a LOGIT average: members predict, and the
predictions are mixed. Neither one has ever averaged the PARAMETERS of two independently fitted runs, and
the two mechanisms are unrelated — a logit average keeps N forward passes at serve time and cannot move a
SOLO number, while this produces one network with one forward pass, which is the only class of change
that has ever moved this board by more than noise.

IT REFUSES RATHER THAN RESHAPES. Members must agree on the architecture options, on the split that
defines the holdout (folds, split seed, fold index), on the table they were SCORED against, and on the
state dict itself key for key and shape for shape. Any of those differing makes the average a different
network read by a different ruler, which is the arithmetic-on-two-questions failure this project has
already paid for once; and a `fold -1` member is refused outright because a full-data fit saw the
holdout, so one of them in the pot silently leaks the number the soup is about to be read on.

THAT ONE REFUSAL HAS A RELEASE, AND ONLY THAT ONE. `ship_only` averages full-data members, because the
leak it guards is a leak INTO A NUMBER and an all-full-data soup has no such number: there is no holdout
for it, `holdout_backfill` has nothing to write and `ledger75` no row to price, so the only ruler it can
ever be read by is the board. A MIX stays refused and needs no extra code to stay refused — `fold` sits in
`SPLIT`, so members disagreeing on it already fail the signature, and the mixed pot is the case the guard
was actually written for. The flag therefore widens nothing except the deliberate all-`-1` ship.

THE SOUP CARRIES ITS FIRST MEMBER'S RECIPE AND NONE OF ITS CLAIMS. `holdout` and `epoch` are cleared:
the average never ran an epoch and has no score until something reads it. The weight FINGERPRINT is
dropped rather than copied, because it is a witness of a specific set of tensors and copying it forward
would let `witness_check` pass a checkpoint whose weights it never saw. `package_members` is the only
consumer that indexes it, so a soup that earns a seat is re-fingerprinted then, deliberately.

IT IS A RUN OF ITS OWN, AND IT NAMES ITS PARENTS. Copying the first member's payload used to copy its
`run` stamp too, so every soup claimed to BE its first member; its parents lived only in arm scripts and
ledger prose. It now mints its own identity with every member's run id as a parent, in the checkpoint and
in a record beside it — a record with a recipe and a run line and no epochs, the same shape a run killed
before its first epoch leaves.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

from core import cli
from kneemri.runs.holdout_record import UNKNOWN_EPOCH
from kneemri.runs.run_identity import RunIdentity
from kneemri.runs.run_metrics import RunMetrics

logger = logging.getLogger(__name__)

ARCHITECTURE = (
    "variant",
    "pool",
    "slice_pooling",
    "slot_pooling",
    "image_size",
    "slices",
    "whole_stack",
    "unfreeze_last",
    "band_low",
    "band_high",
    "encoder_layer",
    "fuse_blocks",
    "lora_rank",
)
"""The options that decide what the tensors MEAN, so two checkpoints agreeing on all of them can be added.

Twelve keys rather than the whole option record, because the record also holds the rate, the epochs, the
label table and forty other things that a soup is deliberately averaging ACROSS — those are the axis, not
the mismatch."""

SPLIT = ("folds", "split_seed", "fold")
"""What decides WHICH studies were held out. A soup of members split differently has no single holdout."""

FULL_DATA_FOLD = "-1"
"""A fit that held nothing out. One in the pot leaks the holdout into every number the soup then reads."""

MINIMUM_MEMBERS = 2
"""One checkpoint averaged with itself is that checkpoint; the refusal says so rather than writing a copy."""

DROPPED = ("fingerprint", "fingerprint_config")
"""Claims about weights that no longer exist. Dropped, never carried."""

WITNESS = "witness"
"""The end-to-end claim, nested inside the recipe rather than beside it — and dropped for the same reason.

A witness is what TRAINING predicted for eight named studies, so it belongs to one set of tensors exactly
as the fingerprint does. Carrying the first member's forward made every soup REFUSED by `witness_check`
at 0.10-0.15 against a 0.0312 bound, which is inside the 0.05-0.5 band that check documents as the
signature of a real reading defect — a false alarm wearing the costume of the one failure it exists to
catch. Measured on `basinsoup_k3_s2026`: its witness was byte-identical to `basinsoup_c101`'s and 0.2497
and 0.2776 from the other two children, so the distance being reported was the average differing from one
child, which is what souping is for.

A soup is therefore UNVERIFIED and never refused, and that is the honest state rather than a gap. A
witness has to be recorded by the run that produced the weights; minting one for the average through the
inference path would compare that path against itself and pass by construction."""


class IncompatibleSoupError(ValueError):
    """Raised when checkpoints cannot be averaged, naming the key and the two values that disagree."""


class WeightSoup:
    """A uniform weight-space average of independently fitted runs that share an architecture."""

    def __init__(self, members: Sequence[Path], *, ship_only: bool = False) -> None:
        if len(members) < MINIMUM_MEMBERS:
            message = f"a soup needs at least {MINIMUM_MEMBERS} members, got {len(members)}"
            raise IncompatibleSoupError(message)
        self.members = list(members)
        self.ship_only = ship_only
        self.payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in self.members]

    def signature(self, payload: dict[str, object]) -> tuple[str, ...]:
        """The architecture, split and scoring ruler one checkpoint claims, as comparable strings.

        The ruler is `eval_labels` when the run named one and its training table otherwise — the same
        precedence `holdout_backfill` uses, because a soup read against a table none of its members were
        scored on is a number about a different question.
        """
        options = payload["options"]
        if not isinstance(options, dict):
            message = "checkpoint records no options, so nothing can be asserted about its architecture"
            raise IncompatibleSoupError(message)
        evaluation = str(options.get("eval_labels"))
        ruler = evaluation if evaluation not in {"None", ""} else str(options.get("labels"))
        return (*(str(options.get(key)) for key in (*ARCHITECTURE, *SPLIT)), Path(ruler).name)

    def agreed(self) -> tuple[str, ...]:
        """The one signature every member shares, or a refusal naming the first member that differs."""
        first = self.signature(self.payloads[0])
        if first[len(ARCHITECTURE) + SPLIT.index("fold")] == FULL_DATA_FOLD and not self.ship_only:
            message = f"{self.members[0].stem} is a full-data fit and has already seen the holdout"
            raise IncompatibleSoupError(message)
        for path, payload in zip(self.members[1:], self.payloads[1:], strict=True):
            other = self.signature(payload)
            if other != first:
                named = zip((*ARCHITECTURE, *SPLIT, "ruler"), first, other, strict=True)
                differing = [key for key, mine, theirs in named if mine != theirs]
                message = f"{path.stem} disagrees with {self.members[0].stem} on {differing}"
                raise IncompatibleSoupError(message)
        return first

    def averaged(self) -> dict[str, Tensor]:
        """The mean of every tensor, key by key, over members already asserted compatible."""
        self.agreed()
        states = [payload["model"] for payload in self.payloads]
        first = states[0]
        for path, state in zip(self.members[1:], states[1:], strict=True):
            if set(state) != set(first):
                missing = sorted(set(first) ^ set(state))[:4]
                message = f"{path.stem} has a different state dict: {missing}"
                raise IncompatibleSoupError(message)
            shapes = [key for key in first if state[key].shape != first[key].shape]
            if shapes:
                message = f"{path.stem} disagrees on the shape of {shapes[:4]}"
                raise IncompatibleSoupError(message)
        return {
            key: torch.stack([state[key].to(torch.float32) for state in states]).mean(0).to(first[key].dtype)
            for key in first
        }

    def write(self, out: Path) -> Path:
        """The soup beside its members: the first member's recipe, averaged weights, and no borrowed claims."""
        payload = dict(self.payloads[0])
        payload["model"] = self.averaged()
        payload["holdout"] = float("nan")
        payload["epoch"] = UNKNOWN_EPOCH
        payload["soup"] = [path.stem for path in self.members]
        parents = {f"member{index}": str(path) for index, path in enumerate(self.members)}
        payload["run"] = RunIdentity.begun(parents, out, resuming=False).stamp()
        RunMetrics(out).begin(parents, run=payload["run"])
        for key in DROPPED:
            payload.pop(key, None)
        recipe = payload.get("config")
        payload["config"] = (
            {key: value for key, value in recipe.items() if key != WITNESS} if isinstance(recipe, dict) else recipe
        )
        torch.save(payload, out)
        return out


def main() -> None:
    """Build one soup: `python -m kneemri.models.weight_soup --members a.pt b.pt --out soup.pt`."""
    command = cli.Command(WeightSoup.__doc__, logger)
    command.parser.add_argument("--members", type=Path, nargs="+", required=True)
    command.parser.add_argument("--out", type=Path, required=True)
    command.parser.add_argument("--ship-only", action="store_true")
    arguments = command.parsed()
    soup = WeightSoup(arguments.members, ship_only=arguments.ship_only)
    logger.info("souping %d members sharing %s", len(arguments.members), soup.agreed())
    logger.info("wrote %s", soup.write(arguments.out))


if __name__ == "__main__":
    main()
