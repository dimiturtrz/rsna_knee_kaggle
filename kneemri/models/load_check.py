"""Does every registry entry actually RECEIVE its pretrained weights?

`python -m kneemri.models.load_check`

MEASURED 2026-09-04. `RADIODINO_BASE` was loaded through `AutoModel.from_pretrained`, because its
`config.json` says `ViTModel` — and the weights beside that config are keyed the way timm keys them, so
transformers built the right architecture, matched not one key, and returned a ViT-B/16 with every
parameter at its random initialisation. `radiodino_corpus` then trained three epochs of it (0.6501,
0.6714, 0.6813) against a pre-registration about MEDICAL PRETRAINING whose lowest branch would have
closed the medical-image row of the grid and retired MRI-CORE. Sixteen checks guard what gets SHIPPED
and what gets QUOTED. None of them asked whether the weights arrived.

`refuse_unloaded` is the gate and it fires at load time, so nothing can descend a random trunk again.
This is the SWEEP: the gate answers for the entry a run happens to touch, and this answers for all of
them at once, which is the question a registry raises and a single run never does.

NOT IN THE MERGE GATE, and that is deliberate rather than an omission. Reaching the answer means
fetching every trunk in the registry — several GB — so a CI job would either download them each run or
quietly skip and report a pass. `registry_to_checkpoints` covers the part that costs one JSON per entry
(each one's declared normalization); this covers the part that costs the weights, and it is run by hand
when an entry is added or a builder changes, like the other checks whose subject is the card or the
board rather than the tree.

WIDTH IS CHECKED IN THE SAME PASS because it is the same class of defect and free once the trunk is in
memory: a registry `dim` that disagrees with the loaded trunk's own `hidden_size` builds a head at the
wrong width, and the entry says one number while the weights say another.
"""

import logging
from dataclasses import dataclass

from core import cli
from kneemri.models.backbone import ENCODERS, Encoder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrunkLoad:
    """What one registry entry's load produced, as two numbers and a message."""

    variant: str
    declared_dim: int
    width: int | None
    """The loaded trunk's own feature width, or None where the load never got that far."""

    failure: str = ""
    """What the loader said when it refused, empty where it did not."""

    @property
    def defect(self) -> str:
        """The one line naming what is wrong with this entry, empty where nothing is.

        A refusal and a width disagreement are reported as the same KIND of thing — the entry claims a
        model the weights are not — because the consequence is identical: a trunk that trains, logs a
        curve, and answers a question about something other than what the entry is named for.
        """
        if self.failure:
            return f"{self.variant}: {self.failure}"
        if self.width != self.declared_dim:
            return f"{self.variant}: entry declares dim {self.declared_dim}, the loaded trunk is {self.width} wide"
        return ""


@dataclass(frozen=True)
class LoadCensus:
    """Every registry entry's load, and the verdict over them."""

    loads: tuple[TrunkLoad, ...]

    @staticmethod
    def loaded(variant: str, encoder: Encoder) -> TrunkLoad:  # devtools-ignore: test-mirror
        """One entry, fetched and built. No mirror test: it downloads a checkpoint, which the `small`
        gate forbids in a unit test — the verdict it feeds IS tested, over `TrunkLoad` directly."""
        try:
            trunk = encoder.load()
        except (ValueError, RuntimeError, OSError) as refusal:
            return TrunkLoad(variant, encoder.dim, width=None, failure=str(refusal))
        width = getattr(trunk.config, "hidden_size", None)
        return TrunkLoad(variant, encoder.dim, width=width)

    @property
    def defects(self) -> tuple[str, ...]:
        """Every entry whose weights did not arrive as the entry describes them."""
        return tuple(line for load in self.loads if (line := load.defect))

    def report(self) -> int:
        """The census, and a nonzero status where any entry is not the model it is named for."""
        for load in self.loads:
            logger.info("  %-20s dim %4d  loaded %s", load.variant, load.declared_dim, load.width or "REFUSED")
        for defect in self.defects:
            logger.error("DEFECT %s", defect)
        if self.defects:
            return 1
        logger.info("OK: %d entries, every one built from its own weights", len(self.loads))
        return 0


def main() -> int:
    """Load every registry entry and refuse a registry holding one that is not what it says."""
    command = cli.Command(__doc__, logger)
    command.parsed()
    census = LoadCensus(tuple(LoadCensus.loaded(variant, encoder) for variant, encoder in sorted(ENCODERS.items())))
    return census.report()


if __name__ == "__main__":
    raise SystemExit(main())
