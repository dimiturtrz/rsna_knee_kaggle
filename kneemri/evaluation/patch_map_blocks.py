"""Join the per-block sweeps of one split into the single map file a run erases from.

A sweep of the train split runs in blocks because the subset copy of one block is already tens of GB,
while `PatchErasing.loaded` takes ONE path. The join is by identifier, never by position: each block
holds its own studies in its own order.
"""

import logging
from pathlib import Path

import numpy as np

from core import cli

logger = logging.getLogger(__name__)

STUDIES = "studies"
MAPS = "maps"
TRUTH = "truth"
TISSUE = "tissue"


class PatchMapBlocks:
    """The block files of one sweep, and the single file their maps concatenate into."""

    def __init__(self, blocks: list[Path]) -> None:
        self.blocks = blocks

    def joined(self) -> dict[str, np.ndarray]:
        """Every block's studies and maps in block order, refusing a study two blocks both claim.

        A duplicate is a sweep whose offsets overlapped, which would give one study two rows and leave
        the join ambiguous at load time — silent, and it would erase the wrong cells for that study.
        """
        held = [np.load(block) for block in self.blocks]
        names = np.concatenate([block[STUDIES] for block in held])
        if len(set(names.tolist())) != len(names):
            message = f"{len(names) - len(set(names.tolist()))} studies appear in more than one block"
            raise ValueError(message)
        return {
            STUDIES: names,
            MAPS: np.concatenate([block[MAPS] for block in held]),
            TRUTH: np.concatenate([block[TRUTH] for block in held]),
            TISSUE: held[0][TISSUE],
        }


def main() -> None:
    """Join a sweep's block files into one map file."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--blocks", type=Path, nargs="+", required=True)
    command.parser.add_argument("--out", type=Path, required=True)
    arguments = command.parsed()

    joined = PatchMapBlocks(sorted(arguments.blocks)).joined()
    destination: Path = arguments.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        studies=joined[STUDIES],
        maps=joined[MAPS],
        truth=joined[TRUTH],
        tissue=joined[TISSUE],
    )
    logger.info("wrote %s: %d studies, maps %s", destination, len(joined[STUDIES]), joined[MAPS].shape)


if __name__ == "__main__":
    main()
