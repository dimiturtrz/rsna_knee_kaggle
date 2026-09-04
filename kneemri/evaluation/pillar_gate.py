"""The plumbing oracle for the Pillar-0 member: `python -m kneemri.evaluation.pillar_gate`.

WHY IT IS NOT A FLAG ON `foreign_gate`. The reading, the AUC and the report are shared and imported from
there; what differs is the BUILDER — a volumetric trunk fed whole series has no read size, so `--res`
has nothing to say to it, and a `--family` switch inside one `main` would be an `if` whose two branches
disagree about their own arguments. Two commands over one `FeedGate` is the split that keeps the scoring
identical while letting each family state its own feed.

WHY THIS ORACLE MATTERS MORE HERE THAN IT DID FOR RAPTOR. Their adapter reads a full series at a real
depth; our cache holds twelve slices per slot, so the depth axis handed to a trunk that expects a
resolved volume is mostly interpolation. That is a feed shortfall this project has no prior for, and a
909-study read costs card hours before it would surface. The 58 annotated studies are also the ONLY
honest ruler available: `foreign_overlap` records that our corpus minus exactly those 58 is what these
checkpoints train on, so anything else this member scores is in-sample.

AND IT IS TWO ORACLES AT ONCE. The archive publishes a per-label profile as well as the macro, with
Effusion highest and MCL lowest. `PILLAR_LABELS` is transcribed from that table and a permuted head
loads perfectly cleanly, so the per-finding rows below are what confirms or refutes the transcription —
at no extra cost, since the same forward produces both.
"""

import logging
from pathlib import Path

import torch

from core import cli
from kneemri.evaluation.foreign_gate import FeedGate
from kneemri.evaluation.supervision_gap import SupervisionGap
from kneemri.inference.pillar_member import GOLD, PillarMember, VolumeFeed

logger = logging.getLogger(__name__)


def main() -> None:
    """Score the Pillar-0 checkpoint through our feed and set it beside the AUC its archive publishes."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--checkpoint", type=Path, required=True, help="the published pillar0 .pt")
    command.parser.add_argument("--cache", type=Path, required=True, help="the processed directory")
    command.parser.add_argument("--key", required=True, help="the cache key, e.g. 336px_12sl")
    command.parser.add_argument("--train", type=Path, required=True, help="train.csv, which carries the gold rows")
    command.parser.add_argument("--device", default="cuda")
    command.parser.add_argument("--out", type=Path, default=None, help="npz of the probabilities, for blending")
    arguments = command.parsed()

    member = PillarMember.rebuild(arguments.checkpoint)
    feed = VolumeFeed.of(arguments.cache, arguments.key)
    identifiers, annotations = SupervisionGap().annotated(arguments.train)
    positions = {name: index for index, name in enumerate(VolumeFeed.order(arguments.cache, arguments.key))}
    rows = [positions[name] for name in identifiers]

    gate = FeedGate(member, feed, torch.device(arguments.device), GOLD)
    title = f"pillar0 through OUR {arguments.key} feed, {len(rows)} annotated studies"
    reading = gate.reading(rows, annotations)
    if arguments.out is not None:
        reading.save(arguments.out, identifiers)
    command.report(reading.lines(title))


if __name__ == "__main__":
    main()
