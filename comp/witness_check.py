"""Re-derive a member's witness through the INFERENCE path: `python comp/witness_check.py`.

The fingerprint proves the weights are the weights. It pins image size and group and nothing else, so
it cannot prove that inference READS the model the way training did — and four defects walked through
that gap in a single session, every one of them loading cleanly and verifying cleanly:

  whole-stack reading served three slices at a time      (knee-y8y)
  the decode resolution taken from SlotReader's default  (knee-nwh)
  the slot list taken from a module constant, twice      (knee-gv3, knee-nwh)

A witness closes it by being an END-TO-END claim rather than a structural one. Training recorded what
it predicted for eight named studies; this runs those same studies through `Ensemble.predict_member`
and refuses the member if the numbers move.

WHAT IS AND IS NOT UNDER TEST. The witness pins the READING — decode resolution, slot scheme and
order, slice depth, pooling, and whether the whole stack is read. It deliberately does NOT pin the
test-time window policy: training scores one middle window while a submission may average several, and
that difference is a choice rather than a defect. So the check feeds exactly the middle window, which
makes both paths read the same pixels and leaves any remaining difference attributable to the reading.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.paths import DataRoot  # noqa: E402
from kneemri.backbone import ENCODERS  # noqa: E402
from kneemri.ensemble import Ensemble  # noqa: E402
from kneemri.members import Member, MemberPackage  # noqa: E402
from kneemri.study_cache import StudyCache  # noqa: E402

logger = logging.getLogger(__name__)

TOLERANCE = 2e-3
"""The fingerprint's tolerance, for the same reason: above GPU noise, far below any reading change."""


def check(package: MemberPackage, member: Member, root: Path, device: torch.device) -> str:
    """Re-derive one member's witness, returning a human-readable verdict."""
    witness = member.witness
    if not witness:
        return "UNVERIFIED - no witness (trained before the check existed, or a published member)"

    cache = StudyCache(root, slices=member.slices, image_size=member.image_size, scheme=member.scheme)
    corpus, identifiers = cache.load()
    index = {name: position for position, name in enumerate(identifiers)}
    studies = [str(name) for name in witness["studies"]]  # pyright: ignore[reportGeneralTypeIssues]
    missing = [name for name in studies if name not in index]
    if missing:
        return f"UNVERIFIED - {len(missing)} witnessed studies are absent from this cache"

    positions = np.array([index[name] for name in studies], dtype=np.int64)
    # The MIDDLE window, exactly what Trainer.predict reads: a whole-stack member wants its whole
    # stack, a windowed one wants one group. Feeding precisely that makes predict_member take a single
    # pass in both cases, so the comparison is about the READING and not about window averaging.
    width = member.slices if member.whole_stack else member.group
    start = (corpus.slices - width) // 2
    images = torch.from_numpy(np.ascontiguousarray(corpus.images[positions, :, start : start + width])).to(device)
    mask = torch.from_numpy(corpus.mask[positions]).to(device)

    encoder = ENCODERS[member.variant]
    model = package.load(member, AutoModel.from_pretrained(encoder.repository), encoder.dim, device)
    ensemble = Ensemble(package, {member.identifier: model}, device)
    got = ensemble.predict_member(member, images, mask, overlap=True)
    expected = np.asarray(witness["predictions"], dtype=np.float64)  # pyright: ignore[reportGeneralTypeIssues]

    difference = float(np.abs(got - expected).max())
    if difference > TOLERANCE:
        return f"REFUSED - inference reads this member differently than training did (max|diff| {difference:.3g})"
    return f"ok (max|diff| {difference:.3g} over {len(studies)} studies)"


def main() -> None:
    """Check every member's witness in a packaged directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("comp/members"))
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    package = MemberPackage(arguments.root)
    root = DataRoot.from_config(arguments.paths).processed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    verdicts = {member.identifier: check(package, member, root, device) for member in package.members}
    for identifier, verdict in verdicts.items():
        logger.info("  %s: %s", identifier, verdict)

    refused = [name for name, verdict in verdicts.items() if verdict.startswith("REFUSED")]
    unverified = [name for name, verdict in verdicts.items() if verdict.startswith("UNVERIFIED")]
    logger.info(
        "%d ok, %d unverified, %d refused",
        len(verdicts) - len(refused) - len(unverified),
        len(unverified),
        len(refused),
    )
    if refused:
        sys.exit(1)


if __name__ == "__main__":
    main()
