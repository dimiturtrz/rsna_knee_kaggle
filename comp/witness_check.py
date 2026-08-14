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

from core.paths import DataRoot
from kneemri.backbone import ENCODERS
from kneemri.ensemble import Ensemble
from kneemri.members import Member, MemberPackage
from kneemri.study_cache import StudyCache

logger = logging.getLogger(__name__)

MATCHED_TOLERANCE = 2e-3
"""The bound when both paths run the member's own arithmetic — the fingerprint's, for its reason."""

CROSS_PRECISION_FACTOR = 4.0
"""How many epsilons apart a half-precision witness and a float32 re-derivation legitimately land.

MEASURED 2026-08-13, four members re-derived on CPU (float32) against their recorded witnesses:

    schedule_8_s2026      float16   1.59e-3   1.6x eps
    intensity392_ws_s2026 float16   1.62e-3   1.7x eps
    sliceattn_s2026       float16   2.34e-3   2.4x eps
    extended_ws_s2026     bfloat16  1.56e-2   2.0x eps

The residual tracks the WITNESS's epsilon across two dtypes eight times apart, at 1.6-2.4x of it —
which is why this is a factor on eps rather than a constant. One eps was the wrong model: eps is the
spacing of a single rounding and a twelve-block transformer accumulates thousands. Four leaves margin
over the widest observed without reaching the 0.05-0.5 that every real reading defect produced.
"""


def tolerance(precision: torch.dtype, device: torch.device) -> float:
    """How far the two paths may legitimately differ, from the ARITHMETIC rather than picked.

    This was a flat 2e-3, inherited from the fingerprint and documented as "above GPU noise" — true
    when training ran in float32 or float16, and false the day bf16 landed. bfloat16 keeps float32's
    exponent and spends only seven bits on the mantissa, so its eps is 2^-7 = 7.8e-3: FOUR TIMES the
    gate. The check refused the first bf16 member it ever saw, correctly by its own rule and wrongly in
    fact, and a gate that cries wolf is a gate someone switches off before a real refusal arrives.

    TWO REGIMES, because the device decides which. On cuda the re-derivation runs under the member's
    own autocast dtype, so the arithmetic MATCHES and any remaining difference is kernel
    nondeterminism. On cpu autocast is off entirely, so a half-precision witness is being compared
    against float32 and the gap is the dtype's, not the reading's — a weaker check, and worth having
    because a gate that needs a free GPU is a gate that gets skipped exactly when a member is being
    packaged.

    IT STILL DISCRIMINATES, which is the only thing that makes a loosened bound acceptable. The four
    reading defects this check exists for moved predictions by 0.05 to 0.5, because feeding a model a
    different slot list or a quarter of its stack changes what it sees rather than how precisely it
    adds. Those also CONCENTRATE in some labels while sparing others, where this residual moved all
    twelve about equally.
    """
    if device.type == "cuda":
        return MATCHED_TOLERANCE
    return CROSS_PRECISION_FACTOR * float(torch.finfo(precision).eps)


def check(package: MemberPackage, member: Member, root: Path, device: torch.device) -> str:
    """Re-derive one member's witness, returning a human-readable verdict."""
    witness = member.witness
    if not witness:
        return "UNVERIFIED - no witness (trained before the check existed, or a published member)"

    cache = StudyCache(root, slices=member.slices, image_size=member.image_size, routing=member.routing)
    corpus, identifiers = cache.load()
    index = {name: position for position, name in enumerate(identifiers)}
    # `witness` is a dict[str, object] by declaration, so its members are narrowed before use rather
    # than suppressed per call site. A malformed witness is UNVERIFIED and never an empty study list:
    # that would have reported "ok over 0 studies", which is a pass this check cannot afford to give.
    named = witness["studies"]
    if not isinstance(named, list) or not named:
        return "UNVERIFIED - the witness carries no readable list of studies"
    studies = [str(name) for name in named]
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
    bound = tolerance(member.precision, device)
    if difference > bound:
        return (
            "REFUSED - inference reads this member differently than training did "
            f"(max|diff| {difference:.3g} > {bound:.3g})"
        )
    return f"ok (max|diff| {difference:.3g} of {bound:.3g} over {len(studies)} studies)"


def main() -> None:
    """Check every member's witness in a packaged directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("comp/members"))
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    # A GATE THAT NEEDS THE GPU IS A GATE THAT GETS SKIPPED. This reads eight studies through one
    # small encoder — seconds of work — and defaulting to cuda meant it could not run at all while a
    # training arm owned the card, which is exactly when a member is most likely to be packaged. The
    # verdict is device-independent: TOLERANCE is set above GPU noise precisely so a reading change is
    # what moves it, and cpu-vs-cuda is not a reading change.
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    package = MemberPackage(arguments.root)
    root = DataRoot.from_config(arguments.paths).processed()
    selected = arguments.device
    if selected == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(selected)

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
