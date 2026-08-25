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

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from core import cli
from kneemri.device import VisibleCards

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kneemri.inference.ensemble import Ensemble
from kneemri.inference.members import Member, MemberPackage
from kneemri.models.backbone import ENCODERS

logger = logging.getLogger(__name__)

EPS_FACTOR = 4.0
"""How many epsilons apart a witness and its re-derivation legitimately land, on either device.

MEASURED 2026-08-13, four members re-derived on CPU (float32) against their recorded witnesses:

    schedule_8_s2026      float16   1.59e-3   1.6x eps
    intensity392_ws_s2026 float16   1.62e-3   1.7x eps
    sliceattn_s2026       float16   2.34e-3   2.4x eps
    extended_ws_s2026     bfloat16  1.56e-2   2.0x eps

The residual tracks the WITNESS's epsilon across two dtypes eight times apart, at 1.6-2.4x of it —
which is why this is a factor on eps rather than a constant. One eps was the wrong model: eps is the
spacing of a single rounding and a twelve-block transformer accumulates thousands. Four leaves margin
over the widest observed without reaching the 0.05-0.5 that every real reading defect produced.

MEASURED AGAIN 2026-08-19 on the card (knee-djk8 leg b), where the arithmetic MATCHES and the residual
is only kernel nondeterminism: three float16 members land at EXACTLY 0, and the bf16 members land at
0.63-2.47x bf16 eps. That is the same 1.6-2.4x band the cross-precision calibration measured, so the
two regimes never needed two numbers — see `tolerance`.
"""


def tolerance(precision: torch.dtype) -> float:
    """How far the two paths may legitimately differ, from the ARITHMETIC rather than picked.

    This was a flat 2e-3, inherited from the fingerprint and documented as "above GPU noise" — true
    when training ran in float32 or float16, and false the day bf16 landed. bfloat16 keeps float32's
    exponent and spends only seven bits on the mantissa, so its eps is 2^-7 = 7.8e-3: FOUR TIMES the
    gate. The check refused the first bf16 member it ever saw, correctly by its own rule and wrongly in
    fact, and a gate that cries wolf is a gate someone switches off before a real refusal arrives.

    ONE REGIME, and it took a second measurement to learn that. The device does decide what the
    residual IS: on cuda the re-derivation runs under the member's own autocast dtype, so the
    arithmetic matches and what is left is kernel nondeterminism, while on cpu autocast is off and a
    half-precision witness is being compared against float32. That argued for two bounds, and the cuda
    one was a flat 2e-3 on the premise that a matched arithmetic drifts by almost nothing. The premise
    is true for float16 — three members re-derive at exactly 0 — and false for bfloat16, whose seven
    mantissa bits put the same nondeterminism at 0.63-2.47x eps, so the gate refused bf16 members for
    BEING bf16. Both regimes measured into the same 1.6-2.5x eps band, which is the finding: the
    residual scales with the mantissa in both, the device branch was never carrying the information,
    and the dtype alone sets the bound.

    IT STILL DISCRIMINATES, which is the only thing that makes a loosened bound acceptable. The four
    reading defects this check exists for moved predictions by 0.05 to 0.5, because feeding a model a
    different slot list or a quarter of its stack changes what it sees rather than how precisely it
    adds. Those also CONCENTRATE in some labels while sparing others, where this residual moved all
    twelve about equally.
    """
    return EPS_FACTOR * float(torch.finfo(precision).eps)


CONCENTRATION_FACTOR = 0.5
"""A label counts as carrying the residual when its own max reaches this share of the widest label's."""


def spread(difference: NDArray[np.float64]) -> str:
    """How many of the twelve labels the residual actually lives in — the arithmetic/defect discriminator.

    A max alone cannot tell the two apart, and on 2026-08-19 that left knee-djk8 stuck: seven members
    refused at 0.00495-0.0193, the shape they were recorded under reproduced exactly, and two runs agreed
    to every digit, so the residual is deterministic and unexplained. The bead's remaining question is
    whether it SPREADS or CONCENTRATES. A rounding difference touches every column about equally, because
    every column comes off the same GEMMs; the four reading defects this check exists for moved a FEW
    findings by 0.05-0.5, because feeding a model the wrong slices changes what it can see about some
    anatomy and not the rest.

    Counted against the widest label rather than against a fixed level, so the sentence reads the same
    whether the residual is 1e-3 or 1e-1: it answers "in how many places", never "how big".

    Reported on the PASSING verdicts too, because a shape means nothing without its control: the same
    member reads 0.0135 as a pass on cpu and as a refusal on cuda, and only the two shapes side by side
    say whether those are the same residual seen through two bounds or two different residuals.
    """
    per_label = difference.max(axis=0)
    widest = float(per_label.max())
    if widest == 0.0:
        return "exact in every label"
    carrying = int((per_label >= CONCENTRATION_FACTOR * widest).sum())
    return f"{carrying} of {len(per_label)} labels carry it"


def check(package: MemberPackage, member: Member, root: Path, device: torch.device, batch: int | None = None) -> str:
    """Re-derive one member's witness, returning a human-readable verdict.

    `batch` is the SHAPE the witness was produced under, and it is not decoration. Trainer.predict
    scores the holdout in batches of the run's `--batch` (12 for every member on disk), so the eight
    witnessed studies were the first eight rows of a twelve-row forward. This check feeds eight. A
    half-precision GEMM tiles by shape, so the two paths reduce in different orders and each carries
    its own rounding — which is the leading candidate for the 0.0135-0.0156 that CUDA reads on five
    bf16 members while CPU, comparing against a single float32 re-derivation, reads 0.0022-0.0156.
    Rows are independent here (per-token LayerNorm, attention within a study), so repeating the last
    study up to the batch reproduces the GEOMETRY without needing the studies that padded it.
    """
    witness = member.witness
    if not witness:
        # The third cause was missing here for as long as it mattered most. Five all-data members read
        # as "trained before the check existed" while the real reason was that `--fold -1` took its
        # witness from a holdout it did not have (knee-52mu) — a wrong cause that reads as a known one
        # is worse than no message, because it retires the question. Fixed in `Fitting.witnessed`, so a
        # member fitted after 2026-08-24 hitting this line is one of the first two cases.
        return "UNVERIFIED - no witness (a published member, or fitted before 2026-08-24)"

    corpus, identifiers = member.cache(root).load()
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
    kept = len(positions)
    recorded = witness.get("batch")
    shape = batch or (recorded if isinstance(recorded, int) else None)
    if shape and shape > kept:
        positions = np.concatenate([positions, np.repeat(positions[-1:], shape - kept)])
    # The MIDDLE window, exactly what Trainer.predict reads: a whole-stack member wants its whole
    # stack, a windowed one wants one group. Feeding precisely that makes predict_member take a single
    # pass in both cases, so the comparison is about the READING and not about window averaging.
    width = member.slices if member.whole_stack else member.group
    start = (corpus.slices - width) // 2
    images = torch.from_numpy(corpus.window(positions, start=start, width=width)).to(device)
    mask = torch.from_numpy(corpus.mask[positions]).to(device)

    encoder = ENCODERS[member.variant]
    model = package.load(member, encoder.load(), encoder.dim, device)
    ensemble = Ensemble(package, {member.identifier: model}, device)
    got = ensemble.predict_member(member, images, mask, overlap=True)[:kept]
    expected = np.asarray(witness["predictions"], dtype=np.float64)  # pyright: ignore[reportGeneralTypeIssues]

    residual = np.abs(got - expected)
    difference = float(residual.max())
    bound = tolerance(member.precision)
    geometry = f", batch {shape}" if shape else ", batch unrecorded"
    if difference > bound:
        return (
            "REFUSED - inference reads this member differently than training did "
            f"(max|diff| {difference:.3g} > {bound:.3g}{geometry}; {spread(residual)})"
        )
    return f"ok (max|diff| {difference:.3g} of {bound:.3g} over {len(studies)} studies{geometry}; {spread(residual)})"


def main() -> None:
    """Check every member's witness in a packaged directory."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument(
        "--root", type=Path, default=None, help="a package to check; default is the one that ships"
    )
    # A GATE THAT NEEDS THE GPU IS A GATE THAT GETS SKIPPED. This reads eight studies through one
    # small encoder — seconds of work — and defaulting to cuda meant it could not run at all while a
    # training arm owned the card, which is exactly when a member is most likely to be packaged. The
    # verdict is device-independent: TOLERANCE is set above GPU noise precisely so a reading change is
    # what moves it, and cpu-vs-cuda is not a reading change.
    command.parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    # A CHECK THAT CAN ONLY REFUSE CANNOT REQUIRE. Every member of the mounted kit predates the witness,
    # so the summary read "0 ok, 3 unverified, 0 refused" and exited 0 — success-shaped output over a
    # package nothing had been proven about. Re-fitting those members is the real fix (knee-stp) and it
    # needs the card, so failing by default would leave the gate permanently red for a reason it cannot
    # fix; an automated caller asks for the strong claim instead.
    command.parser.add_argument("--require-witness", action="store_true", help="a member with no witness fails")
    # THE SHAPE THE WITNESS WAS PRODUCED UNDER, for members recorded before the witness carried it.
    # Every member on disk was scored at --batch 12, which is readable from its metrics.jsonl and not
    # from the packaged member, so the geometry leg is runnable on them only by being told. New runs
    # record it and need no flag.
    command.parser.add_argument("--batch", type=int, default=None, help="feed this batch shape (default: as recorded)")
    # THE MEMBERS ONE ENTRY ACTUALLY MOUNTS, because the package holds every member ever fitted and an
    # entry seats a handful. On CPU the whole package takes longer than the timeout it is run under, so
    # the gate that the conventions require before every submission could not finish while the card was
    # busy — which is exactly when it is needed. Naming the entry's members turns it back into a check.
    command.parser.add_argument("--members", nargs="+", default=None, help="check only these identifiers")
    arguments = command.parsed()

    package = MemberPackage.at(arguments.root, arguments.paths)
    root = command.rooted(arguments).processed()
    selected = arguments.device
    if selected == "auto":
        selected = VisibleCards().device().type
    device = torch.device(selected)

    wanted = set(arguments.members or ())
    members = [member for member in package.members if not wanted or member.identifier in wanted]
    missing = sorted(wanted - {member.identifier for member in members})
    if missing:
        # A NAME THE PACKAGE DOES NOT HOLD IS THE FAILURE THIS FLAG COULD OTHERWISE HIDE: a typo would
        # silently narrow the check to nothing and report "0 ok, 0 refused", which exits clean.
        message = f"not in the package: {', '.join(missing)}"
        raise SystemExit(message)

    # REPORTED AS EACH ONE LANDS, not after the last. On CPU this takes minutes per member, and the
    # dictionary comprehension that used to wrap it meant a run killed by a timeout printed NOTHING —
    # every verdict it had already computed died with it, so fifteen minutes of work read as no output
    # at all rather than as most of an answer.
    # COUNTED AGAINST A TOTAL, because a killed run and a finished one printed the SAME SHAPE of log.
    # On 2026-08-25 a 900 s cap cut this at member 22 of 40 and the eighteen it never reached left no
    # trace: no line, no count, no summary. Eighteen absent verdicts read as eighteen members that do
    # not exist, and the seven that actually ship were among them. The index is what makes a truncated
    # log say so on its own — "22/40" ends a run that stopped, "40/40" ends one that finished.
    verdicts: dict[str, str] = {}
    logger.info("checking %d members on %s", len(members), device.type)
    for position, member in enumerate(members, start=1):
        verdicts[member.identifier] = check(package, member, root, device, arguments.batch)
        logger.info("  [%d/%d] %s: %s", position, len(members), member.identifier, verdicts[member.identifier])

    unreached = [member.identifier for member in members if member.identifier not in verdicts]
    if unreached:
        # Unreachable while the loop is exhaustive, and kept anyway: the claim "checked" must be the
        # loop's own arithmetic rather than a property a future edit is trusted to preserve.
        message = f"{len(unreached)} members produced no verdict: {', '.join(unreached)}"
        raise SystemExit(message)

    refused = [name for name, verdict in verdicts.items() if verdict.startswith("REFUSED")]
    unverified = [name for name, verdict in verdicts.items() if verdict.startswith("UNVERIFIED")]
    verified = len(verdicts) - len(refused) - len(unverified)
    logger.info("%d ok, %d unverified, %d refused", verified, len(unverified), len(refused))
    if verdicts and not verified:
        logger.info(
            "NOTHING HERE IS VERIFIED: no member of %s carries a witness, so this run made no claim "
            "about how inference reads them. Not a pass — an unanswered question.",
            arguments.root,
        )
    if refused or (arguments.require_witness and unverified):
        sys.exit(1)


if __name__ == "__main__":
    main()
