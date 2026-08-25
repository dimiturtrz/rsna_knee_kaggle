"""What a pre-flight check SAYS, and whether it refuses the run — apart from what it detects.

Its own module rather than another class in `arm_check`, because that file crossed the god-file bar and
this is the seam the split falls on: the detectors ask the tree questions, and this answers the two the
tree cannot — which voice to say a finding in, and whether a finding is worth stopping for.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArmVerdict:
    """One check's findings, and the two voices this tool says them in.

    THE ASYMMETRY, written once rather than eight times. Seventy arms predate whichever rule an advisory
    check enforces and nearly all have already run, so naming each on every sweep buries the REFUSED and
    UNCACHED lines that actually cost a queue slot — and a pre-flight check nobody reads by the third
    sweep is how one earns a `--no-verify`. Such a defect is only actionable on the arm about to be ARMED,
    which is exactly the run that says `--only`; the sweep still says the COUNT, because a number that
    never falls is the erosion record.

    IT CARRIES ITS OWN REFUSAL POLICY, which used to be spelled in four places — two inside the check that
    detected the finding, so `--require-caches` could reach UNCACHED and nothing could reach the others.
    The reason for each is a property of the FINDING rather than of the run, so `ArmCheck.verdicts` states
    them beside the rules and `main` composes an exit code without knowing one of them.
    """

    label: str
    found: Mapping[str, str]
    clause: str = ""
    bead: str = ""
    always: bool = False
    """Name every finding on every sweep — for the checks whose findings each cost a queue slot."""
    refuses: bool = False
    arming_refuses: bool = False

    def report(self, total: int, *, arming: bool) -> bool:
        """Log every finding by name, or one counted line; answer whether they REFUSE this run."""
        reference = f" ({self.bead})" if self.bead else ""
        if self.always or arming:
            for name, detail in sorted(self.found.items()):
                logger.info("%s %s: %s%s", self.label, name, detail, reference)
        elif self.found:
            logger.info(
                "%s: %d of %d arms %s; --only <arm> names them%s",
                self.label,
                len(self.found),
                total,
                self.clause,
                reference,
            )
        return bool(self.found) and (self.refuses or (self.arming_refuses and arming))
