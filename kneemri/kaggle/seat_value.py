"""What a candidate member would add to the champion roster, card-free.

`python -m kneemri.kaggle.seat_value --candidates a.pt b.pt`

THE SECOND RULER, and the one that was missing. Every arm is read against a MATCHED control — the same
recipe with one field changed — which is the right question for `did this knob do anything` and the
wrong one for `does this earn a seat`. The two came apart once the roster stopped being the control's
family. `depth_regions_narrow` beat the fixed control `wsfold0_e6` by +0.0097 and beat its own
predecessor `depth_regions` by +0.0005; only the first number was going to be reported.

`res392_ws` is the case that forces the distinction. It scores 0.8523 alone, it was a member of the four
that scored 0.925 on the board, and over 22 same-split members the greedy takes it NINTH — the first
pick that does not raise the mix. Solo score ranked it into a submission; marginal contribution would
not have.

THE ROSTER IS READ FROM THE PIN, not typed. `comp/champion.json` holds the members of the entry that
actually scored, and `champion_check` is what keeps it true. A hand-listed roster in an arm header is a
copy that goes stale silently, and every delta under a stale roster is against the wrong base.

READ IT AS A SIGN. The holdout under-prices diversity against the board by 2.4x (knee-7d1), so a small
negative here is inside the region where this ruler is known to be wrong — `dinov3_base_lora`, a
genuinely different encoder, reads -0.0013. What transfers is the ordering, not the value.
"""

import argparse
import logging
from pathlib import Path

from core.paths import DataRoot
from kneemri.evaluation.member_curve import COMBINE_SEED, MemberCurve
from kneemri.evaluation.record_mix import MixReading, RecordMix
from kneemri.evaluation.scores import LabelScores
from kneemri.kaggle.champion_check import Champion

logger = logging.getLogger(__name__)


class SeatValue:
    """The pinned champion's members, and what each candidate would add to them."""

    def __init__(self, champion: Champion, processed: Path) -> None:
        self.champion = champion
        self.processed = processed

    def roster(self) -> list[Path]:
        """The champion's members as checkpoints on disk.

        A member the pin names but the directory does not hold is left for `RecordMix.split` to refuse.
        Reporting it here as well would put the same failure in two places with two wordings, and the
        one downstream is the one that also catches a member present but recordless.
        """
        return [self.processed / f"{name}.pt" for name in self.champion.members]

    def report(self, candidates: list[Path], combine_seed: int = COMBINE_SEED) -> list[str]:
        """Each candidate's marginal contribution to the champion, best first."""
        roster = self.roster()
        mixed = RecordMix(list(dict.fromkeys([*roster, *candidates]))).per_member()
        curve = MemberCurve(LabelScores(), combine_seed)
        return MixReading(mixed, roster).lines(curve, greedy=False, paired=False)


def main() -> None:
    """Price candidate members against the entry that actually scored."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True, help="checkpoints to price")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    parser.add_argument("--combine-seed", type=int, default=COMBINE_SEED)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    champion = Champion.recorded()
    logger.info("%s scored %.3f on the board with %d members", champion.entry, champion.board, len(champion.members))
    seats = SeatValue(champion, DataRoot.from_config(arguments.paths).processed())
    for line in seats.report(arguments.candidates, arguments.combine_seed):
        logger.info(line)


if __name__ == "__main__":
    main()
