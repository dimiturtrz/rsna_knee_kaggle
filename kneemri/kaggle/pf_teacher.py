"""Write the pf line's teacher over the TRAINING studies: `python -m kneemri.kaggle.pf_teacher --reader <csv> …`.

The pf line (board 0.958) is nartaa's CoAtNet and heliosli's R6 reader fused per finding in rank space at
heliosli's Raptor shares. `nartaa_teacher.npz` already holds the CoAtNet half; `kneemri-r6teacher` writes the
R6 half as each Kaggle worker's `predictions.csv`. This joins them into one teacher file of the same shape,
so a distillation recipe needs only a different `teacher_targets`.

Each model is ranked over the studies IT scored, and a study the reader did not reach before its deadline
takes the CoAtNet's rank — the rule heliosli's own blend applies to a failed study. The 224-crop CoAtNet that
shares the board entry's Raptor half has no training-split pass, so the Raptor side here is the 384 alone.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from core import cli
from core.findings import STUDY, TARGETS
from kneemri.teacher_file import TeacherFile

logger = logging.getLogger(__name__)

RAPTOR_SHARE = {"Synovitis": 0.80, "Lateral OA": 0.20, "PF OA": 0.60, "Baker's": 0.35}
"""heliosli's per-finding Raptor share of the (Raptor + reader) pair, as `kneemri-s6r7pf` ships it."""

RAPTOR_SHARE_ELSEWHERE = 0.50

SHARES = MappingProxyType({target: RAPTOR_SHARE.get(target, RAPTOR_SHARE_ELSEWHERE) for target in TARGETS})


@dataclass(frozen=True)
class PfTeacher:
    """The per-finding rank fusion of the CoAtNet's readings with the R6 reader's."""

    shares: Mapping[str, float] = SHARES

    def blended(self, raptor: pd.DataFrame, reader: pd.DataFrame) -> pd.DataFrame:
        """One row per CoAtNet study, each column `share * rank(raptor) + (1 - share) * rank(reader)`."""
        studies = raptor.set_index(STUDY)[list(TARGETS)]
        raptor_ranks = studies.rank(pct=True)
        reader_ranks = reader.set_index(STUDY)[list(TARGETS)].rank(pct=True).reindex(studies.index)
        missing = int(reader_ranks.isna().any(axis=1).sum())
        if missing:
            logger.info("%d of %d studies carry no reader rank and take the CoAtNet's", missing, len(studies))
        reader_ranks = reader_ranks.fillna(raptor_ranks)
        weights = pd.Series(self.shares)[list(TARGETS)]
        return (raptor_ranks * weights + reader_ranks * (1.0 - weights)).reset_index()

    @staticmethod
    def readings(parts: Sequence[Path]) -> pd.DataFrame:
        """The reader's predictions from every worker's file, one row per study."""
        frame = pd.concat([pd.read_csv(part, dtype={STUDY: str}) for part in parts], ignore_index=True)
        if frame[STUDY].duplicated().any():
            message = f"{int(frame[STUDY].duplicated().sum())} studies appear in more than one worker's file"
            raise ValueError(message)
        return frame[[STUDY, *TARGETS]]


def main() -> None:
    """Fuse the CoAtNet teacher file with the R6 workers' predictions and write the pf teacher file."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--reader", type=Path, nargs="+", required=True, help="R6 worker predictions")
    command.parser.add_argument("--raptor", type=Path, default=None, help="the CoAtNet teacher npz")
    command.parser.add_argument("--out", type=Path, default=None, help="destination npz")
    arguments = command.parsed()
    processed = command.rooted(arguments).processed()
    stored = np.load(arguments.raptor or processed / "nartaa_teacher.npz")
    raptor = pd.DataFrame(stored[TeacherFile.SCORES], columns=list(TARGETS))
    raptor.insert(0, STUDY, stored[TeacherFile.IDENTIFIERS].astype(str))
    fused = PfTeacher().blended(raptor, PfTeacher.readings(arguments.reader))
    destination = arguments.out or processed / "pf_teacher.npz"
    TeacherFile(destination).write(fused[STUDY].tolist(), fused[list(TARGETS)].to_numpy(dtype=np.float32))
    logger.info("wrote %d studies to %s", len(fused), destination)


if __name__ == "__main__":
    main()
