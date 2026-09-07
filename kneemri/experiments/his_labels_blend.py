"""Blend his soft-label table toward our teacher, on his study set and in his column order.

BCE with a soft target is linear in the target, so training his script on ``(1 - w) * y + w * t`` is
exactly ``(1 - w) * BCE(logit, y) + w * BCE(logit, t)``: distillation toward our teacher without one
line of his code changing. Only the rows he TRAINS on are blended; rows in the held-out fold keep his
raw values, so the OOF truth his writer records (``…twelve-finding-model.py:545``) and the teacher
never meet on a study the arm is scored by.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float

logger = logging.getLogger(__name__)

PARQUET_TWIN = ".parquet"
"""The suffix his trainer's existence check looks for beside the blend it is actually served."""

UID = "StudyInstanceUID"
HIS_POSITIVE = 0.5
OUR_POSITIVE = 0.75
FINDINGS = 12


@dataclass(frozen=True)
class Folds:
    """His fold map, read with the predicate his script uses (``:393-394``)."""

    path: Path

    def members(self, fold: int) -> set[str]:
        """The study ids assigned to ``fold`` — the ones his run holds out, and this run leaves raw."""
        with self.path.open() as handle:
            fmap = json.load(handle)["folds"]
        return {uid for uid, assigned in fmap.items() if assigned == fold}


@dataclass(frozen=True)
class Blend:
    """His table blended toward our teacher at ``weight``, on the rows he trains on and only those."""

    his: pd.DataFrame
    teacher: Path
    held_out: set[str]
    weight: float

    def columns(self) -> list[str]:
        """His twelve findings, in his order — the count is what makes a positional mapping legal."""
        found = [column for column in self.his.columns if column != UID]
        if len(found) != FINDINGS:
            message = f"his table has {len(found)} finding columns, expected {FINDINGS}"
            raise ValueError(message)
        return found

    def table(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (blended table in his layout, per-column marginal over the rows that changed)."""
        columns = self.columns()
        with np.load(self.teacher, allow_pickle=True) as loaded:
            scores = pd.DataFrame(loaded["scores"], index=loaded["identifiers"].astype(str), columns=columns)
        out = self.his.copy()
        out[UID] = out[UID].astype(str)
        out = out.set_index(UID)
        train = [uid for uid in out.index if uid not in self.held_out]
        missing = [uid for uid in train if uid not in scores.index]
        if missing:
            message = f"{len(missing)} of his train studies have no teacher row, first {missing[0]}"
            raise ValueError(message)
        raw = out.loc[train, columns].to_numpy(dtype=np.float64)
        taught = scores.loc[train, columns].to_numpy(dtype=np.float64)
        mixed = (1.0 - self.weight) * raw + self.weight * taught
        out.loc[train, columns] = mixed
        return out.reset_index(), self._marginal(columns, raw, taught, mixed)

    @staticmethod
    def _marginal(
        columns: list[str],
        raw: Float[np.ndarray, "studies findings"],
        taught: Float[np.ndarray, "studies findings"],
        mixed: Float[np.ndarray, "studies findings"],
    ) -> pd.DataFrame:
        """What decides the weight: a moved positive rate is a relabel, a moved mean is a distillation."""
        return pd.DataFrame(
            {
                "his_mean": raw.mean(0),
                "blend_mean": mixed.mean(0),
                "teacher_mean": taught.mean(0),
                "his_pos@.5": (raw > HIS_POSITIVE).mean(0),
                "blend_pos@.5": (mixed > HIS_POSITIVE).mean(0),
                "his_pos@.75": (raw >= OUR_POSITIVE).mean(0),
                "blend_pos@.75": (mixed >= OUR_POSITIVE).mean(0),
                "corr(y,t)": [np.corrcoef(raw[:, index], taught[:, index])[0, 1] for index in range(len(columns))],
            },
            index=columns,
        )


def main() -> None:
    """Write the blended table beside his, named for the weight, and log the marginal shift."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--his", type=Path, required=True, help="his labels_llm_soft.csv")
    parser.add_argument("--teacher", type=Path, required=True, help="teacher_clean_full.npz")
    parser.add_argument("--fold-file", type=Path, required=True, help="his fold0_909.json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True, help="csv; his --labels takes the parquet twin path")
    parsed = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    held = Folds(parsed.fold_file).members(parsed.fold)
    table, marginal = Blend(pd.read_csv(parsed.his), parsed.teacher, held, parsed.weight).table()
    out: Path = parsed.out
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    # His trainer tests os.path.exists on the parquet BEFORE it reads it, and that test runs ahead of
    # foreign_run's csv-beside shim - the shim redirects the READ, never the existence check. So a
    # blend written as csv alone made the trainer print its no-labels line and SystemExit(0), which
    # reads as a clean run everywhere: exit 0, no traceback, ledger row still queued, card idle. The
    # twin is an existence marker only; the shim serves the csv standing beside it.
    out.with_suffix(PARQUET_TWIN).touch()
    n_held = sum(uid in held for uid in table[UID])
    logger.info(
        "wrote %s: %d rows, %d held-out rows left at his raw values, weight %.2f",
        out,
        len(table),
        n_held,
        parsed.weight,
    )
    logger.info("\n%s", marginal.round(3).to_string())


if __name__ == "__main__":
    main()
