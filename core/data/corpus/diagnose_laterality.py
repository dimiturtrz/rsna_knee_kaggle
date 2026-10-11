"""How was each study's side decided? `python -m core.data.corpus.diagnose_laterality --out <csv>`.

Both medial/lateral target pairs favour medial on our holdout (meniscus −0.043, OA −0.066), and the
five side-defined targets are exactly what a broken laterality normalisation would damage. That is a
hypothesis with an innocent rival: medial pathology is commoner and more often named in reports, which
would produce the same asymmetry through the labels alone.

Splitting the holdout by HOW each study's side was decided separates them. This needs only headers — one
read per series — so it costs seconds and no cache rebuild.
"""

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from core import cli
from core.data.corpus.acquisition import Acquisition
from core.data.corpus.laterality import Laterality
from core.data.corpus.study_cache import CorpusSplit
from core.findings import STUDY

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SideEvidence:
    """What one study's series say about its side: the tags, the centres, and what each rule resolves."""

    acquisitions: list[Acquisition]

    def tagged(self) -> list[str]:
        """Every series' own laterality tag, in probe order, untagged series skipped."""
        return [a.laterality_tag for a in self.acquisitions if a.laterality_tag]

    def centres(self) -> list[float]:
        """Every series' image-centre x, where its header carried one."""
        return [a.centre_x for a in self.acquisitions if a.centre_x is not None]

    def geometric(self) -> str | None:
        """The side geometry alone gives, read from EVERY series with its tag stripped.

        Not only the untagged ones: computing it from the untagged remainder makes the "both available"
        set mean "studies whose series disagree about having a tag", which is a biased subsample and not
        the accuracy question being asked.
        """
        if not self.centres():
            return None
        return Laterality().of_study([replace(a, laterality_tag=None) for a in self.acquisitions])

    def side(self) -> str | None:
        """The side the production rule resolves, tag first."""
        return Laterality().of_study(self.acquisitions)

    def source(self) -> str:
        """Which rule decided the side: the tag, the geometry, or neither."""
        if self.tagged():
            return "tag"
        return "unresolved" if self.side() is None else "geometry"

    def row(self) -> dict[str, object]:
        """The audit columns this evidence fills, empty strings where a rule gave nothing."""
        tagged, centres = self.tagged(), self.centres()
        return {
            "side": self.side() or "",
            "source": self.source(),
            "tag_side": next(iter(tagged), ""),
            "geometry_side": self.geometric() or "",
            "series": len(self.acquisitions),
            "median_centre_x": float(pd.Series(centres, dtype=float).median()),
        }


class LateralityAudit:
    """Resolves every study's side and records which rule decided it."""

    @staticmethod
    def resolve_one(argument: tuple[str, str]) -> dict[str, object]:
        """One study: the side the tag gives, the side geometry gives, and which one was used."""
        identifier, directory = argument
        series = [path for path in sorted(Path(directory).iterdir()) if path.is_dir()]
        evidence = SideEvidence([Acquisition.probe(identifier, path.name, path) for path in series])
        return {STUDY: identifier, **evidence.row()}

    def run(self, source: Path, identifiers: list[str], workers: int) -> pd.DataFrame:
        """Resolve every study in parallel; only headers are read, so this is seconds not hours."""
        arguments = [(identifier, str(source / identifier)) for identifier in identifiers]
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done, row in enumerate(pool.map(self.resolve_one, arguments, chunksize=8), start=1):
                rows.append(row)
                if done % 500 == 0:
                    logger.info(f"  resolved {done}/{len(identifiers)}")
        return pd.DataFrame(rows)


def main() -> None:
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--out", type=Path, required=True)
    command.parser.add_argument("--workers", type=int, default=20)
    arguments = command.parsed()

    split = CorpusSplit(command.rooted(arguments).raw())
    table = LateralityAudit().run(split.source, split.identifiers(), arguments.workers)
    table.to_csv(arguments.out, index=False)

    logger.info(f"\nresolved {len(table)} studies -> {arguments.out}")
    logger.info("%s", table["source"].value_counts().to_dict())
    both = table[(table.tag_side != "") & (table.geometry_side != "")]
    if len(both):
        agree = float((both.tag_side == both.geometry_side).mean())
        logger.info(f"tag and geometry both available on {len(both)} studies; they agree on {agree:.3f}")
    logger.info("%s", table["side"].value_counts().to_dict())


if __name__ == "__main__":
    main()
