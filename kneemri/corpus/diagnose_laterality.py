"""How was each study's side decided? `python -m kneemri.corpus.diagnose_laterality --out <csv>`.

Both medial/lateral target pairs favour medial on our holdout (meniscus −0.043, OA −0.066), and the
five side-defined targets are exactly what a broken laterality normalisation would damage. That is a
hypothesis with an innocent rival: medial pathology is commoner and more often named in reports, which
would produce the same asymmetry through the labels alone.

Splitting the holdout by HOW each study's side was decided separates them. This needs only headers — one
read per series — so it costs seconds and no cache rebuild.
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd

from core.paths import DataRoot
from kneemri.corpus.acquisition import Acquisition
from kneemri.corpus.laterality import Laterality

logger = logging.getLogger(__name__)


class LateralityAudit:
    """Resolves every study's side and records which rule decided it."""

    @staticmethod
    def resolve_one(argument: tuple[str, str]) -> dict[str, object]:
        """One study: the side the tag gives, the side geometry gives, and which one was used."""
        identifier, directory = argument
        acquisitions = [
            Acquisition.probe(identifier, series.name, series)
            for series in sorted(Path(directory).iterdir())
            if series.is_dir()
        ]
        laterality = Laterality()
        tagged = [a.laterality_tag for a in acquisitions if a.laterality_tag]
        centres = [a.centre_x for a in acquisitions if a.centre_x is not None]
        # Geometry is computed from EVERY series, not only the untagged ones. Computing it from the
        # untagged remainder makes the "both available" set mean "studies whose series disagree about
        # having a tag", which is a biased subsample and not the accuracy question being asked.
        stripped = [replace(a, laterality_tag=None) for a in acquisitions]
        geometric = laterality.of_study(stripped) if centres else None
        side = laterality.of_study(acquisitions)
        source = "tag" if tagged else ("geometry" if side is not None else "unresolved")
        return {
            "StudyInstanceUID": identifier,
            "side": side or "",
            "source": source,
            "tag_side": tagged[0] if tagged else "",
            "geometry_side": geometric or "",
            "series": len(acquisitions),
            "median_centre_x": float(pd.Series(centres).median()) if centres else float("nan"),
        }

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = DataRoot.from_config(arguments.paths)
    raw = root.raw()
    identifiers = pd.read_csv(raw / "train.csv")["StudyInstanceUID"].tolist()
    table = LateralityAudit().run(raw / "train_series", identifiers, arguments.workers)
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
