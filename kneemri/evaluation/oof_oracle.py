"""Does OUR pipeline hand a published member the same pixels its author did?

`python -m kneemri.evaluation.oof_oracle --checkpoint published/m_013dc75703.pt`

THE ONE MEASUREMENT THAT PRICES OUR DATA LAYER AGAINST GROUND TRUTH, and it needs no labels and no
leaderboard. Each published checkpoint carries `oof`: the study identifiers it scored and the twelve
probabilities it produced for each, written by the author's own pipeline. Running the same weights
through ours over the same studies makes any difference attributable to the READING alone — routing,
order, laterality, crop, percentile, resize. The architecture is already proven separately by the
fingerprint oracle, which feeds noise and is blind to pixels.

A faithful pipeline agrees to about 1e-6, as the fingerprint does. Ours agreed to max|diff| 0.025 with
per-study correlation 0.99, and on identical weights the graft scored 0.888 where public forks sit near
0.899. Those two numbers are the same defect seen locally and on the board.

WHAT THIS INSTRUMENT COST TO NOT HAVE. It existed once as `scratchpad/oof_oracle.py`, was never
committed, and is gone — with the weights it ran on. Every hypothesis about the residual since has been
argued from a number nobody could recompute, and two of them (slot tie-break, mismatched reading rules)
were killed by facts that were sitting inside these checkpoints the whole time. The repository already
had the rule that a one-off run becomes a committed entrypoint; this is what the exception cost.

WHY IT REPORTS PER-STUDY AND NOT ONE NUMBER. A max|diff| says a defect exists and nothing about where.
The residual is compared against covariates that cost nothing to compute — which knee, how many series,
whether any slot was contested — because a defect that concentrates on left knees is a laterality bug
and one that concentrates on contested studies is a routing bug, and one scalar cannot tell them apart.

NOTE ON THE NAME. The published `fold: -1` members trained on every study, so their `oof` is not held
out at all. That costs this oracle nothing: it compares two pipelines running ONE set of weights over
ONE set of studies, and generalisation never enters. The name is theirs and is kept so the field it
reads stays findable.

FETCHING THE INPUTS (out of the repo, as all data is):

    cd "$(python -c 'from core.paths import DataRoot; print(DataRoot.from_config("paths.yaml").processed())')/published"
    kaggle datasets download pilkwang/rsna-knee-weights --unzip
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from numpy.typing import NDArray

from core.paths import DataRoot
from kneemri.corpus.slot_reader import Band
from kneemri.corpus.slot_ties import SlotTies
from kneemri.corpus.slots import SCHEMES
from kneemri.corpus.study_cache import StudyCache
from kneemri.inference.holdout_predictions import HoldoutPredictions
from kneemri.inference.members import Member

logger = logging.getLogger(__name__)

FAITHFUL = 1e-6
"""What two pipelines computing the same function agree to. The fingerprint oracle reaches it, so it is
an observed standard here rather than an aspiration."""

WORST = 12
"""How many disagreeing studies to name. Enough to look at, few enough to read."""


class OutOfFoldOracle:
    """A published member's own recorded predictions against what our pipeline produces."""

    def recorded(self, checkpoint: Path) -> tuple[list[str], Float[np.ndarray, "studies targets"]]:
        """The identifiers and probabilities the author's pipeline wrote into this checkpoint.

        `weights_only=False` because the payload is a config and two plain lists rather than tensors —
        and this reads files WE fetched from a named dataset and stored ourselves.
        """
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        stored = saved["oof"]
        return list(stored["ids"]), np.asarray(stored["pred"], dtype=np.float64)

    def corpus(self, published: Path) -> tuple[list[str], list[NDArray[np.float64]]]:
        """Every study in the corpus, with one prediction per SEED, held out — a ruler that costs nothing.

        The twenty published checkpoints are four seeds over five folds, and their `oof` id-sets
        partition all 4,407 studies exactly. So stacking them by fold gives, for every study in the
        corpus, four predictions from models that did not train on it.

        WHY THIS MATTERS MORE THAN THE RESIDUAL IT SITS NEXT TO. Our own ruler is one fold — 881 studies,
        62 Fracture positives, a sampling noise floor of 0.0053 that has decided what counts as a result
        for this whole project. This one is the entire corpus: five times the studies, so about 0.0024.
        Every question that is arithmetic over member predictions — per-label combining above all — can
        be asked here at more than twice the resolution, on a CPU, today.

        WHAT IT CANNOT DO: it holds only THEIR members. Ours were never fitted fold-wise over the whole
        corpus, so a comparison between ours and theirs still belongs on the shared fold-0 holdout. This
        ruler answers questions about combining, not about whose weights are better.

        Returned as one array per seed, aligned to the returned identifiers, so the caller sees exactly
        the shape `LabelCombine` and `MemberCurve` already take.
        """
        by_fold: dict[tuple[str, ...], list[tuple[int, list[str], NDArray[np.float64]]]] = {}
        for checkpoint in sorted(published.glob("*.pt")):
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            identifiers = list(saved["oof"]["ids"])
            key = tuple(sorted(identifiers))
            values = np.asarray(saved["oof"]["pred"], dtype=np.float64)
            by_fold.setdefault(key, []).append((int(saved["config"]["seed"]), identifiers, values))

        seeds = sorted({seed for fold in by_fold.values() for seed, _, _ in fold})
        names: list[str] = []
        stacked: list[list[NDArray[np.float64]]] = [[] for _ in seeds]
        for fold in by_fold.values():
            members = {seed: (identifiers, values) for seed, identifiers, values in fold}
            if set(members) != set(seeds):
                continue
            order = members[seeds[0]][0]
            names += order
            for column, seed in enumerate(seeds):
                identifiers, values = members[seed]
                # REORDERED, never assumed: two members of one fold hold the same studies, and nothing
                # promises they wrote them in the same order.
                position = {name: index for index, name in enumerate(identifiers)}
                stacked[column].append(values[[position[name] for name in order]])
        return names, [np.concatenate(rows) for rows in stacked]

    def agrees(self, member: Member, checkpoint: Path) -> list[str]:
        """Whether the reading a published member RECORDS matches the one we will serve it under.

        Checked before any pixel is decoded, because a mismatch here explains a residual completely and
        costs nothing to find — and because the two beads that chased this residual for three days were
        both answerable from these fields alone.
        """
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = saved["config"]
        served = member.reading
        differences = []
        # AGAINST THE MEMBER, not against our module constants. This method took a `Member` and never
        # touched it, comparing their config with CROP_MM and SLICE_BAND — which asked the same question
        # as "does it match the reading we will serve it under" only while nothing of ours could differ
        # from the default. The 80 mm cache on disk ended that.
        #
        # A recorded field the accessor DROPS is the failure this is really watching for: a member whose
        # file states a reading and whose `reading` reports our default is exactly how the same value
        # has gone missing nine times in this project, and here it would be charged to the reader as a
        # residual nobody could explain.
        if "crop_mm" not in config:
            differences.append(f"they record no crop_mm; we will serve {served.crop.mm}")
        elif float(config["crop_mm"]) != served.crop.mm:
            differences.append(f"crop_mm {config['crop_mm']} vs the {served.crop.mm} we will serve")
        if "band" not in config:
            differences.append(f"they record no band; we will serve {served.band.recorded()}")
        elif Band.of(config["band"]) != served.band:
            differences.append(f"band {config['band']} vs the {served.band.recorded()} we will serve")
        # THEIR slot order against OURS, not against their own config: a published checkpoint records
        # its slots at the top level and not inside `config`, so comparing the two config fields asked a
        # question about their file rather than about the difference between their pipeline and ours.
        ours = [slot.name for slot in SCHEMES[member.scheme]]
        if list(saved["slots"]) != ours:
            differences.append(f"slot ORDER {saved['slots']} vs ours {ours}")
        return differences

    def covariates(self, identifiers: list[str], series: pd.DataFrame) -> dict[str, NDArray[np.float64]]:
        """Per-study properties that cost nothing and can each explain a different defect.

        `series_count` separates a routing defect from a decode one: a study with a single series has no
        routing to get wrong. `contested` is the same question sharpened — those are the studies where
        two series compete for one slot. Neither needs a label, a GPU or a pixel.
        """
        counts = series.groupby("StudyInstanceUID")["SeriesInstanceUID"].count()
        return {"series_count": np.array([float(counts.get(name, 0)) for name in identifiers])}

    def routing(
        self, identifiers: list[str], series_root: Path, planes: dict[str, str]
    ) -> dict[str, NDArray[np.float64]]:
        """Per-study ROUTING covariates, which is where the residual pointed.

        `series_count` correlated +0.647 with the residual over 24 studies, and that is a routing
        signature rather than a decode one — a decode difference spreads evenly, while a study with two
        series has almost no routing to get wrong and one with nine has a great deal. But series_count
        also tracks study complexity in general, so it cannot say WHICH part of routing.

        These three can. `unrouted` counts series no slot accepts at all — 17.4% of the corpus, the
        population `knee-t92` measured and nothing has ever tied to an outcome. `empty` counts slots left
        with no candidate, which the presence mask then declares absent. `contested` counts slots two
        series competed for, the population whose tie-break was refuted this morning — included
        precisely so the refuted candidate gets a second, independent chance to show itself.

        Headers only, no pixels, so this costs seconds beside the forward pass it explains.
        """
        from kneemri.corpus.slots import SlotScheme  # noqa: PLC0415 - CLI-side analysis, not the library path
        from kneemri.corpus.study import Study  # noqa: PLC0415 - as above

        scheme = SlotScheme()
        unrouted: list[float] = []
        empty: list[float] = []
        contested: list[float] = []
        for name in identifiers:
            study = Study.from_directory(series_root / name, planes)
            routed = scheme.assign(study.acquisitions, planes)
            unrouted.append(float(len(scheme.unrouted(study.acquisitions, planes))))
            empty.append(float(len(scheme.slots) - len(routed)))
            contested.append(float(sum(scheme.contested(study.acquisitions, planes).values())))
        return {
            "unrouted_series": np.array(unrouted),
            "empty_slots": np.array(empty),
            "contested_slots": np.array(contested),
        }

    def report(
        self,
        identifiers: list[str],
        ours: Float[np.ndarray, "studies targets"],
        theirs: Float[np.ndarray, "studies targets"],
        covariates: dict[str, NDArray[np.float64]],
    ) -> list[str]:
        """The residual, its shape across studies, and what it correlates with.

        The headline is `max|diff|` against 1e-6, and everything after it exists so the number can be
        acted on: a residual spread evenly over every study is a decode or normalisation difference,
        and one concentrated in a handful is a routing difference on those studies.
        """
        per_study = np.abs(ours - theirs).max(axis=1)
        lines = [
            f"{len(identifiers)} studies, max|diff| {per_study.max():.6f}, mean {per_study.mean():.6f}",
            f"  faithful would be {FAITHFUL:.0e}; studies above 0.01: "
            f"{int((per_study > 0.01).sum())} of {len(identifiers)}",  # noqa: PLR2004 - a readable threshold, not a rule
        ]
        for name, values in covariates.items():
            finite = np.isfinite(values) & np.isfinite(per_study)
            if finite.sum() > 1 and np.std(values[finite]) > 0:
                correlation = float(np.corrcoef(values[finite], per_study[finite])[0, 1])
                lines.append(f"  corr(residual, {name}) = {correlation:+.3f}")
        order = np.argsort(-per_study)[:WORST]
        lines.append(f"  the {len(order)} worst studies:")
        lines += [f"    {identifiers[index]}  {per_study[index]:.4f}" for index in order]
        return lines


def main() -> None:
    """Score one published member through our pipeline and compare against its own recorded output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="a published member carrying `oof`")
    parser.add_argument("--studies", type=int, default=0, help="0 uses every study the checkpoint recorded")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = DataRoot.from_config(arguments.paths)
    oracle = OutOfFoldOracle()
    identifiers, theirs = oracle.recorded(arguments.checkpoint)
    member = Member.from_checkpoint(arguments.checkpoint)

    for difference in oracle.agrees(member, arguments.checkpoint):
        logger.info("READING DIFFERS: %s", difference)

    corpus, reference = member.cache(root.processed()).load()
    # THE STUDIES THEY SCORED, in THEIR order, and only those our cache holds — a row compared against
    # the wrong study's answer is a residual that no amount of staring at the reader will explain.
    position = {name: index for index, name in enumerate(reference)}
    keep = [index for index, name in enumerate(identifiers) if name in position]
    if arguments.studies:
        keep = keep[: arguments.studies]
    logger.info("%d of %d recorded studies are in our cache", len(keep), len(identifiers))

    from kneemri.inference.holdout_slab import HoldoutSlab  # noqa: PLC0415 - CLI-only

    rows = np.array([position[identifiers[index]] for index in keep])
    slab = HoldoutSlab(corpus, rows)
    device = torch.device(arguments.device)
    predictions = HoldoutPredictions(device)
    from transformers import AutoModel  # noqa: PLC0415 - CLI-only, and it downloads

    from kneemri.models.backbone import ENCODERS  # noqa: PLC0415 - CLI-only

    encoder = ENCODERS[member.variant]
    model = predictions.load(
        arguments.checkpoint, AutoModel.from_pretrained(encoder.repository), encoder.dim, member.slot_count
    )
    ours = predictions.predict(model, slab, member.window(corpus.slices, overlap=False), member.precision)

    chosen = [identifiers[index] for index in keep]
    series = pd.read_csv(root.raw() / "train_series.csv")
    covariates = oracle.covariates(chosen, series)
    covariates.update(
        oracle.routing(chosen, root.raw() / "train_series", StudyCache.planes_of(root.raw() / "train_series.csv"))
    )
    for line in oracle.report(chosen, ours, theirs[keep], covariates):
        logger.info(line)
    logger.info("(contested-slot rate for context: %s)", SlotTies().disagreeing_order(root.raw() / "train_series.csv"))


if __name__ == "__main__":
    main()
