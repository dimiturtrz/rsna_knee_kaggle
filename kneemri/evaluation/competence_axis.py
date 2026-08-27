"""Whether the whole member pool differs along ONE trade-off axis — the question a seat search cannot ask.

WHY THIS EXISTS. Nine seat candidates priced against the champion spread 0.0012 between best and worst,
a fifth of the 0.0062 arm-vs-arm floor, and `2026-08-27_every-no-card-axis-is-priced-and-flat.md` reads
the mounted pool as exhausted. `seat_value` says WHICH permutation is best; it cannot say why every
permutation lands in the same place. `member_distance` gets closer — it reads how far a candidate's
ORDERING sits from an incumbent's — but it is pairwise and label-free, so it answers "is this a twin"
rather than "does the pool have more than one direction in it at all".

THE HYPOTHESIS IT TESTS, pre-registered in knee-54yp before this module existed: the members differ
along a single centre-versus-periphery trade-off. `band_wide` against `band_dense` (`slices 16→20`, the
only diff) buys the centre (PF OA +0.0194, Medial OA +0.0122) and pays with the periphery (MCL -0.0111,
ACL -0.0037, Lateral Meniscus -0.0007) for a mean of +0.0025 — a trade, not a gain. If every member on
disk sits somewhere on that one line, then no permutation of them can move the mean, because the mean is
what the line is orthogonal to. The way past the champion would then be a member OFF the axis, and that
is a claim about ARCHITECTURE rather than about seats.

WHAT IT MEASURES. A members x findings matrix of holdout AUCs, DOUBLE-CENTRED, then its singular values.

Double-centred, and both centrings are load-bearing:

  * By FINDING (columns) — removes how hard each finding is. Lateral Meniscus is near 0.65 for everybody
    and ACL near 0.86 for everybody; left in, the first component is that profile and says nothing about
    how members differ.
  * By MEMBER (rows) — removes how good each member is. This is the confound that would otherwise make
    the answer yes by construction: a pool containing one weak member and one strong one has a large
    first component that loads every finding with the same sign, and calling that a "trade-off axis" is
    reading member QUALITY back as pool structure. The quality direction is reported separately, as the
    share of variance it takes before it is removed, so it is priced rather than hidden.

WHAT MAKES THE ANSWER FALSIFIABLE. A first component always exists and always takes more variance than
the rest; with twelve findings and per-member AUC noise on a few hundred studies, a share of 0.25 could
be nothing. So the share is read against a NULL that shuffles each finding's column independently across
members — which destroys any coupling between findings while keeping every finding's own spread exactly
as measured. The null's 95th percentile is the bar. Below it, the pool has no shared axis and the flat
seat search has some other explanation; well above it, the pool has one and the loadings say what it is.

WHAT IT DOES NOT DO. It does not price a seat, name a member to build, or say a high-share pool cannot
be improved by permutation — a one-axis pool with members spread WIDELY along it still has a best point.
It says whether the spread that exists is one-dimensional, which is a statement about what a new member
would have to be, not about which existing one to seat.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.data.corpus.anatomy import ScaleProfile
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import LabelScores

logger = logging.getLogger(__name__)

SHUFFLE_SEED = 2026
"""The null's generator, fixed so a rerun of the same pool prints the same bar."""

NULL_RESAMPLES = 500
"""Shuffles behind the bar. The statistic is one SVD of a 12-column matrix, so this costs milliseconds."""

NULL_PERCENTILE = 95.0
"""How far into the null's own tail an observed share must sit before the axis is called real."""

EXTREMES = 3
"""Members named at each end of the axis — enough to see what the ends ARE, few enough to stay a report."""

RANKED = 12
"""Members printed from a subset ranking — a shortlist a seat search can actually be run over."""

MINIMUM_MEMBERS = 3
"""Below this a double-centred matrix has at most one degree of freedom left and every share reads 1.0."""


@dataclass(frozen=True)
class AxisReading:
    """What the pool's spread looks like once difficulty and member quality are taken out of it."""

    findings: list[str]
    share: float
    """Fraction of the double-centred variance the first component takes."""
    null: float
    """The same share at the null's `NULL_PERCENTILE` — the bar, not a p-value."""
    quality: float
    """Fraction of the finding-centred variance that was pure member quality, removed before the SVD."""
    loadings: dict[str, float]
    spread: dict[str, float]
    """How much each finding varies across members at all, once difficulty and quality are removed.

    Without it a loading is unreadable. A near-zero loading has two completely different causes — the
    finding varies across members but ORTHOGONALLY to the shared axis, or it does not vary at all — and
    only the second says a seat permutation can never move that finding. Printed beside the loading so
    the two are never confused, which is the mistake this reading exists to be safe against.
    """
    extremes: list[tuple[str, float]]

    @property
    def resolved(self) -> bool:
        """Whether the pool has a shared axis at all — the one verdict this reading is allowed to state."""
        return self.share > self.null

    def lines(self) -> list[str]:
        """The reading, with the bar beside the number so the share is never quoted alone."""
        ordered = sorted(self.loadings.items(), key=lambda pair: pair[1])
        profile = ScaleProfile.of(self.loadings)
        verdict = (
            "ONE shared axis: the pool's spread is a trade, and a permutation of it cannot move the mean"
            if self.resolved
            else "NO shared axis above the null: the members differ in ways that do not line up"
        )
        return [
            f"{len(self.extremes)} members x {len(self.findings)} findings",
            f"member quality took {self.quality:.1%} of the spread and was removed before the axis was read",
            f"component 1: {self.share:.1%} of what is left, against a null bar of {self.null:.1%}",
            f"  {verdict}",
            "  loading (spread), most negative first: "
            + ", ".join(f"{finding} {loading:+.3f} ({self.spread[finding]:.4f})" for finding, loading in ordered),
            profile.line(),
            "  ends of the axis: "
            + ", ".join(f"{name} {value:+.3f}" for name, value in self.extremes[:EXTREMES])
            + "  ...  "
            + ", ".join(f"{name} {value:+.3f}" for name, value in self.extremes[-EXTREMES:]),
        ]


class CompetenceAxis:
    """The members' per-finding competence, read for its shape rather than for its best subset."""

    def __init__(self, scores: LabelScores, seed: int = SHUFFLE_SEED) -> None:
        self.scores = scores
        self.seed = seed

    def table(self, pool: Pool) -> tuple[list[str], Float[np.ndarray, "members findings"]]:
        """One AUC per member per finding, with any finding undefined for ANY member dropped and named.

        Dropped whole rather than imputed: a NaN comes from a finding with one class present on the
        shared studies, and filling it with the column mean would place that member at the centre of the
        very axis being measured — an invented zero that can only make the pool look more one-dimensional
        than it is. The findings that survive ARE the denominator, which is why they travel in the
        reading rather than being counted off the twelve.
        """
        rows = [self.scores.per_label(member, pool.truth) for member in pool.predictions]
        findings = [finding for finding in rows[0] if all(np.isfinite(row[finding]) for row in rows)]
        if dropped := [finding for finding in rows[0] if finding not in findings]:
            logger.warning("undefined for at least one member on the shared studies, dropped: %s", ", ".join(dropped))
        return findings, np.array([[row[finding] for finding in findings] for row in rows], dtype=np.float64)

    def read(
        self,
        names: list[str],
        matrix: Float[np.ndarray, "members findings"],
        findings: list[str],
        resamples: int = NULL_RESAMPLES,
    ) -> AxisReading:
        """Double-centre, take the first component, and price it against a column-shuffled null."""
        if len(names) < MINIMUM_MEMBERS:
            message = (
                f"{len(names)} members cannot carry an axis — a double-centred pair has one direction by construction"
            )
            raise ValueError(message)
        centred = matrix - matrix.mean(axis=0, keepdims=True)
        residual = self._residual(matrix)
        quality = 1.0 - float((residual**2).sum() / (centred**2).sum())
        left, values, right = np.linalg.svd(residual, full_matrices=False)
        loadings = right[0] * np.sign(right[0][np.argmax(np.abs(right[0]))])
        projections = left[:, 0] * values[0] * np.sign(right[0][np.argmax(np.abs(right[0]))])
        order = np.argsort(projections)
        return AxisReading(
            findings=findings,
            share=self._share(values),
            null=self._null(residual, resamples),
            quality=quality,
            loadings=dict(zip(findings, loadings.tolist(), strict=True)),
            spread=dict(zip(findings, residual.std(axis=0).tolist(), strict=True)),
            extremes=[(names[index], float(projections[index])) for index in order],
        )

    def strongest(
        self,
        names: list[str],
        matrix: Float[np.ndarray, "members findings"],
        findings: list[str],
        wanted: tuple[str, ...],
    ) -> list[tuple[str, float]]:
        """Members ranked by their competence on a NAMED subset of findings, quality already removed.

        The selection rule every seat search on this board has used is the holdout MEAN, and the mean is
        the direction the pool is saturated in — member quality takes four fifths of the spread, so a
        ranking on it re-reads the same ordering the ledger already has. This ranks on the RESIDUAL: how
        much better a member is on these findings than its own overall level predicts. A member that is
        mediocre everywhere and unusually good on the meniscus sorts to the top here and nowhere else.

        Which is only worth doing for a subset named IN ADVANCE by a different measurement. Ranking on
        whichever findings look interesting after seeing the matrix is choosing and scoring on the same
        data, and this project has a name for the number that comes out of that.

        It ranks; it does not price. What a residual-strong member is worth in a roster is `seat_value`'s
        question, against the champion, and nothing here substitutes for it.
        """
        if missing := [finding for finding in wanted if finding not in findings]:
            message = f"not in this pool's scored findings: {', '.join(missing)}"
            raise ValueError(message)
        columns = [findings.index(finding) for finding in wanted]
        strength = self._residual(matrix)[:, columns].mean(axis=1)
        order = np.argsort(-strength)
        return [(names[index], float(strength[index])) for index in order]

    @staticmethod
    def _residual(matrix: Float[np.ndarray, "members findings"]) -> Float[np.ndarray, "members findings"]:
        """The matrix with finding difficulty and member quality both removed — every reading's input."""
        centred = matrix - matrix.mean(axis=0, keepdims=True)
        return centred - centred.mean(axis=1, keepdims=True)

    @staticmethod
    def _share(values: NDArray[np.float64]) -> float:
        """The first singular value's share of the total variance — the statistic, in one place."""
        return float(values[0] ** 2 / (values**2).sum())

    def _null(self, residual: Float[np.ndarray, "members findings"], resamples: int) -> float:
        """The share a pool with NO shared axis reaches, at `NULL_PERCENTILE`.

        Each finding's column is permuted across members independently, so every finding keeps the exact
        spread it was measured with and only the pairing between findings is destroyed. That is the
        precise null the claim needs: not "members are identical" — which no pool satisfies and which
        would make any share look enormous — but "members differ, and the differences do not line up".
        """
        generator = np.random.default_rng(self.seed)
        shares = []
        for _ in range(resamples):
            drawn = np.column_stack([generator.permutation(column) for column in residual.T])
            recentred = drawn - drawn.mean(axis=0, keepdims=True)
            recentred -= recentred.mean(axis=1, keepdims=True)
            shares.append(self._share(np.linalg.svd(recentred, compute_uv=False).astype(np.float64)))
        return float(np.percentile(shares, NULL_PERCENTILE))


def main() -> None:
    """Read whether a pool of holdout records spans one trade-off axis, using no card and no labels work."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--records", type=Path, nargs="+", required=True, help="runs sharing ONE split")
    command.parser.add_argument(
        "--largest-pool",
        action="store_true",
        help="keep the largest set that shares a split AND a label table instead of refusing on the first "
        "record that cannot be mixed — what makes a glob of the processed root runnable",
    )
    command.parser.add_argument("--resamples", type=int, default=NULL_RESAMPLES, help="shuffles behind the null bar")
    command.parser.add_argument(
        "--on",
        nargs="+",
        default=(),
        help="also rank the members by their residual competence on these findings — how much better each "
        "is there than its own overall level predicts. Name them from a measurement made BEFORE this one; "
        "a subset chosen off the matrix is chosen and scored on the same data.",
    )
    arguments = command.parsed()
    mix = RecordMix(list(arguments.records))
    if arguments.largest_pool and (narrowed := mix.narrow()):
        logger.warning("not mixable with the majority, dropped: %s", " ".join(narrowed))
    pool = mix.per_member()
    logger.info(
        "%d of %d records became members, over %d shared studies",
        len(pool.names),
        len(arguments.records),
        len(pool.truth),
    )
    axis = CompetenceAxis(LabelScores())
    findings, matrix = axis.table(pool)
    command.report(axis.read(pool.names, matrix, findings, arguments.resamples).lines())
    if wanted := tuple(arguments.on):
        ranked = axis.strongest(pool.names, matrix, findings, wanted)
        command.report(
            [
                f"residual competence on {', '.join(wanted)} — a RANKING, not a price:",
                *[f"  {name} {value:+.4f}" for name, value in ranked[:RANKED]],
            ]
        )


if __name__ == "__main__":
    main()
