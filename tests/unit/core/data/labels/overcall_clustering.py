"""Per-study over-call dispersion, over cell frames this test writes itself."""

import numpy as np
import pandas as pd

from core.data.labels.overcall_clustering import INDEPENDENT, OvercallClustering

POSITIVE = "YES"
UNCERTAIN = "UNK"


def _cells(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """(study, verdict, truth) triples in the shape `ReaderAgreement.cells` hands back."""
    return pd.DataFrame(rows, columns=["study", "verdict", "truth"])


def test_counts():
    """Only the reader's POSITIVE calls are the denominator, and a wrong one is the numerator."""
    cells = _cells(
        [
            ("a", POSITIVE, 1.0),
            ("a", POSITIVE, 0.0),
            ("a", UNCERTAIN, 0.0),
            ("b", POSITIVE, 0.0),
        ]
    )

    counts = OvercallClustering(cells).counts()

    assert list(counts.index) == ["a", "b"]
    assert counts.loc["a", "called"] == 2, "the UNK cell is not a call and cannot be an over-call"
    assert counts.loc["a", "overcalled"] == 1.0
    assert counts.loc["b", "overcalled"] == 1.0


def test_dispersion():
    """Counts drawn at the pooled rate with no per-study latent land near the independence null.

    The tolerance is wide on purpose: 200 studies of 6 calls is a sampling estimate of a variance ratio,
    not an identity, and a tight assertion here would fail on the seed rather than on the mechanism.
    """
    generator = np.random.default_rng(2026)
    rows = [(f"s{study}", POSITIVE, 0.0 if generator.random() < 0.3 else 1.0) for study in range(200) for _ in range(6)]

    ratio = OvercallClustering.dispersion(OvercallClustering(_cells(rows)).counts())

    assert abs(ratio - INDEPENDENT) < 0.4, "independent cells are not clustered"


def test_dispersion_reads_high_when_the_over_call_is_a_property_of_the_study():
    """Half the studies over-call everything and half nothing — the maximum a per-study latent can be."""
    rows = [(f"s{study}", POSITIVE, 0.0 if study % 2 else 1.0) for study in range(60) for _ in range(6)]

    ratio = OvercallClustering.dispersion(OvercallClustering(_cells(rows)).counts())

    assert ratio > 3.0, "a fully study-driven over-call is far wider than Binomial"


def test_resampled():
    """The interval is seeded, so a reading quoted from it can be quoted again unchanged."""
    rows = [(f"s{study}", POSITIVE, float(study % 3 > 0)) for study in range(40) for _ in range(5)]
    clustering = OvercallClustering(_cells(rows))
    counts = clustering.counts()

    drawn = clustering.resampled(counts)

    assert np.array_equal(drawn, clustering.resampled(counts)), "same seed, same interval"
    low, high = np.percentile(drawn, (2.5, 97.5))
    assert low <= OvercallClustering.dispersion(counts) <= high


def test_report():
    """Clustered and independent readings each print the decision they license, not just a number."""
    clustered = [(f"s{study}", POSITIVE, 0.0 if study % 2 else 1.0) for study in range(60) for _ in range(6)]
    generator = np.random.default_rng(7)
    flat = [(f"s{study}", POSITIVE, 0.0 if generator.random() < 0.3 else 1.0) for study in range(200) for _ in range(6)]

    assert any("CLUSTERED" in line for line in OvercallClustering(_cells(clustered)).report())
    assert any("INDEPENDENT within" in line for line in OvercallClustering(_cells(flat)).report())
