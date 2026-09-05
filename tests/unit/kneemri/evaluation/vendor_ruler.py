import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.scores import LabelScores
from kneemri.evaluation.vendor_ruler import MIN_STRATUM, SEATS, VendorRuler
from kneemri.evaluation.vendor_strata import Stratum

STUDIES = 400
MEMBERS = 10


def _pool() -> Pool:
    """Ten members of deliberately DECREASING quality over the same truth, so the pooled order is known."""
    generator = np.random.default_rng(0)
    truth = (generator.random((STUDIES, len(TARGETS))) > 0.5).astype(np.float32)
    predictions = [
        (truth + generator.normal(0.0, 0.4 + 0.1 * index, truth.shape)).astype(np.float64) for index in range(MEMBERS)
    ]
    return Pool([f"member{index}" for index in range(MEMBERS)], predictions, truth)


def _ruler(strata: list[Stratum] | None = None) -> VendorRuler:
    pool = _pool()
    everything = np.arange(STUDIES, dtype=np.int64)
    return VendorRuler(pool, strata or [Stratum("all", everything)], LabelScores(), np.random.default_rng(1))


def test_scored():
    """Members and truth are restricted TOGETHER, and the built-in quality order comes back."""
    scored = _ruler().scored(np.arange(0, 200, dtype=np.int64))

    assert len(scored) == MEMBERS
    assert scored[0] > scored[-1], "member0 carries the least noise by construction"
    assert all(0.0 <= value <= 1.0 for value in scored)


def test_agreement():
    """Spearman over midranks: identical is 1, reversed is -1, and a constant ordering is undefined."""
    ordering = [0.9, 0.8, 0.7, 0.6]

    assert VendorRuler.agreement(ordering, ordering) == pytest.approx(1.0)
    assert VendorRuler.agreement(ordering, list(reversed(ordering))) == pytest.approx(-1.0)
    assert np.isnan(VendorRuler.agreement(ordering, [0.5] * 4)), "a flat side has no ordering to agree with"


def test_null_band():
    """The band a random subset of a given size earns, which is what a vendor must fall below."""
    ruler = _ruler()
    whole = ruler.scored(np.arange(STUDIES, dtype=np.int64))

    low, high = ruler.null_band(MIN_STRATUM, whole)

    assert low <= high
    assert high <= 1.0
    assert low > 0.0, "random subsets of the same corpus still broadly agree with the pooled order"


def test_seats():
    """The names a seat search would take, under one ordering — the best `SEATS`, not the first."""
    ruler = _ruler()
    scored = list(reversed([float(index) for index in range(MEMBERS)]))

    assert ruler.seats(scored) == {f"member{index}" for index in range(SEATS)}


def test_line():
    """A stratum under the floor states its size and declines; one above it reports rho beside its band."""
    ruler = _ruler()
    whole = ruler.scored(np.arange(STUDIES, dtype=np.int64))
    seats = ruler.seats(whole)

    small = ruler.line(Stratum("tiny", np.arange(10, dtype=np.int64)), whole, seats)
    assert "not read" in small and "10 studies" in small

    real = ruler.line(Stratum("Siemens", np.arange(200, dtype=np.int64)), whole, seats)
    assert "rho" in real and "null" in real and "null band" in real
    assert f"top-{SEATS} kept" in real


def test_report():
    """The header carries the denominators, and every stratum gets exactly one row."""
    strata = [Stratum("a", np.arange(200, dtype=np.int64)), Stratum("b", np.arange(200, STUDIES, dtype=np.int64))]

    lines = VendorRuler(_pool(), strata, LabelScores(), np.random.default_rng(2)).report()

    assert f"{MEMBERS} members over {STUDIES} shared studies" in lines[0]
    assert sum(line.strip().startswith(("a ", "b ")) for line in lines) == 2
