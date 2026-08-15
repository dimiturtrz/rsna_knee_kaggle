"""The benchmark's arithmetic, on a workload that needs no corpus."""

from kneemri.corpus.decode_bench import DecodeBench

STRATEGIES = 5
"""serial, threads at 4 and 8, processes at 4 and 8."""


def square(item: str) -> int:
    """Trivial, picklable work — a process pool must be able to reach it by name."""
    return len(item) ** 2


def test_timings():
    """Every strategy is timed over the same items, and the warm-up is not one of the entries."""
    measured = DecodeBench().timings(square, ["a", "bb", "ccc"])

    assert len(measured) == STRATEGIES
    assert "serial" in measured and "processes(4)" in measured
    assert all(seconds >= 0.0 for seconds in measured.values())


def test_report():
    """Each strategy against serial, since the question is whether a pool earns its complexity."""
    lines = DecodeBench().report({"serial": 20.0, "processes(4)": 10.0}, studies=100)

    assert "serial" in lines[0]
    assert "0.20s/study" in lines[0]
    assert "1.00x" in lines[0], "serial is its own baseline"
    assert "2.00x" in lines[1], "half the time is twice the speed"
