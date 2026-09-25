"""What a completed kernel's log says about whether its output is the entry it was named for."""

import json
from pathlib import Path

from kneemri.kaggle.run_check import KernelLog

WHOLE = (
    "kit 53f81d3\n"
    "loaded and fingerprint-checked 7 of 7 members from /kaggle/input/datasets/dimiturnt/kneemri-members\n"
    "wrote 3 rows\n"
)
DROPPED = (
    "member vjepa_warm_s2026 (vjepa2l) would not load - dropping it from this entry\n"
    "loaded and fingerprint-checked 6 of 7 members from /kaggle/input/datasets/dimiturnt/kneemri-members\n"
    "ENTRY IDENTITY CHANGED: named 7 and ran 6 - Score this as the smaller ensemble\n"
)


def test_of():
    """Kaggle's JSON record form and the same run as plain text reduce to the same lines."""
    records = json.dumps([{"stream_name": "stderr", "data": line} for line in WHOLE.splitlines()])
    assert KernelLog.of(records).lines == KernelLog.of(WHOLE).lines


def test_read(tmp_path: Path):
    """A log on disk is read as `kaggle kernels output` leaves it."""
    path = tmp_path / "kneemri-vjepaseat.log"
    path.write_text(WHOLE, encoding="utf-8")
    assert KernelLog.read(path).seats() == (7, 7)


def test_seats():
    """The census is the run's own count, and absent when the run never got as far as printing it."""
    assert KernelLog.of(DROPPED).seats() == (6, 7)
    assert KernelLog.of("kit 53f81d3\n").seats() is None


def test_dropped():
    """A dropped seat names the member and the encoder it wanted, so the mount can be found."""
    assert KernelLog.of(DROPPED).dropped() == ["vjepa_warm_s2026 wanted vjepa2l"]
    assert KernelLog.of(WHOLE).dropped() == []


def test_changed():
    """The run's own announcement that it served something other than the entry it was named for."""
    assert KernelLog.of(DROPPED).changed()
    assert not KernelLog.of(WHOLE).changed()


def test_fell_back():
    """A two-producer run that shipped one producer's table says so, and a whole run does not."""
    fallback = WHOLE + "his program did not produce a table - shipping our six alone\n"
    assert KernelLog.of(fallback).fell_back()
    assert not KernelLog.of(WHOLE).fell_back()
    assert any("re-ship" in refusal for refusal in KernelLog.of(fallback).faults())

    stage_one_died = WHOLE + "wrote fallback submission.csv\nhis program exited 0 after 153s\n"
    assert any("stand-in" in refusal for refusal in KernelLog.of(stage_one_died).faults())


def test_faults():
    """A whole run refuses nothing; the farseat shape refuses on the drop, the mark and the count.

    That shape is `farseat_v1` verbatim (ref 55758082, 2026-08-25): six of seven members, scored under the
    seven-member name, and read for five days as evidence closing the cross-family diversity axis.
    """
    assert KernelLog.of(WHOLE).faults() == []
    refusals = KernelLog.of(DROPPED).faults()
    assert len(refusals) == 3, refusals
    assert any("vjepa_warm_s2026" in refusal for refusal in refusals)
    assert any("6 of 7" in refusal for refusal in refusals)
