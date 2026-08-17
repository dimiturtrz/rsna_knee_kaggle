from pathlib import Path

import pydicom

from kneemri.corpus.acquisition import Acquisition

from ..conftest import write_series


def _probe(directory: Path) -> Acquisition:
    return Acquisition.probe("study", "series", directory)


def test_probe(series: Path):
    """The header yields the acquisition axes: PD is fluid-sensitive, ScanOptions FS is suppressed."""
    acquisition = _probe(series)
    assert acquisition.weighting == "PD"
    assert acquisition.fat_suppressed
    assert acquisition.pixel_spacing == 0.3
    assert acquisition.laterality_tag == "L"
    assert acquisition.centre_x is not None


def test_probe_recovers_weighting_from_tr_te_when_undescribed(tmp_path: Path):
    """With nothing in the description, TR and TE decide: short TR is T1, long TR with long TE is T2."""
    t1 = _probe(write_series(tmp_path / "t1", description="knee", repetition=500.0, echo=12.0, scan_options=""))
    t2 = _probe(write_series(tmp_path / "t2", description="knee", repetition=3000.0, echo=90.0, scan_options=""))
    pd = _probe(write_series(tmp_path / "pd", description="knee", repetition=3000.0, echo=20.0, scan_options=""))
    assert (t1.weighting, t2.weighting, pd.weighting) == ("T1", "T2", "PD")
    assert not t1.fluid_sensitive
    assert t2.fluid_sensitive and pd.fluid_sensitive


def test_probe_settles_gradient_echo_before_tr(tmp_path: Path):
    """GRE is read from the sequence, BEFORE TR gets a vote — its TR is short by design and reads as T1.

    A gradient-echo series called T1 is not a naming slip: it changes which slot the series is routed
    into, so a GRE sequence would compete for a slot whose contrast it does not have.
    """
    directory = write_series(tmp_path / "gre", description="knee", repetition=500.0, echo=12.0, scan_options="")
    for name in sorted(path.name for path in directory.glob("*.dcm")):
        data = pydicom.dcmread(str(directory / name))
        data.ScanningSequence = "GR"
        data.save_as(str(directory / name))

    assert _probe(directory).weighting == "GRE", "short TR alone would have said T1"


def test_probe_says_unknown_rather_than_guessing_without_tr(tmp_path: Path):
    """No description, no sequence and no TR leaves nothing to decide on — so it says so.

    UNK is a real answer here: the slot predicates can decline a series whose weighting is unknown,
    which they cannot do for one confidently mislabelled T1.
    """
    directory = write_series(tmp_path / "bare", description="knee", scan_options="")
    for name in sorted(path.name for path in directory.glob("*.dcm")):
        data = pydicom.dcmread(str(directory / name))
        del data.RepetitionTime
        data.save_as(str(directory / name))

    acquisition = _probe(directory)

    assert acquisition.weighting == "UNK"
    assert not acquisition.fluid_sensitive, "an unknown weighting is not treated as fluid-bright"


def test_probe_of_a_directory_with_no_slices(tmp_path: Path):
    """A series directory holding no DICOM is probed rather than skipped, and every axis reads absent.

    Probed because the routing scheme scores candidates and needs one to score: raising here would make
    a single empty series directory fail a whole study, and the axes it returns (UNK, unsuppressed, no
    spacing) lose every slot predicate on their own.
    """
    empty = tmp_path / "nothing"
    empty.mkdir()

    acquisition = _probe(empty)

    assert acquisition.files == ()
    assert acquisition.weighting == "UNK"
    assert (acquisition.pixel_spacing, acquisition.laterality_tag, acquisition.centre_x) == (None, None, None)
    assert not acquisition.fat_suppressed
    assert acquisition.slice_count == 0


def test_probe_treats_an_empty_laterality_tag_as_absent(tmp_path: Path):
    """Type 2C means the tag may be present and empty — which `hasattr` would call present."""
    directory = write_series(tmp_path / "blank", laterality="")
    assert _probe(directory).laterality_tag is None


def test_ordered_files(series: Path):
    """Ordering follows the geometry, and the file names here deliberately run the other way."""
    acquisition = _probe(series)
    assert acquisition.ordered_files() != acquisition.files
    assert sorted(acquisition.ordered_files()) == sorted(acquisition.files)
    assert acquisition.ordered_files() == tuple(reversed(acquisition.files))


def test_fluid_sensitive(series: Path):
    """PD and T2 are the fluid-bright weightings."""
    assert _probe(series).fluid_sensitive


def test_slice_count(series: Path):
    """The slice count is the tie-break when a slot has several candidate series."""
    assert _probe(series).slice_count == 8
