from pathlib import Path

import pydicom

from kneemri.corpus.acquisition import Acquisition, SeriesHeader

from ..conftest import write_series


def _probe(directory: Path) -> Acquisition:
    return Acquisition.probe("study", "series", directory)


def _header(directory: Path) -> SeriesHeader:
    return SeriesHeader.of(directory / sorted(path.name for path in directory.glob("*.dcm"))[0])


def test_of(series: Path):
    """A header is read WITHOUT its pixels — a probe over 4,400 studies must not pay to decode them."""
    header = SeriesHeader.of(series / sorted(path.name for path in series.glob("*.dcm"))[0])
    assert header.header.SeriesDescription
    assert "PixelData" not in header.header


def test_of_a_file_that_is_not_dicom_reads_as_an_empty_header(tmp_path: Path):
    """`force=True` means an unreadable file yields a header with nothing in it, not a raise.

    Every axis then answers absent by the same five rules the empty-directory case uses, and the slot
    predicates decline the series — which is what a series nothing can be said about deserves.
    """
    junk = tmp_path / "junk.dcm"
    junk.write_bytes(b"not a dicom")

    header = SeriesHeader.of(junk)

    assert header.weighting == "UNK"
    assert (header.pixel_spacing, header.laterality_tag, header.centre_x) == (None, None, None)
    assert not header.fat_suppressed


def test_weighting(tmp_path: Path):
    """The description decides where it says so, and separators do not hide the word it says it with."""
    assert _header(write_series(tmp_path / "d", description="SAG_PD-FSE")).weighting == "PD"
    assert _header(write_series(tmp_path / "t2", description="ax t2 fse")).weighting == "T2"


def test_fat_suppressed(tmp_path: Path):
    """ScanOptions is matched as whole tokens: GE's SAT_GEMS is spatial saturation, not fat suppression."""
    assert _header(write_series(tmp_path / "fs", description="knee", scan_options="FS")).fat_suppressed
    assert not _header(write_series(tmp_path / "sat", description="knee", scan_options="SAT_GEMS")).fat_suppressed


def test_pixel_spacing(tmp_path: Path):
    """Row spacing in mm — the number that, with the crop extent, fixes the effective resolution."""
    assert _header(write_series(tmp_path / "s")).pixel_spacing == 0.3


def test_laterality_tag(tmp_path: Path):
    """Present-but-empty is Type 2C's own case, and reads absent rather than as a side."""
    assert _header(write_series(tmp_path / "l", laterality="L")).laterality_tag == "L"
    assert _header(write_series(tmp_path / "blank", laterality="")).laterality_tag is None


def test_centre_x(tmp_path: Path):
    """The CENTRE, not the corner: half a field of view is enough to reverse the sign near the midline."""
    # An AXIAL series, where the in-plane axes carry x: a sagittal one has no x component in either
    # cosine, so the corner and the centre coincide there and the distinction cannot show.
    header = _header(write_series(tmp_path / "c", normal=(0.0, 0.0, 1.0), spacing=0.3, size=64))
    corner = float(header.header.ImagePositionPatient[0])

    assert corner == 0.0
    assert header.centre_x == 0.3 * 64 / 2, "half a 64-column field at 0.3 mm — the corner would say 0.0"


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


def test_ordered_files_keeps_sorting_around_one_unreadable_header(tmp_path: Path):
    """One file with no readable key costs its OWN slot, not the whole series' ordering.

    Abandoning the sort for the series is what this used to do, and file names here are arbitrary UIDs
    exactly as they are in the corpus — so the central band would then sample by name, which says
    nothing about depth, while the presence mask still reported the slot acquired. Unmeasurable on this
    corpus (0 of 1682 series over 300 studies hold a keyless file) and cheap insurance for the hidden
    test, where a vendor we have never seen decides what the tags say.
    """
    directory = write_series(tmp_path / "mixed", slices=6)
    acquisition = _probe(directory)
    keyless = acquisition.files[2]
    (directory / keyless).write_bytes(b"not a dicom")

    ordered = _probe(directory).ordered_files()

    assert ordered[2] == keyless, "the file nothing can be said about stays where it was"
    assert list(ordered[:2] + ordered[3:]) == [name for name in reversed(acquisition.files) if name != keyless], (
        "and every file that HAS a key is still sorted by geometry"
    )


def test_fluid_sensitive(series: Path):
    """PD and T2 are the fluid-bright weightings."""
    assert _probe(series).fluid_sensitive


def test_slice_count(series: Path):
    """The slice count is the tie-break when a slot has several candidate series."""
    assert _probe(series).slice_count == 8
