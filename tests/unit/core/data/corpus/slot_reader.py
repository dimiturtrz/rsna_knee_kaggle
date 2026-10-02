from pathlib import Path

import numpy as np
import pydicom
import torch

from core.data.corpus.acquisition import Acquisition
from core.data.corpus.crop_centre import Crop, ImageCentre, TissueCentre
from core.data.corpus.depth_match import Plane
from core.data.corpus.slot_reader import (
    ADJACENT,
    SPREAD,
    Adjacent,
    Band,
    FileMemo,
    Reading,
    SlotReader,
    Spacing,
    Spread,
)
from tests.unit.conftest import write_series


def test_sharing(tmp_path: Path):
    """A shared memo collects the reader's file opens, and the pixels are the same either way."""
    acquisition = Acquisition.probe("study", "series", write_series(tmp_path / "series", slices=4, size=32))
    memo = FileMemo()

    alone = SlotReader(image_size=32).read(acquisition, 2)
    shared = SlotReader(image_size=32).sharing(memo).read(acquisition, 2)

    assert alone is not None and shared is not None
    assert torch.equal(alone, shared), "a memo changes what is opened twice, never what comes back"
    assert memo.planes, "and the shared reader left its opens in the memo"


def test_ordered(tmp_path: Path):
    """The header parse happens once however many read groups ask for it, and the answer is the same."""
    acquisition = Acquisition.probe("study", "series", write_series(tmp_path / "series", slices=6, size=32))
    memo = FileMemo()

    first = memo.ordered(acquisition)
    assert first == acquisition.ordered_files(), "the memo answers what the acquisition answers"
    assert memo.ordered(acquisition) is first, "and the second ask is the same object, not a second parse"
    assert memo.header_opens == 1, "the counter records parses, not asks — that is what a census reads"


def test_plane(tmp_path: Path):
    """A FAILED decode is kept too — a file that will not open must not be opened again per group."""
    acquisition = Acquisition.probe("study", "series", write_series(tmp_path / "series", slices=2, size=32))
    memo = FileMemo()

    memo.plane(acquisition, "broken.dcm", None)

    assert memo.planes[(acquisition.series, "broken.dcm")] is None
    assert memo.slice_opens == 1, "a failed open is still an open — the census must not read it as free"
    assert memo.opened(acquisition, "broken.dcm"), "a failure is a recorded answer, not a missing one"


def test_opened(tmp_path: Path):
    """Whether some other read group has already decoded this file of this series."""
    acquisition = Acquisition.probe("study", "series", write_series(tmp_path / "series", slices=2, size=32))
    memo = FileMemo()

    assert not memo.opened(acquisition, "one.dcm")
    memo.plane(acquisition, "one.dcm", Plane(np.zeros((4, 4), dtype=np.float32), 16))
    assert memo.opened(acquisition, "one.dcm")


def test_read(tmp_path: Path):
    """A series becomes a fixed-shape uint8 stack, physically cropped and percentile-normalised."""
    acquisition = Acquisition.probe("study", "series", write_series(tmp_path / "series", slices=10, size=128))
    stack = SlotReader(image_size=32).read(acquisition, slices=3)
    assert stack is not None
    assert stack.shape == (3, 32, 32)
    assert stack.dtype == torch.uint8
    assert int(stack.max()) > int(stack.min()), "a normalised stack must use its range"


def test_read_returns_none_when_the_series_is_empty(tmp_path: Path):
    """Nothing decodable means the slot is ABSENT, which the mask can say — not black, which it cannot."""
    empty = tmp_path / "empty"
    empty.mkdir()
    acquisition = Acquisition("study", "series", empty, (), "UNK", False, None, None, None)
    assert SlotReader().read(acquisition, slices=3) is None


def test_read_repeats_the_last_index_when_the_band_is_thin(tmp_path: Path):
    """Asking for more slices than the central band holds pads rather than failing."""
    acquisition = Acquisition.probe("study", "series", write_series(tmp_path / "thin", slices=2, size=32))
    stack = SlotReader(image_size=16).read(acquisition, slices=6)
    assert stack is not None
    assert stack.shape == (6, 16, 16)


def test_cropped(tmp_path: Path):
    """Everything expensive and nothing size-dependent — so ONE read serves every size, exactly.

    This is the whole reason the split exists: a submission carrying members that read 392, 336 and 224
    px currently decodes every DICOM three times, because `predict_groups` decodes once per pixel group.
    Reading once and resizing per target is only worth doing if it is EXACT, and byte-identity against a
    native read at each size is what makes it so — a downsample between sizes would instead move every
    member that already scores.
    """
    acquisition = Acquisition.probe("study", "series", write_series(tmp_path / "series", slices=10, size=128))
    reader = SlotReader(image_size=32)

    stack = reader.cropped(acquisition, 3)
    assert stack is not None

    for size in (32, 16):
        native = SlotReader(image_size=size).read(acquisition, 3)
        assert native is not None
        assert torch.equal(reader.sized(stack, size), native), f"one read must equal a native read at {size}"


def test_cropped_fills_a_failed_slice_from_its_neighbour(tmp_path: Path):
    """A slice that will not decode is filled from the nearest one that did — never from zeros.

    Never from zeros because a zero plane is a candidate for the majority shape, and a black slice served
    behind a mask that says ACQUIRED is the failure this reader has already been bitten by twice. The
    neighbour is what the model would have been shown anyway if the band had landed one file over.
    """
    directory = write_series(tmp_path / "torn", slices=10, size=64)
    acquisition = Acquisition.probe("study", "series", directory)
    # THE PIXELS ARE TORN, THE HEADER IS NOT — a file whose header is unreadable also loses the series
    # its geometric order (`ordered_files` falls back to name order), which would move which slice the
    # band samples and hide what this test is about.
    torn = directory / acquisition.ordered_files()[4]
    data = pydicom.dcmread(str(torn))
    data.PixelData = data.PixelData[: len(data.PixelData) // 2]
    data.save_as(str(torn))

    stack = SlotReader(image_size=16).cropped(acquisition, slices=3)

    assert stack is not None
    assert stack.shape == (3, 64, 64), "one torn slice does not change the stack the slot is served at"
    assert stack[1].any(), "the failed middle slice carries pixels, not a black plane"
    assert np.array_equal(stack[1], stack[0]), "and they are its nearest decoded neighbour's"
    assert not np.array_equal(stack[2], stack[0]), "the slices that did decode are still their own"


def test_cropped_rescales_an_eight_bit_slice_onto_its_series(tmp_path: Path):
    """A slice stored one byte per pixel under a 16-bit header is served on its neighbours' scale.

    The raw corpus holds two such files. Before, one failed to decode and was replaced by its
    neighbour; read at its true depth, it was 25-30% darker than the stack it sat in.
    """
    directory = write_series(tmp_path / "eight", slices=10, size=64)
    acquisition = Acquisition.probe("study", "series", directory)
    ordered = acquisition.ordered_files()
    reader = SlotReader(image_size=16)
    neighbour = pydicom.dcmread(str(directory / ordered[reader.band_indices(10, 3)[0]])).pixel_array
    eight = directory / ordered[reader.band_indices(10, 3)[1]]
    data = pydicom.dcmread(str(eight))
    data.PixelData = (neighbour // 16).astype(np.uint8).tobytes()
    data.save_as(str(eight))

    stack = reader.cropped(acquisition, slices=3)

    assert stack is not None
    assert not np.array_equal(stack[1], stack[0]), "its own pixels, not a neighbour fill"
    assert np.allclose(stack[1], stack[0], atol=16), "on the neighbour's scale, to within one byte step"


def test_cropped_returns_none_when_no_sampled_slice_decodes(tmp_path: Path):
    """A series with files that all fail is ABSENT, which the mask can say — not black, which it cannot.

    Distinct from the empty-directory path: here the series HAS files and an `Acquisition` probed from
    real headers, so the reader gets past `ordered_files` and only the decode loop can report the loss.
    """
    directory = write_series(tmp_path / "unreadable", slices=4, size=64)
    acquisition = Acquisition.probe("study", "series", directory)
    for name in acquisition.ordered_files():
        (directory / name).write_bytes(b"not a dicom")

    assert SlotReader(image_size=16).cropped(acquisition, slices=3) is None


def test_cropped_lets_the_majority_shape_decide(tmp_path: Path):
    """One odd-sized slice is dropped — it must not drop every OTHER slice by being sampled first.

    Slices of one series can differ in matrix size (multi-echo, some reformats), and those are not
    stackable, so the odd one out is zeroed. The authority for "odd" used to be the FIRST sampled plane,
    which is a property of where the band happens to land and not of the series: a series whose first
    sampled slice is the outlier had every other slice zeroed instead, and the presence mask still
    reported the slot acquired — so the model was served a black slot as if it were a real one.
    """
    directory = write_series(tmp_path / "mixed", slices=10, size=64)
    acquisition = Acquisition.probe("study", "series", directory)
    odd = directory / acquisition.ordered_files()[1]
    data = pydicom.dcmread(str(odd))
    data.Rows, data.Columns = 32, 32
    data.PixelData = np.full((32, 32), 500, np.uint16).tobytes()
    data.save_as(str(odd))

    stack = SlotReader(image_size=16).cropped(acquisition, slices=3)

    assert stack is not None
    assert stack.shape == (3, 64, 64), "the shape the SERIES mostly has, not the one sampled first"
    assert not stack[0].any(), "the odd slice out is dropped"
    assert stack[1].any() and stack[2].any(), "and it takes no good slice with it"


def test_sized(tmp_path: Path):
    """The stack becomes uint8 on the requested grid, whatever the reader's own size is."""
    acquisition = Acquisition.probe("study", "series", write_series(tmp_path / "series", slices=10, size=128))
    reader = SlotReader(image_size=32)
    stack = reader.cropped(acquisition, 3)
    assert stack is not None

    small = reader.sized(stack, 16)

    assert small.shape == (3, 16, 16)
    assert small.dtype == torch.uint8


def test_band_indices():
    """Indices spread over the central band, repeating the last when the band is thinner than asked.

    PUBLIC because the band is a reading rule and not a DICOM one — a second corpus arrives as whole
    pickled volumes and cannot enter through `cropped`, but must be sampled by the SAME rule or the two
    corpora differ on an axis that is measured while claiming not to.
    """
    reader = SlotReader(band=Band(0.0, 1.0))

    assert reader.band_indices(10, 3) == [0, 4, 9], "the whole stack, ends included"
    assert SlotReader().band_indices(11, 3) == [2, 5, 8], "the historic band keeps its own indices"
    assert SlotReader(band=Band(0.1, 0.9)).band_indices(11, 3) == [1, 5, 9]
    assert reader.band_indices(2, 5) == [0, 1, 1, 1, 1], "a thin band pads the deepest rather than failing"
    assert SlotReader().band_indices(2, 3) == [1, 1, 1], "a band too thin to have two ends takes the middle"
    assert len(reader.band_indices(37, 12)) == 12, "always exactly as many planes as the cache holds"


def test_indices():
    """The base rule is a strategy and not a default — a spacing that forgot to answer must not read."""
    try:
        Spacing().indices(0, 9, 3)
    except NotImplementedError:
        return
    raise AssertionError("the base spacing must not silently answer for a rule that was never written")


def test_spread_indices():
    """Evenly across the band, de-duplicated — the historic rule every member on disk was fitted on."""
    assert Spread().indices(0, 9, 3) == [0, 4, 9]
    assert Spread().indices(4, 6, 5) == [4, 5, 6], "a band thinner than asked de-duplicates rather than repeating"


def test_adjacent_indices():
    """A contiguous run centred in the band, slid inside rather than clipped when it will not fit."""
    assert Adjacent().indices(0, 29, 9) == list(range(10, 19)), "consecutive, and centred on the band"
    assert Adjacent().indices(20, 29, 9) == list(range(20, 29)), "a run that would hang off the end slides back"
    assert Adjacent().indices(4, 6, 9) == [4, 5, 6], "a band shorter than the run gives what it has, to pad"


def test_spacing_suffix():
    """The historic spacing adds nothing to a cache name; any other one names its own file."""
    assert Reading().spacing_suffix() == "", "the default must keep resolving to the arrays already on disk"
    assert Reading(spacing=ADJACENT).spacing_suffix() == "_adjacent"
    assert Reading(spacing=ADJACENT).suffix().endswith("_adjacent"), "and it must reach the cache key"


def test_band_indices_spacing():
    """The same band under the two rules: spread samples are never adjacent, adjacent ones always are."""
    spread = SlotReader(band=Band(0.1, 0.9), spacing=SPREAD).band_indices(30, 9)
    adjacent = SlotReader(band=Band(0.1, 0.9), spacing=ADJACENT).band_indices(30, 9)

    assert min(np.diff(spread)) > 1, "the historic rule cannot show two consecutive millimetres, at any width"
    assert list(np.diff(adjacent)) == [1] * 8, "which is the whole mechanism the adjacent rule exists for"
    assert len(adjacent) == 9
    assert SlotReader(spacing=ADJACENT).band_indices(2, 3) == [1, 1, 1], "a band too thin still takes the middle"


def _offset_stack() -> np.ndarray:
    """A stack whose only signal is a block well away from the image centre."""
    stack = np.zeros((4, 200, 200), np.float32)
    stack[:, 120:160, 130:170] = 1.0
    return stack


def test_window():
    """The extent it takes, clamped into the frame, and None whenever the crop does not apply."""
    stack = np.zeros((2, 200, 200), np.float32)
    reader = SlotReader(crop_mm=100.0)

    taken = reader.window(stack, 1.0)
    assert taken is not None
    assert (taken[0].start, taken[0].stop, taken[1].start, taken[1].stop) == (50, 150, 50, 150)

    edge = SlotReader(crop_mm=100.0, centre=TissueCentre()).window(_edge_stack(), 1.0)
    assert edge is not None
    assert edge[0].stop - edge[0].start == 100, "clamped rather than short — a short crop is a different mm/pixel"
    assert edge[0].start == 0

    assert reader.window(stack, None) is None, "unknown spacing is not a physical extent"
    assert reader.window(stack, 0.0) is None
    assert reader.window(stack, 1.0, 400.0) is None, "a crop wider than the frame does nothing"
    assert reader.window(stack, 1.0, 4.0) is None, "and one below a handful of pixels is nonsense"


def _edge_stack() -> np.ndarray:
    """A stack whose tissue sits hard against the top edge, so an unclamped window would run off it."""
    stack = np.zeros((2, 200, 200), np.float32)
    stack[:, 0:20, 0:20] = 1.0
    return stack


def test_crop_centre_default_is_unchanged():
    """The default reader crops EXACTLY where it always did — 99 GB of caches depend on it.

    Changing where the crop is taken changes the pixels, so a silent switch would invalidate every
    cached array and every member fitted on one, and no fingerprint would catch it: each is checked
    against its own reading.
    """
    stack = _offset_stack()
    reader = SlotReader(crop_mm=100.0, image_size=64)

    default = reader._crop(stack, spacing=1.0)
    centred = SlotReader(crop_mm=100.0, image_size=64, centre=ImageCentre())._crop(stack, spacing=1.0)

    assert np.array_equal(default, centred), "the default IS the image centre"
    tissue = SlotReader(crop_mm=100.0, image_size=64, centre=TissueCentre())._crop(stack, spacing=1.0)
    assert not np.array_equal(default, tissue), "and a tissue centre really does read different pixels"
    assert tissue.shape == default.shape, "the physical extent is the same either way"


def test_narrowed():
    """A tight extent is taken from the stack the wide read already produced, not decoded again.

    That is what lets SCALE be a slot axis rather than a second decode: 80mm is a subset of 130mm, so
    the multi-scale scheme costs encoder passes and no extra I/O.
    """
    stack = np.zeros((2, 200, 200), np.float32)
    reader = SlotReader(crop_mm=130.0, image_size=64)

    assert reader.narrowed(stack, 1.0, None).shape == stack.shape, "no extent named means untouched"
    assert reader.narrowed(stack, 1.0, 200.0).shape == stack.shape, "a WIDER extent cannot widen a crop"

    tighter = reader.narrowed(stack, 1.0, 80.0)
    assert tighter.shape[1] == 80, "at 1 mm per pixel an 80 mm extent is 80 pixels"
    assert tighter.shape[1] < stack.shape[1]


def test_band_suffix():
    """The historic band names nothing; any other names itself, because it is a different cache.

    Empty by design and not by accident: those arrays are memory-mapped by running trainers and every
    member on disk was fitted on them, so a default that suddenly named itself would send a run looking
    for a file that does not exist.
    """
    assert Band().suffix() == "", "the historic reading must keep resolving to the file it always did"
    assert Band(0.1, 0.9).suffix() == "_0.1-0.9band"
    assert Band(0.0, 1.0).suffix() == "_0-1band", "the whole stack is a band like any other"


def test_recorded():
    """A manifest holds a list, because JSON has no tuples."""
    assert Band(0.1, 0.9).recorded() == [0.1, 0.9]


def test_of():
    """Read back from the list a manifest holds."""
    assert Band.of([0.1, 0.9]) == Band(0.1, 0.9)
    assert Band.of(Band().recorded()) == Band(), "a round trip through the manifest is the same reading"


def test_reader():
    """The geometry builds the reader, so a call site cannot reach past it and forget an axis."""
    reading = Reading(crop=Crop(mm=80.0, centre="tissue"), band=Band(0.1, 0.9))
    reader = reading.reader(224)

    assert (reader.crop_mm, reader.image_size, reader.band) == (80.0, 224, Band(0.1, 0.9))
    assert isinstance(reader.centre, TissueCentre), "the centre is the STRATEGY the crop names, not its name"


def test_reading_suffix():
    """Both axes compose into the cache's name, and the historic pair still names nothing."""
    assert Reading().suffix() == "", "the historic geometry must keep resolving to the file it always did"
    assert Reading(crop=Crop(mm=80.0), band=Band(0.1, 0.9)).suffix() == "_80mm_0.1-0.9band"
