"""Header census of the competition's TEST mount, printed beside TRAIN's — an instrument, not an entry.

The 0.010 to the leader is invisible to every offline ruler this project owns, and one thing that looks
exactly like that from the inside is a test population drawn differently from train. No offline number
can see it, because every offline number is computed on train. A kernel run spends no submission slot,
so this question is answerable for free and has never been asked.

It writes NO submission.csv, loads no member, and takes no GPU. It reads ONE DICOM header per series with
`stop_before_pixels=True` and prints, for train and test alike, the distribution of the acquisition facts
a model is sensitive to: manufacturer, field strength, sequence and plane, slice count, in-plane spacing,
slice thickness, and series per study.

READ THE DENOMINATOR, NOT THE RATE. Every table below prints `n / total` and the number of series whose
header could not be read. A distribution over the series a header parse happened to succeed on is not a
distribution over the mount, and that distinction has cost this project a P1 three times.
"""

import collections
import logging
import sys
from pathlib import Path

import pydicom

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("census")

SPLIT_DIRECTORIES = {"train": ("train_series", "train_images"), "test": ("test_series", "test_images")}
"""The two spellings the competition mount has used for each split; the first that exists wins."""

TAGS = {
    "manufacturer": "Manufacturer",
    "model": "ManufacturerModelName",
    "field_strength": "MagneticFieldStrength",
    "sequence": "SeriesDescription",
    "orientation": "ImageOrientationPatient",
    "thickness": "SliceThickness",
    "spacing": "PixelSpacing",
    "rows": "Rows",
    "columns": "Columns",
    "tr": "RepetitionTime",
    "te": "EchoTime",
    "fat_sat": "ScanOptions",
}

PLANE_AXES = ("sagittal", "coronal", "axial")
"""The plane a direction cosine triple points at, indexed by which component is largest."""

ORIENTATION_LENGTH = 6
"""ImageOrientationPatient is two direction cosines: three components each."""

TOP_ROWS = 12
"""How many values of a categorical tag to print. Enough to see a tail, short enough to read."""

QUANTILES = (0.0, 0.1, 0.5, 0.9, 1.0)
"""min / p10 / median / p90 / max — the shape of a numeric tag, not just its centre."""

CATEGORICAL = ("manufacturer", "model", "field_strength", "plane", "sequence", "fat_sat")
NUMERIC = ("slices", "thickness", "spacing", "rows", "columns", "tr", "te")


def competition_root() -> Path:
    """The mounted competition directory, found without walking the study tree beneath it."""
    for candidate in sorted(Path("/kaggle/input").iterdir()):
        if any((candidate / name).is_dir() for names in SPLIT_DIRECTORIES.values() for name in names):
            return candidate
    message = "no competition mount under /kaggle/input holds a train/test series directory"
    raise FileNotFoundError(message)


def split_root(root: Path, split: str) -> Path | None:
    """The directory holding one subdirectory per study for this split, or None if the mount lacks it."""
    for name in SPLIT_DIRECTORIES[split]:
        if (root / name).is_dir():
            return root / name
    return None


def plane_of(orientation: object) -> str:
    """The acquisition plane named by the direction cosines, or 'unknown' when the tag is absent.

    The normal of the image plane is the cross product of the two row/column cosines, and the axis its
    largest component points along is the plane's name. Computed rather than read off SeriesDescription
    because a description is free text and this is geometry.
    """
    if not isinstance(orientation, (list, pydicom.multival.MultiValue)) or len(orientation) != ORIENTATION_LENGTH:
        return "unknown"
    cosines = [float(value) for value in orientation]
    row, column = cosines[:3], cosines[3:]
    normal = [
        row[1] * column[2] - row[2] * column[1],
        row[2] * column[0] - row[0] * column[2],
        row[0] * column[1] - row[1] * column[0],
    ]
    return PLANE_AXES[max(range(len(PLANE_AXES)), key=lambda axis: abs(normal[axis]))]


def series_header(series: Path) -> dict[str, object] | None:
    """One header per series, read without pixels, plus the series' own slice count.

    None when the directory holds no readable DICOM at all — counted as a failure rather than dropped,
    because a silent drop turns the denominator into the subset that happened to parse.
    """
    files = sorted(series.glob("*.dcm")) or sorted(path for path in series.iterdir() if path.is_file())
    if not files:
        return None
    try:
        header = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
    except Exception:  # noqa: BLE001 - a census must survive one unreadable series and say so
        return None
    read = {name: getattr(header, attribute, None) for name, attribute in TAGS.items()}
    read["plane"] = plane_of(read.pop("orientation"))
    read["slices"] = len(files)
    return read


def census(root: Path, split: str) -> tuple[list[dict[str, object]], int, int]:
    """Every series of one split, as headers, with the counts needed to state a denominator."""
    directory = split_root(root, split)
    if directory is None:
        return [], 0, 0
    studies = sorted(path for path in directory.iterdir() if path.is_dir())
    headers: list[dict[str, object]] = []
    attempted = 0
    for study in studies:
        for series in sorted(path for path in study.iterdir() if path.is_dir()) or [study]:
            attempted += 1
            read = series_header(series)
            if read is not None:
                headers.append({**read, "study": study.name})
    return headers, attempted, len(studies)


def report(name: str, headers: list[dict[str, object]], attempted: int, studies: int) -> None:
    """One split's distributions, every one of them carrying its own denominator."""
    logger.info("")
    logger.info("=== %s: %d series read of %d attempted, over %d studies ===", name, len(headers), attempted, studies)
    if not headers:
        return
    per_study = collections.Counter(str(header["study"]) for header in headers)
    logger.info("series per study: %s", dict(sorted(collections.Counter(per_study.values()).items())))

    for field in CATEGORICAL:
        seen = collections.Counter(str(header[field]) for header in headers)
        logger.info("--- %s (%d distinct) ---", field, len(seen))
        for value, count in seen.most_common(TOP_ROWS):
            logger.info("  %6d  %5.1f%%  %s", count, 100.0 * count / len(headers), value[:70])

    for field in NUMERIC:
        values: list[float] = []
        for header in headers:
            raw = header[field]
            first = raw[0] if isinstance(raw, (list, pydicom.multival.MultiValue)) and len(raw) else raw
            try:
                values.append(float(first))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
        if not values:
            logger.info("--- %s: 0 of %d parsed as a number ---", field, len(headers))
            continue
        values.sort()
        spread = [values[int(fraction * (len(values) - 1))] for fraction in QUANTILES]
        logger.info(
            "--- %s: n=%d of %d  min %.3f  p10 %.3f  median %.3f  p90 %.3f  max %.3f",
            field,
            len(values),
            len(headers),
            *spread,
        )


def main() -> None:
    """Census both splits and print them one after the other, for a reader to difference by eye."""
    root = competition_root()
    logger.info("competition mount: %s", root)
    for split in ("train", "test"):
        headers, attempted, studies = census(root, split)
        report(split, headers, attempted, studies)
    logger.info("")
    logger.info("no submission written: this kernel is an instrument and spends no slot")


if __name__ == "__main__":
    main()
