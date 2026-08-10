"""Probe the Kaggle runtime and the corpus, and leave a scoreable submission behind.

Answers, in one unlimited-cost kernel push, the questions that would otherwise be learned hours into a
real run: what accelerator we actually got, which libraries are present offline, where the data mounts,
how fast a DICOM series reads off the mount, and the corpus facts that decide the data layer
(knee-61q, and items 1/5/6/7/8 of knee-25n — none of which need the 247 GB local download).

The submission is written FIRST and never overwritten: a probe that crashes halfway still leaves a valid
file, so the run is never wasted. It is a copy of sample_submission.csv, so the expected leaderboard
score is exactly 0.5 — the mean of twelve AUCs at chance. Any other value falsifies our reading of
either the file format or the metric.
"""

import hashlib
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

WORKING = Path("/kaggle/working")
RULE = "=" * 78


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}", flush=True)


def find_input() -> Path:
    """Locate the competition mount rather than assuming it.

    Version 1 assumed `/kaggle/input/<competition-slug>` and died on FileNotFoundError before any other
    probe ran. The mount name is not guaranteed to equal the slug, so it is discovered from the one file
    that must exist — and the whole listing is printed either way, because a wrong guess here is exactly
    the class of thing this kernel exists to settle.
    """
    root = Path("/kaggle/input")
    print(f"contents of {root}:")
    for entry in sorted(root.glob("*")):
        print(f"  {entry.name}/")
        for child in sorted(entry.glob("*"))[:12]:
            print(f"      {child.name}{'/' if child.is_dir() else ''}")

    matches = sorted(root.glob("*/sample_submission.csv"))
    if not matches:
        matches = sorted(root.glob("**/sample_submission.csv"))
    if not matches:
        raise FileNotFoundError(f"no sample_submission.csv anywhere under {root}")
    found = matches[0].parent
    print(f"\ncompetition mount resolved to: {found}")
    return found


INPUT = find_input()


def write_submission() -> None:
    """A valid file before anything else runs — a crash later still leaves something scoreable."""
    section("0. SUBMISSION (written first, deliberately)")
    sample = pd.read_csv(INPUT / "sample_submission.csv")
    sample.to_csv(WORKING / "submission.csv", index=False)
    print(f"wrote submission.csv  rows={len(sample)}  cols={list(sample.columns)}")
    print("expected LB = 0.5000 exactly (constant prediction => every AUC at chance)")


def probe_runtime() -> None:
    section("1. RUNTIME")
    print(f"python  {sys.version.split()[0]}")
    print(f"cpus    {os.cpu_count()}")
    try:
        meminfo = dict(
            (parts[0].rstrip(":"), parts[1])
            for parts in (line.split() for line in Path("/proc/meminfo").read_text().splitlines())
        )
        print(
            f"ram     total={int(meminfo['MemTotal']) / 1e6:.1f} GB  available={int(meminfo['MemAvailable']) / 1e6:.1f} GB"
        )
    except (OSError, KeyError) as exc:
        print(f"ram     unreadable ({exc})")

    try:
        import torch
    except ImportError as exc:
        print(f"torch   ABSENT ({exc})")
        return
    print(f"torch   {torch.__version__}  cuda_build={torch.version.cuda}")
    print(f"cuda    available={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(
            f"  gpu{index}  {properties.name}  sm_{properties.major}{properties.minor}  "
            f"{properties.total_memory / 1e9:.1f} GB"
        )
    print(f"arch_list {torch.cuda.get_arch_list() if torch.cuda.is_available() else 'n/a'}")
    print(f"bf16_supported {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'n/a'}")


def probe_libraries() -> None:
    section("2. LIBRARIES PRESENT OFFLINE (absent ones must ship as wheels in the kit)")
    for name in (
        "pydicom",
        "timm",
        "transformers",
        "monai",
        "polars",
        "cv2",
        "SimpleITK",
        "nibabel",
        "einops",
        "sklearn",
        "scipy",
        "PIL",
        "zarr",
    ):
        try:
            module = __import__(name)
            print(f"  {name:14s} {getattr(module, '__version__', 'present')}")
        except ImportError:
            print(f"  {name:14s} ABSENT")


def probe_mounts() -> None:
    section("3. MOUNTS")
    for root in sorted(Path("/kaggle/input").glob("*")):
        print(f"  {root}")
    print(f"\ncompetition files at {INPUT}:")
    for entry in sorted(INPUT.iterdir()):
        kind = "dir" if entry.is_dir() else f"{entry.stat().st_size} B"
        print(f"  {entry.name:24s} {kind}")


def probe_tables() -> None:
    """The corpus facts that decide the data layer, from CSVs alone — no pixels, no download."""
    section("4. TABLES")
    train = pd.read_csv(INPUT / "train.csv")
    print(f"train.csv  shape={train.shape}")
    print(f"  columns: {list(train.columns)}")

    label_columns = [c for c in train.columns if c not in ("StudyInstanceUID", "Report")]
    if "Report" in train.columns:
        has_report = train["Report"].notna().sum()
        print(f"  studies with a Report: {has_report} / {len(train)}")
    if label_columns:
        labelled = train[label_columns].notna().all(axis=1).sum()
        print(f"  studies with all {len(label_columns)} labels: {labelled} / {len(train)}")
        print("\n  per-label prevalence among labelled studies (the rare ones are the expensive ones):")
        for column in label_columns:
            values = train[column].dropna()
            if len(values):
                print(
                    f"    {column:20s} n={len(values):6d}  positive_rate={values.mean():.4f}  "
                    f"distinct={sorted(values.unique())[:6]}"
                )

    if "Report" in train.columns:
        section("5. REPORT DUPLICATES (the leak that inflates a holdout — knee-92z)")
        digests = train["Report"].fillna("").map(lambda text: hashlib.sha1(text.encode("utf-8")).hexdigest())
        sizes = Counter(digests)
        multiples = Counter(size for size in sizes.values() if size > 1)
        print(f"  distinct reports: {len(sizes)} over {len(train)} studies")
        print(f"  studies in a duplicate group: {sum(size for size in sizes.values() if size > 1)}")
        print(f"  duplicate group sizes (size: count): {dict(sorted(multiples.items())[:15])}")
        largest = sizes.most_common(3)
        for digest, size in largest:
            sample_text = train.loc[digests == digest, "Report"].iloc[0]
            print(f"\n  group of {size}, first 200 chars:\n    {str(sample_text)[:200]!r}")

    section("6. SERIES TABLE (does the shipped acquisition pair carry one axis or two? — knee-64b)")
    series = pd.read_csv(INPUT / "train_series.csv")
    print(f"train_series.csv  shape={series.shape}  columns={list(series.columns)}")
    if {"Fluid_Sensitive", "Fat_Suppression"} <= set(series.columns):
        agreement = (series["Fluid_Sensitive"] == series["Fat_Suppression"]).mean()
        print(f"  Fluid_Sensitive == Fat_Suppression on {agreement:.6f} of rows")
        print(f"  crosstab:\n{pd.crosstab(series['Fluid_Sensitive'], series['Fat_Suppression'])}")
        print("  -> agreement of 1.0 means the two columns are ONE axis; recover both from the header.")
    if "Anatomical_Plane" in series.columns:
        print(f"\n  planes: {series['Anatomical_Plane'].value_counts().to_dict()}")
        print(
            f"  series per study: mean={series.groupby('StudyInstanceUID').size().mean():.2f}  "
            f"max={series.groupby('StudyInstanceUID').size().max()}"
        )


def probe_one_series() -> None:
    """Read cost off the mount, and whether the geometry tags the ordering depends on are actually there."""
    section("7. ONE SERIES OFF THE MOUNT (read cost + geometry tags — knee-7ve, knee-b0c, knee-e5e)")
    try:
        import pydicom
    except ImportError:
        print("pydicom ABSENT — it must ship as a wheel in the kit")
        return

    series_root = INPUT / "train_series"
    if not series_root.exists():
        print(f"{series_root} absent; skipping")
        return

    started = time.time()
    study = next(series_root.iterdir())
    series = next(study.iterdir())
    files = sorted(series.iterdir())
    listed = time.time() - started
    print(f"study={study.name[-12:]}  series={series.name[-12:]}  slices={len(files)}  list={listed:.2f}s")

    started = time.time()
    datasets = [pydicom.dcmread(str(path)) for path in files]
    read = time.time() - started
    print(f"read {len(files)} slices in {read:.2f}s  ({read / max(len(files), 1) * 1000:.0f} ms/slice)")

    first = datasets[0]
    for tag in (
        "Modality",
        "Manufacturer",
        "SeriesDescription",
        "ScanningSequence",
        "SequenceName",
        "ScanOptions",
        "RepetitionTime",
        "EchoTime",
        "Laterality",
        "PixelSpacing",
        "Rows",
        "Columns",
        "SliceThickness",
        "ImageOrientationPatient",
        "ImagePositionPatient",
    ):
        print(f"  {tag:24s} {getattr(first, tag, '<absent>')}")

    pixels = first.pixel_array
    print(f"  pixel_array shape={pixels.shape} dtype={pixels.dtype} min={pixels.min()} max={pixels.max()}")

    spacing = getattr(first, "PixelSpacing", None)
    if spacing is not None:
        print(f"  field of view = Rows*PixelSpacing = {float(first.Rows) * float(spacing[0]):.1f} mm")

    positions = [getattr(d, "ImagePositionPatient", None) for d in datasets]
    orientations = [getattr(d, "ImageOrientationPatient", None) for d in datasets]
    print(f"  slices carrying ImagePositionPatient: {sum(p is not None for p in positions)}/{len(datasets)}")
    print(f"  slices carrying ImageOrientationPatient: {sum(o is not None for o in orientations)}/{len(datasets)}")

    if all(p is not None for p in positions) and orientations[0] is not None:
        row_axis = [float(v) for v in orientations[0][:3]]
        col_axis = [float(v) for v in orientations[0][3:]]
        normal = [
            row_axis[1] * col_axis[2] - row_axis[2] * col_axis[1],
            row_axis[2] * col_axis[0] - row_axis[0] * col_axis[2],
            row_axis[0] * col_axis[1] - row_axis[1] * col_axis[0],
        ]
        projections = [sum(float(p[i]) * normal[i] for i in range(3)) for p in positions]
        filename_rank = list(range(len(files)))
        geometric_rank = [i for i, _ in sorted(enumerate(projections), key=lambda pair: pair[1])]
        # Spearman rho without scipy: 1 - 6*sum(d^2)/(n(n^2-1))
        rank_of = {index: position for position, index in enumerate(geometric_rank)}
        n = len(files)
        squared = sum((filename_rank[i] - rank_of[i]) ** 2 for i in range(n))
        rho = 1 - 6 * squared / (n * (n * n - 1)) if n > 1 else float("nan")
        print(f"  Spearman rho(filename order, geometric k) = {rho:.4f}   <- expect ~0; sorting by name is wrong")
        print(f"  k range: {min(projections):.1f} .. {max(projections):.1f} mm")


def main() -> None:
    write_submission()
    for probe in (probe_runtime, probe_libraries, probe_mounts, probe_tables, probe_one_series):
        try:
            probe()
        except Exception as exc:  # a probe that fails must not cost the other probes  # noqa: BLE001
            print(f"\n!! {probe.__name__} FAILED: {type(exc).__name__}: {exc}", flush=True)
    section("PROBE COMPLETE — submission.csv is valid")


if __name__ == "__main__":
    main()
