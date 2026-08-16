from pathlib import Path

import pandas as pd

from core.paths import COMPETITION
from kneemri.corpus.crop_centre import Crop
from kneemri.corpus.slot_reader import Reading
from kneemri.corpus.study_cache import StudyCache
from kneemri.experiments.arm_check import ArmCheck

from ..conftest import write_series

GOOD = "--fold 0 --epochs 12 --one-cycle --slice-pooling max --whole-stack --labels /tmp/labels.csv"
SLICES = 4
SIZE = 28


def test_refused():
    """A real invocation passes, a typo'd flag is caught, and the parser's own complaint is reported.

    The pure half on purpose: `invocations` shells out to bash, which the unit gate forbids, while this
    is where the logic lives. Splitting them is what makes the check testable at all.
    """
    refused = ArmCheck().refused(
        {
            "good": GOOD,
            "typo": GOOD + " --slice-poolng max",
            "bad_value": "--fold 0 --slice-pooling nonsense",
        }
    )

    assert "good" not in refused, "a real arm's flags must survive the trainer's parser"
    assert "typo" in refused, "a misspelled flag is exactly what this exists to catch before a queue"
    assert "bad_value" in refused, "a value outside the choices is refused too"
    assert refused["typo"], "the parser's own complaint is reported, not a generic failure"


def test_uncached(tmp_path: Path):
    """An arm reading a cache nobody has decoded is named, and one reading a decoded cache is not.

    The failure this catches costs the same as a flag typo: `main` selects the card before it maps the
    cache, so an undecoded reading holds the GPU and then raises on a memory map. It has to stay silent
    without a data root, because `paths.yaml` is gitignored and CI has none.
    """
    cache = StudyCache(tmp_path / "processed" / COMPETITION, slices=SLICES, image_size=SIZE)
    cache.directory.mkdir(parents=True)
    for path in (cache.images_path, cache.mask_path, cache.index_path):
        path.write_text("decoded", encoding="utf-8")
    paths = tmp_path / "paths.yaml"
    paths.write_text(f"data: {tmp_path.as_posix()}\n", encoding="utf-8")
    # POSIX separators: the flags go through `shlex.split`, which eats a Windows backslash as an escape
    # and hands the parser a path that resolves to nothing — under which this test passes vacuously.
    reading = f"{GOOD} --paths {paths.as_posix()} --slices {SLICES} --image-size {SIZE}"

    uncached = ArmCheck().uncached(
        {
            "decoded": reading,
            "other_crop": f"{reading} --crop-mm 80",
            "rootless": f"{GOOD} --paths {(tmp_path / 'absent.yaml').as_posix()}",
        }
    )

    assert "decoded" not in uncached, "the reading this arm asks for is on disk"
    tighter = StudyCache(cache.directory, slices=SLICES, image_size=SIZE, reading=Reading(crop=Crop(mm=80.0)))
    assert uncached["other_crop"] == tighter.key, (
        "a different physical crop is a different cache, and the key is what says which"
    )
    assert "rootless" not in uncached, "no data root resolves, so nothing can be claimed about it"


def _rooted(tmp_path: Path) -> str:
    """An arm's flags pointed at a real data root, at a resolution a test can afford to decode."""
    paths = tmp_path / "paths.yaml"
    paths.write_text(f"data: {tmp_path.as_posix()}\n", encoding="utf-8")
    # POSIX separators: the flags go through `shlex.split`, which eats a Windows backslash as an escape.
    return f"{GOOD} --paths {paths.as_posix()} --slices {SLICES} --image-size {SIZE}"


def test_readings(tmp_path: Path):
    """Every accepted arm's root and cache, resolved once — the parse both consumers would repeat."""
    rootless = f"{GOOD} --paths {(tmp_path / 'absent.yaml').as_posix()}"
    readings = ArmCheck().readings({"real": _rooted(tmp_path), "rootless": rootless, "typo": f"{GOOD} --fld 0"})

    assert set(readings) == {"real"}, "no data root and no valid flags are both silence, not a claim"
    assert readings["real"].key == StudyCache(tmp_path, slices=SLICES, image_size=SIZE).key


def test_key(tmp_path: Path):
    """A reading names the file it would read, which is the string an UNCACHED report prints."""
    reading = ArmCheck().readings({"real": _rooted(tmp_path)})["real"]

    assert reading.key == f"{SIZE}px_{SLICES}sl"


def test_decoded(tmp_path: Path):
    """A reading answers whether its cache is on disk, rather than three path checks at each caller."""
    reading = ArmCheck().readings({"real": _rooted(tmp_path)})["real"]
    assert not reading.decoded()

    cache = StudyCache(tmp_path / "processed" / COMPETITION, slices=SLICES, image_size=SIZE)
    cache.directory.mkdir(parents=True, exist_ok=True)
    for path in (cache.images_path, cache.mask_path, cache.index_path):
        path.write_text("decoded", encoding="utf-8")

    assert reading.decoded()


def test_truncated(tmp_path: Path):
    """The smoke's reading is the arm's own over the first N studies — a different file, by name."""
    reading = ArmCheck().readings({"real": _rooted(tmp_path)})["real"]

    smoked = reading.truncated(2)

    assert smoked.key == f"{SIZE}px_{SLICES}sl_first2", "the truncation is in the name, not only the contents"
    assert reading.key != smoked.key, "so a smoke decode can never land on the corpus it is a prefix of"


def _corpus(tmp_path: Path, studies: int) -> None:
    """A raw root with the two shipped tables and one series per study, as a decode expects to find it."""
    raw = tmp_path / "raw" / COMPETITION
    identifiers = [f"study{index}" for index in range(studies)]
    for identifier in identifiers:
        write_series(raw / "train_series" / identifier / "sag", slices=4, size=32)
    pd.DataFrame({"StudyInstanceUID": identifiers}).to_csv(raw / "train.csv", index=False)
    # Keyed by the series DIRECTORY, which is what `Study.from_directory` looks the plane up by — a
    # table naming anything else routes every series nowhere and the decode reports a corpus of zeros.
    pd.DataFrame({"SeriesInstanceUID": ["sag"], "Anatomical_Plane": ["Sagittal"]}).to_csv(
        raw / "train_series.csv", index=False
    )


def test_decode(tmp_path: Path):
    """A reading decodes ITS OWN cache out of raw, which is what makes an untried axis smokeable."""
    _corpus(tmp_path, studies=2)
    reading = ArmCheck().readings({"real": _rooted(tmp_path)})["real"].truncated(1)

    # ONE worker: the pool is what a decode costs, not the pixels. Eight interpreters re-importing
    # torch under Windows' spawn, for two studies, once cost a unit run seven and a half hours.
    reading.decode(workers=1)

    assert reading.decoded(), "all three files, so the smoke that follows has something to map"


def test_decode_smoke_caches(tmp_path: Path):
    """The truncated cache is built before the smoke, once per reading, and never for a refused arm."""
    _corpus(tmp_path, studies=2)
    flags = _rooted(tmp_path)
    check = ArmCheck()

    built = check.decode_smoke_caches(
        {"real": flags, "same": flags, "typo": f"{flags} --fld 0", "broken": flags},
        {"broken": "refused by the parser"},
        first=1,
        workers=1,
    )

    assert built == [f"{SIZE}px_{SLICES}sl_first1"], "two arms on one reading decode it once, and only once"
    assert check.readings({"real": flags})["real"].truncated(1).decoded()
    assert not check.readings({"real": flags})["real"].decoded(), "the FULL corpus is untouched by a smoke"


def test_invocations(tmp_path: Path):
    """An arm is SOURCED, so the shell finds what only the shell can — and its flags come back.

    Sourcing rather than reading the file is the point: an arm lost two hours to a quoting error once,
    and no regex finds that. Arms that never call run_train are absent rather than reported, because a
    lint that complains about files it cannot check gets switched off.
    """
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "real.sh").write_text(
        'source "${ARM_LIB}"\nrequire "$LABELS"\nrun_train "real" --fold 0 --epochs 8\n', encoding="utf-8"
    )
    (arms / "no_train.sh").write_text('source "${ARM_LIB}"\necho "this arm drives uv directly"\n', encoding="utf-8")
    # An arm its own gate refuted: still committed as the record of what was tried, never run again.
    (arms / "retired.sh").write_text(
        'source "${ARM_LIB}"\nretired "gated out"\nrun_train "retired" --fold 0 --epochs 8\n', encoding="utf-8"
    )

    found = ArmCheck(arms, Path("experiments/stub_lib.sh")).invocations()

    assert found == {"real": "--fold 0 --epochs 8"}, (
        "a retired arm reaches no parser, so it is never reported UNCACHED for a cache nobody should build"
    )
