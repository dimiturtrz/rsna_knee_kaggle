from pathlib import Path

from core.paths import COMPETITION
from kneemri.corpus.crop_centre import Crop
from kneemri.corpus.slot_reader import Reading
from kneemri.corpus.study_cache import StudyCache
from kneemri.evaluation.arm_check import ArmCheck

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

    found = ArmCheck(arms, Path("experiments/stub_lib.sh")).invocations()

    assert found == {"real": "--fold 0 --epochs 8"}
