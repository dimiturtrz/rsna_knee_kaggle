"""The cache key an arm asks for, from argv, on the stream a shell can capture.

`build_cache --print-key` exists so the arm's completion gate checks the file the builder writes rather
than a name reassembled in bash. It was printed through `logging`, which writes to STDERR — so the key
appeared on the terminal, looked right to anyone who ran it by hand, and `KEY="$(… --print-key)"`
captured the empty string. On 2026-08-15 a decode wrote all 4407 studies and its mask, and the arm
reported `ARM FAILED build_cache_ (no mask written)` over it.

The stream is the contract here, not the string: a key on the wrong stream is a gate that checks
`mask_.npy` forever, and every reading axis added since would ride the same fault.
"""

from pathlib import Path

import pandas as pd
import pytest

from core.data.corpus import build_cache
from core.data.corpus.crop_centre import CENTRES, DEFAULT_CENTRE, Crop
from core.data.corpus.slot_reader import Band, Reading
from core.data.corpus.slots import CANONICAL_SCHEME, FIRST_MAXIMAL, FOREIGN_SCHEMES, SCHEMES, TIES, Routing
from core.data.corpus.study_cache import StudyCache
from core.paths import COMPETITION

SLICES = 12


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """The two shipped tables `main` reads before it can name a cache."""
    raw = tmp_path / "raw" / COMPETITION
    raw.mkdir(parents=True)
    pd.DataFrame({"StudyInstanceUID": ["s0", "s1"]}).to_csv(raw / "train.csv", index=False)
    pd.DataFrame({"SeriesInstanceUID": ["e0"], "Anatomical_Plane": ["SAG"]}).to_csv(
        raw / "train_series.csv", index=False
    )
    (tmp_path / "processed" / COMPETITION).mkdir(parents=True)
    (tmp_path / "paths.yaml").write_text(f"data: {tmp_path.as_posix()}\n", encoding="utf-8")
    return tmp_path


def _key(root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *flags: str) -> str:
    """What a shell would capture from `$(python -m core.data.corpus.build_cache --print-key …)`: stdout alone."""
    argv = ["build_cache", "--print-key", "--paths", str(root / "paths.yaml"), *flags]
    monkeypatch.setattr("sys.argv", argv)
    build_cache.main()
    return capsys.readouterr().out.strip()


def test_the_key_reaches_stdout(root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """The default reading names the historic cache, and it arrives where a command substitution reads."""
    assert _key(root, monkeypatch, capsys) == StudyCache(root, slices=SLICES).key


def test_every_reading_axis_reaches_the_key(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """A flag that changes which pixels are decoded must change the name they are decoded under.

    Five axes select the file — size, depth, physical crop, the point it is taken about, and the slice
    band — and the gate that waits for a build spelled only the first two. An axis missing from the key
    is a decode that overwrites the arrays every member on disk was fitted on. The flags themselves now
    come from `CacheFlags`, which every cache-reading CLI shares, so this also pins that they are all
    asking for the same file: the builder and the trainer had already diverged on how to spell the band.
    """
    printed = _key(
        root,
        monkeypatch,
        capsys,
        *("--image-size", "448"),
        *("--slices", "6"),
        *("--crop-mm", "80"),
        *("--centre", "tissue"),
        *("--band-low", "0.1"),
        *("--band-high", "0.9"),
    )

    reading = Reading(crop=Crop(mm=80.0, centre="tissue"), band=Band(0.1, 0.9))
    assert printed == StudyCache(root, slices=6, image_size=448, reading=reading).key
    assert printed != _key(root, monkeypatch, capsys), "a different reading cannot share a name"


def test_the_scheme_reaches_the_key(root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """The slot scheme is the axis the bash gate DID carry, and the only one it did — keep it carried."""
    printed = _key(root, monkeypatch, capsys, "--scheme", "extended")

    assert printed == StudyCache(root, slices=SLICES, routing=Routing(scheme="extended")).key
    assert "extended" in printed


OWN_SCHEMES = tuple(name for name in SCHEMES if name not in FOREIGN_SCHEMES)
"""The schemes `build_cache` will decode. It refuses the foreign ones outright — it reads OUR study list
whatever `--scheme` says, so naming another corpus there writes our studies under that corpus's filename
(knee-48br). Their keys are exercised by their own builders; the refusal itself is pinned in
`tests/unit/core/data/corpus/build_cache.py`, so no name in `SCHEMES` goes unexercised."""

READING_VOCABULARIES = {
    "--scheme": (OWN_SCHEMES, CANONICAL_SCHEME),
    "--centre": (tuple(CENTRES), DEFAULT_CENTRE),
    "--tie": (TIES, FIRST_MAXIMAL),
}
"""The reading axes a CLI offers a CLOSED SET of names for, each with the name that keeps the historic
cache.

The tests above spell ONE name of each — `extended`, `tissue` — and that is the hole this closes
(knee-29s5): a name added to a registry and offered by `CacheFlags` but not carried into the key is a
decode that lands on the arrays every member on disk was fitted on, and it fails at the commit that
adds the name rather than the run that first asks for it. Walking the vocabulary is what makes that
true; naming today's alternatives is what left `multiscale` and both tie rules unexercised."""


@pytest.mark.parametrize(("flag", "vocabulary_and_historic"), READING_VOCABULARIES.items())
def test_every_reading_a_run_can_name_reaches_the_key(
    flag: str,
    vocabulary_and_historic: tuple[tuple[str, ...], str],
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Every name argv accepts either IS the historic reading or is named for itself — no third case."""
    vocabulary, historic = vocabulary_and_historic
    assert historic in vocabulary, f"{flag} defaults to a name it does not offer"
    default = _key(root, monkeypatch, capsys)

    for name in vocabulary:
        printed = _key(root, monkeypatch, capsys, flag, name)
        if name == historic:
            assert printed == default, f"{flag} {name} is the historic reading and must keep its name"
            continue
        assert printed != default, f"{flag} {name} decodes different pixels under the historic name"
        assert name in printed, f"{flag} {name} is not readable back off the name it decoded under"


def test_a_truncated_decode_reaches_the_key(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """A partial decode is a different cache, so it is named for itself — the smoke path for a new axis.

    Without this the only way to try a reading nothing has decoded was a two-hour full build, and a
    truncated one written under the full name would destroy the arrays every member was fitted on.
    """
    printed = _key(root, monkeypatch, capsys, "--cache-studies", "60")

    assert printed == StudyCache(root, slices=SLICES, first=60).key
    assert printed.endswith("_first60")
    assert printed != _key(root, monkeypatch, capsys), "a partial corpus cannot share the full one's name"


def test_a_fold_restriction_is_refused_without_a_directory(root: Path, monkeypatch: pytest.MonkeyPatch):
    """The truncation the key CANNOT carry has to be separated by directory, and that is now required.

    `--holdout-fold` decodes one fold's holdout — a fifth of the corpus — and the key says nothing about
    it, so into the processed root it lands on the full cache's own filename. It cannot be named for
    itself either: `train.py` has no fold flag, so a suffix would make the arrays it already reads
    unaddressable. So `--out` is the separation, and asking without it is refused rather than obeyed.
    """
    monkeypatch.setattr("sys.argv", ["build_cache", "--paths", str(root / "paths.yaml"), "--holdout-fold", "0"])

    with pytest.raises(SystemExit) as refusal:
        build_cache.main()

    assert refusal.value.code == 2, "argparse's own usage error, so the shell sees a refusal not a traceback"
    assert list((root / "processed" / COMPETITION).iterdir()) == [], "and nothing was decoded"


def test_print_key_builds_nothing(root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """Asking for a name must not start a decode: the gate calls this before every build, and twice.

    `main` reads the shipped tables before it answers, so the cheap question sits upstream of the
    expensive work — and `open_memmap(mode="w+")` recreates a file ZEROED, which would make asking the
    gate's own question destroy the cache it is asking about.
    """
    _key(root, monkeypatch, capsys, "--crop-mm", "80")

    assert list((root / "processed" / COMPETITION).iterdir()) == [], "no array, no mask, no index"
