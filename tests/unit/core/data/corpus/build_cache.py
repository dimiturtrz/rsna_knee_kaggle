"""The corpus builders' COMMAND LINES, exercised through the argv they actually parse.

WHY `--help` IS THE TEST. `build_rijeka` was uncallable for as long as `CacheFlags` has owned a
`--spacing` flag: the builder declared its own float `--spacing` for millimetres-per-pixel, argparse
raised `conflicting option string` while CONSTRUCTING the parser, and the CLI died before it could read
one argument. Nothing caught it, because no test ever built these parsers — the modules are shells whose
only untested line was the one that broke. `--help` reaches parser construction and exits before any
filesystem work, so it is the whole of the defect and none of the decode.
"""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.data.corpus.build_cache import main
from core.data.corpus.slots import FOREIGN_SCHEMES

BUILDERS = ("build_cache", "build_merged", "build_mrnet", "build_rijeka")


@pytest.mark.parametrize("builder", BUILDERS)
def test_main(builder: str, monkeypatch: pytest.MonkeyPatch):
    """Every builder's parser CONSTRUCTS, which is the step `--spacing` collided in."""
    monkeypatch.setattr("sys.argv", [builder, "--help"])

    with pytest.raises(SystemExit) as exit_code:
        import_module(f"core.data.corpus.{builder}").main()

    assert exit_code.value.code == 0, f"{builder} cannot build its own parser"


@pytest.mark.parametrize("scheme", FOREIGN_SCHEMES)
def test_foreign_scheme_refused(scheme: str, monkeypatch: pytest.MonkeyPatch):
    """`build_cache` decodes OUR study list whatever `--scheme` says, so a foreign name is refused.

    `--scheme` selects a FILENAME and nothing else consults it, and for `merged` the shape matches ours
    exactly — six slots either way — so neither the log line nor the array can catch the mistake. Only
    the refusal can. It cost 59.7 GB (knee-48br).
    """
    monkeypatch.setattr("sys.argv", ["build_cache", "--scheme", scheme])

    with pytest.raises(SystemExit) as exit_code:
        main()

    assert exit_code.value.code == 2, f"--scheme {scheme} must be refused, not decoded under our corpus"


def _listing(tmp_path: Path, *identifiers: str) -> Path:
    path = tmp_path / "wanted.txt"
    path.write_text("\n".join(identifiers) + "\n", encoding="utf-8")
    return path


def test_only_studies_refused_without_an_out_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A restricted decode under the FULL cache's name destroys it — the same guard --holdout-fold has."""
    monkeypatch.setattr("sys.argv", ["build_cache", "--only-studies", str(_listing(tmp_path, "a"))])

    with pytest.raises(SystemExit) as exit_code:
        main()

    assert exit_code.value.code == 2, "a partial corpus must not be written under the full cache's name"


Call = tuple[object, ...]


class _Split:
    """Four studies, a named subset of three, and one plane."""

    def __init__(self, calls: list[Call], raw: Path, name: str) -> None:
        self.calls = calls
        self.raw = raw
        self.source = raw / name

    def identifiers(self) -> list[str]:
        return ["a", "b", "c", "d"]

    def named(self, path: Path) -> SimpleNamespace:
        self.calls.append(("named", path.name))
        return SimpleNamespace(chosen=["b", "c", "d"])

    def planes(self) -> dict[str, str]:
        return {"s": "Sagittal"}


class _Folds:
    """A fold whose holdout is the first and third of whatever it is handed."""

    def __init__(self, calls: list[Call]) -> None:
        self.calls = calls

    def assign(self, table: str, identifiers: list[str]) -> str:
        self.calls.append(("assign", Path(table).name, tuple(identifiers)))
        return "assigned"

    def holdout(self, assigned: str, fold: int) -> tuple[None, list[int]]:
        self.calls.append(("holdout", assigned, fold))
        return None, [0, 2]


class _Cache:
    """A cache that records the build instead of decoding."""

    key = "KEY123"
    images_path = Path("images.npy")

    def __init__(self, calls: list[Call]) -> None:
        self.calls = calls

    def enrolled(self, identifiers: list[str]) -> list[str]:
        return identifiers

    def shape(self, count: int) -> tuple[int, int]:
        return (count, 6)

    def build(self, source: Path, identifiers: list[str], planes: dict[str, str], *, workers: int) -> None:
        self.calls.append(("build", source.name, tuple(identifiers), tuple(planes), workers))


class _Corpus:
    """The collaborators main reaches for, each recording into one shared list of calls."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.calls: list[Call] = []

    def split(self, raw: Path, name: str) -> _Split:
        return _Split(self.calls, raw, name)

    def folds(self) -> _Folds:
        return _Folds(self.calls)

    def cache(self, arguments: object, directory: Path) -> _Cache:
        self.calls.append(("cache", directory.relative_to(self.tmp_path).as_posix()))
        return _Cache(self.calls)

    def directory(self, path: Path, planes: dict[str, str]) -> SimpleNamespace:
        return SimpleNamespace(metadata=lambda: f"meta:{path.name}")

    def write(self, entries: list[str], path: Path) -> None:
        self.calls.append(("write", tuple(entries), path.relative_to(self.tmp_path).as_posix()))


def _drive(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path, *flags: str
) -> tuple[list[str], list[tuple[object, ...]]]:
    """Run main with every decode, fold and filesystem collaborator faked; return the log and the calls made."""
    import contextlib  # noqa: PLC0415 - test-local
    import logging  # noqa: PLC0415 - test-local

    from core import cli  # noqa: PLC0415 - test-local
    from core.data.corpus import build_cache  # noqa: PLC0415 - test-local

    corpus = _Corpus(tmp_path)
    calls = corpus.calls
    rooted = SimpleNamespace(raw=lambda: tmp_path / "raw", processed=lambda: tmp_path / "processed")
    monkeypatch.setattr(cli.Command, "rooted", lambda self, arguments: rooted)
    monkeypatch.setattr(build_cache, "CorpusSplit", corpus.split)
    monkeypatch.setattr(build_cache, "ReportFolds", corpus.folds)
    monkeypatch.setattr(build_cache, "Wakelock", contextlib.nullcontext)
    monkeypatch.setattr(build_cache.CacheFlags, "cache", staticmethod(corpus.cache))
    monkeypatch.setattr(build_cache.Study, "from_directory", staticmethod(corpus.directory))
    monkeypatch.setattr(build_cache.StudyMetadata, "write", staticmethod(corpus.write))
    monkeypatch.setattr("sys.argv", ["build_cache", *flags])
    caplog.set_level(logging.INFO, logger=build_cache.__name__)
    build_cache.main()
    return [record.getMessage() for record in caplog.records if record.name == build_cache.__name__], calls


def test_main_builds_the_restricted_studies(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
):
    """--only-studies narrows first, then --holdout-fold narrows THAT list, and the build gets what is left."""
    listing = _listing(tmp_path, "b")
    lines, calls = _drive(
        monkeypatch,
        caplog,
        tmp_path,
        "--only-studies",
        str(listing),
        "--holdout-fold",
        "1",
        "--out",
        str(tmp_path / "out"),
        "--workers",
        "3",
    )
    assert lines == [
        f"restricted to {listing}: 3 studies",
        "restricted to fold 1's holdout: 2 studies",
        "building (2, 6) uint8 -> images.npy",
        "CACHE BUILD COMPLETE",
    ]
    assert calls == [
        ("named", "wanted.txt"),
        ("assign", "train.csv", ("b", "c", "d")),
        ("holdout", "assigned", 1),
        ("cache", "out"),
        ("build", "train_series", ("b", "d"), ("s",), 3),
    ]


def test_main_prints_the_key_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    """--print-key answers on stdout and decodes nothing."""
    lines, calls = _drive(monkeypatch, caplog, tmp_path, "--print-key")
    assert (lines, calls, capsys.readouterr().out) == ([], [("cache", "processed")], "KEY123\n")


def test_main_writes_only_the_metadata(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
):
    """--metadata-only writes one row per study under the processed root and never opens a cache."""
    lines, calls = _drive(monkeypatch, caplog, tmp_path, "--metadata-only")
    assert lines == ["CANONICAL METADATA COMPLETE (no decode requested)"]
    assert calls == [("write", ("meta:a", "meta:b", "meta:c", "meta:d"), "processed/metadata/canonical_metadata.csv")]
