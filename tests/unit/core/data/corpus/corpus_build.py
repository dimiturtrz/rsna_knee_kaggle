"""The foreign builders' shared command line, driven through each builder's own `main` and argv."""

import logging
from importlib import import_module
from pathlib import Path

import pytest

from core import cli
from core.data.corpus.corpus_build import BuildSite, CorpusBuild
from core.data.corpus.slots import (
    CANONICAL_SCHEME,
    FASTMRI_SCHEME,
    KMAR_SCHEME,
    MRNET_SCHEME,
    RIJEKA_SCHEME,
    Routing,
)
from core.data.corpus.study_cache import StudyCache
from core.data.corpus.study_metadata import StudyMetadata

FOREIGN_BUILDERS = {
    "build_mrnet": ("MRNetCorpus", MRNET_SCHEME, "--archive"),
    "build_rijeka": ("RijekaCorpus", RIJEKA_SCHEME, "--archive"),
    "build_kmar": ("KmarStudies", KMAR_SCHEME, "--archive"),
    "build_fastmri": ("FastmriCorpus", FASTMRI_SCHEME, "--root"),
}
"""Each foreign builder: the corpus class it constructs, the one scheme it accepts, its source flag."""

DEEPER = 6
KMAR = CorpusBuild(KMAR_SCHEME, "KMAR", metadata="kmar_metadata.csv")


class NamedCacheError(Exception):
    """Raised by the stand-in corpus the moment it is handed the cache a builder named."""

    def __init__(self, cache: StudyCache) -> None:
        super().__init__(cache.images_path)
        self.cache = cache


def _named(*positional: object, **_keywords: object) -> None:
    """A corpus constructor that stops the build at the cache, before any decode."""
    raise NamedCacheError(next(value for value in positional if isinstance(value, StudyCache)))


def _builder_argv(builder: str, tmp_path: Path, scheme: str, *flags: str) -> list[str]:
    source_flag = FOREIGN_BUILDERS[builder][2]
    (tmp_path / "paths.yaml").write_text(f"data: {tmp_path.as_posix()}\n", encoding="utf-8")
    return [
        builder,
        *("--paths", str(tmp_path / "paths.yaml"), source_flag, str(tmp_path)),
        *("--out", str(tmp_path / "out"), "--scheme", scheme, *flags),
    ]


@pytest.mark.parametrize("builder", FOREIGN_BUILDERS)
def test_every_foreign_builder_names_its_own_cache(builder: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The argv a builder accepts names the same file it always did — scheme-suffixed, under `--out`."""
    corpus_class, scheme, _ = FOREIGN_BUILDERS[builder]
    module = import_module(f"core.data.corpus.{builder}")
    monkeypatch.delenv("KNEEMRI_DATA", raising=False)
    monkeypatch.setattr(module, corpus_class, _named)
    monkeypatch.setattr("sys.argv", _builder_argv(builder, tmp_path, scheme, "--slices", str(DEEPER)))

    with pytest.raises(NamedCacheError) as named:
        module.main()

    expected = StudyCache(tmp_path / "out", slices=DEEPER, routing=Routing(scheme=scheme))
    assert named.value.cache.images_path == expected.images_path


@pytest.mark.parametrize("builder", FOREIGN_BUILDERS)
def test_every_foreign_builder_refuses_our_scheme(
    builder: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """The canonical scheme writes onto our own arrays, so every foreign builder refuses it by name."""
    corpus_class, scheme, _ = FOREIGN_BUILDERS[builder]
    module = import_module(f"core.data.corpus.{builder}")
    monkeypatch.setattr(module, corpus_class, _named)
    monkeypatch.setattr("sys.argv", _builder_argv(builder, tmp_path, CANONICAL_SCHEME))

    with pytest.raises(SystemExit) as refusal:
        module.main()

    assert refusal.value.code == 2
    assert f"--scheme {scheme} is required" in capsys.readouterr().err


def _site(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scheme: str) -> BuildSite:
    monkeypatch.delenv("KNEEMRI_DATA", raising=False)
    monkeypatch.setattr("sys.argv", _builder_argv("build_kmar", tmp_path, scheme))
    command = cli.Command("a test builder", logging.getLogger(__name__))
    KMAR.opening(command).add_argument("--archive", type=Path, required=True)
    return KMAR.opened(command)


def test_opening():
    """The parser handed back already carries every cache flag, so the builder adds only its own source."""
    command = cli.Command("a test builder", logging.getLogger(__name__))

    parser = KMAR.opening(command)

    assert parser is command.parser
    assert parser.parse_args(["--scheme", KMAR_SCHEME, "--slices", str(DEEPER)]).slices == DEEPER


def test_opened(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """The builder's own flags parse beside the cache flags, `--out` places the cache, and only its scheme passes."""
    site = _site(monkeypatch, tmp_path, KMAR_SCHEME)

    assert site.arguments.archive == tmp_path
    assert site.cache.images_path == StudyCache(tmp_path / "out", slices=12, routing=Routing(KMAR_SCHEME)).images_path
    with pytest.raises(SystemExit) as refusal:
        _site(monkeypatch, tmp_path, MRNET_SCHEME)
    assert refusal.value.code == 2
    assert f"--scheme {KMAR_SCHEME} is required" in capsys.readouterr().err


class RecordingCorpus:
    """A corpus that states no metadata and records whether it was decoded."""

    def __init__(self) -> None:
        self.decoded = False

    def metadata(self) -> list[StudyMetadata]:
        return []

    def build(self) -> None:
        self.decoded = True


@pytest.mark.parametrize(("decode", "closing"), [(True, "KMAR BUILD COMPLETE"), (False, "KMAR LABELS COMPLETE")])
def test_run(
    decode: bool,
    closing: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    """Metadata is always written; the decode runs only when asked, and the closing line says which happened."""
    site = _site(monkeypatch, tmp_path, KMAR_SCHEME)
    corpus = RecordingCorpus()

    with caplog.at_level(logging.INFO):
        KMAR.run(site, corpus, lambda: [], decode=decode)

    assert (site.root.processed() / "metadata" / "kmar_metadata.csv").exists()
    assert corpus.decoded is decode
    assert closing in caplog.text
