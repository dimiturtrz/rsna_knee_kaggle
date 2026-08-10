from pathlib import Path

import pytest

from core.paths import COMPETITION, DataRoot


@pytest.fixture
def config(tmp_path: Path) -> Path:
    written = tmp_path / "paths.yaml"
    written.write_text(f"data: {(tmp_path / 'root').as_posix()}\n", encoding="utf-8")
    return written


def test_from_config(config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("KNEEMRI_DATA", raising=False)  # env override wins when set — resolve from paths.yaml here
    assert DataRoot.from_config(config).raw("x") == tmp_path / "root" / "raw" / "x"


def test_from_config_prefers_the_env_override(config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A runner without the dataset points KNEEMRI_DATA at an empty dir so data-gated tests skip."""
    monkeypatch.setenv("KNEEMRI_DATA", str(tmp_path / "elsewhere"))
    assert DataRoot.from_config(config).raw("x") == tmp_path / "elsewhere" / "raw" / "x"


def test_raw(tmp_path: Path):
    """`raw` never creates anything — the download is read-only."""
    assert not DataRoot(tmp_path).raw("dataset").exists()


def test_raw_defaults_to_the_competition(tmp_path: Path):
    assert DataRoot(tmp_path).raw() == tmp_path / "raw" / COMPETITION


def test_processed(tmp_path: Path):
    """Derived artefacts are written, so the directory has to exist by the time the caller has the path."""
    assert DataRoot(tmp_path).processed("dataset").is_dir()
