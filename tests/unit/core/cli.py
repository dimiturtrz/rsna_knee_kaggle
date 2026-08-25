import logging
from pathlib import Path

import pytest

from core import cli


@pytest.fixture
def config(tmp_path: Path) -> Path:
    written = tmp_path / "paths.yaml"
    written.write_text(f"data: {(tmp_path / 'root').as_posix()}\n", encoding="utf-8")
    return written


@pytest.fixture
def command() -> cli.Command:
    return cli.Command("what this answers", logging.getLogger("band_coverage"))


def test_parser_describes_the_command_with_the_docstring_it_was_given(command: cli.Command):
    assert command.parser.description == "what this answers"


def test_parser_carries_paths_so_no_command_declares_it_again(command: cli.Command):
    """The whole point: `--paths` was spelled out in forty-five modules before this existed."""
    assert command.parser.parse_args([]).paths == cli.PATHS


def test_parser_leaves_the_command_its_own_arguments(command: cli.Command):
    """Exposed as an attribute rather than wrapped, so no passthrough is needed per argparse feature."""
    command.parser.add_argument("--sample", type=int, default=3)
    parsed = command.parser.parse_args(["--sample", "9", "--paths", "elsewhere.yaml"])
    assert (parsed.sample, parsed.paths) == (9, Path("elsewhere.yaml"))


def test_parsed(command: cli.Command, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("sys.argv", ["band_coverage", "--paths", "elsewhere.yaml"])
    assert command.parsed().paths == Path("elsewhere.yaml")


def test_parsed_configures_logging_so_a_command_cannot_print_nothing(
    command: cli.Command, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """A command that parsed without configuring logging looks like an empty result, not a missing line."""
    monkeypatch.setattr("sys.argv", ["band_coverage"])
    command.parsed()
    with caplog.at_level(logging.INFO):
        command.report(["a line"])
    assert "a line" in caplog.text


def test_rooted(command: cli.Command, config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("KNEEMRI_DATA", raising=False)  # the env override wins when set; resolve from the file here
    arguments = command.parser.parse_args(["--paths", str(config)])
    assert command.rooted(arguments).raw("x") == tmp_path / "root" / "raw" / "x"


def test_report(command: cli.Command, caplog: pytest.LogCaptureFixture):
    """Through the caller's, so a report line still names the command that produced it."""
    with caplog.at_level(logging.INFO):
        command.report(["first", "second"])
    assert [record.name for record in caplog.records] == ["band_coverage", "band_coverage"]
    assert [record.message for record in caplog.records] == ["first", "second"]


def test_report_accepts_a_generator_rather_than_a_built_list(command: cli.Command, caplog: pytest.LogCaptureFixture):
    """Every ported command hands it `X.report(...)`, which yields rather than returning a list."""
    with caplog.at_level(logging.INFO):
        command.report(f"line {index}" for index in range(2))
    assert [record.message for record in caplog.records] == ["line 0", "line 1"]
