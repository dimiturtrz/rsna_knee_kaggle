"""The file test, against the shapes actually written in this tree."""

from pathlib import Path

from kneemri.experiments.arm_shadow import ArmShadow


def write_arms(directory: Path) -> Path:
    """One arm through `uv`, one reaching `.venv` directly, one reaching a sibling environment."""
    arms = directory / "arms"
    arms.mkdir()
    (arms / "safe.sh").write_text("UV_NO_SYNC=1 uv run python -m kneemri.thing\n", encoding="utf-8")
    (arms / "direct.sh").write_text('"$REPO/.venv/Scripts/python.exe" -m kneemri.thing\n', encoding="utf-8")
    (arms / "sibling.sh").write_text('"$REPO/.venv-arm/bin/python" -m kneemri.thing\n', encoding="utf-8")
    return arms


def test_unshadowed(tmp_path: Path) -> None:
    """Read from the FILE, since an arm that reaches its interpreter directly may print nothing at all.

    Both slash flavours and both environments count: the eight arms in this tree are spelled several
    ways, and a pattern matching only `.venv/Scripts` would pass the tree while the hazard stood.
    """
    found = ArmShadow(write_arms(tmp_path)).unshadowed()

    assert [arm.name for arm in found] == ["direct.sh", "sibling.sh"]


def test_reported(tmp_path: Path) -> None:
    """Named, never skipped — and silent on a clean tree, so the lint's own output stays readable."""
    lines = ArmShadow(write_arms(tmp_path)).reported()

    assert len(lines) == 3, "one line an arm, plus what to do about them"
    assert "direct.sh" in lines[0]
    assert "UV_PROJECT_ENVIRONMENT=.venv-arm" in lines[-1], "a blind `uv run` rewrite moves the env too"

    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "safe.sh").write_text("uv run python -m kneemri.thing\n", encoding="utf-8")
    assert ArmShadow(clean).reported() == []
