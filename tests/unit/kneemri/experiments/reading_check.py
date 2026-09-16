"""The mirror for the pre-registered-reading check: every public method called, and asserted."""

from pathlib import Path

from kneemri.experiments.reading_check import ArmDeclaration, ReadingDrift

DARK = "TRAIN='--fold -1 --ema'\n"


def _arm(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.sh"
    path.write_text(body, encoding="utf-8")
    return path


def test_read(tmp_path: Path) -> None:
    path = _arm(tmp_path, "one", "# reading: gold/one*\n")
    assert ArmDeclaration.read(path).text.startswith("# reading:")


def test_name(tmp_path: Path) -> None:
    assert ArmDeclaration.read(_arm(tmp_path, "one", "x\n")).name == "one"


def test_readings(tmp_path: Path) -> None:
    body = "# reading: gold/one*\n#\n# reading:   \n# reading: a flat mean is the finding\n# reading: board/one\n"
    assert ArmDeclaration.read(_arm(tmp_path, "one", body)).readings() == ["gold/one*", "board/one"]


def test_unmeasurable(tmp_path: Path) -> None:
    assert ArmDeclaration.read(_arm(tmp_path, "one", "# unmeasurable: bought no ruler\n")).unmeasurable()
    assert not ArmDeclaration.read(_arm(tmp_path, "two", "# bead: knee-x\n")).unmeasurable()


def test_dark(tmp_path: Path) -> None:
    assert ArmDeclaration.read(_arm(tmp_path, "one", DARK)).dark()
    assert not ArmDeclaration.read(_arm(tmp_path, "two", "TRAIN='--fold 0'\n")).dark()
    assert not ArmDeclaration.read(_arm(tmp_path, "three", "# this is the fold-0 twin of --fold -1\n")).dark()


def _drift(tmp_path: Path) -> ReadingDrift:
    models = tmp_path / "models"
    processed = tmp_path / "processed"
    models.mkdir()
    processed.mkdir()
    return ReadingDrift(models, processed, tmp_path / "arms")


def test_declarations(tmp_path: Path) -> None:
    drift = _drift(tmp_path)
    _arm(drift.arms, "b", "x\n")
    _arm(drift.arms, "a", "y\n")
    assert [arm.name for arm in drift.declarations()] == ["a", "b"]


def test_ran(tmp_path: Path) -> None:
    drift = _drift(tmp_path)
    arm = ArmDeclaration.read(_arm(drift.arms, "one", "x\n"))
    assert not drift.ran(arm)
    (drift.models / "one_s2026.pt").write_bytes(b"")
    assert drift.ran(arm)


def test_arm_names(tmp_path: Path) -> None:
    drift = _drift(tmp_path)
    _arm(drift.arms, "one", "x\n")
    _arm(drift.arms, "one_336", "x\n")
    assert drift.arm_names() == frozenset({"one", "one_336"})


def test_ran_ignores_a_longer_named_sibling(tmp_path: Path) -> None:
    """`one_336` writes `one_336_s2026.pt`; that is not evidence that `one` ran."""
    drift = _drift(tmp_path)
    arm = ArmDeclaration.read(_arm(drift.arms, "one", "x\n"))
    _arm(drift.arms, "one_336", "x\n")
    (drift.models / "one_336_s2026.pt").write_bytes(b"")
    assert not drift.ran(arm)
    (drift.models / "one_s2026.pt").write_bytes(b"")
    assert drift.ran(arm)


def test_found(tmp_path: Path) -> None:
    drift = _drift(tmp_path)
    assert not drift.found("gold/one*")
    (drift.processed / "gold").mkdir()
    (drift.processed / "gold" / "one_s2026.npy").write_bytes(b"")
    assert drift.found("gold/one*")


def test_found_in_models(tmp_path: Path) -> None:
    """A holdout array lives beside its checkpoint in the MODELS root, which is the commonest ruler there is."""
    drift = _drift(tmp_path)
    assert not drift.found("one_*sl_s2026.holdout.npz")
    (drift.models / "one_20sl_s2026.holdout.npz").write_bytes(b"")
    assert drift.found("one_*sl_s2026.holdout.npz")


def test_found_absolute(tmp_path: Path) -> None:
    """An ABSOLUTE declaration resolves directly — `Path.glob` raises on one, which killed the whole check.

    Five arms declare `D:/models/…holdout.npz`, and from 2026-09-07 `84660bf5` every invocation of
    `reading_check` exited 1 on `NotImplementedError: Non-relative patterns are unsupported`, reporting
    nothing at all (D193 §1).
    """
    drift = _drift(tmp_path)
    array = drift.models / "abs_s2026.holdout.npz"
    assert not drift.found(str(array))
    array.write_bytes(b"")
    assert drift.found(str(array))
    assert drift.found(str(drift.models / "abs_*.holdout.npz"))


def test_missing(tmp_path: Path) -> None:
    drift = _drift(tmp_path)
    _arm(drift.arms, "one", "# reading: gold/one*\n")
    assert drift.missing() == []

    (drift.models / "one_s2026.pt").write_bytes(b"")
    assert "one declared 'gold/one*'" in drift.missing()[0]


def test_undeclared(tmp_path: Path) -> None:
    drift = _drift(tmp_path)
    _arm(drift.arms, "dark", DARK)
    _arm(drift.arms, "said", DARK + "# unmeasurable: knowingly\n")
    _arm(drift.arms, "held", "TRAIN='--fold 0'\n")
    assert drift.undeclared() == ["dark"]


def test_complaints(tmp_path: Path) -> None:
    drift = _drift(tmp_path)
    _arm(drift.arms, "dark", DARK)
    _arm(drift.arms, "said", DARK + "# unmeasurable: knowingly\n")
    assert drift.complaints() == ["dark fits --fold -1 and declares neither a reading nor why it has none"]


def test_report(tmp_path: Path) -> None:
    drift = _drift(tmp_path)
    _arm(drift.arms, "dark", DARK)
    _arm(drift.arms, "one", "# reading: gold/one*\n")
    lines = drift.report()
    assert "dark fits --fold -1" in lines[0]
    assert "2 arms: 1 declare a reading, 1 are dark with none declared" in lines[-1]


def test_taken(tmp_path: Path) -> None:
    drift = _drift(tmp_path)
    arm = ArmDeclaration.read(_arm(drift.arms, "one", "# reading: one*.npz\n"))
    assert drift.taken(arm) == []

    (drift.models / "one_lr3.npz").write_bytes(b"")
    assert drift.taken(arm) == ["one*.npz"]


def test_taken_is_silent_once_the_arm_has_run(tmp_path: Path) -> None:
    drift = _drift(tmp_path)
    arm = ArmDeclaration.read(_arm(drift.arms, "one", "# reading: one*.npz\n"))
    (drift.models / "one_lr3.npz").write_bytes(b"")
    (drift.models / "one_s2026.pt").write_bytes(b"")
    assert drift.taken(arm) == []


def test_already_taken(tmp_path: Path) -> None:
    drift = _drift(tmp_path)
    _arm(drift.arms, "one", "# reading: one*.npz\n")
    _arm(drift.arms, "two", "# reading: two*.npz\n")
    (drift.models / "one_lr3.npz").write_bytes(b"")
    assert drift.already_taken("two") == []

    lines = drift.already_taken("one")
    assert len(lines) == 1
    assert "one declared 'one*.npz' and it ALREADY matches" in lines[0]
    assert "one_lr3.npz" in lines[0]


def test_matches(tmp_path: Path) -> None:
    drift = _drift(tmp_path)
    assert drift.matches("nothing*.npz") == ""

    (drift.models / "one_lr3.npz").write_bytes(b"")
    resolved = drift.matches("one*.npz")
    assert "one_lr3.npz" in resolved
    assert "Z)" in resolved
