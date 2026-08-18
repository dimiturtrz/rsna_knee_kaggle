import json
from pathlib import Path

from kneemri.kaggle.champion_check import Champion

RECORD = {
    "entry": "solotop_v1",
    "kernel": "kneemri-solotop",
    "submission": 55607054,
    "board": 0.925,
    "kit": "e4cffcf",
    "members": ["res392_ws_s2026", "lr4e-5_ws_s2026"],
}

SCRIPT = '''"""A kernel whose PROSE names a member it no longer loads: res392_ws_s2026 was here."""

def main() -> None:
    chosen = ("res392_ws_s2026", "lr4e-5_ws_s2026")
    KernelRun(names=chosen, overlap=False).run()
'''


def _pinned(tmp_path: Path, members: list[str] | None = None) -> Path:
    path = tmp_path / "champion.json"
    held = dict(RECORD, members=members) if members else RECORD
    path.write_text(json.dumps(held), encoding="utf-8")
    return path


def _kernels(tmp_path: Path, script: str = SCRIPT) -> Path:
    root = tmp_path / "kernels" / RECORD["kernel"]
    root.mkdir(parents=True)
    (root / f"{RECORD['kernel']}.py").write_text(script, encoding="utf-8")
    return tmp_path / "kernels"


def _roster(tmp_path: Path, names: list[str]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "members.roster"
    body = "# what the live package holds\n" + "".join(f"{name}.pt  # why\n" for name in names)
    path.write_text(body, encoding="utf-8")
    return path


def test_recorded(tmp_path: Path):
    """The champion is READ, never inferred from whichever entry scored most recently."""
    champion = Champion.recorded(_pinned(tmp_path))

    assert champion.entry == "solotop_v1"
    assert champion.board == 0.925
    assert champion.members == ("res392_ws_s2026", "lr4e-5_ws_s2026")


def test_named_by_kernel(tmp_path: Path):
    """Parsed as Python, so a member name surviving only in a docstring is not a member."""
    champion = Champion.recorded(_pinned(tmp_path))

    named = champion.named_by_kernel(_kernels(tmp_path))

    assert named == ("res392_ws_s2026", "lr4e-5_ws_s2026")


def test_drift(tmp_path: Path):
    """The two ways the best entry dies quietly: the kernel stops naming it, the roster stops shipping it.

    Both leave every other gate green — the package builds, the kit verifies, the kernel runs — which is
    why this is asserted rather than reviewed.
    """
    champion = Champion.recorded(_pinned(tmp_path))
    kernels = _kernels(tmp_path)
    whole = _roster(tmp_path, ["res392_ws_s2026", "lr4e-5_ws_s2026"])

    assert champion.drift(whole, kernels) == []

    short = _roster(tmp_path / "short", ["lr4e-5_ws_s2026"])
    dropped = champion.drift(short, kernels)

    assert len(dropped) == 1
    assert "res392_ws_s2026" in dropped[0]
    assert "roster" in dropped[0]


def test_drift_sees_a_kernel_that_stopped_naming_a_member(tmp_path: Path):
    """A `chosen` tuple edited in place, rather than copied into a new kernel, is the expected failure."""
    champion = Champion.recorded(_pinned(tmp_path))
    edited = SCRIPT.replace('("res392_ws_s2026", "lr4e-5_ws_s2026")', '("lr4e-5_ws_s2026",)')

    found = champion.drift(
        _roster(tmp_path, ["res392_ws_s2026", "lr4e-5_ws_s2026"]),
        _kernels(tmp_path, edited),
    )

    assert len(found) == 1
    assert "no longer names 'res392_ws_s2026'" in found[0]
