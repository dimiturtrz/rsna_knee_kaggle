import json
from pathlib import Path

import pytest

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

    assert champion.drift((whole,), kernels) == []

    short = _roster(tmp_path / "short", ["lr4e-5_ws_s2026"])
    dropped = champion.drift((short,), kernels)

    assert len(dropped) == 1
    assert "res392_ws_s2026" in dropped[0]
    assert "roster" in dropped[0]

    second = _roster(tmp_path / "second", ["res392_ws_s2026"])
    assert champion.drift((short, second), kernels) == [], "a pin spanning both shelves is still buildable"


def _board(tmp_path: Path) -> Path:
    """The board record as it is actually written: prose rows interleaved among the entries."""
    path = tmp_path / "board.jsonl"
    rows = [
        {"note": "how to read this file"},
        {
            "id": "greedy7_v1",
            "board": 0.933,
            "kernel": "kneemri-greedy7",
            "kit": "e7b65a0",
            "members_named": ["small_batch6_s2026", "calib_span_s2026"],
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_from_board(tmp_path: Path):
    """Any scored entry can be priced against, not only the best one — the champion may be unscoreable."""
    entry = Champion.from_board("greedy7_v1", _board(tmp_path))

    assert entry.entry == "greedy7_v1"
    assert entry.board == 0.933
    assert entry.members == ("small_batch6_s2026", "calib_span_s2026")
    assert entry.kit == "e7b65a0"


def test_from_board_names_the_field_a_half_written_row_is_missing(tmp_path: Path):
    """A row can be present and unusable, and the complaint must say which field sent the reader back.

    31 of 48 rows were written without `members_named`, so this path raised a bare KeyError naming
    neither the entry nor the field — on exactly the entries at the top of the board.
    """
    path = tmp_path / "half.jsonl"
    path.write_text(json.dumps({"id": "thin_v1", "board": 0.9, "kernel": "k"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="records no members_named") as raised:
        Champion.from_board("thin_v1", path)

    assert "thin_v1" in str(raised.value), "the entry, so the reader knows which kernel log to open"


def test_from_board_carries_an_unknown_kit_as_none_and_still_prices(tmp_path: Path):
    """`"kit": null` is a row DECLINING to invent a sha, and it must neither refuse nor become "None".

    Fourteen entries predate per-entry kernel directories, so no log holds their kit. Refusing them
    would cost the one use they are still good for — nothing that prices an entry reads the kit — and
    requiring the field would make `str(None)` read like a commit to every check downstream.
    """
    path = tmp_path / "nulled.jsonl"
    row = {"id": "nulled_v1", "board": 0.9, "kernel": "k", "members_named": ["a", "b"], "kit": None}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    entry = Champion.from_board("nulled_v1", path)

    assert entry.kit is None, "the unknown stays an unknown rather than becoming the string 'None'"
    assert entry.members == ("a", "b"), "and the entry is still a usable pricing target"


def test_from_board_skips_the_prose_rows(tmp_path: Path):
    """A row carrying only a note is not an entry, and it is not an error either."""
    with pytest.raises(ValueError, match="no board entry named nosuch") as raised:
        Champion.from_board("nosuch", _board(tmp_path))

    assert "the record holds greedy7_v1" in str(raised.value), "only real entries are offered as the way out"


def test_outranked(tmp_path: Path):
    """A pin the board has moved past is a defect, and a disqualified row scoring higher is not.

    THE HOLE THIS CLOSES: `drift` asks whether the pin can be rebuilt and passes on a pin naming
    yesterday's entry, which is how the pin came to lag twice — once by three hours, once through five
    submissions — while every gate stayed green. The scope filter is the other half: an `excluded` row is
    on the board with a stated reason it may never be a champion (a grafted fusion, a kit failure), so it
    must not raise this complaint however it scored.
    """
    path = tmp_path / "board.jsonl"
    rows = [
        {"note": "how to read this file"},
        {"id": "tied_v1", "board": 0.938, "scope": "ensemble"},
        {"id": "grafted_v1", "board": 0.944, "scope": "excluded"},
        {"id": "better_v1", "board": 0.941, "scope": "ensemble"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    champion = Champion(entry="votefit", kernel="kneemri-votefit", board=0.938, members=("a",), kit="aa79cd3")

    found = champion.outranked(path)

    assert len(found) == 1, "a tie is not outranked, and the grafted 0.944 is disqualified by scope"
    assert "better_v1" in found[0]
    assert "0.941" in found[0]

    ahead = Champion(entry="better", kernel="k", board=0.941, members=("a",), kit="aa79cd3")

    assert ahead.outranked(path) == [], "a pin holding the maximum is current"


def test_drift_sees_a_kernel_that_stopped_naming_a_member(tmp_path: Path):
    """A `chosen` tuple edited in place, rather than copied into a new kernel, is the expected failure."""
    champion = Champion.recorded(_pinned(tmp_path))
    edited = SCRIPT.replace('("res392_ws_s2026", "lr4e-5_ws_s2026")', '("lr4e-5_ws_s2026",)')

    found = champion.drift(
        (_roster(tmp_path, ["res392_ws_s2026", "lr4e-5_ws_s2026"]),),
        _kernels(tmp_path, edited),
    )

    assert len(found) == 1
    assert "no longer names 'res392_ws_s2026'" in found[0]
