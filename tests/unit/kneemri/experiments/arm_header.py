"""What an arm's header declares, over shells these tests write themselves."""

import json
from pathlib import Path

from kneemri.experiments.arm_header import ArmHeader


def test_text(tmp_path: Path):
    """The file named after the arm first, the shell that PRINTED the name second, empty for neither.

    The second clause is the one that was missing: `final_fit_six.sh` prints six member names and none
    of them is its filename, so every declaration in the tree's most expensive arm went unread.
    """
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "solo.sh").write_text("# bead: knee-yuqf\n", encoding="utf-8")
    six = tmp_path / "final_fit_six.sh"
    six.write_text("# gate: kneemri-finalfit >= 0.936\n", encoding="utf-8")
    headers = ArmHeader(arms)
    headers.record("member_ff", six)

    assert "knee-yuqf" in headers.text("solo")
    assert "0.936" in headers.text("member_ff"), "a six-member arm's header is reached through its shell"
    assert headers.text("rung_12") == "", "a synthesised recipe has no shell, and that is not a fault"


def test_record(tmp_path: Path):
    """A name is bound to the shell that printed it, which is the only link a multi-member arm has."""
    arms = tmp_path / "arms"
    arms.mkdir()
    shell = tmp_path / "pair.sh"
    shell.write_text("# control  band_dense = 0.8578\n", encoding="utf-8")
    headers = ArmHeader(arms)

    assert headers.controls(["one_ff"]) == {}, "unrecorded, the name resolves to no shell at all"
    headers.record("one_ff", shell)
    assert headers.controls(["one_ff"]) == {"one_ff": "band_dense"}


def test_beads(tmp_path: Path):
    """ONE REP PER CLASS of how an arm can name the issue that priced it: named, unnamed, or absent.

    The header line is the whole record — nothing else on disk ties an arm to the argument that bought
    its card, and the 140 mm build ran forty minutes on a cost with no expected gain beside it. An arm
    whose file does not exist is dropped rather than reported None, because `invocations` can carry a
    queued rung (`rung_12`) whose SCRIPT is named `rung.sh`; reporting that as unbeaded would refuse
    every parameterised arm for a file it was never going to have.
    """
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "priced.sh").write_text("#!/bin/bash\n# bead: knee-yuqf\nrun_train priced\n", encoding="utf-8")
    (arms / "bare.sh").write_text("#!/bin/bash\nrun_train bare\n", encoding="utf-8")

    found = ArmHeader(arms).beads(["priced", "bare", "rung_12"])

    assert found == {"priced": "knee-yuqf", "bare": None}, "a rung with no file of its own is not an unbeaded arm"


def test_controls(tmp_path: Path):
    """ONE REP PER CLASS of how a header can name a control: declared, prose, absent.

    The `=` is what separates a declaration from a sentence — five headers in the tree open prose with
    "# control is the same recipe...", and reading those as a control name would diff every arm against
    a run called `is`.
    """
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "declared.sh").write_text("# control  band_dense = 0.8578 (best epoch 8)\n", encoding="utf-8")
    (arms / "prose.sh").write_text("# control is the same recipe with one flag moved\n", encoding="utf-8")
    (arms / "silent.sh").write_text("# bead: knee-yuqf\n", encoding="utf-8")

    found = ArmHeader(arms).controls(["declared", "prose", "silent"])

    assert found == {"declared": "band_dense"}, "a sentence beginning with the word is not a declaration"


def test_deliberate(tmp_path: Path):
    """The flags an arm says it moves on purpose — one line each, so moving three is visible as three."""
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "two.sh").write_text("# deliberate: --epochs\n# deliberate: --backbone-lr\n", encoding="utf-8")
    (arms / "silent.sh").write_text("# bead: knee-yuqf\n", encoding="utf-8")
    headers = ArmHeader(arms)

    assert headers.deliberate("two") == {"--epochs", "--backbone-lr"}
    assert headers.deliberate("silent") == set(), "an undeclared diff is the defect, so silence declares nothing"


def test_gated(tmp_path: Path):
    """An arm whose own precondition the board does not meet, beside one that is met and one ungated.

    MEASURED 2026-08-23: `final_fit_six.sh` stated a 0.936 floor in its header, `kneemri-finalfit` read
    0.934, and the arm ran roughly five hours of card anyway — a stop condition in a bash comment has
    no consumer. The met and ungated arms are here because a check that names every arm names nothing.
    """
    arms = tmp_path / "arms"
    arms.mkdir()
    board = tmp_path / "board.jsonl"
    board.write_text(json.dumps({"id": "finalfit_v1", "kernel": "kneemri-finalfit", "board": 0.934}), encoding="utf-8")
    (arms / "expensive.sh").write_text("# gate: kneemri-finalfit >= 0.936\n", encoding="utf-8")
    (arms / "licensed.sh").write_text("# gate: kneemri-finalfit >= 0.930\n", encoding="utf-8")
    (arms / "ungated.sh").write_text("# bead: knee-yuqf\n", encoding="utf-8")

    found = ArmHeader(arms).gated(["expensive", "licensed", "ungated"], board)

    assert list(found) == ["expensive"], "only the arm the board refuses is named"
    assert "0.934" in found["expensive"], "and the sentence carries the reading it was refused on"
