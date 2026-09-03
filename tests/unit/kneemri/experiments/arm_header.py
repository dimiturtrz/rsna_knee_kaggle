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


def test_control_scores(tmp_path: Path):
    """The number beside the name, which is what picks a run out of its split-suffixed siblings.

    THE HYPHEN CASE IS THE REGRESSION. `lr4e-5_ws` is the control three headers call "the control every
    other member reads", and the name class excluded `-` until 2026-09-03, so it matched nothing and
    every arm reading it was reported as declaring no control at all.
    """
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "hyphen.sh").write_text("# control  lr4e-5_ws = 0.8525 (fold 0)\n", encoding="utf-8")
    (arms / "bare.sh").write_text("# control  wsfold0_e6 =\n", encoding="utf-8")
    headers = ArmHeader(arms)

    assert headers.controls(["hyphen", "bare"]) == {"hyphen": "lr4e-5_ws", "bare": "wsfold0_e6"}
    assert headers.control_scores(["hyphen", "bare"]) == {"hyphen": 0.8525}, "a checkpoint quotes no score"


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


def test_shell(tmp_path: Path):
    """Which file printed a name — the arm's own, else the shell recorded for it, else nothing.

    Two invocations printed by ONE shell are two runs an author wrote side by side; from two shells
    they are two experiments. Nothing in a flag line carries that distinction, so the TWIN check reads
    it from here, and a wrong answer costs a legitimate multi-run arm a refusal it cannot argue with.
    """
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "solo.sh").write_text("# bead: knee-yuqf\n", encoding="utf-8")
    pair = tmp_path / "throughput.sh"
    pair.write_text("run_train throughput_bf16\nrun_train throughput_compiled\n", encoding="utf-8")
    headers = ArmHeader(arms)
    headers.record("throughput_bf16", pair)
    headers.record("throughput_compiled", pair)

    assert headers.shell("solo") == arms / "solo.sh"
    assert headers.shell("throughput_bf16") == headers.shell("throughput_compiled") == pair
    assert headers.shell("rung_12") is None, "a synthesised recipe was printed by no shell of the tree"


def test_twins(tmp_path: Path):
    """The arms this one says it re-runs flag for flag, its variable being code rather than a flag.

    `swin_wholetop` carries `swin_supervised`'s exact invocation across `EncoderSurgery.blocks()`
    (knee-94ec), and the identity IS the experiment. Undeclared, that pair is a duplicate; declared, it
    is a before/after across something a flag line cannot express.
    """
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "after.sh").write_text("# twin-of: before\n# twin-of: also_before\n", encoding="utf-8")
    (arms / "before.sh").write_text("# bead: knee-yuqf\n", encoding="utf-8")
    headers = ArmHeader(arms)

    assert headers.twins("after") == {"before", "also_before"}
    assert headers.twins("before") == set(), "an undeclared twin is a duplicate, so silence declares nothing"


def test_paired(tmp_path: Path):
    """The two ways an author says a shared flag line is deliberate, and the pair that says neither.

    Either side's `# twin-of:` line covers the pair, because the arm written second is the one that knows
    the identity is deliberate. One shell printing both names covers it too — a multi-run arm's variable
    is the environment, which no flag line carries. Anything else is one recipe under two names.
    """
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "recoded.sh").write_text("# twin-of: staged\n", encoding="utf-8")
    (arms / "staged.sh").write_text("# bead: knee-yuqf\n", encoding="utf-8")
    (arms / "copy.sh").write_text("# bead: knee-yuqf\n", encoding="utf-8")
    shell = tmp_path / "throughput.sh"
    shell.write_text("# bead: knee-yuqf\n", encoding="utf-8")
    headers = ArmHeader(arms)
    headers.record("bf16", shell)
    headers.record("compiled", shell)

    assert headers.paired("recoded", "staged"), "the declaring arm's line covers the pair"
    assert headers.paired("staged", "recoded"), "and it covers the pair read from the other side too"
    assert headers.paired("bf16", "compiled"), "two runs an author wrote side by side in one shell"
    assert not headers.paired("copy", "staged"), "no declaration and two shells is a duplicate"
    assert not headers.paired("rung_8", "rung_12"), "two shell-less recipes are not paired by both being absent"
