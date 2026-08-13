from pathlib import Path

from kneemri.arm_check import ArmCheck

GOOD = "--fold 0 --epochs 12 --one-cycle --slice-pooling max --whole-stack --labels /tmp/labels.csv"


def test_refused():
    """A real invocation passes, a typo'd flag is caught, and the parser's own complaint is reported.

    The pure half on purpose: `invocations` shells out to bash, which the unit gate forbids, while this
    is where the logic lives. Splitting them is what makes the check testable at all.
    """
    refused = ArmCheck().refused(
        {
            "good": GOOD,
            "typo": GOOD + " --slice-poolng max",
            "bad_value": "--fold 0 --slice-pooling nonsense",
        }
    )

    assert "good" not in refused, "a real arm's flags must survive the trainer's parser"
    assert "typo" in refused, "a misspelled flag is exactly what this exists to catch before a queue"
    assert "bad_value" in refused, "a value outside the choices is refused too"
    assert refused["typo"], "the parser's own complaint is reported, not a generic failure"


def test_invocations(tmp_path: Path):
    """An arm is SOURCED, so the shell finds what only the shell can — and its flags come back.

    Sourcing rather than reading the file is the point: an arm lost two hours to a quoting error once,
    and no regex finds that. Arms that never call run_train are absent rather than reported, because a
    lint that complains about files it cannot check gets switched off.
    """
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "real.sh").write_text(
        'source "${ARM_LIB}"\nrequire "$LABELS"\nrun_train "real" --fold 0 --epochs 8\n', encoding="utf-8"
    )
    (arms / "no_train.sh").write_text('source "${ARM_LIB}"\necho "this arm drives uv directly"\n', encoding="utf-8")

    found = ArmCheck(arms, Path("experiments/stub_lib.sh")).invocations()

    assert found == {"real": "--fold 0 --epochs 8"}
