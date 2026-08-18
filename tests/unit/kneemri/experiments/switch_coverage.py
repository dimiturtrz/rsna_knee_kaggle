from pathlib import Path

from kneemri.experiments.switch_coverage import COVERAGE, NOT_A_RECIPE, SwitchCoverage
from kneemri.training.train import TrainingOptions


def test_armed(tmp_path: Path):
    """The armed set comes out of the shell files, so an arm that drops a flag drops its coverage."""
    (tmp_path / "an_arm.sh").write_text("run_train --slice-pooling max --whole-stack\n", encoding="utf-8")

    armed = SwitchCoverage(tmp_path).armed()

    assert armed == {"slice_pooling", "whole_stack"}, "the flags passed, translated to option names"


def test_unaccounted():
    """No training switch may be reachable, stamped, and never run by anything.

    Five were, on 2026-08-18: `--pool` focal, `--anatomical-prior`, `--normalise-by-count`, `--tie` and
    `--head-lr`. A knob in that state fails at the queue, after the card is held — the exact price
    `arm_check` exists to stop paying. A new option now lands here first: arm it, cover it with a smoke
    recipe, or say in `NOT_A_RECIPE` why it is not a method.
    """
    assert not SwitchCoverage().unaccounted()


def test_coverage_recipes_are_flags_the_trainer_accepts():
    """A recipe that the parser refuses would report itself as a broken switch, blaming the knob."""
    for name, flags in COVERAGE.items():
        parsed = TrainingOptions.parser().parse_args(["--labels", "x.csv", *flags.split()])
        options = TrainingOptions(**vars(parsed))
        assert getattr(options, name) != getattr(TrainingOptions(), name), f"{name} recipe leaves the default"


def test_nothing_is_both_covered_and_written_off():
    """A knob claiming both a smoke recipe and a reason for having none is a contradiction to resolve."""
    assert not set(COVERAGE) & set(NOT_A_RECIPE)


def test_no_entry_names_an_option_that_no_longer_exists():
    """A renamed option would leave its entry behind, accounting for a field nothing has any more.

    The partition would still come out total — the stale name subtracts nothing — so the RENAMED option
    lands in the hole silently, which is the state this module was built to make impossible.
    """
    options = set(vars(TrainingOptions()))

    assert not (set(COVERAGE) | set(NOT_A_RECIPE)) - options


def test_labels():
    """The label table is read off an arm's own flag line, and absent when no arm printed one."""
    assert SwitchCoverage.labels({"a": "--labels /data/x.csv --whole-stack"}) == "--labels /data/x.csv"
    assert SwitchCoverage.labels({"a": "--whole-stack"}) == ""


def test_invocations():
    """A recipe without `--labels` is refused by the parser, and reads as a broken switch.

    Taken from an arm rather than written here: the stub fabricates one root under CI and `lib.sh`
    resolves another on this machine, so any spelling of that path in this module is wrong somewhere.
    """
    printed = {"an_arm": "--labels /data/labels.csv --whole-stack"}

    assert SwitchCoverage().invocations(printed)["switch_tie"] == "--labels /data/labels.csv --tie last"
    assert SwitchCoverage().invocations({})["switch_tie"] == "--tie last", "no arm printed one, so nothing to add"
