import re

import pytest

from kneemri.training.progress_schedule import FULL, NEGATIVE, OVERLAP, ROLES, ProgressSchedule

RAMPED = ProgressSchedule(start=0.0, mid=1.0, end=0.0, ramp=0.3, tail=0.15)


def test_value_default_is_exactly_one_everywhere() -> None:
    schedule = ProgressSchedule()
    assert all(schedule.value(progress) == FULL for progress in (0.0, 0.3, 0.5, 0.99, 1.0))


def test_value() -> None:
    assert RAMPED.value(0.0) == 0.0
    assert RAMPED.value(0.15) == pytest.approx(0.5)
    assert RAMPED.value(0.3) == pytest.approx(1.0)
    assert RAMPED.value(0.6) == 1.0
    assert RAMPED.value(0.85) == 1.0
    assert RAMPED.value(0.925) == pytest.approx(0.5)
    assert RAMPED.value(1.0) == pytest.approx(0.0)


def test_value_clips_progress_outside_the_run() -> None:
    assert RAMPED.value(-0.5) == 0.0
    assert RAMPED.value(1.5) == pytest.approx(0.0)


def test_value_zero_ramp_is_a_step() -> None:
    step = ProgressSchedule(start=0.0, mid=0.7, ramp=0.0)
    assert step.value(0.0) == pytest.approx(0.7)
    assert step.value(1.0) == pytest.approx(0.7)


def test_value_mid_above_one_is_allowed() -> None:
    assert ProgressSchedule(start=1.0, mid=1.5, end=1.0, ramp=0.5, tail=0.5).value(0.5) == pytest.approx(1.5)


@pytest.mark.parametrize(
    ("knobs", "message"),
    [
        ({"start": -0.1}, NEGATIVE),
        ({"end": -1.0}, NEGATIVE),
        ({"ramp": 0.6, "tail": 0.5}, OVERLAP),
        ({"ramp": -0.1}, OVERLAP),
    ],
)
def test_invalid_schedules_are_refused(knobs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        ProgressSchedule(**knobs)


def test_recorded() -> None:
    assert RAMPED.recorded() == (0.0, 1.0, 0.0, 0.3, 0.15)


def test_knob_help() -> None:
    texts = {knob: ProgressSchedule.knob_help(knob, "the subject") for knob in ROLES}
    assert set(ROLES) == {"start", "mid", "end", "ramp", "tail"}
    assert all(text.startswith(ROLES[knob]) and "the subject" in text for knob, text in texts.items())
