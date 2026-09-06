import datetime as dt

from kneemri.kaggle.slot_pace import DAILY_SLOTS, SlotPace

NOW = dt.datetime(2026, 9, 6, 11, 42, tzinfo=dt.UTC)


def _pace(*stamps: dt.datetime) -> SlotPace:
    return SlotPace(stamps, NOW)


def test_spent():
    """Slots are counted by FIRING date, and yesterday's submissions are not today's allowance.

    The pending one is the point: `56052777` was fired at 10:21Z and had no score and no record row at
    11:42Z, and the arithmetic done off `comp/board.jsonl` therefore read the day as empty.
    """
    pace = _pace(
        dt.datetime(2026, 9, 6, 0, 21, tzinfo=dt.UTC),
        dt.datetime(2026, 9, 6, 10, 21, tzinfo=dt.UTC),
        dt.datetime(2026, 9, 5, 23, 19, tzinfo=dt.UTC),
    )
    assert pace.spent() == 2, "two fired today, one yesterday"


def test_left():
    """What is left is the allowance minus the day's firings, and it never goes negative."""
    assert _pace().left() == DAILY_SLOTS
    fired = tuple(dt.datetime(2026, 9, 6, hour, tzinfo=dt.UTC) for hour in range(6))
    assert SlotPace(fired, NOW).left() == 0, "six firings against five slots is zero left, not minus one"


def test_interval():
    """The gap is what is LEFT of the day over the slots left, so an unspent mark compresses the rest."""
    three = _pace(*(dt.datetime(2026, 9, 6, hour, tzinfo=dt.UTC) for hour in (0, 10)))
    assert abs(three.interval().total_seconds() / 3600 - 3.93) < 0.01, "11h48m over three slots"

    later = SlotPace(three.submitted, dt.datetime(2026, 9, 6, 15, 30, tzinfo=dt.UTC))
    assert later.interval() < three.interval(), "the same three slots later in the day are closer together"


def test_deadline():
    """The day's last mark is 23:30 UTC, the owner's directive, on the current UTC date."""
    assert _pace().deadline() == dt.datetime(2026, 9, 6, 23, 30, tzinfo=dt.UTC)


def test_marks():
    """Every remaining mark, and the last of them is the deadline itself rather than a step past it."""
    marks = _pace(dt.datetime(2026, 9, 6, 0, 21, tzinfo=dt.UTC)).marks()
    assert len(marks) == DAILY_SLOTS - 1
    assert marks == sorted(marks), "earliest first"
    assert abs((marks[-1] - _pace().deadline()).total_seconds()) < 1, "the last mark IS 23:30Z"


def test_lines():
    """A spent day says so and names the reset; a live day prints the compressed interval."""
    spent = SlotPace(tuple(dt.datetime(2026, 9, 6, hour, tzinfo=dt.UTC) for hour in range(5)), NOW)
    assert "no slot left" in spent.lines()[0]

    live = _pace(dt.datetime(2026, 9, 6, 0, 21, tzinfo=dt.UTC)).lines()
    assert "1 of 5 spent today, 4 left" in live[0]
    assert "23:30Z" in live[1], "the marks are printed, not just the count"
