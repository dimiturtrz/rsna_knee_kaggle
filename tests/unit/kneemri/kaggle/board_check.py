"""The submission record read against the board that made it — completeness, not correctness of recipe."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from kneemri.evaluation.board_transfer import SCOPES
from kneemri.kaggle.board_check import SEARCH, SELECTION_RULE_DATE, BoardCensus, BoardRow, LiveBoard

SEARCHED: Mapping[str, object] = {
    "id": "greedy_v1",
    "ref": "1",
    "date": "2026-08-31",
    "offline": 0.8704,
    "offline_raw": 0.8731,
    "selection": SEARCH,
    "board": 0.932,
    "scope": "ensemble",
}
"""`peak8`'s shape, corrected: the row the rule was written for, as the rule says it must be written."""

CLEAN: Sequence[Mapping[str, object]] = [
    {"note": "a note carries commentary and no submission, so nothing here counts it"},
    {"id": "alpha_v1", "ref": "1", "date": "2026-08-20", "offline": 0.86, "board": 0.93, "scope": "ensemble"},
    {
        "id": "beta_v1",
        "ref": "2",
        "date": "2026-08-24",
        "offline": None,
        "board": 0.937,
        "scope": "ensemble",
        "members_named": ["calib_span_ff", "band_dense_s2026"],
    },
    {"ref": "3", "date": "2026-08-12", "board": 0.5, "scope": "excluded", "why": "kit failure"},
]
LIVE: dict[str, float | None] = {"1": 0.93, "2": 0.937, "3": 0.5}


def _census(
    tmp_path: Path,
    rows: Sequence[Mapping[str, object]] | None = None,
    prose: str = "alpha beta",
    scripts: Mapping[str, str] | None = None,
) -> BoardCensus:
    """A record and a ledger written into the test's own directory — it touches nothing it did not make."""
    record = tmp_path / "board.jsonl"
    record.write_text("".join(json.dumps(row) + "\n" for row in (rows if rows is not None else CLEAN)))
    ledger = tmp_path / "SUBMISSIONS.md"
    ledger.write_text(prose)
    kernels = tmp_path / "kernels"
    for name, body in (scripts or {}).items():
        directory = kernels / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.py").write_text(body)
    return BoardCensus(record, ledger, kernels)


def test_parsed():
    """A JSONL line as a row, including the excluded line that carries no id and must still be reachable."""
    row = BoardRow.parsed(CLEAN[2])
    assert (row.identifier, row.ref, row.board, row.offline) == ("beta_v1", "2", 0.937, None)
    assert row.members_named == ("calib_span_ff", "band_dense_s2026")

    assert row.names_key, "the key was there"

    excluded = BoardRow.parsed(CLEAN[3])
    assert excluded.identifier == "3", "a row with no id is identified by its ref, never by an empty string"
    assert excluded.members_named == (), "no members named is an empty tuple, not a crash"
    assert not excluded.names_key, "and an absent key is distinguishable from an empty one"
    assert (excluded.kernel, excluded.members) == (None, None)
    assert (excluded.selection, excluded.offline_raw) == (None, None), "a row that says nothing parses as silence"

    searched = BoardRow.parsed(SEARCHED)
    assert (searched.selection, searched.offline_raw, searched.date) == (SEARCH, 0.8731, "2026-08-31")
    assert (searched.foreign, searched.fusion) == (None, None), "a row without the blend fields leaves them unsaid"

    blended = BoardRow.parsed({**SEARCHED, "foreign": {"nartaa384": 0.6667}, "fusion": "probit"})
    assert (blended.foreign, blended.fusion) == ({"nartaa384": 0.6667}, "probit")


def test_script(tmp_path: Path):
    """The kernel script an entry shipped from, by its own field and by the id-stem fallback."""
    declared = BoardRow.parsed({"id": "chosen_v2", "ref": "1", "kernel": "kneemri-chosen"})
    assert declared.script(tmp_path) == tmp_path / "kneemri-chosen" / "kneemri-chosen.py"

    silent = BoardRow.parsed({"id": "chosen_v2", "ref": "1"})
    assert silent.script(tmp_path) == declared.script(tmp_path), "a null kernel is not an unreachable row"

    assert BoardRow.parsed({"id": "solo", "ref": "1"}).script(tmp_path).name == "kneemri-solo.py", (
        "an id carrying no version mark is its own stem"
    )


def test_disagrees():
    """A row read against a live score, including the null cases a float compare would crash on."""
    row = BoardRow.parsed(CLEAN[1])
    assert not row.disagrees(0.93)
    assert not row.disagrees(0.9302), "half a tick is transcription rounding, not a defect"
    assert row.disagrees(0.928)
    assert row.disagrees(None), "recorded a score, the board holds none"

    scoreless = BoardRow.parsed({"id": "none_v1", "ref": "9", "board": None})
    assert scoreless.disagrees(0.93), "recorded none, the board scored it"
    assert not scoreless.disagrees(None), "no score either side is agreement, not a defect"


def test_all_data():
    """The seats that held nothing out, by the roster's own naming rule."""
    assert BoardRow.parsed(CLEAN[2]).all_data() == ["calib_span_ff"], "the _ff suffix, and only it"
    named = {"id": "x", "ref": "9", "members_named": ["final_ws_e12"]}
    assert BoardRow.parsed(named).all_data() == ["final_ws_e12"], "the one member that predates the suffix"
    assert BoardRow.parsed(CLEAN[1]).all_data() == [], "an entry naming no members says nothing either way"


def test_rows(tmp_path: Path):
    """Every entry, the commentary skipped — commentary is a note with no submission behind it."""
    assert [row.identifier for row in _census(tmp_path).rows()] == ["alpha_v1", "beta_v1", "3"]


def test_rows_keeps_an_annotated_submission(tmp_path: Path):
    """A ROW EXPLAINING ITSELF MUST NOT VANISH. Skipping on the note rather than on the missing ref
    hid four priced entries from every consumer, two of them at the board's own maximum, and every
    gate stayed green over it — `submit_check` cleared a set it had already spent a slot on.
    """
    annotated = [*CLEAN, {"id": "noted_v1", "ref": "9", "board": 0.94, "scope": "ensemble", "note": "why"}]
    assert "noted_v1" in [row.identifier for row in _census(tmp_path, annotated).rows()]


def test_refless(tmp_path: Path):
    """A row no live listing can reach, which is what the whole join key exists to prevent."""
    assert _census(tmp_path).refless() == []
    assert _census(tmp_path, [{"id": "orphan_v1", "board": 0.9, "scope": "ensemble"}]).refless() == ["orphan_v1"]


def test_duplicated(tmp_path: Path):
    """One submission recorded twice is two points in every fit that reads this file."""
    assert _census(tmp_path).duplicated() == []
    twice = [*CLEAN, {"id": "alpha_v2", "ref": "1", "board": 0.93, "scope": "ensemble"}]
    assert _census(tmp_path, twice).duplicated() == ["1"]


def test_unpriceable(tmp_path: Path):
    """An offline number over a pool that has seen the holdout it was scored on."""
    assert _census(tmp_path).unpriceable() == [], "beta names an all-data seat and records no offline"
    priced = [{"id": "bad_v1", "ref": "1", "offline": 0.87, "board": 0.93, "members_named": ["calib_span_ff"]}]
    found = _census(tmp_path, priced).unpriceable()
    assert len(found) == 1 and "calib_span_ff" in found[0], "the offending seat is named, not just counted"


def test_governed():
    """The rule's own scope: a row that prices something, written after the rule existed."""
    assert BoardRow.parsed(SEARCHED).governed(SELECTION_RULE_DATE)

    early = BoardRow.parsed({**SEARCHED, "date": "2026-08-24"})
    assert not early.governed(SELECTION_RULE_DATE), (
        "a pre-rule row was searched over a pool the measured band does not describe"
    )

    priceless = BoardRow.parsed({**SEARCHED, "offline": None})
    assert not priceless.governed(SELECTION_RULE_DATE), "no offline number is no claim to correct"

    assert BoardRow.parsed({**SEARCHED, "date": SELECTION_RULE_DATE}).governed(SELECTION_RULE_DATE), (
        "the day the rule was written is the first day it binds"
    )


def test_undeclared(tmp_path: Path):
    """A priced row the rule reaches that never says how its seats were chosen — silence is the finding."""
    assert _census(tmp_path, [SEARCHED]).undeclared() == []
    assert _census(tmp_path).undeclared() == [], "alpha predates the rule and is not defaulted into it"

    silent: Mapping[str, object] = {key: value for key, value in SEARCHED.items() if key != "selection"}
    found = _census(tmp_path, [silent]).undeclared()
    assert len(found) == 1 and "greedy_v1" in found[0] and "knee-jsgb" in found[0], "the row and the rule are named"

    invented = _census(tmp_path, [{**SEARCHED, "selection": "curated"}]).undeclared()
    assert len(invented) == 1, "a value outside the vocabulary declares nothing a check can read"


def test_misscoped(tmp_path: Path):
    """A scope in neither reader's vocabulary — green here, and silently absent from the fit there.

    MEASURED 2026-09-04: four rows carried `scope: "included"`, the negation of this file's EXCLUDED
    sentinel written into a field that names the DENOMINATOR instead. Three carried honest ensemble
    offline numbers and sat outside `board_transfer`'s fit while every check reported clean.
    """
    assert _census(tmp_path, [SEARCHED]).misscoped() == []

    for known in sorted(SCOPES):
        assert _census(tmp_path, [{**SEARCHED, "scope": known}]).misscoped() == [], f"{known} is vocabulary"

    invented = _census(tmp_path, [{**SEARCHED, "scope": "included"}]).misscoped()
    assert len(invented) == 1, "a value no reader knows is the defect, not a new scope"
    assert "included" in invented[0] and "greedy_v1" in invented[0], "the row and the value it carries"
    assert "ensemble" in invented[0], "the message says what may be written instead"


def test_uncorrected(tmp_path: Path):
    """The raw search-selected score — the row that was written the day after the rule forbade it."""
    assert _census(tmp_path, [SEARCHED]).uncorrected() == []
    assert _census(tmp_path, [{**SEARCHED, "selection": "designed"}]).uncorrected() == [], (
        "a hand-picked set was not selected against a field and carries no selection bias"
    )

    raw = _census(tmp_path, [{**SEARCHED, "offline": 0.8731}]).uncorrected()
    assert len(raw) == 1, "0.8731 raw against a raw 0.8731 is a correction of zero"
    assert "0.8700 to 0.8706" in raw[0], "the message says what the row should have carried"
    assert "+0.0000" in raw[0] and "knee-jsgb" in raw[0], "what it carries, and the rule it breaks"

    unstated = _census(tmp_path, [{key: v for key, v in SEARCHED.items() if key != "offline_raw"}]).uncorrected()
    assert len(unstated) == 1 and "offline_raw" in unstated[0], "a claim of correction is not arithmetic"

    over = _census(tmp_path, [{**SEARCHED, "offline": 0.8600}]).uncorrected()
    assert len(over) == 1, "an over-correction buys a transfer residual and is refused too"


def test_unnamed(tmp_path: Path):
    """A priced entry the prose ledger never mentions; an excluded one is exempt by design."""
    assert _census(tmp_path).unnamed() == []
    assert _census(tmp_path, prose="alpha only").unnamed() == ["beta_v1"]
    assert "3" not in _census(tmp_path, prose="").unnamed(), "an excluded row earns no write-up"


def test_hollow(tmp_path: Path):
    """An empty members_named, which reads as recovered to one consumer and missing to another."""
    assert _census(tmp_path).hollow() == [], "absent is silence and silence is allowed"
    no_seats: list[str] = []
    empty = [{"id": "weak_v1", "ref": "1", "board": 0.9, "scope": "ensemble", "members_named": no_seats}]
    found = _census(tmp_path, empty).hollow()
    assert len(found) == 1 and "weak_v1" in found[0]


def test_miscounted(tmp_path: Path):
    """Seats named against seats recorded — two independently sourced numbers, so disagreement is signal."""
    assert _census(tmp_path).miscounted() == [], "beta names two and records no count to contradict"
    short = [{"id": "part_v1", "ref": "1", "members": 6, "members_named": ["a", "b"], "scope": "ensemble"}]
    found = _census(tmp_path, short).miscounted()
    assert len(found) == 1 and "names 2" in found[0] and "members=6" in found[0]

    whole = [{"id": "part_v1", "ref": "1", "members": 2, "members_named": ["a", "b"], "scope": "ensemble"}]
    assert _census(tmp_path, whole).miscounted() == []


def test_recoverable(tmp_path: Path):
    """An unnamed row whose surviving script names its members anyway — and the ones that honestly cannot."""
    literal = 'chosen = ("a", "b", "c")\nKernelRun(names=chosen)\n'
    unnamed: list[Mapping[str, object]] = [{"id": "gamma_v1", "ref": "1", "board": 0.9, "scope": "ensemble"}]

    assert _census(tmp_path, unnamed).recoverable() == [], "no script is no claim"

    found = _census(tmp_path, unnamed, scripts={"kneemri-gamma": literal}).recoverable()
    assert len(found) == 1 and "3 of them" in found[0], "the id stem alone reaches the script"

    ranked = _census(tmp_path, unnamed, scripts={"kneemri-gamma": "KernelRun(members=3)\n"}).recoverable()
    assert ranked == [], "strongest(N) is a ranking a parse cannot replay, so silence is honest there"

    dropped: list[Mapping[str, object]] = [{"ref": "1", "board": 0.5, "scope": "excluded", "why": "kit"}]
    assert _census(tmp_path, dropped, scripts={"kneemri-1": literal}).recoverable() == []


def test_unrecorded(tmp_path: Path):
    """A submission that scored and reached no record — the failure this tool was written for."""
    assert _census(tmp_path).unrecorded(LIVE) == []
    assert _census(tmp_path).unrecorded({**LIVE, "99": 0.938}) == ["99"]
    assert _census(tmp_path).unrecorded({**LIVE, "99": None}) == [], "still scoring cannot yet be recorded"


def test_pending(tmp_path: Path):
    """The submission in flight: reported so it is not forgotten, and never counted as a defect."""
    assert _census(tmp_path).pending(LIVE) == []
    assert _census(tmp_path).pending({**LIVE, "99": None}) == ["99"]
    assert _census(tmp_path).pending({**LIVE, "99": 0.938}) == [], "a scored orphan is a refusal, not a wait"


def test_phantom(tmp_path: Path):
    """A recorded ref the board does not hold — a typo in the join key reads as a whole record."""
    assert _census(tmp_path).phantom(LIVE) == []
    assert _census(tmp_path).phantom({"1": 0.93, "2": 0.937}) == ["3"]
    assert _census(tmp_path).phantom({"3": 0.94}) == [], "older than the listing reaches is out of reach"


def test_mismatched(tmp_path: Path):
    """A board number transcribed wrongly, and the null-against-scored case a float compare would miss."""
    assert _census(tmp_path).mismatched(LIVE) == []
    assert len(_census(tmp_path).mismatched({**LIVE, "1": 0.928})) == 1
    assert len(_census(tmp_path).mismatched({**LIVE, "1": None})) == 1, "recorded a score, the board holds none"


def test_faults(tmp_path: Path):
    """Every defect at once, and the live half omitted entirely when no listing was taken."""
    assert _census(tmp_path).faults(LIVE) == []
    assert _census(tmp_path).faults(None) == []
    assert len(_census(tmp_path).faults({**LIVE, "99": 0.938})) == 1
    assert _census(tmp_path).faults({**LIVE, "99": None}) == [], "an entry in flight blocks nothing"
    assert _census(tmp_path).faults(None) == [], "without a listing the offline rules still pass"

    rawline = [*CLEAN, {**SEARCHED, "ref": "4", "offline": 0.8731}]
    found = _census(tmp_path, rawline, prose="alpha beta greedy").faults(None)
    assert len(found) == 1 and "greedy_v1" in found[0], "a raw search-selected row refuses with no board listing"


def test_taken():
    """The live listing through an injected lister — the real round trip is the only untested line."""
    board = LiveBoard.taken(lambda: [("1", "0.93", "2026-09-06 10:21:29"), ("4", "", "")])
    assert board.scores == {"1": 0.93, "4": None}, "a blank score is None, not 0.0"
