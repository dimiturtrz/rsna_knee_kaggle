"""Which members a kernel script asks for, parsed rather than run."""

from pathlib import Path

from kneemri.kaggle.blend_shape import Fusion
from kneemri.kaggle.kernel_script import ALL_OURS, Blend, KernelScript


def _script(tmp_path: Path, body: str) -> KernelScript:
    written = tmp_path / "kernel.py"
    written.write_text(body, encoding="utf-8")
    return KernelScript(written)


def test_call(tmp_path: Path):
    """The `KernelRun(...)` a script builds, and None when it builds none rather than a guess."""
    assert _script(tmp_path, "KernelRun(members=1).run()\n").call() is not None
    assert _script(tmp_path, "MemberPackage.find(mounts)\n").call() is None, "graft picks members another way"
    assert _script(tmp_path, "SeatedEntry(names=chosen).run()\n").call() is not None, "builds its KernelRun in the kit"


def test_member_names(tmp_path: Path):
    """A literal selection resolves — through one variable — and a ranked one reads as the whole package.

    `members=N` is `strongest(N)` over recorded holdouts, which a parse cannot evaluate, so None is the
    answer that keeps the check conservative: every member is a candidate and every mount must be there.
    """
    assert _script(tmp_path, "KernelRun(names=('a', 'b'), overlap=False).run()\n").member_names() == ["a", "b"]

    indirect = "chosen = ('res392_ws_s2026', 'sliceattn_s2026')\nKernelRun(names=chosen).run()\n"
    assert _script(tmp_path, indirect).member_names() == ["res392_ws_s2026", "sliceattn_s2026"], (
        "the chosen entry names its members through a variable"
    )

    seated = "chosen = ('final_ws_cf', 'dense_cf')\nSeatedEntry(names=chosen, weight=1.0 / 3.0).run()\n"
    assert _script(tmp_path, seated).member_names() == ["final_ws_cf", "dense_cf"]

    assert _script(tmp_path, "KernelRun(members=2).run()\n").member_names() is None
    assert _script(tmp_path, "KernelRun().run()\n").member_names() is None
    assert _script(tmp_path, "KernelRun(names=picked()).run()\n").member_names() is None, "not a literal"

    twice = "chosen = ('a', 'b')\nKernelRun(names=chosen).run()\nKernelRun(names=('c', 'a')).run()\n"
    assert _script(tmp_path, twice).member_names() == ["a", "b", "c"], "every run's members, once each"
    bound = {
        "module": "chosen = ('a', 'b')\nSeatedEntry(names=chosen).run()\n",
        "function": "def main():\n    chosen = ('a', 'b')\n    SeatedEntry(names=chosen).run()\n",
    }
    for scope, body in bound.items():
        assert _script(tmp_path, body).member_names() == ["a", "b"], f"chosen bound at {scope} scope"
    unbound = "def main():\n    SeatedEntry(names=chosen).run()\n"
    assert _script(tmp_path, unbound).member_names() is None, "an unbound name is not a literal"

    table = (
        "def main():\n    seven = ('c', 'a')\n"
        "    for names, share, out in ((('a',), 0.25, 'x.csv'), (('b',), 0.25, 'y.csv'), (seven, 0.3, 'z.csv')):\n"
        "        KernelRun(names=names).run()\n"
    )
    assert _script(tmp_path, table).member_names() == ["a", "b", "c"], "every row of an unpacked loop table"
    ranked = "for names, out in ((picked(), 'x.csv'),):\n    KernelRun(names=names).run()\n"
    assert _script(tmp_path, ranked).member_names() is None, "a row that is not literal"

    ranked_second = "KernelRun(names=('a',)).run()\nKernelRun(members=2).run()\n"
    assert _script(tmp_path, ranked_second).member_names() is None, "one ranked run makes the whole set unknown"


def test_foreign_seats(tmp_path: Path):
    """Every way a kernel names its raptor checkpoint resolves to the board's name for it; no call is no seat."""
    default = "RaptorMember.loaded(RaptorEntry.weights(mounts=mounts), device)\n"
    assert _script(tmp_path, default).foreign_seats() == {"raptor_v4_swa"}, "no filename is DEFAULT_WEIGHTS"

    literal = 'RaptorEntry.weights("raptor_ft_effv2l480.pt", mounts)\n'
    assert _script(tmp_path, literal).foreign_seats() == {"raptor_ft_effv2l480"}

    variable = 'WEIGHTS = "raptor_ft_repl2x2_a_swa.pt"\nRaptorEntry.weights(WEIGHTS, mounts)\n'
    assert _script(tmp_path, variable).foreign_seats() == {"raptor_ft_repl2x2_a_swa"}

    looped = (
        'FOREIGN = ("raptor_ft_coatnet_v4_full_swa.pt", "raptor_ft_effv2l480.pt")\n'
        "kept = [name for name in directories]\n"
        "members = {name: RaptorEntry.weights(name, mounts=mounts) for name in FOREIGN}\n"
    )
    assert _script(tmp_path, looped).foreign_seats() == {"raptor_v4_swa", "raptor_ft_effv2l480"}, (
        "the loop AROUND the call binds the name, not another loop reusing it"
    )

    seated = "SeatedEntry(names=('a',)).run()\n"
    assert _script(tmp_path, seated).foreign_seats() == {"raptor_v4_swa"}, "a seated entry with no weights= is default"

    chosen = 'SeatedEntry(names=("a",), weights="raptor_ft_effv2l480.pt").run()\n'
    assert _script(tmp_path, chosen).foreign_seats() == {"raptor_ft_effv2l480"}

    assert _script(tmp_path, "KernelRun(names=('a',)).run()\n").foreign_seats() == set()
    assert _script(tmp_path, "RaptorEntry.weights(picked(), mounts)\n").foreign_seats() == set(), "not a literal"


SEAT = 'load("nartaa_seat.csv")\n'


def test_marked_seats(tmp_path: Path):
    """Any marker literal names its seat, a runner script as much as a table; none at all is an empty set."""
    assert _script(tmp_path, SEAT + 'run("helios_s6.py")\n').marked_seats() == {"nartaa384", "heliosli_s6"}
    assert _script(tmp_path, 'run("nartaa_0945.py")\n').marked_seats() == {"nartaa224"}
    assert _script(tmp_path, 'KernelRun(names=("a",)).run()\n').marked_seats() == set()


def test_blend(tmp_path: Path):
    """Each blend shape prices its seats; anything the parse cannot state as scalars covering every seat is None."""
    assert _script(tmp_path, "KernelRun(members=2).run()\n").blend() == ALL_OURS, "no foreign marker: all ours"

    combined = 'OURS = 2.0 / 3.0\nRaptorSeat.combined("ours.csv", working / "nartaa_seat.csv", (1.0 - OURS,))\n'
    assert _script(tmp_path, combined).blend() == Blend.of({"nartaa384": 1 / 3}, "rank"), "constant arithmetic"

    seat = 'RaptorSeat(weight=0.5).blended(read("ours.csv"), read("reader_seat.csv"))\n'
    assert _script(tmp_path, seat).blend() == Blend.of({"goodpjw_convnext": 0.5}, "rank")

    probit = (
        'TABLES = ("ours.csv", "nartaa_seat.csv", "ours_sub.csv")\nSHARES = (1 / 6, 0.5, 1 / 3)\n'
        "z = sum(w * ndtri(read(t)) for t, w in zip(TABLES, SHARES))\n"
    )
    assert _script(tmp_path, probit).blend() == Blend.of({"nartaa384": 0.5, "heliosli_s6": 1 / 3}, "probit")
    assert _script(tmp_path, probit.replace("ndtri", "rankdata")).blend() is None, "a zip outside probit is no blend"

    sliced = 'T = ("ours.csv", "nartaa_seat.csv")\nRaptorSeat.combined(T[0], T[1:], (0.25,))\n'
    assert _script(tmp_path, sliced).blend() == Blend.of({"nartaa384": 0.25}, "rank")
    two_readers = (
        'T = ("ours.csv", "seven.csv", "r7_sub.csv", "r6_sub.csv")\n'
        "RaptorSeat.combined(T[0], T[1:], (1 / 3, 1 / 6, 1 / 6))\n"
    )
    assert _script(tmp_path, two_readers).blend() == Blend.of({"heliosli_r7": 1 / 6, "heliosli_r6": 1 / 6}, "rank"), (
        "R6 is a foreign seat, not ours"
    )
    computed = 'T = ("ours.csv", "nartaa_seat.csv")\nRaptorSeat.combined(T[0], T[k:], (0.25,))\n'
    assert _script(tmp_path, computed).blend() is None, "a computed slice bound"

    uncovered = combined.replace("OURS = 2.0", 'run("helios_s6.py")\nOURS = 2.0')
    assert _script(tmp_path, uncovered).blend() is None, "a marked seat no blend prices"
    per_finding = SEAT + 'RaptorSeat.combined("ours.csv", "nartaa_seat.csv", shares_for(finding))\n'
    assert _script(tmp_path, per_finding).blend() is None, "a share the parse cannot evaluate"
    twice = combined + 'RaptorSeat.combined("ours.csv", "nartaa_seat.csv", (0.5,))\n'
    assert _script(tmp_path, twice).blend() is None, "two different blends cover the seat"


def test_of():
    """Rounded to four places and ordered by seat, so two spellings of one blend are equal."""
    assert Blend.of({"z": 2.0 / 3.0, "a": 0.5}, "probit") == Blend((("a", 0.5), ("z", 0.6667)), Fusion.PROBIT)
    assert Blend.of({}, "rank") == ALL_OURS


def test_described():
    """The reader's spelling: a seat-to-share dict and the fusion word."""
    assert Blend.of({"nartaa384": 0.5}, "rank").described() == "foreign {'nartaa384': 0.5}, fusion rank"
    assert ALL_OURS.described() == "foreign {}, fusion rank"


def test_differences():
    """One phrase per differing axis, mine first; none when equal."""
    third = Blend.of({"nartaa384": 0.5}, "rank")
    assert third.differences(third) == []
    assert third.differences(Blend.of({"nartaa384": 0.5}, "probit")) == ["fusion rank vs probit"]
    assert ALL_OURS.differences(Blend.of({"nartaa384": 0.5}, "probit")) == [
        "foreign {} vs {'nartaa384': 0.5}",
        "fusion rank vs probit",
    ]
