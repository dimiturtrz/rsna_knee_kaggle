"""The member curve's arithmetic, on constructed predictions whose right answer is known by hand."""

from pathlib import Path

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.member_curve import MemberCurve
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.scores import LabelScores
from kneemri.inference.vote import RankAverage

STUDIES = 40
SEED = 2026


def _truth() -> np.ndarray:
    """Half the studies positive on every label."""
    column = np.array([1.0] * (STUDIES // 2) + [0.0] * (STUDIES // 2), dtype=np.float32)
    return np.repeat(column[:, None], len(TARGETS), axis=1)


def _member(strength: float, seed: int) -> np.ndarray:
    """A member that ranks the positives above the negatives with the given reliability."""
    generator = np.random.default_rng(seed)
    signal = _truth()[:, 0].astype(np.float64) * strength
    return np.repeat((signal + generator.normal(0.0, 1.0, STUDIES))[:, None], len(TARGETS), axis=1)


def _names(members: list[np.ndarray]) -> list[str]:
    """Names for a pool whose test does not care which member is which."""
    return [f"m{index}.pt" for index in range(len(members))]


def test_sizes():
    """Three members give three singletons, three pairs and one triple."""
    members = [_member(1.0, seed) for seed in (1, 2, 3)]
    by_size = MemberCurve(LabelScores()).sizes(Pool(_names(members), members, _truth()))
    assert [len(by_size[size]) for size in (1, 2, 3)] == [3, 3, 1]


def test_sizes_shows_independent_members_helping():
    """Averaging ranks of independent noisy members beats any one of them.

    This is the property the whole experiment exists to measure, so a curve that could not detect it
    would be measuring nothing. The members here are independent by construction; real ones share
    training data and are not, which is exactly why the real curve has to be run rather than assumed.
    """
    members = [_member(1.0, seed) for seed in (1, 2, 3)]
    by_size = MemberCurve(LabelScores()).sizes(Pool(_names(members), members, _truth()))
    assert np.mean(by_size[3]) > np.mean(by_size[1])


def test_sizes_is_flat_for_duplicated_members():
    """A member added twice adds nothing — the guard against reading ensembling into a copy."""
    duplicated = [_member(1.0, 1)] * 3
    by_size = MemberCurve(LabelScores()).sizes(Pool(_names(duplicated), duplicated, _truth()))
    assert np.mean(by_size[3]) == np.mean(by_size[1])


def test_report():
    """One line per size, carrying the mean, the range and the step from the previous size."""
    lines = MemberCurve(LabelScores()).report({1: [0.80, 0.82], 2: [0.85]})
    assert lines[0].startswith("1 member(s): mean 0.8100")
    assert "+0.0400 over 1" in lines[1]


def test_subsets():
    """Every subset of one size, carrying the INDICES that identify it and not just its score."""
    curve = MemberCurve(LabelScores())
    truth = _truth()
    members = [_member(2.0, SEED), _member(0.5, SEED + 1)]

    pairs = curve.subsets(Pool(_names(members), members, truth), size=1)
    assert [subset for _, subset in pairs] == [(0,), (1,)], "one entry per member, each naming itself"
    assert all(np.isfinite(score) for score, _ in pairs)
    assert [subset for _, subset in curve.subsets(Pool(_names(members), members, truth), size=2)] == [(0, 1)]


def test_best():
    """The winning subset is NAMED at every size, which is what a submission is built from.

    `report` said five members reached 0.8601 and never said WHICH five, so the best number this
    analysis produced could not be turned into an entry.
    """
    curve = MemberCurve(LabelScores())
    truth = _truth()
    members = [_member(2.0, SEED), _member(0.2, SEED + 1)]

    lines = curve.best(Pool(["strong.pt", "weak.pt"], members, truth))
    assert len(lines) == 2
    assert "strong.pt" in lines[0] and "weak.pt" not in lines[0], "the best single member is the strong one"
    assert "strong.pt" in lines[1] and "weak.pt" in lines[1], "the only pair is both of them"


def test_oracle():
    """The ceiling is at least the plain average, and strictly above it when members differ per label.

    Constructed so member A is perfect on the first label and useless on the second while member B is
    the reverse. Averaging them cannot beat either on the label it owns, so the per-label ceiling must
    exceed the average - which is exactly the case the measurement exists to detect.
    """
    truth = np.zeros((6, len(TARGETS)), dtype=np.float32)
    truth[3:, 0] = 1.0
    truth[3:, 1] = 1.0
    first = np.tile(np.array([[0.1], [0.2], [0.3], [0.7], [0.8], [0.9]]), (1, len(TARGETS)))
    second = first.copy()
    second[:, 0] = first[::-1, 0]  # B is inverted on label 0
    first[:, 1] = second[::-1, 1].copy()  # A is inverted on label 1

    ceiling = MemberCurve(LabelScores()).oracle(Pool(["a.pt", "b.pt"], [first, second], truth))

    assert ceiling["oracle"] >= ceiling["average"], "a ceiling below the thing it bounds is not a ceiling"
    assert ceiling["oracle"] > ceiling["average"], "members that differ per label must show headroom"
    assert ceiling["average"] >= 0.0


def test_routing():
    """One row per k, the last one naming itself the flat rank-average this reading is measured against."""
    generator = np.random.default_rng(29)
    truth = _truth()
    signal = np.repeat((truth[:, 0].astype(np.float64) * 3.0)[:, None], len(TARGETS), axis=1)
    members = [signal + generator.normal(0.0, 1.0, signal.shape) for _ in range(3)]

    lines = MemberCurve(LabelScores()).routing(Pool(_names(members), members, truth))

    assert len(lines) == len(members), "one row per k, and k runs to every member"
    assert lines[0].startswith("top- 1 per label:")
    assert "(= the flat rank-average)" in lines[-1], "the matched control is the last row, one estimator"


def test_labels():
    """The mix's weakest label leads, and its headroom line prices exactly 1/12 of the lift it names."""
    generator = np.random.default_rng(23)
    truth = _truth()
    strong = np.repeat((truth[:, 0].astype(np.float64) * 4.0)[:, None], len(TARGETS), axis=1)
    # ONE label deliberately blinded to chance, which is the hole this reading exists to surface.
    strong[:, 3] = generator.normal(0.0, 1.0, STUDIES)
    members = [strong, strong + generator.normal(0.0, 0.01, strong.shape)]

    lines = MemberCurve(LabelScores()).labels(Pool(_names(members), members, truth))

    assert lines[0].startswith("mix of 2:")
    assert lines[1].strip().startswith(TARGETS[3]), "the blinded label must be reported first"
    weakest = float(lines[1].split("  n=")[0].split()[-1])
    lifted = float(lines[1].split("(")[1].rstrip(")"))
    assert lifted == pytest.approx((0.90 - weakest) / len(TARGETS), abs=5e-5)


def test_paired():
    """Reports one block per member against the first, and stars only intervals that exclude zero."""
    generator = np.random.default_rng(11)
    truth = np.zeros((120, len(TARGETS)), dtype=np.float32)
    truth[generator.choice(120, 40, replace=False), :] = 1.0
    signal = truth + generator.normal(0, 1.0, truth.shape)
    control = (signal + generator.normal(0, 0.05, truth.shape)).astype(np.float64)
    # NOT a rescaling of `signal`: AUC is invariant to any increasing map, so `signal * 3` ranks
    # identically and would be no better at all. Extra TRUTH is what actually separates.
    better = (signal + truth * 2.0).astype(np.float64)

    lines = MemberCurve(LabelScores()).paired(Pool(["control.pt", "better.pt"], [control, better], truth))
    text = "\n".join(lines)

    assert "paired deltas, better.pt against control.pt:" in text
    assert "*" in text, "a member built to be strictly better must clear the interval somewhere"


def test_logged():
    """The whole reading as lines: the curve, the named subsets, the ceiling and the reachable curve.

    Asserted by what each section CONTRIBUTES rather than by a line count, so adding a reading extends
    the test instead of breaking it — and the paired block is checked by its absence when unasked,
    because a flag that reports the same thing either way is not a flag.
    """
    generator = np.random.default_rng(19)
    truth = np.zeros((80, len(TARGETS)), dtype=np.float32)
    truth[40:, :] = 1.0
    members = [generator.random((80, len(TARGETS))) for _ in range(2)]
    members[0][:, 0] = truth[:, 0] * 0.5 + generator.random(80) * 0.1

    curve = MemberCurve(LabelScores(), combine_seed=5)
    lines = curve.logged(Pool(["a.pt", "b.pt"], members, truth), paired=False)

    assert any(line.startswith("2 member(s):") for line in lines), "the size curve"
    assert any("a.pt + b.pt" in line for line in lines), "the subsets, named for packaging"
    assert any(line.startswith("oracle ") and "headroom" in line for line in lines), "the ceiling"
    assert any("(= the flat rank-average)" in line for line in lines), "the reachable per-label curve"
    assert not any("paired deltas" in line for line in lines)

    assert any("paired deltas" in line for line in curve.logged(Pool(["a.pt", "b.pt"], members, truth), paired=True))


def test_votes():
    """Every rule is scored on the SAME members, and the shipped one reports the number it ships.

    The comparison is only readable if the flat row IS `RankAverage.combine` — a row computed some other
    way would make each difference below it partly a difference in the baseline, which is the error this
    whole comparison exists to avoid.
    """
    members = [_member(1.0, seed) for seed in (1, 2, 3)]
    truth = _truth()
    lines = MemberCurve(LabelScores(), SEED).votes(Pool(_names(members), members, truth))

    assert [line.split()[0] for line in lines] == ["RankAverage", "LogitAverage", "QualityWeightedRank"]
    shipped = LabelScores().mean(RankAverage().combine(members), truth)
    assert lines[0].split()[1] == f"{shipped:.4f}", "the flat row is the rule a submission runs"


def test_quality_weighted():
    """The fitted rule reports a SPLIT-HALF number, and the report carries that same number.

    Split-half because the weights come off the holdout they would otherwise be scored on — the in-sample
    optimism that killed per-label selection. A number scored where it was fitted is not a result.
    """
    members = [_member(1.0, seed) for seed in (1, 2, 3)]
    truth = _truth()
    curve = MemberCurve(LabelScores(), SEED)

    pool = Pool(_names(members), members, truth)

    honest = curve.quality_weighted(pool)
    assert 0.0 <= honest <= 1.0
    assert curve.votes(pool)[2].split()[1] == f"{honest:.4f}", "the report prints the honest score"


def test_greedy():
    """Forward selection takes the strongest member first and names what each step added."""
    curve = MemberCurve(LabelScores())
    truth = _truth()
    members = [_member(0.2, SEED), _member(2.0, SEED + 1), _member(0.3, SEED + 2)]

    lines = curve.greedy(Pool(["weak.pt", "strong.pt", "middling.pt"], members, truth))

    assert len(lines) == 3, "every member is eventually taken, so the curve shows where it turned"
    assert "+strong.pt" in lines[0], "the best single member is chosen first"
    assert all(line.startswith(f"greedy {step}") for step, line in enumerate(lines, start=1))


def test_swap():
    """A candidate takes the seat of the member it can most afford to displace, and says whose it was."""
    curve = MemberCurve(LabelScores())
    truth = _truth()
    roster = Pool(["strong.pt", "weak.pt"], [_member(2.0, SEED), _member(0.05, SEED + 1)], truth)
    candidates = Pool(["good.pt", "noise.pt"], [_member(2.0, SEED + 2), _member(0.05, SEED + 3)], truth)

    lines = curve.swap(roster, candidates)

    assert lines[0].startswith("holding 2 members"), "the count is held, which is what makes it a seat"
    assert "good.pt for weak.pt" in lines[1], "the strong candidate displaces the weak incumbent, not the strong one"
    assert lines[1].split("(")[1].startswith("+"), "replacing the roster's weakest member with a strong one pays"
    assert "noise.pt" in lines[2], "the weaker candidate is priced too, and reported second"
    assert lines[-2].endswith("never been shown to pick a board tick"), "the resolution bound is stated"
    assert lines[-2].startswith("2 of 2 clear +0.0040"), "both beat an incumbent that is pure noise"
    assert "saturates" in lines[-1], "a best that clears the bound is real but capped, and the line says so"


def test_swap_unresolvable():
    """A candidate that cannot improve a healthy roster is counted OUT of the resolvable set, not filtered.

    The count is the half of the trailing line that can be wrong in the direction that costs a slot: an
    argmax reads like a submission whether or not the board could tell it apart from the incumbent.
    """
    curve = MemberCurve(LabelScores())
    truth = _truth()
    roster = Pool(["a.pt", "b.pt"], [_member(2.0, SEED), _member(2.0, SEED + 1)], truth)
    candidates = Pool(["noise.pt"], [_member(0.05, SEED + 2)], truth)

    lines = curve.swap(roster, candidates)

    assert "noise.pt" in lines[1], "it is still priced and still reported"
    assert lines[-2].startswith("0 of 1"), "priced, ordered, and named as no move the board could read"
    assert "3x" in lines[-1], "a negative best is priced at what the board charges, not left looking free"
    assert "not a free look" in lines[-1], "which is the reading knee-c0u9 refuted"


def test_swap_prices_nothing_when_no_candidate_carries_a_record():
    """An empty candidate pool is a STATE, not an error — `final_ws_e12` can never carry a record.

    It used to raise `max() iterable argument is empty` AFTER a twelve-epoch fit had already written its
    checkpoint, so a crash in a reporting step read in the log exactly like a training failure.
    """
    curve = MemberCurve(LabelScores())
    truth = _truth()
    roster = Pool(["a.pt", "b.pt"], [_member(2.0, SEED), _member(2.0, SEED + 1)], truth)

    lines = curve.swap(roster, Pool([], [], truth))

    assert lines == ["no candidate carries a holdout record, so no seat can be priced"]


def test_board_cost():
    """The three bands a best-seat delta can fall in, each said in the words that decide a slot.

    Read straight rather than through `swap`, because the branch is the rule and building three rosters
    that land in three bands would test the fixture's noise as much as the rule.
    """
    assert "saturates" in MemberCurve.board_cost(0.0040), "above the bound, real but capped"
    assert "cannot tell it from the incumbent" in MemberCurve.board_cost(0.0004), "a positive inside the bound"
    assert "-0.0030" in MemberCurve.board_cost(-0.0010), "a negative is quoted at what the board charges"


def test_split():
    """Both halves come back NAMED, because a swap line has to say which incumbent it displaces."""
    truth = _truth()
    pool = Pool(["in.pt", "out.pt"], [_member(2.0, SEED), _member(2.0, SEED + 1)], truth)

    held, candidates = pool.split([Path("in.pt")])

    assert (held.names, candidates.names) == (["in.pt"], ["out.pt"])
    with pytest.raises(ValueError, match="no holdout record"):
        pool.split([Path("absent.pt")])


def test_marginal():
    """A candidate is priced against a FIXED roster, best first, and a useless one reads negative."""
    curve = MemberCurve(LabelScores())
    truth = _truth()
    roster = Pool(["a.pt", "b.pt"], [_member(2.0, SEED), _member(2.0, SEED + 1)], truth)
    candidates = [_member(0.05, SEED + 2), _member(2.0, SEED + 3)]

    lines = curve.marginal(roster, Pool(["noise.pt", "good.pt"], candidates, truth))

    assert lines[0].startswith("roster of 2:"), "the base the deltas are against is stated, not implied"
    assert "+good.pt" in lines[1], "the candidate that adds most is reported first"
    assert lines[2].split("(")[1].startswith("-"), "a member that only adds noise reads as a negative seat"
    assert "[" in lines[1] and "]" in lines[1], "every priced seat carries its own paired band, not a borrowed floor"
    assert lines[-1].endswith(("readable", "a decision")), "and the table closes by naming the selection trap"


def test_selection_caveat():
    """The line refuses to let an argmax read as a decision — and says so plainly when nothing clears."""
    empty = MemberCurve.selection_caveat(0, 36)
    some = MemberCurve.selection_caveat(3, 36)

    assert "adds nothing readable" in empty[0], "no band clearing zero is the unambiguous case, said plainly"
    assert "36" in empty[0], "the denominator is on the line, so the reader is never guessing at it"
    assert "never as a decision" in some[0], "a star is evidence about the pair, not a licence to ship"


def test_diverse():
    """The second seat goes to the STRANGER, not to the twin the mean would rather have."""
    rng = np.random.default_rng(SEED + 40)
    curve = MemberCurve(LabelScores())
    truth = _truth()
    strong = _member(2.0, SEED)
    twin = strong + rng.normal(0.0, 0.02, strong.shape)
    stranger = _member(1.6, SEED + 7)

    lines = curve.diverse(Pool(["strong.pt", "twin.pt", "stranger.pt"], [strong, twin, stranger], truth))

    # The pair is asserted as a PAIR rather than by name: `twin` is `strong` under a 0.02 nudge, so which
    # of the two rounds higher on the holdout is noise, and pinning it would make this test a statement
    # about a seed. What the rule promises is that the OTHER one waits until last.
    assert lines[0].startswith("diverse 1:"), "the seed is the strongest member, whichever of the pair that is"
    assert "+stranger.pt" in lines[1], "the seat rule takes the least-correlated member, not the best one"
    assert {"strong.pt", "twin.pt"} == {lines[0].split("+")[1].split()[0], lines[2].split("+")[1].split()[0]}, (
        "the near-duplicate is taken last, which is the whole point of the rule"
    )
    assert "(nearest 0.9" in lines[2], "each step reports the agreement that decided it"


def test_forward():
    """The greedy's path is the data behind the printed lines — each seat's score and who took it."""
    curve = MemberCurve(LabelScores())
    members = [_member(0.2, SEED), _member(2.0, SEED + 1), _member(0.3, SEED + 2)]

    path = curve.forward(Pool(["weak.pt", "strong.pt", "middling.pt"], members, _truth()))

    assert len(path) == 3, "every member is eventually seated"
    assert path[0][1] == 1, "the best single member takes the first seat"
    assert sorted(index for _, index in path) == [0, 1, 2], "no member is seated twice"


def test_honest():
    """The seat count is chosen on studies it is NOT scored on, and the in-sample bias is stated."""
    curve = MemberCurve(LabelScores())
    members = [_member(strength, SEED + step) for step, strength in enumerate((2.0, 1.5, 0.05))]

    lines = curve.honest(Pool(["strong.pt", "good.pt", "noise.pt"], members, _truth()))

    assert lines[0].startswith("honest seats:"), "the reading says how it was split before it says a number"
    assert len(lines) == 1 + len(members) + 1, "one line per seat, plus the header and the verdict"
    assert all("bias" in line for line in lines[1 : 1 + len(members)])
    assert lines[-1].startswith("best honest count "), "the count is the answer, and it is named as held-out"


def test_dropped():
    """A member that ranks the truth BACKWARDS must read as the roster's most profitable removal.

    The reading exists to price the displaced half of a swap, so the sign convention is the assertion:
    positive means the roster scores higher without that member, and the payer sorts to the top.
    """
    members = [_member(2.0, 1), _member(2.0, 2), _member(-2.0, 3)]
    pool = Pool(_names(members), members, _truth())

    lines = MemberCurve(LabelScores()).dropped(pool)

    assert lines[0].startswith("roster of 3 at "), "the whole roster's score is the baseline the deltas are from"
    assert lines[0].endswith("read on the cut0.5 ruler:"), "a leave-one-out is a macro and macros name their cut"
    assert lines[1].startswith("  -m2.pt: "), "the backwards member is the one worth dropping, so it sorts first"
    assert "(+" in lines[1], "dropping it must RAISE the roster"
    assert all("(-" in line for line in lines[2:]), "dropping either honest member must cost"
