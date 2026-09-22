from kneemri.experiments.arm_budget import ArmBudget
from kneemri.experiments.arm_check import ArmCheck
from kneemri.training.run_setup import RunSetup

GOOD = "--fold 0 --epochs 12 --one-cycle --slice-pooling max --whole-stack --labels /tmp/labels.csv"


def budget_of(invocations: dict[str, str]) -> ArmBudget:
    """The arms, through the same parse-then-resolve `arm_check` hands this class its budgets from.

    Asserts the parse, rather than skipping a `None`: every invocation here is written to parse, so one
    that does not is a broken FIXTURE and has to fail loudly instead of quietly leaving an arm out of a
    dict the assertions then read as an absence.
    """
    check = ArmCheck()
    budgets = {}
    for name, flags in invocations.items():
        options = check._parsed(flags)[0]
        assert options is not None, f"{name} does not parse — the fixture is broken, not the check"
        budgets[name] = RunSetup(options).budget()
    return ArmBudget(budgets)


def test_overstated():
    """A cap copied off a bigger recipe is reported; one the slices justify, and one absent, are not.

    `depth_regions_narrow` is the live case: `depth_regions` with the slices dropped to 12 and the
    20-slice cap of 28 carried along, ten GB above what a 12-slice step reserves. Free on an empty card
    and a wait-then-crawl once a co-tenant holds a third of it.
    """
    overstated = budget_of(
        {
            "stale_cap": GOOD + " --slices 12 --vram-gb 28",
            "derived": GOOD + " --slices 20 --vram-gb 28",
            "rounded_up": GOOD + " --slices 12 --vram-gb 18",
            "no_cap": GOOD + " --slices 12",
        }
    ).overstated()

    assert round(overstated["stale_cap"]) == 10, "a 20-slice cap on a 12-slice recipe is what this catches"
    assert "derived" not in overstated, "asking UNDER the derived need is not an overstatement"
    assert "rounded_up" not in overstated, "a cap is a bound, so one slice of rounding is not a defect"
    assert "no_cap" not in overstated, "an arm that names no cap gets the derived one and says nothing"


def test_unshareable():
    """An arm is answerable for what its OWN recipe demands, never for the project-wide fallback.

    Asking `capped()` instead reported 477 of 490 arms, which is a fact about `DEFAULT_SHARE` being
    0.8 — above this share on its own, so every silent arm tripped and no author could act on it.
    """
    unshareable = budget_of(
        {
            "fat": GOOD + " --slices 24",
            "lean": GOOD + " --slices 4",
            "capped_low": GOOD + " --slices 24 --vram-gb 10",
            "capped_high": GOOD + " --slices 4 --vram-gb 25",
        }
    ).unshareable()

    assert unshareable["fat"] > 0, "a 24-slice step demands more than half this card"
    assert "lean" not in unshareable, "a small recipe shares, and says nothing"
    assert "capped_low" not in unshareable, "a named cap IS the demand — that is the flag's whole effect"
    assert round(unshareable["capped_high"]) == 9, "25 GB named against a 15.92 GB share"
