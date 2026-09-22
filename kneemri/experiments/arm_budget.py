"""What an arm asks of the card, read off its parsed recipe.

Apart from `arm_check` because the two questions it answers are about the DEVICE rather than about the
arm's text — every other check there reads flags, beads, caches or shell bytes, and these two are the
only ones that resolve a `DeviceBudget`. Splitting them is also what kept `arm_check.py` under the
god-file ceiling when `unshareable` was added.
"""

from __future__ import annotations

from kneemri.training.device_budget import (
    CAP_HEADROOM,
    CARD_GB,
    DESKTOP_RESERVE_GB,
    GB_PER_SLICE,
    SHARED_SHARE,
    DeviceBudget,
)


class ArmBudget:
    """The VRAM questions, over the arms whose flags parsed.

    Takes RESOLVED BUDGETS rather than configs or flag strings. `arm_check` has already parsed every
    arm once, so re-deriving five hundred invocations here is five hundred parses; and a budget is all
    either question reads, so taking a `TrainConfig` would put this module on `config_tree`'s fan-in as
    its ninth importer and make that a god-module.
    """

    def __init__(self, budgets: dict[str, DeviceBudget]) -> None:
        self.budgets = budgets

    def overstated(self) -> dict[str, float]:
        """The arms asking for more VRAM than their own slice count needs, by how many GB.

        A HAND-TYPED CAP IS A THIRD STATEMENT OF A FACT `--slices` ALREADY MAKES, and the third one goes
        stale: `depth_regions_narrow` is `depth_regions` with the slices dropped to 12 and the 20-slice
        cap of 28 copied along, 10 GB above what a 12-slice step reserves. That was free on an empty
        card — `DeviceBudget.await_headroom` waits for `free >= vram_gb` and an overstatement just
        passes — and stopped being free when `biohub_kaggle` took ~13 GB of the 32: the arm now polls
        for memory it never needed, burns the wait, and starts into the PCIe-paging crawl the cap
        exists to prevent. `depth_regions_span` was queued at 28 on a 12-slice recipe and caught by
        hand; this is that catch, mechanised.

        Measured against `DeviceBudget.needed`, which already carries one slice of slack. Only
        OVERSTATEMENT is reported: asking under the derived need is what `band_dense` does at 28
        against 28.26, and it peaked at 26.92.

        IT REPORTS 38 OF 50 ARMS, AND THAT IS THE TRUE ANSWER RATHER THAN A LOOSE BAR. I first read the
        count as noise and tried to legislate a threshold under which slack was "inside the 20 / 26 /
        28 convention" — but that convention is exactly the defect: every one of those numbers was
        picked as a fraction of the CARD, months before `DeviceBudget.needed` derived a cap from the
        RECIPE, so 26 on a 12-slice arm is 8.4 GB over need for the same reason 28 is 10.4 — not
        because one was copied and the other rounded. There is no honest line between them. So this
        returns all of them and `main` says it once, in a sentence naming the worst, instead of
        thirty-eight times: the flag is obsolete tree-wide, and the fix is to delete it tree-wide
        (knee-ylp0), not to tune a bar until the report is quiet.
        """
        asked: dict[str, float] = {}
        for name, budget in self.budgets.items():
            if not budget.vram_gb:
                continue
            needed = budget.needed()
            if budget.vram_gb > needed + GB_PER_SLICE:
                asked[name] = round(budget.vram_gb - needed, 2)
        return asked

    def unshareable(self) -> dict[str, float]:
        """The arms demanding more of the card than this project's share of it, by GB over.

        THE COMPLAINT WAS THAT 349 ARMS NAME NO `--vram-gb` AND THEREFORE CLAIM THE WHOLE CARD, and the
        premise is wrong: `capped` falls back to `max(needed() * CAP_HEADROOM, DEFAULT_SHARE * whole)`,
        so a silent arm is capped at its own derived need or at 0.8 of the card, whichever is larger.
        Nothing is uncapped. Measured 2026-09-22 over the 490 invocations in the tree: 69 name a number,
        421 are silent, and of those 421 the 0.8 floor — 25.5 GB — is what 219 of them actually hold.

        WHICH IS WHY THE OBVIOUS FIX IS THE WRONG ONE. Requiring every arm to name a cap cannot lower
        one: a hand-typed number below the derived need does not make the run smaller, it makes it
        raise, which is this project's documented preference over a PCIe crawl but is not sharing. The
        silent arms' own derived need has a median of 22.4 GB and a p90 of 32.5, and **200 of the 490
        derive ABOVE the card and clamp to `whole - DESKTOP_RESERVE_GB`** — for those, no value of any
        flag both fits the recipe and leaves a co-tenant alive. The lever that reaches the owner's share
        is `--slices`, `--image-size` and the batch, because `needed()` follows the recipe; the flag is
        a third statement of a fact the recipe already makes (`overstated`, knee-ylp0).

        IT MEASURES THE ARM'S OWN DEMAND, NOT `capped`. Asking `capped` reported 477 of 490, and that
        number is about `DEFAULT_SHARE` rather than about any arm: the 0.8 fallback is 25.5 GB, already
        above this share, so every silent arm tripped by construction and an author could do nothing
        about it. The fallback is one project-wide constant and the owner's call (knee-l2d7o); what an
        arm answers for is what its own recipe demands, which is the named cap or the derived need.

        So this reports the arms that cannot run under the sharing directive AS WRITTEN, which is the
        question `overstated` does not ask — that one compares an arm against ITSELF, and an arm can sit
        exactly at its own derived need and still hold four fifths of a shared card. ADVISORY while the
        count is in the hundreds: it graduates to refusing at `--only` once the tree is clean, the same
        way every other gate here did, because a pre-flight that refuses seven arms in ten is one that
        earns a `--no-verify`.
        """
        holding: dict[str, float] = {}
        share = SHARED_SHARE * CARD_GB
        for name, budget in self.budgets.items():
            demand = min(budget.vram_gb or budget.needed() * CAP_HEADROOM, CARD_GB - DESKTOP_RESERVE_GB)
            if demand > share:
                holding[name] = round(demand - share, 2)
        return holding
