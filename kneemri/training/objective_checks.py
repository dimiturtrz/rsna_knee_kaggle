"""Every way the loss flags fail to compose, raised before the card is touched.

A refusal here is not the same question as which objective the flags ask for, and holding both in one
place put that dispatch over the complexity ceiling. Each guard names a flag the reader can drop or
must add, because the failure they all guard is a run that trains one mechanism under another's name
and reports a number about the wrong thing.
"""

from dataclasses import dataclass

from kneemri.training.config_tree import TrainConfig

LONE_BATCH = 1
"""The micro-batch at which every comparison term in `objective.py` is STRUCTURALLY zero.

Not a threshold and not a floor picked for comfort: at two studies a label still forms one pair, and one
thin pair is a recipe judgement. At ONE study there is no second study to compare against, so the term is
absent rather than weak — `PairMass.between(batch, batch)` forms only the self-pair, which carries
`s_i - s_j = 0` and drops out, and `Margin` finds no label holding both a positive and
a negative and returns its zero. `--accumulate` does not repair it: accumulation sums the gradients of
micro-batch losses each computed alone, so it never forms a cross-micro-batch pair."""


@dataclass
class ObjectiveChecks:
    """The loss flags of one run, and whether they can be honoured together."""

    options: TrainConfig

    def composed(self) -> None:
        """Every way the loss flags fail to compose, raised before the card is touched.

        SEPARATE from `objective` because the two answer different questions — which objective, and
        whether the flags asking for it can be honoured together — and because holding both put the
        dispatch over the complexity ceiling, which is the gate saying the same thing.

        The ORDER is the content. `--auc-margin` goes first because it replaces the term every other
        objective is built on, so its conflict is named against IT. Dissent precedes the pair-memory rule
        so a dissenting arm is not told about a `--rank-weight` it never meant to give; feature precedes it
        so `--feature-weight --pair-memory` names the dropped flag rather than an unfed pool. The batch
        guard is LAST: every message above names a flag the reader can drop, and it names one to ADD.
        """
        options = self.options
        guards = (
            (options.targets.auc_margin, self._check_margin_alone),
            (options.teacher.dissent_weight, self._check_dissent_alone),
            (options.targets.contrast_weight, self._check_contrast_alone),
            (options.teacher.feature_weight, self._check_feature_alone),
            (options.targets.pair_memory, self._check_memory_fed),
            (options.teacher.distill_weight and options.targets.pair_memory, self._check_distill_unpooled),
            (options.optim.batch <= LONE_BATCH and not options.targets.pair_memory, self._check_pairs_can_form),
        )
        for armed, check in guards:
            if armed:
                check()

    def _check_memory_fed(self) -> None:
        """`--pair-memory` with no pooled term to widen, LOUD for `objective`'s reason.

        A memory with no term to feed is a silent no-op that trains the plain recipe under the arm's
        name, and would be read as a refutation of the mechanism rather than of the invocation.
        """
        targets = self.options.targets
        if targets.rank_weight or targets.contrast_weight or targets.auc_margin:
            return
        message = (
            f"--pair-memory {targets.pair_memory} widens the pool of a ranking, contrastive, or "
            "margin term and none of --rank-weight, --contrast-weight, or --auc-margin was given, "
            "so there is nothing for it to widen"
        )
        raise ValueError(message)

    def _check_distill_unpooled(self) -> None:
        """`--distill-weight` beside `--pair-memory`, which would drop the teacher on every remembered pair.

        `Remembered` widens a pooled term and knows nothing about a teacher, so the two together would
        train a distilling arm whose remembered pairs silently carry no teacher at all — half the
        mechanism, under its name.
        """
        options = self.options
        message = (
            f"--distill-weight {options.teacher.distill_weight} and --pair-memory {options.targets.pair_memory} "
            "are not composed: the remembered pool widens the label term only, so the teacher would be "
            "dropped for every remembered step"
        )
        raise ValueError(message)

    def _check_pairs_can_form(self) -> None:
        """The comparison terms that are silently OFF at a batch of one, raised before the card is touched.

        MEASURED 2026-09-03, and this guard exists because nothing caught it. `ortho_stack` ran nine
        hundred steps of `--rank-weight 1.0 --batch 1 --accumulate 12` with the ranking term contributing
        exactly zero: its epoch-0 loss read 0.6281 where the same recipe at batch 12 read 1.2266, and the
        recipe line recorded `rank_weight: 1.0` throughout, because `Objective.settings` reports the
        CONSTRUCTOR's weight and the constructor was reached. The arm was a two-variable change — the
        trunk AND the objective — against a control that trained the term, and its pre-registered middle
        branch would have CLOSED the project's only open lever on that comparison.

        This is the failure `objective.py` already warns about ("a --rank-weight that never reached the
        constructor trains plain cross-entropy under a ranking arm's name and reports a number that reads
        as a refutation of the mechanism"), arriving by batch size rather than by dispatch, which is why
        the warning did not cover it.

        `--auc-margin` is the worst of the four and is named the same way: it REPLACES cross-entropy, so a
        lone batch leaves the side term as the entire loss.
        """
        options = self.options
        silenced = {
            "--rank-weight": options.targets.rank_weight,
            "--auc-margin": options.targets.auc_margin,
            "--contrast-weight": options.targets.contrast_weight,
            "--dissent-weight": options.teacher.dissent_weight,
        }
        named = [f"{flag} {value}" for flag, value in silenced.items() if value]
        if not named:
            return
        message = (
            f"--batch {options.optim.batch} forms no pairs, so {', '.join(named)} would contribute exactly zero "
            "and the run would descend the plain recipe under the arm's name — pass --pair-memory to form "
            "pairs across steps, or raise --batch (--accumulate cannot repair this: it sums the gradients "
            "of micro-batch losses each computed alone)"
        )
        raise ValueError(message)

    def _check_margin_alone(self) -> None:
        """The four flags `--auc-margin` cannot be composed with, each LOUD for `objective`'s reason.

        Every other objective in this file's chain is cross-entropy PLUS a term, and this one removes
        cross-entropy. So each of those flags would ask for a term to be added to a loss that is no
        longer there, and the sum would descend fine while meaning nothing: the run would report both
        switches landed and no number afterwards would say which objective it actually was.
        """
        options = self.options
        conflicts = {
            "--rank-weight": options.targets.rank_weight,
            "--distill-weight": options.teacher.distill_weight,
            "--feature-weight": options.teacher.feature_weight,
            "--dissent-weight": options.teacher.dissent_weight,
            "--contrast-weight": options.targets.contrast_weight,
        }
        named = [f"{flag} {value}" for flag, value in conflicts.items() if value]
        if named:
            message = (
                f"--auc-margin {options.targets.auc_margin} is not composed with {', '.join(named)}: it REPLACES "
                "the cross-entropy term those all add themselves to, so what ran would be neither the "
                "margin objective nor the ranking one"
            )
            raise ValueError(message)

    def _check_dissent_alone(self) -> None:
        """The three flags `--dissent-weight` cannot be composed with, each LOUD for `objective`'s reason.

        `--distill-weight` reads the same teacher to pull the member TOWARD the pool's ordering, which is
        the opposite of what this term spends the member on; running both descends a sum whose result
        says nothing about either. `--rank-weight` is the same label ordering at a flat pair mass, and
        `Dissent` replaces it — passing it would be silently ignored,
        which is the exact failure the whole flag vocabulary here exists to prevent. `--pair-memory`
        widens the label term's pool with remembered readouts that carry no teacher, so every remembered
        pair would enter at an emphasis this term cannot compute.
        """
        options = self.options
        conflicts = {
            "--distill-weight": options.teacher.distill_weight,
            "--rank-weight": options.targets.rank_weight,
            "--pair-memory": options.targets.pair_memory,
        }
        named = [f"{flag} {value}" for flag, value in conflicts.items() if value]
        if named:
            message = (
                f"--dissent-weight {options.teacher.dissent_weight} is not composed with {', '.join(named)}: it "
                "IS the label ranking term at a re-weighted pair mass, and each of those either "
                "descends that same ordering a second time or feeds it pairs with no teacher"
            )
            raise ValueError(message)

    def _check_contrast_alone(self) -> None:
        """The two flags `--contrast-weight` cannot be composed with, each LOUD for `objective`'s reason.

        Not because the terms disagree — a representation term and a pairwise ordering term compose fine,
        and `--rank-weight` is deliberately absent from this list because it is carried beside `Contrast`
        — but because each of these two is an EXCLUSIVE term, one the builder refuses to pair. A run
        passing both would otherwise have trained one term's recipe under the other arm's name, which is
        the failure every guard in this file exists to prevent.

        `--pair-memory` WAS on this list and is not any more, and it was never a rejected idea: the pool
        is the named fix for this term's small-batch weakness — `contrast_dense` pre-registered exactly
        that suspect against its own -0.0087 — and it was refused only because the chain returned before
        reaching the remembered ranking term. `Remembered(Contrast)` now wraps the pair, so the composition the
        ledger asked for is the one that runs.
        """
        options = self.options
        conflicts = {
            "--distill-weight": options.teacher.distill_weight,
            "--dissent-weight": options.teacher.dissent_weight,
        }
        named = [f"{flag} {value}" for flag, value in conflicts.items() if value]
        if named:
            message = (
                f"--contrast-weight {options.targets.contrast_weight} is not composed with {', '.join(named)}: "
                "the objective chain returns ONE class, so this branch would win and the term those "
                "flags asked for would never be built"
            )
            raise ValueError(message)

    def _check_feature_alone(self) -> None:
        """The four flags `--feature-weight` cannot be composed with, each LOUD for `objective`'s reason.

        Not because a representation target and a pairwise one disagree — `--rank-weight` is deliberately
        absent from this list, since it is carried beside `FeatureDistillation` — but because the other
        three are EXCLUSIVE terms the builder refuses to pair, and `--pair-memory` has no pooled form of
        this term to widen. Each would otherwise train one mechanism under the other's name, which is
        what every guard in this file exists to prevent.
        """
        options = self.options
        conflicts = {
            "--distill-weight": options.teacher.distill_weight,
            "--dissent-weight": options.teacher.dissent_weight,
            "--contrast-weight": options.targets.contrast_weight,
            "--pair-memory": options.targets.pair_memory,
        }
        named = [f"{flag} {value}" for flag, value in conflicts.items() if value]
        if named:
            message = (
                f"--feature-weight {options.teacher.feature_weight} is not composed with {', '.join(named)}: the "
                "objective chain returns ONE class, so only one of these terms would be built and the "
                "run would report the other's recipe under this arm's name"
            )
            raise ValueError(message)
