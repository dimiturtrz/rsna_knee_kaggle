"""Which of the twelve findings a run's loss is allowed to learn from: `LabelFocus`.

Every one of the 107 recorded runs fits all twelve findings through one encoder, and the published
work this project is measured against does not. MRNet, ELNet and OrthoDiffusion are SPECIALISTS —
one model, one or three findings — and they reach 0.965-0.970 on ACL and 0.85-0.90 on meniscus. The
only published model fitting a twelve-finding panel the way we do averages 0.812, which is below our
own mean. So the per-finding gap this project prices against the literature may be the price of
JOINTNESS rather than a defect in the encoder, and nothing here has ever measured that.

This is the instrument that separates them. Restricting the loss to one finding puts the whole
encoder behind it, which is the strongest tilt the mechanism allows: if maximum tilt does not recover
the gap, no partial reweighting will either.

WHY IT IS A MASK ON THE WEIGHTS AND NOT A NARROWER HEAD. The head still emits twelve logits and the
checkpoint still loads under `core.findings.TARGETS`, so a focused member remains a member — it can
be scored by every tool here and mounted by the kernel without a second code path. The narrowing is
a policy the objective HOLDS, exactly as the grade ladder and the side ranking are.

WHY MASKING THE WEIGHTS IS SUFFICIENT, which is the part worth checking rather than assuming. The
cross-entropy term is `elementwise * weights`, normalised by `weights.sum()`. The ranking term forms
`confidence = weights_i * weights_j` and normalises by `pair_weight.sum()`. A zeroed column therefore
drops out of BOTH terms and out of both normalisers, so the surviving finding descends at its natural
scale rather than at a twelfth of it. One mask, and every term follows.

THE CONFOUND, STATED BEFORE THE ARM RUNS. A focused run sees one supervisory signal per study where
the joint run sees twelve, so it is data-starved relative to its control. That makes a WIN strong
evidence — it won despite the starvation — and a LOSS ambiguous between "jointness costs nothing" and
"one finding is too little supervision to fit an encoder on". The follow-up for an ambiguous loss is
an upweight rather than a mask, which starves nothing.
"""

from dataclasses import dataclass, replace

import torch

from core.findings import TARGETS
from kneemri.training.grades import Predicted

FOCUS = "focus_label"
"""The key a run's settings answer the focus question in, so a queued arm can be checked for it."""

UNFOCUSED = "unfocused_weight"
"""The key the PARTIAL policy answers in. Zero is the mask every focused arm before 2026-09-06 ran."""

KEPT = 1.0
"""The weight a named finding keeps. The policy moves the OTHERS, so the target's scale never changes."""


@dataclass(frozen=True)
class LabelFocus:
    """The findings this loss leans on, and what weight the rest keep — all twelve at full weight by default.

    Frozen and defaulted so an objective built without it descends the identical scalar it always did:
    the empty tuple returns the batch itself, by identity, not by multiplying with a mask of ones.

    TWO POLICIES IN ONE OBJECT, and the second exists because the first failed in a way this module
    predicted. `focus_mcl` ran the mask (`unfocused` zero, one finding) and did not merely underperform:
    it regressed its own target label by -0.0576 paired and collapsed to a 0.5332 twelve-column mean.
    The docstring above already named the follow-up for exactly that outcome — "an upweight rather than
    a mask, which starves nothing" — and `unfocused` between zero and one IS that upweight. The named
    findings keep their full scale and the other columns are turned down, so the encoder still sees
    twelve supervisory signals per study and merely spends more of its gradient on two of them.
    """

    findings: tuple[str, ...] = ()
    unfocused: float = 0.0

    SEPARATOR = "+"
    """What joins several findings in ONE cli token, so the flag's value and the recorded setting match.

    A finding's name holds a space ("Lateral OA"), so `nargs="+"` would make the flag's value a list
    that no longer reads back as the string `settings()` records — and `arm_check` compares those two
    as strings. One quoted token keeps the request and its account byte-identical.
    """

    @classmethod
    def parse(cls, text: str | None, unfocused: float = 0.0) -> "LabelFocus":
        """The policy a `--focus-label` value asks for, with the null policy for an absent one."""
        return cls(tuple(text.split(cls.SEPARATOR)) if text else (), unfocused)

    def __post_init__(self) -> None:
        """Reject an unknown finding LOUDLY, because the silent failure here costs a queued arm.

        A mistyped name that fell through to the null policy would train the plain twelve-label recipe
        under the focused arm's name and report a number that reads as a refutation of the mechanism
        rather than of the invocation — the same failure `run_setup` raises on for `--pair-memory`.
        """
        if isinstance(self.findings, str):
            message = f"findings is a tuple of findings, not the string {self.findings!r}"
            raise TypeError(message)
        unknown = [finding for finding in self.findings if finding not in TARGETS]
        if unknown:
            message = f"--focus-label {unknown} is not among the twelve findings: {TARGETS}"
            raise ValueError(message)

    def focused(self, batch: Predicted) -> Predicted:
        """The batch with every unnamed finding's weight turned down, and the batch itself under the null policy.

        Idempotent through `Predicted.shaped` and not through the arithmetic. A zero/one mask squares to
        itself, which is what let both objectives focus at their own entry for free; an unfocused weight
        strictly inside that range does not, so the marker carries the contract the mask used to carry
        by accident.
        """
        if not self.findings or batch.shaped:
            return batch
        mask = torch.full_like(batch.weights, self.unfocused)
        for finding in self.findings:
            mask[:, TARGETS.index(finding)] = KEPT
        return replace(batch, weights=batch.weights * mask, shaped=True)

    def settings(self) -> dict[str, object]:
        """This policy's account, so a checkpoint records WHICH findings were leaned on and HOW HARD."""
        return {
            FOCUS: self.SEPARATOR.join(self.findings),
            UNFOCUSED: self.unfocused if self.findings else 0.0,
        }
