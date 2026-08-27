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

from dataclasses import dataclass

import torch

from core.findings import TARGETS
from kneemri.training.grades import Predicted

FOCUS = "focus_label"
"""The key a run's settings answer the focus question in, so a queued arm can be checked for it."""


@dataclass(frozen=True)
class LabelFocus:
    """The findings this loss may learn from — all twelve under the null policy, which is every member shipped.

    Frozen and defaulted so an objective built without it descends the identical scalar it always did:
    `None` returns the batch itself, by identity, not by multiplying with a mask of ones.
    """

    finding: str | None = None

    def __post_init__(self) -> None:
        """Reject an unknown finding LOUDLY, because the silent failure here costs a queued arm.

        A mistyped name that fell through to the null policy would train the plain twelve-label recipe
        under the focused arm's name and report a number that reads as a refutation of the mechanism
        rather than of the invocation — the same failure `run_setup` raises on for `--pair-memory`.
        """
        if self.finding is not None and self.finding not in TARGETS:
            message = f"--focus-label {self.finding!r} is not one of the twelve findings: {TARGETS}"
            raise ValueError(message)

    def focused(self, batch: Predicted) -> Predicted:
        """The batch with every unfocused finding's weight zeroed, and the batch itself under the null policy.

        Idempotent by construction — the mask holds zeros and ones, so applying it twice is applying it
        once. That is what lets `RankedObjective` focus at its own entry and still delegate to a base
        `__call__` that focuses again, without either caller having to know about the other.
        """
        if self.finding is None:
            return batch
        mask = torch.zeros_like(batch.weights)
        mask[:, TARGETS.index(self.finding)] = 1.0
        return Predicted(
            logits=batch.logits,
            scores=batch.scores,
            weights=batch.weights * mask,
            teacher=batch.teacher,
            embedding=batch.embedding,
        )

    def settings(self) -> dict[str, object]:
        """This policy's account, so a checkpoint records WHICH finding was fitted rather than that one was."""
        return {FOCUS: self.finding or ""}
