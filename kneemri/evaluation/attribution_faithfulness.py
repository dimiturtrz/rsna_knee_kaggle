"""Is the slot-drop occlusion in `attribution` a faithful explanation, or just a number we trust?

`attribution.slot_importance` withholds one acquisition at a time and reports what the held-out AUC
loses. Every reading this project has made about WHAT the model uses rests on that sweep, and nothing
has ever checked it. "Checked" here does not mean re-running it: an occlusion agrees with itself by
construction. It means asking the two questions the attribution-evaluation literature asks of any
attribution, both of which have an answer that could come back bad.

INFIDELITY (Yeh et al., 2019) — does the attribution PREDICT what a perturbation actually costs? A
per-slot attribution is a linear claim: drop slots A and C and the probability should fall by roughly
attr(A) + attr(C). Infidelity is the mean squared gap between that prediction and the measured fall,
over random multi-slot drops. High infidelity means the slots interact — the sequences are redundant,
or one only matters when another is missing — and a per-slot table is then the wrong shape for the
answer, however carefully each cell was measured.

SENSITIVITY (Yeh et al., 2019) — does the attribution survive a perturbation that should not move it?
`sensitivity_max` re-explains under small jitter of the encoder features and reports the largest change.
An attribution that swings under noise smaller than the difference between two scanners is not a fact
about the model, it is a fact about the batch it was computed on.

BOTH ARE RUN ON TWO EXPLANATIONS, which is the point. The ablation attribution is the occlusion
diagnostic in per-study form. The attention mass is what the head itself says it read — free, one
softmax, no second pass. If attention scores as faithfully as ablation, the expensive sweep has a cheap
surrogate; if it does not, the head's attention is not the explanation it is routinely read as, and
`attention_readout`'s table has to be read as a description of the mechanism rather than of importance.
Both are scored with `normalize=True`, so the comparison is about the SHAPE of the attribution and not
about the units it happens to be in.

PIXEL-SPACE SALIENCY IS STILL DECLINED and this library does not reopen it. Integrated Gradients,
GradCAM and DeepLIFT would all import cleanly now; the argument against them was never availability.
Nothing in this supervision says WHERE in a slice a finding is, so a heatmap has no ground truth and
cannot be scored — which is exactly why captum is here for its METRICS and not for its attributions.

THE ENCODER RUNS ONCE. The presence mask enters at the head and never reaches the trunk, so every
perturbation below is a re-run of one projection and one softmax over cached slot features. That is what
makes hundreds of perturbations per study affordable at all, and it is exact rather than an
approximation — the features a masked pass would compute are the features that are cached.
"""

import logging
from dataclasses import dataclass

import torch
from captum.attr import FeatureAblation
from captum.metrics import infidelity, sensitivity_max
from jaxtyping import Float
from torch import Tensor

from core import cli
from core.findings import TARGETS
from kneemri.evaluation.slot_probe import SlotProbe

logger = logging.getLogger(__name__)

PRESENT = 0.5
"""Midpoint of the {0, 1} presence mask, matching `SlotHead`'s own threshold."""

ABSENT = 0.0
"""The baseline a slot is ablated to — a withheld acquisition, which is what the mask means."""

DROP_RATE = 0.3
"""Chance each present slot is withheld in one infidelity perturbation.

Not one slot and not all of them: infidelity asks whether a per-slot table composes, and a perturbation
that touches exactly one slot is the ablation restated, while one that touches all of them leaves the
head nothing to renormalise over. At 0.3 a six-slot study drops one or two slots most of the time,
which is the regime the additive claim is actually used in.
"""

SAMPLES = 20
"""Random multi-slot perturbations per study for infidelity — the estimator's own sample size."""

SENSITIVITY_SAMPLES = 5
"""Re-explanations per study for `sensitivity_max`, which costs one head pass each."""

SENSITIVITY_RADIUS = 0.02
"""Jitter radius as a FRACTION OF THE FEATURES' OWN SPREAD, never an absolute.

captum's default 0.02 is an absolute uniform radius, which is meaningful for a normalised image and
meaningless here: these are raw trunk outputs whose scale is whatever the encoder happens to emit, and
the same 0.02 is a rounding error for one backbone and a different study for another. Scaled by the
batch's own standard deviation it asks one question of every member: does the attribution survive noise
two percent of the size of the signal?
"""


@dataclass(frozen=True)
class Faithfulness:
    """What the two metrics said about one finding's two explanations."""

    finding: str
    ablation_infidelity: float
    attention_infidelity: float
    ablation_sensitivity: float
    attention_sensitivity: float

    def line(self) -> str:
        """This finding's row: lower is better in every column, and the two columns are comparable."""
        return (
            f"  {self.finding:<17} infidelity ablation={self.ablation_infidelity:.4f} "
            f"attention={self.attention_infidelity:.4f} | sensitivity "
            f"ablation={self.ablation_sensitivity:.4f} attention={self.attention_sensitivity:.4f}"
        )


class SlotFaithfulness:
    """Scores the slot attribution with captum's metrics — it produces no attribution of its own.

    `FeatureAblation` over the presence mask IS `attribution.slot_importance`, per study rather than per
    corpus and on the probability rather than on the AUC. It is used here instead of the hand-written
    sweep for one reason: the metrics below have to perturb the same object the attribution was computed
    from, and captum's own ablation is the version they are specified against.
    """

    def __init__(self, samples: int = SAMPLES, sensitivity_samples: int = SENSITIVITY_SAMPLES) -> None:
        self.samples = samples
        self.sensitivity_samples = sensitivity_samples

    def ablation(self, probe: SlotProbe, mask: Float[Tensor, "b s"], target: int) -> Float[Tensor, "b s"]:
        """What withholding each slot costs this finding's probability, per study."""
        return FeatureAblation(probe.probabilities).attribute(mask, baselines=ABSENT, target=target)

    @staticmethod
    def attention(probe: SlotProbe, mask: Float[Tensor, "b s"], target: int) -> Float[Tensor, "b s"]:
        """The head's own attention over the slots, as an explanation to be scored beside the ablation."""
        return probe.weights(mask)[:, target, :]

    def infidelity(
        self,
        probe: SlotProbe,
        mask: Float[Tensor, "b s"],
        attribution: Float[Tensor, "b s"],
        target: int,
    ) -> Float[Tensor, " b"]:
        """Squared gap between what the attribution predicts a drop costs and what it does cost.

        `normalize=True` fits the one scale factor that best relates the attribution to the measured
        drops before taking the gap, which is what lets an attention weight (sums to one) and an ablation
        delta (in probability) be compared at all. Without it this would rank the units, not the shapes.
        """
        return infidelity(
            probe.probabilities,
            self.perturb,
            mask,
            attribution,
            target=target,
            n_perturb_samples=self.samples,
            normalize=True,
        )

    def sensitivity(
        self,
        probe: SlotProbe,
        mask: Float[Tensor, "b s"],
        target: int,
        *,
        attend: bool,
    ) -> Float[Tensor, " b"]:
        """Largest change in the explanation under jitter of the features it was computed from.

        The jitter is applied to the trunk features rather than to the mask, because the mask is binary:
        captum's uniform radius never crosses `PRESENT`, so every "perturbed" explanation would be the
        original one and the metric would report a confident zero for any attribution whatsoever.
        """
        radius = SENSITIVITY_RADIUS * float(probe.features.std())

        # captum hands an explanation_func ONE positional argument holding a TUPLE of its inputs, even
        # when there is a single input — both of `sensitivity_max`'s call sites do it — so this declares
        # the tuple rather than the tensor it wants.
        # It must RETURN a tuple as well: `sensitivity_max` tuplizes the perturbed explanation but not the
        # unperturbed one, so a bare tensor there is iterated ROW-WISE and subtracted against the whole
        # perturbed batch. That fails as a broadcast error about tensor sizes and never mentions shape.
        def explain(inputs: tuple[Float[Tensor, "n s *o d"], ...]) -> tuple[Float[Tensor, "n s"], ...]:
            features = inputs[0]
            jittered = SlotProbe(probe.head, features)
            rows = SlotProbe.tiled(mask, len(features))
            explained = self.attention(jittered, rows, target) if attend else self.ablation(jittered, rows, target)
            return (explained,)

        return sensitivity_max(
            explain,
            (probe.features,),
            perturb_radius=radius,
            n_perturb_samples=self.sensitivity_samples,
        )

    def measure(self, probe: SlotProbe, mask: Float[Tensor, "b s"], target: int) -> Faithfulness:
        """Both metrics on both explanations for one finding, meaned over the studies."""
        ablation, attention = self.ablation(probe, mask, target), self.attention(probe, mask, target)
        return Faithfulness(
            finding=TARGETS[target],
            ablation_infidelity=float(self.infidelity(probe, mask, ablation, target).mean()),
            attention_infidelity=float(self.infidelity(probe, mask, attention, target).mean()),
            ablation_sensitivity=float(self.sensitivity(probe, mask, target, attend=False).mean()),
            attention_sensitivity=float(self.sensitivity(probe, mask, target, attend=True).mean()),
        )

    def perturb(
        self,
        mask: Float[Tensor, "n s"],
    ) -> tuple[Float[Tensor, "n s"], Float[Tensor, "n s"]]:
        """Withhold a random subset of the PRESENT slots, never all of them and never an absent one.

        Withholding an absent slot is not a perturbation — the head already renormalised it away — so a
        naive Bernoulli over all six would spend most of its samples measuring nothing and report the
        attribution as faithful for it. Emptying a study is refused for the reason `MeanSlotHead` floors
        its denominator: a study with no present slot is not a case the corpus contains, and the number
        it would produce is about the floor rather than about the model.

        Returns captum's pair: the perturbation `input - perturbed`, then the perturbed input.
        """
        present = mask >= PRESENT
        dropped = (torch.rand_like(mask) < DROP_RATE) & present
        kept = present & ~dropped
        rescued = ~kept.any(dim=1, keepdim=True) & (present.cumsum(dim=1) == 1) & present
        perturbed = (present & ~dropped | rescued).float()
        return mask - perturbed, perturbed


@dataclass(frozen=True)
class FaithfulnessReport:
    """The rows as prose, with the comparison they were run to make stated rather than left implicit."""

    rows: list[Faithfulness]

    def lines(self) -> list[str]:
        """The per-finding table and the one verdict the two columns together support."""
        lines = ["faithfulness of the slot attribution (lower is better; both columns normalised):"]
        lines.extend(row.line() for row in self.rows)
        lines.append("")
        lines.extend(self._verdict())
        return lines

    def _verdict(self) -> list[str]:
        """Whether the head's free attention is as faithful as the expensive occlusion sweep."""
        cheaper = [row for row in self.rows if row.attention_infidelity <= row.ablation_infidelity]
        return [
            f"attention is at least as faithful as ablation on {len(cheaper)} of {len(self.rows)} findings",
            "an ablation infidelity far above zero means the slots INTERACT and a per-slot table is the "
            "wrong shape for the answer, however carefully each cell was measured",
            "a sensitivity far above zero means the attribution moves under noise smaller than the "
            "difference between two scanners, and is a fact about the batch rather than about the model",
        ]


def main() -> None:
    """Score the slot occlusion of one member on its fold's holdout with captum's two metrics."""
    import json  # noqa: PLC0415 - a main() imports what only it needs, as every entrypoint here does
    import logging as log  # noqa: PLC0415 - as above
    from pathlib import Path  # noqa: PLC0415 - as above

    import numpy as np  # noqa: PLC0415 - as above

    from core.data.corpus.folds import ReportFolds  # noqa: PLC0415 - as above
    from kneemri.device import VisibleCards  # noqa: PLC0415 - as above
    from kneemri.inference.holdout_predictions import HoldoutPredictions  # noqa: PLC0415 - as above
    from kneemri.inference.members import Member  # noqa: PLC0415 - as above
    from kneemri.models.backbone import ENCODERS  # noqa: PLC0415 - as above

    command = cli.Command(__doc__, log.getLogger(__name__))
    command.parser.add_argument("--checkpoint", type=Path, required=True)
    command.parser.add_argument("--fold", type=int, default=0)
    command.parser.add_argument("--folds", type=int, default=5)
    command.parser.add_argument("--split-seed", type=int, default=2026)
    command.parser.add_argument("--studies", type=int, default=64, help="holdout studies read, for wall-clock")
    command.parser.add_argument("--samples", type=int, default=SAMPLES)
    command.parser.add_argument("--device", default="auto", help="auto, cpu, or a cuda index")
    arguments = command.parsed()

    root = command.rooted(arguments)
    member = Member.from_checkpoint(arguments.checkpoint)
    corpus, identifiers = member.cache(root.processed()).load()
    folds = ReportFolds(arguments.folds, arguments.split_seed)
    _, holdout = folds.holdout(folds.assign(str(root.raw() / "train.csv"), identifiers), arguments.fold)
    holdout = holdout[: arguments.studies]

    device = VisibleCards().device() if arguments.device == "auto" else torch.device(arguments.device)
    encoder = ENCODERS[member.variant]
    model = HoldoutPredictions(device).load(arguments.checkpoint, encoder.load(), encoder.dim, member.slot_count)
    window = member.window(corpus.slices, overlap=True)
    # ONE ENCODER PASS FOR THE WHOLE MEASUREMENT, over the member's own centred window: what follows
    # perturbs the mask, which the trunk never sees, so re-running it per perturbation would buy nothing.
    start = window.starts[len(window.starts) // 2]
    pixels = torch.from_numpy(np.ascontiguousarray(corpus.images[holdout])).to(device)
    flags = torch.from_numpy(corpus.mask[holdout]).to(device)
    with torch.no_grad():
        features = model.slot_features(pixels[:, :, start : start + window.width], window.size).float()

    faithfulness = SlotFaithfulness(arguments.samples)
    probe = SlotProbe(model.head, features)
    rows = [faithfulness.measure(probe, flags, target) for target in range(len(TARGETS))]
    destination = root.processed() / f"attribution_faithfulness_{arguments.checkpoint.stem}.json"
    destination.write_text(json.dumps([vars(row) for row in rows], indent=2, sort_keys=True), encoding="utf-8")

    command.report(FaithfulnessReport(rows).lines())
    command.report([f"written: {destination}"])


if __name__ == "__main__":
    main()
