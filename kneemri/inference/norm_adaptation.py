"""Tuning a member's LayerNorm affines to the study in front of it, in the seconds before it reads it.

WHY THIS IS A STRATEGY OBJECT AND NOT A FLAG. Every serving loop in this repo is `no_grad`-wrapped
(`study_reading.py:58`, `holdout_predictions.py:114`, `ensemble.py:460`), and adaptation needs a
gradient. A flag would have to reach inside that wrapper and turn it off, which is how a read stops
being a read. Instead the strategy owns an ADAPT PASS that runs BEFORE the wrapper, hands back the same
member, and undoes itself afterwards — so the default costs nothing and changes no number, and the
no-grad forward that produces every published figure is untouched.

THE PUBLISHED NUMBER, AND ITS DENOMINATOR. CARVE (arXiv:2607.15556, July 2026) is the only published
multi-label medical classification TTA result: AUROC +0.034 to +0.036 absolute, episodic per scan, Adam
1e-5, two steps, LN affines only. That is lever-sized against a board gap of 0.010 and it sits on our
head type. The same paper states the ceiling: depth/protocol shift alone costs -0.125 AUROC BEFORE
adaptation, and their conclusion is that adaptation is bounded by the transferred representation. This
tunes within what the trunk already serves; it does not repair a pipeline mismatch.

THE OBJECTIVE IS THE RESULT, NOT A DETAIL. Plain mean per-label Bernoulli entropy is UNPUBLISHED as a
loss and is the form CARVE measures FAILING (accuracy 0.633 -> 0.605), and cross-label softmax entropy
on a multi-label head provably suppresses co-occurring positives (ML-TTA Prop. 1). Both are easier to
write than the one that works, which is why they are named here rather than left out.

WHAT OUR SERVING LOOP GIVES US FOR FREE. `StudyReadings.predict` reads ONE STUDY A FORWARD, as a batch
of its own windows. So the episodic-per-scan regime is what this repo already does, and the
same-batch-same-domain assumption the whole TTA family rests on holds by construction rather than by
care. It also means batch-1 entropy estimation — the setting InTEnt calls unreliable — is not the
setting we are in.
"""

import logging
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

logger = logging.getLogger(__name__)

CARDINALITY_BOUNDS = (0, 8)
"""How many findings one study may be adapted toward — our own 5th and 95th percentiles, not CARVE's.

Measured 2026-09-09 over all 4406 studies of `report_labels_v2.csv` at the 0.5 midpoint: cardinality
mean 3.661, sd 2.476, percentiles p5 0, p25 2, p50 3, p75 5, p95 8.

THE LOWER BOUND IS 0 AND THAT IS A DEPARTURE. CARVE's k_min is at least one because a CT findings task
assumes something is there to find; over a tenth of our studies (p10 = 0) are clean knees. A k_min of 1
would force the objective to sharpen the single most probable label on every healthy patient — it would
manufacture a positive precisely where the member was right to be quiet.
"""

NEGATIVE_WEIGHT = 0.8
"""How hard the labels outside the top-k are pushed down, against 1.0 for the ones inside it (CARVE)."""

ADAPT_STEPS = 2
"""Gradient steps per study (CARVE 5.1, episodic single-scan). TENT's repo default is one, online."""

ADAPT_LR = 1e-5
"""Adam, per CARVE 5.1's episodic setting.

NOT SAR's 1e-3, which is SGD-momentum at batch 64 in an ONLINE stream. The two published settings are
coherent internally and must not be crossed: the only batch-small episodic numbers in print are four
orders below the online ones.
"""

FROZEN_LAST_BLOCKS = 3
"""Trunk blocks whose LayerNorms are left alone, counting from the output (SAR C.2, blocks 9-11 of 12).

MoETTA App. H freezes the FIRST block instead. Neither paper adapts a readout-head norm or the
patch-embed norm, and neither says it excluded them — so leaving the head frozen follows the practice
and not a statement, which is why it is written down here rather than assumed.
"""

PROBABILITY_FLOOR = 1e-6
"""Keeps the log finite when a member is fully certain, which fp16 sigmoid reaches routinely."""

Forward = Callable[[], Float[Tensor, "studies labels"]]
"""One graded read of the studies being adapted to, supplied by the caller as a closure.

A CLOSURE RATHER THAN A TENSOR, because the two serving loops do not agree on a call signature.
`StudyReadings` calls `member(batch)`; `HoldoutPredictions` calls `model(pixels, flags, size)` inside a
walk over views and window starts. Taking pixels here would force one of them to be rewritten in the
other's shape, and the thing they DO agree on is that a graded forward can be produced on demand.
"""


UNBOUNDED_EPISODE = 0
"""What an adaptation declares when it constrains nothing about how many studies share a read."""

EPISODE_STUDIES = 1
"""Studies that may share ONE adapt pass under CARVE's episodic setting: exactly one.

DECLARED BY THE STRATEGY BECAUSE ONLY THE STRATEGY KNOWS IT. `HoldoutPredictions` reads studies eight
at a time, and eight studies sharing one gradient step is the SAME leak as a shared top-k one level up:
study N's findings move the affines that read study N+1. The batch top-k was a bug I fixed; a batched
gradient would have re-introduced it under a different name. The reading loop therefore takes its
episode size from here rather than from its own batch constant.
"""


@runtime_checkable
class Adaptation(Protocol):
    """Whatever may tune a member to the studies in front of it, and must put it back afterwards.

    A CONTEXT MANAGER RATHER THAN A CALL, because the undo is the load-bearing half. An adapt pass that
    returns leaves the member changed for every study after it, which turns an episodic recipe into an
    online one silently — and online is the cell CARVE measures as harmful on a multi-label head.
    Restoring in a `finally` makes the reset unskippable rather than remembered.

    IT LIVES BESIDE ITS IMPLEMENTATIONS RATHER THAN INSIDE A READING LOOP because there are now two
    loops. It was `StudyReadings`'s parameter vocabulary while that was the only caller; a contract two
    modules speak is shared vocabulary, and leaving it in one of them would have made the other import
    a reading loop it never calls.
    """

    @property
    def episode(self) -> int:
        """Studies that may share one adapt pass, or 0 for no constraint.

        READ-ONLY, because the implementations are frozen dataclasses and a read-write attribute here
        would exclude every one of them: an episode size is chosen when the strategy is constructed and
        a reading loop that could move it mid-read is the online regime this whole module refuses.
        """
        ...

    def on(self, member: object, logits: Forward) -> AbstractContextManager[None]: ...


class NotAdaptableError(TypeError):
    """A member that cannot be adapted, raised rather than silently served unadapted.

    A member arrives through a Protocol that promises only `to` and a call, so the parameters this
    needs are not in the contract it was handed. Serving it unadapted would be the worse failure: the
    reading would carry the name of an adapted run and the values of an unadapted one, which is the
    exact shape of the `farseat_v1` defect — a complete run, a real score, and a member that never was
    what the row says it was.
    """


@dataclass(frozen=True)
class NoAdaptation:
    """The served function is the trained one. The default, and the control every reading is against."""

    episode: int = UNBOUNDED_EPISODE

    @contextmanager
    def on(self, member: object, logits: Forward) -> Iterator[None]:
        """Serve the member exactly as it was trained."""
        del member, logits
        yield


@dataclass(frozen=True)
class CardinalityAwareLN:
    """CARVE's objective over a member's LayerNorm affines, episodic: adapt to one study, then forget it.

    THE OBJECTIVE IS NOW THE PUBLISHED ONE — Eq. 8 and Eq. 9 arrived with D270 (2026-09-09) and the
    reduction was corrected against them, which was NOT a formality: the draft written from prose meaned
    both terms over the label count instead of over their own sets, quartering the sharpening term.
    Two departures remain, both deliberate and both stated where they live: `CARDINALITY_BOUNDS` admits
    k̂ = 0 for the clean knees CARVE's CT task cannot have, and the count is taken from ONE graded pass
    rather than from CARVE's average over the lowest-entropy 25% of eight weak views — the adapt pass in
    this repo is one view at one start, which is what keeps it minutes rather than hours. The view
    average is the next thing to try if a reading lands short; the arithmetic no longer is.

    EPISODIC IS WHY THERE IS NO ENTROPY GUARD. SAR carries a moving average of batch entropy and resets
    to source weights below e_0 = 0.2, because an online stream can drift for a thousand batches before
    anybody notices. Resetting after every study makes drift unrepresentable — the guard would be
    guarding a state that cannot occur, and a check that cannot fire is worse than no check.
    """

    episode: int = EPISODE_STUDIES
    steps: int = ADAPT_STEPS
    lr: float = ADAPT_LR
    negative_weight: float = NEGATIVE_WEIGHT
    bounds: tuple[int, int] = CARDINALITY_BOUNDS
    frozen_last_blocks: int = FROZEN_LAST_BLOCKS

    @staticmethod
    def _reachable(member: object) -> nn.Module:
        """The member as something whose parameters can be reached, or a refusal that says why not.

        A member arrives through a Protocol promising only `to` and a call, so the wider claim this needs
        is CHECKED here rather than assumed by every method that relies on it.
        """
        if not isinstance(member, nn.Module):
            message = f"{type(member).__name__} is not an nn.Module, so its LayerNorm affines cannot be reached"
            raise NotAdaptableError(message)
        return member

    def affines(self, member: object) -> list[nn.Parameter]:
        """Every LayerNorm weight and bias in the trunk, output-side blocks excepted.

        The normalization COMPUTATION is untouched and there is nothing to decide about statistics:
        LayerNorm keeps no running or batch statistics, so "recompute per batch?" is not a question this
        family has. Only the affine moves.
        """
        norms = [module for _, module in self._reachable(member).named_modules() if isinstance(module, nn.LayerNorm)]
        frozen = self.frozen_last_blocks
        adapted = norms[: -frozen or None] if len(norms) > frozen else norms
        return [parameter for norm in adapted for parameter in (norm.weight, norm.bias) if parameter is not None]

    def cardinality(self, probability: Float[Tensor, "studies labels"]) -> Float[Tensor, " studies"]:
        """How many findings each study is adapted toward — its own expected count, clipped to our range.

        ONE COUNT PER STUDY, never one for the batch. Both call sites can hand this several studies at
        once (`HoldoutPredictions.predict` reads them eight at a time), and a shared top-k across a batch
        would adapt every study toward whatever the batch's most common findings were — a leak from a
        study's neighbours into its own reading, which is the failure the episodic regime exists to
        prevent and which no reset can undo.
        """
        low, high = self.bounds
        return probability.detach().sum(dim=1).round().clamp(low, high)

    def wanted(self, probability: Float[Tensor, "studies labels"]) -> Bool[Tensor, "studies labels"]:
        """The labels each study is sharpened toward: its own top-k̂, k̂ varying study by study."""
        order = probability.argsort(dim=1, descending=True)
        rank = torch.empty_like(order)
        arranged = torch.arange(probability.shape[1], device=probability.device).expand_as(order)
        rank.scatter_(1, order, arranged)
        return rank < self.cardinality(probability).unsqueeze(1)

    def loss(self, logits: Float[Tensor, "studies labels"]) -> Float[Tensor, ""]:
        """CARVE Eq. 9: sharpen each study's own top-k̂ and suppress the rest, at `negative_weight`.

        EACH TERM IS MEANED OVER ITS OWN SET, and that is the equation rather than a preference:

            L = -(1/|P|) sum_{j in P} log p_j - lambda_neg (1/|N|) sum_{j in N} log(1 - p_j)

        An earlier draft divided BOTH sums by the label count, which is the plausible-looking spelling and
        is wrong in a direction that matters. Our k̂ has median 3 of 12, so |P|/L is about a quarter: the
        sharpening term arrived at roughly a quarter of its weight while suppression arrived at 10/12 of
        its own, and the objective this repo would have priced was mostly a push-everything-down. A NULL
        or a REGRESSION from that arm would have been a fact about the arithmetic, not about CARVE
        (D270 supplied Eq. 8-9; M265 is answered).

        THE EMPTY SET IS OURS, not CARVE's, and it is why the denominators are guarded. CARVE's k_min is 1
        so |P| is never zero; ours is 0 because over a tenth of our studies read clean, and a mean over no
        labels is a division by zero rather than a small number. A study with nothing to sharpen
        contributes suppression alone.

        THE ROWS ARE STUDIES, NOT WINDOWS, and that is a correction rather than a choice. A member pools
        its windows INTERNALLY — `StudyReadings.Member.__call__` is typed `b n` and `StudyModel` takes the
        window as an argument — so no per-window logit ever reaches this. An earlier draft meaned over a
        window axis that does not exist at the member's output; in the real path it would have been handed
        a 1-D tensor, and only a toy member with a different shape hid it.
        """
        probability = torch.sigmoid(logits.float()).clamp(PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
        wanted = self.wanted(probability)
        sharpen = self._over_set(-torch.log(probability), wanted)
        suppress = self._over_set(-torch.log1p(-probability), ~wanted)
        return (sharpen + self.negative_weight * suppress).mean()

    @staticmethod
    def _over_set(
        term: Float[Tensor, "studies labels"], chosen: Bool[Tensor, "studies labels"]
    ) -> Float[Tensor, " studies"]:
        """One study's mean of `term` over the labels `chosen` selects, or zero where it selects none."""
        counted = chosen.sum(dim=1)
        return (term * chosen).sum(dim=1) / counted.clamp(min=1)

    @contextmanager
    def on(self, member: object, logits: Forward) -> Iterator[None]:
        """Adapt to this study, hand the member over to be read, then put it back as it was."""
        affines = self.affines(member)
        snapshot = [parameter.detach().clone() for parameter in affines]
        try:
            self._descend(logits, affines)
            yield
        finally:
            with torch.no_grad():
                for parameter, original in zip(affines, snapshot, strict=True):
                    parameter.copy_(original)

    def _descend(self, logits: Forward, affines: list[nn.Parameter]) -> None:
        """Take the study's gradient steps, leaving every other parameter exactly where it was."""
        optimiser = torch.optim.Adam(affines, lr=self.lr)
        was_grading = [(parameter, parameter.requires_grad) for parameter in affines]
        for parameter in affines:
            parameter.requires_grad = True
        for _ in range(self.steps):
            optimiser.zero_grad(set_to_none=True)
            self.loss(logits()).backward()
            optimiser.step()
        for parameter, grading in was_grading:
            parameter.requires_grad = grading
