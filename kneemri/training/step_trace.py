"""What one epoch's optimiser steps did, held on the device until the epoch ends.

`fit_epoch` used to leave one number behind it — the mean loss — plus five quantiles. How large each
step's gradient was, whether one went non-finite, what rate each group ran at, and how well the model
fit the studies it was descending on were all in hand inside the loop and thrown away, so a run that
slid could never say whether its steps had grown. The trace keeps them as device tensors and reads them
back ONCE, after the loop, so the hottest loop in the project gains no synchronisation.

It also owns the per-study loss block that sat in `Trainer`, which is what lets it grow without pushing
`trainer.py` past the god-file ceiling.
"""

import numpy as np
import torch
from jaxtyping import Float, Int
from numpy.typing import NDArray
from torch import Tensor

from kneemri.evaluation.scores import MIDPOINT, LabelScores
from kneemri.runs.run_record import StepRow
from kneemri.training.grades import Predicted
from kneemri.training.loss_quantiles import EpochLosses

WEIGHT_FLOOR = 1e-6
"""Below this a study's summed cell weight is treated as this, so a fully masked study reads as zero loss."""

UNREAD = float("nan")


class StepTrace:
    """One epoch's steps: per-study losses, per-step loss, gradient norm and rates, and the fit readout."""

    def __init__(self) -> None:
        self.quantiles: dict[str, float] = {}
        self.opened(np.empty(0, dtype=np.int64))

    def opened(self, drawn: Int[NDArray[np.int64], " n"]) -> None:
        """Start an epoch over the positions the sampler DREW, forgetting the previous epoch's steps."""
        self.drawn = drawn
        self._studies: list[Tensor] = []
        self._losses: list[Tensor] = []
        self._norms: list[Tensor] = []
        self._rates: list[list[float]] = []
        self._fit: list[Tensor] = []
        self._truth: list[Tensor] = []
        self._weights: list[Tensor] = []

    def judged(self, judged: Predicted) -> None:
        """Keep one micro-batch's per-study loss and its predictions against the targets that judged them.

        The per-study loss is the same weighted BCE the scalar is built from, averaged over labels within
        each study, on the UNSHARED loss so the distribution matches the number the epoch reports. Under
        `no_grad` so none of it can feed back into training.
        """
        with torch.no_grad():
            elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
                judged.logits, judged.scores, reduction="none"
            )
            weights = judged.weights.float()
            self._studies.append((elementwise * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=WEIGHT_FLOOR))
            self._fit.append(judged.logits.detach().float().sigmoid())
            self._truth.append(judged.scores.detach().float())
            self._weights.append(weights)

    def stepped(self, loss: Float[Tensor, ""], norm: Float[Tensor, ""], optimiser: torch.optim.Optimizer) -> None:
        """Keep one optimiser step: the last micro-batch's loss, the pre-clip norm, and each group's rate."""
        self._losses.append(loss.detach().float())
        self._norms.append(norm.detach().float())
        self._rates.append([float(group["lr"]) for group in optimiser.param_groups])

    def closed(self) -> None:
        """Reduce the epoch's per-study losses to the quantiles the record carries."""
        losses = EpochLosses()
        if self._studies:
            losses.collect(torch.cat(self._studies).double().cpu().numpy())
        self.quantiles = losses.quantiles()

    def loss_fields(self) -> dict[str, float]:
        """The per-study loss quantiles under the record's names, `loss_q25` … `loss_q95`."""
        return {f"loss_{name}": value for name, value in self.quantiles.items()}

    def rows(self, epoch: int) -> list[StepRow]:
        """Every optimiser step of the epoch, in order, as the step stream records it."""
        losses, norms = self._host(self._losses), self._host(self._norms)
        return [
            StepRow(epoch=epoch, step=step, loss=float(loss), grad_norm=float(norm), lr=rates)
            for step, (loss, norm, rates) in enumerate(zip(losses, norms, self._rates, strict=True))
        ]

    def optimised(self) -> dict[str, object]:
        """The epoch's gradient norms summarised over its FINITE steps, the count that were not, and the last rates."""
        norms = self._host(self._norms)
        finite = norms[np.isfinite(norms)]
        return {
            "grad_norm_mean": float(finite.mean()) if len(finite) else UNREAD,
            "grad_norm_max": float(finite.max()) if len(finite) else UNREAD,
            "nonfinite_steps": int((~np.isfinite(norms)).sum()),
            "lr": self._rates[-1] if self._rates else [],
        }

    def fit_auc(self) -> float:
        """Mean AUC of the training predictions as they were made, over the cells the objective weighted.

        Read under augmentation, mixup and weights still moving, so it is a fit-side TREND to set beside
        the holdout AUC, never a number to compare with it.
        """
        if not self._fit:
            return UNREAD
        predicted = torch.cat(self._fit).double().cpu().numpy()
        truth = torch.cat(self._truth).cpu().numpy() > MIDPOINT
        weighted = torch.cat(self._weights).cpu().numpy() > 0
        aucs = [
            LabelScores.area_under_curve(
                predicted[weighted[:, column], column], truth[weighted[:, column], column].astype(np.int64)
            )
            for column in range(predicted.shape[1])
        ]
        defined = [auc for auc in aucs if not np.isnan(auc)]
        return float(np.mean(defined)) if defined else UNREAD

    @staticmethod
    def _host(values: list[Tensor]) -> NDArray[np.float64]:
        return torch.stack(values).double().cpu().numpy() if values else np.empty(0, dtype=np.float64)
