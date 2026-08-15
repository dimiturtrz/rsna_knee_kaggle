"""Supervising a study by its whole report, as an auxiliary term beside the twelve labels.

`kneemri.embeddings.report_embeddings` writes one vector per study offline. This is the training-time half: line
the vectors up with the corpus, and add a term that pays when the study representation points somewhere
other than its report.

The term is a **cosine** distance rather than a squared error. The target came out of an L2-normalised
text encoder and then a projection, so its length carries how far a report sits from the corpus mean —
a quantity about report LENGTH and verbosity as much as about pathology. Direction is the part worth
predicting, and an image cannot be expected to know how talkative the radiologist was.

The default is a no-op object rather than `None`, so the training loop never asks whether report
supervision is switched on.
"""

import os

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor, nn

REPORT_WEIGHT = 0.3
"""How hard the auxiliary term pulls beside the twelve-label loss.

Below one on purpose: the labels are the thing being scored, and the report term exists to shape the
representation rather than to become the objective. It is a starting point with an argument, not a
swept value — sweeping it before the term has shown any signal would be measuring a method that does
not work yet.
"""


class NoReportSupervision:
    """No report term — the default, and a zero the optimiser can add without a branch."""

    def __call__(self, predicted: Float[Tensor, "b r"], positions: NDArray[np.int64]) -> Float[Tensor, ""]:
        """Nothing to pay: report supervision is off for this run."""
        return predicted.new_zeros(())


class ReportSupervision(NoReportSupervision):
    """Pays when a study's representation points away from its report's embedding."""

    def __init__(self, vectors: Float[np.ndarray, "n r"], weight: float = REPORT_WEIGHT) -> None:
        self.vectors = vectors
        self.weight = weight

    @classmethod
    def from_file(
        cls, path: os.PathLike[str], identifiers: list[str], weight: float = REPORT_WEIGHT
    ) -> "ReportSupervision":
        """Read the npz and reorder it into the CORPUS's study order, failing loudly on a gap.

        Reordering rather than trusting the file's order is the discipline the label loader follows:
        both are keyed by StudyInstanceUID, and a silent misalignment would train every study against
        another study's report while everything continued to run.

        The term loads its own targets because there is nothing else to do with them - a separate
        loader class would be a second subject in this file and a type that exists to be passed once.
        """
        stored = np.load(path, allow_pickle=True)
        position = {str(name): index for index, name in enumerate(stored["identifiers"])}
        missing = [name for name in identifiers if name not in position]
        if missing:
            message = f"{len(missing)} studies have no report embedding, first {missing[0]}"
            raise KeyError(message)
        rows = np.array([position[name] for name in identifiers])
        return cls(stored["vectors"][rows].astype(np.float32), weight)

    def __call__(self, predicted: Float[Tensor, "b r"], positions: NDArray[np.int64]) -> Float[Tensor, ""]:
        """One minus the cosine similarity to each study's own report vector, at the configured weight."""
        target = torch.from_numpy(self.vectors[positions]).to(predicted.device, predicted.dtype)
        return self.weight * (1.0 - nn.functional.cosine_similarity(predicted, target, dim=-1)).mean()
