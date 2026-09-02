"""Where a run's weights COME FROM when they are a member on disk rather than a published encoder.

Its own module rather than a block in `train.py` because the decision it encodes is not the trainer's:
a run that starts from a fitted head is a different EXPERIMENT from one that starts from a fresh one,
and that difference has a published mechanism behind it (see `WarmStart`).
"""

from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from kneemri.training.init_scope import InitScope, WholeModel

STATE = "model"
"""The payload key holding a member's parameters — `MemberRecord.payload` writes it under this name."""


class WarmStart:
    """The weights a run begins from, loaded STRICTLY into the model this run's flags describe.

    WHY THIS EXISTS. `radnet_family` fine-tuned a trunk end to end and read 0.6716; a ridge probe over
    that same trunk's frozen features read 0.8197. One set of weights, two readings 0.148 apart, which is
    the signature of fine-tuning distorting the pretrained features rather than of an empty trunk. Kumar
    et al. (arXiv:2202.10054) measure exactly that and give the fix a name: fit the head first over a
    frozen trunk, THEN unfreeze — features move about twenty times less, and the result beats both the
    probe and the direct fine-tune. Davila et al. (arXiv:2406.10050) reproduce the ordering across five
    medical modalities and find it most reliable on DenseNet-121.

    Two stages need one thing our trainer could not express: stage two starting from stage one's
    checkpoint. That is all this is.

    STRICT, AND THE DESIGN IS NOT READ BACK. The checkpoint's own config is deliberately ignored — the
    flags of THIS run describe the model, and the weights either fit it or they do not. Loading
    leniently would let a checkpoint from a different pooling, width or slot count land partially and
    train something nobody asked for, which is this project's recurring failure with the serial number
    filed off: a value recorded in a config while the consumer reads something else. A mismatch raises
    here, before the card is spent.
    """

    def __init__(self, checkpoint: Path, scope: InitScope | None = None) -> None:
        self.checkpoint = checkpoint
        self.scope = scope or WholeModel()

    def applied(self, model: nn.Module) -> str:
        """Load the checkpoint into `model` and return the line the run should log about it.

        Returns rather than logs, so the caller keeps one logging site and a test can assert on what
        was said — the provenance of a run's starting weights belongs in its log, because from the next
        epoch onward nothing distinguishes a warm start from a cold one.
        """
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or STATE not in payload:
            message = f"{self.checkpoint} holds no {STATE!r} state, so it is not a member checkpoint"
            raise ValueError(message)
        state = cast("dict[str, Tensor]", payload[STATE])
        carried = self.scope.carried(state)
        # STILL STRICT over everything it takes: `strict=False` suppresses only the ABSENCE of a name, and
        # a shape that disagrees raises here exactly as it did before. What the scope may change is which
        # names are asked for at all — so a tensor landing in the wrong parameter remains impossible, and
        # a checkpoint missing something the scope did NOT drop is the failure it always was.
        incomplete = model.load_state_dict(carried, strict=False)
        unasked = set(incomplete.missing_keys) - self.scope.dropped(state)
        if incomplete.unexpected_keys or unasked:
            message = (
                f"{self.checkpoint} does not fit this model: "
                f"{sorted(unasked)} absent, {sorted(incomplete.unexpected_keys)} unknown"
            )
            raise ValueError(message)
        epoch = payload.get("epoch")
        cold = len(state) - len(carried)
        left = f", {cold} head tensors left cold" if cold else ""
        return f"warm start from {self.checkpoint.name} ({len(carried)} tensors{left}, its epoch {epoch})"
