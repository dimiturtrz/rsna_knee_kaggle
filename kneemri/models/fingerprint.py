"""The portable identity of a fitted member: what it answers to a seeded synthetic question.

Weights read through the wrong preprocessing produce predictions, not errors. The submission is well
formed, the log says nothing, and the difference is a number no output of the run reveals. This module
is the one thing that makes such a member fail loudly, so the question it asks has to be a constant of
the QUESTION and not of any member that answers it — which is why the seed and the batch live here,
beside the code that uses them, rather than being restated by each caller. `member_record.py` restated
the seed once; a stamped seed that drifts from the asked seed describes a bag nobody fed.
"""

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn


@dataclass(frozen=True)
class Fingerprint:
    """The seeded question every member is asked, and the machinery that asks it."""

    seed: int = 2026
    """The seed of the synthetic bag. A constant of the question, not of the member that answers it."""

    batch: int = 2
    """Two studies, so the masked branch of the softmax is exercised alongside the unmasked one."""

    def of(
        self, model: nn.Module, n_slot: int, image_size: int, group: int, device: torch.device
    ) -> NDArray[np.float32]:
        """This model's output on the fixed synthetic bag.

        The input is generated from the seed rather than read — identical on any machine — and pushed
        through the whole forward path: byte scaling, ImageNet normalisation, resize, encoder, slot
        attention. Any of those differing moves the output by order one.

        float32 throughout: autocast would make the value depend on which device happened to run it,
        and the point of the number is that it does not.
        """
        generator = torch.Generator().manual_seed(self.seed)
        imgs = torch.randint(
            0,
            256,
            (self.batch, n_slot, group, image_size, image_size),
            generator=generator,
            dtype=torch.uint8,
        ).to(device)
        mask = torch.ones(self.batch, n_slot, device=device)
        mask[1, -1] = 0.0
        was_training = model.training
        model.eval()
        with torch.no_grad():
            out = model(imgs, mask, image_size).float().cpu().numpy()
        if was_training:
            model.train()
        return out


FINGERPRINT = Fingerprint()
"""The one question. Spelled once so a stamped seed cannot drift from the asked seed."""
