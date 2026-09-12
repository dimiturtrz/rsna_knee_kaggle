"""What the OPTIMISER says a run got, as opposed to what its flags asked for.

Three flags reach the optimiser through `Adaptation` and are silent everywhere else: `--weight-decay`,
`--layer-decay` and `--backbone-lr`. A value that never arrives builds a perfectly working optimiser at
the inherited setting, trains to convergence, and reports the OLD recipe's number under the new arm's
name — the failure shape `kneemri.training.landed` exists to catch, and the one it cannot catch by
reading the request back to itself. The only object that cannot lie about them is the optimiser, so the
witnesses are read off its parameter groups rather than off the config that built them.

They live here rather than on `Trainer` because they are one responsibility and none of them touches the
model, the data or the step: given the groups and the anchor rate, the answers are fully determined.
"""

import logging

import torch

from kneemri.models.model import NO_LAYER_DECAY

logger = logging.getLogger(__name__)

NEEDED_FOR_A_RATIO = 2
"""A ratio needs two backbone groups; fewer means the trunk is frozen and no decay is observable."""


class OptimiserWitness:
    """The positive witnesses for the three flags whose only consumer is the optimiser.

    Constructed with the anchor rate captured at build time rather than read live, for the reason the
    capture site states: a schedule rewrites the live rate every step, so a late read reports the
    schedule and not the flag.
    """

    def __init__(self, optimiser: torch.optim.Optimizer, backbone_anchor: float) -> None:
        self._optimiser = optimiser
        self._backbone_anchor = backbone_anchor

    def weight_decay(self) -> float:
        """The decay MEASURED off the optimiser, not read back off the flag that asked for it.

        The positive witness `--weight-decay` would otherwise lack, and the same argument
        `encoder_rate_decay` makes one method below: a decay that never reached `Adaptation` builds an
        optimiser at the inherited 0.02, trains perfectly well, and reports the old recipe's number under
        the new arm's name. Read off the HEAD group, which every configuration has — under LoRA the
        backbone groups are empty of base blocks, and the head is the one group `parameter_groups` always
        returns.
        """
        return float(self._optimiser.param_groups[0]["weight_decay"])

    def encoder_rate_decay(self) -> float:
        """The per-block rate ratio MEASURED off the optimiser, not read back off the flag that asked.

        This is the positive witness `--layer-decay` would otherwise lack. Everything else about the
        flag is silent: a decay that never reached `Adaptation` builds the flat two groups, trains
        perfectly well, and reports the flat recipe's number under the decayed arm's name — the exact
        failure shape `landed` exists for. The optimiser's own groups cannot lie about it.

        Read off the TOP TWO backbone groups because they are the pair the decay is defined between,
        and because `unfreeze_last` is clamped to the trunk's depth: counting groups instead would
        report a mismatch for `--unfreeze-last 24` that is only this check's own arithmetic.
        """
        backbone = self._optimiser.param_groups[1:]
        if len(backbone) < NEEDED_FOR_A_RATIO:
            return NO_LAYER_DECAY
        return float(backbone[-2]["lr"]) / float(backbone[-1]["lr"])

    def backbone_rate(self) -> float:
        """The ANCHOR the per-block decay is measured FROM, read off the optimiser rather than the flag.

        `encoder_rate_decay` above witnesses the RATIO between two backbone groups, and a ratio is blind
        to the rate it is a ratio of. A `--backbone-lr` that never reached `Adaptation` builds the groups
        at the inherited rate, decays them by exactly the 0.7 that was asked for, and reports every
        switch landed — so the flag with the largest effect on any foreign-trunk arm was the one flag
        with no positive witness at all.

        That is how `band_deep_matched` floored its training loss 0.189 above its own control with all
        four recipe flags recorded correct, and why its 0.8487 could not be read as a fact about depth:
        no number anywhere could say whether the rate compensation arrived (knee-62sol).

        Taken off the LAST backbone group, which `backbone_groups` builds at `layer_decay ** 0` — the
        anchor itself, and the only group whose rate is the flag rather than a power of it.
        """
        return self._backbone_anchor
