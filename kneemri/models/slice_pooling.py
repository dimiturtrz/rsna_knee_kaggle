"""The slice-pooling vocabulary: what each NAME decides about how a slot's slices are read.

Its own module rather than a block inside `model.py` because three call sites speak it — the model, the
argument parser, and the frozen-feature trial — and because the model file had grown past the god-file
threshold. Shared vocabulary belongs beside neither one consumer nor a leftovers bucket.
"""

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

from kneemri.models.depth_attention import DepthAttention
from kneemri.models.labelwise_attention import LabelwiseAttention
from kneemri.models.labelwise_patch_attention import LabelwisePatchAttention
from kneemri.models.moment_pooling import MOMENTS as MOMENT_PARTS
from kneemri.models.moment_pooling import MomentPooling
from kneemri.models.slice_attention import SliceAttention
from kneemri.models.wide_moment_pooling import WideMomentPooling

ATTEND = "attend"
"""The one slice pooling that carries a learned module, so it is the one spelled in more than one place."""

CHANNELS = "channels"
"""Three consecutive slices become the encoder's three channels — one forward per slot, and the group's
physical thickness is coupled to the slice count."""

MAXIMUM = "max"
"""Each slice encoded ALONE, the slot taking the per-dimension maximum. Three forwards, no coupling."""

DEPTH = "depth"
"""`attend`, preceded by a depthwise mix along the slice axis — the one slice pooling that can see ORDER.

Named beside the two order-blind ones because that is the comparison it exists to make. See
`DepthAttention` for why the axis went untested for this long."""

MOMENTS = "moments"
"""The mean along the slice axis beside its SPREAD — the one pooling that is not a read of the first
moment alone.

Named beside the four that are, because that is the comparison it exists to make: `attend`, `depth` and
`labelwise` all select among slices and measured within +0.0002 of the elementwise maximum. See
`MomentPooling` for the frozen-feature reading that opened the axis and for why it is not the same
instrument."""

MOMENTS_WIDE = "moments_wide"
"""The same two moments as `moments`, handed to the head CONCATENATED instead of mixed back down.

Named beside `moments` because the pair is the measurement. `moment_dense` ran the mixed-down form at the
best solo corner and read 0.8516 against a 0.8634 control; the checkpoint dump then retired both suspects
that band named (the spread half of the mix is alive at 0.79x the first moment's magnitude) and surfaced a
third: the identity initialisation on the mean half is gone, so the head was fed a learned projection of
the pair rather than the pair. The frozen probe that opened this axis read the CONCATENATION. See
`WideMomentPooling`."""

LABELWISE = "labelwise"
"""`attend` with one set of pooling weights PER FINDING instead of one shared by all twelve.

The only pooling that hands the head a diagnosis axis rather than a single vector per slot. See
`LabelwiseAttention` for why a shared query's +0.0002 is the null this predicts."""

LABELWISE_PATCH = "labelwise_patch"
"""`labelwise`, plus a per-finding softmax over every PATCH token of the slot, entering through a zeroed
projection so that at initialisation it IS `labelwise`.

The only pooling that reads inside a slice per diagnosis: every other name sees each slice already reduced
to one vector by the token pooling. See `LabelwisePatchAttention`."""


@dataclass(frozen=True)
class SlicePooling:
    """What a slice-pooling NAME decides: whether slices are encoded alone, and what then pools them.

    One record per name instead of a bool table plus an `== ATTEND` branch repeated in every module that
    builds a head. That branch had already been copied into `head_trial`, so a pooling added here would
    have been accepted by the trial's argument parser and then silently built as the maximum — this
    project's plumbing failure exactly: a flag that lands in the record and never reaches the consumer.
    """

    per_slice: bool
    aggregator: Callable[..., nn.Module] | None = None
    out_multiplier: int = 1
    """How many times the pooled width the aggregator HANDS BACK — 1 for every pooling that preserves it.

    Recorded here rather than asked of the module, because the head is constructed in the same breath as
    the aggregator and a width discovered from a tensor would be discovered one batch too late. It is the
    third shape this vocabulary reconciles, beside the slice axis a pooling removes and the diagnosis axis
    `labelwise` puts in its place."""
    """The module that pools the slice axis away, or None when a bare `amax` does it.

    It may replace that axis with a DIAGNOSIS axis rather than remove it outright — `labelwise` does —
    and `SlotHead.per_diagnosis` is the one place that reconciles the two shapes."""
    reads_patches: bool = False
    """Whether the aggregator also takes the slot's PATCH tokens, not only the token-pooled slices.

    The model keeps the patch tokens past the token pooling only when this is set, so every other pooling
    computes exactly what it did. A cached-feature trial has no patches to give, which is why it refuses
    these names."""

    def build(self, width: int, token_width: int) -> nn.Module | None:
        """The aggregator at the pooled `width`, told the trunk's `token_width` when it reads patches."""
        if self.aggregator is None:
            return None
        return self.aggregator(width, token_width) if self.reads_patches else self.aggregator(width)


SLICE_POOLINGS: dict[str, SlicePooling] = {
    CHANNELS: SlicePooling(per_slice=False),
    MAXIMUM: SlicePooling(per_slice=True),
    ATTEND: SlicePooling(per_slice=True, aggregator=SliceAttention),
    DEPTH: SlicePooling(per_slice=True, aggregator=DepthAttention),
    LABELWISE: SlicePooling(per_slice=True, aggregator=LabelwiseAttention),
    LABELWISE_PATCH: SlicePooling(per_slice=True, aggregator=LabelwisePatchAttention, reads_patches=True),
    MOMENTS: SlicePooling(per_slice=True, aggregator=MomentPooling),
    MOMENTS_WIDE: SlicePooling(per_slice=True, aggregator=WideMomentPooling, out_multiplier=MOMENT_PARTS),
}
"""The vocabulary, mapped to how the slice axis is read — the fact three call sites each decided.

An unknown name USED to mean channel stacking, because `per_slice` asked `in {"max", "attend"}` and
anything else fell through. `members.py` and `holdout_predictions.py` read this string out of a
manifest and hand it straight to `StudyDesign`, so a stale or misspelled name rebuilt a different
function on different pixels and loaded without complaint. The lookup now happens where the model is
BUILT, so it raises there instead."""
