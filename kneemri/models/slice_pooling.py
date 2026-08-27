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
from kneemri.models.slice_attention import SliceAttention

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

LABELWISE = "labelwise"
"""`attend` with one set of pooling weights PER FINDING instead of one shared by all twelve.

The only pooling that hands the head a diagnosis axis rather than a single vector per slot. See
`LabelwiseAttention` for why a shared query's +0.0002 is the null this predicts."""


@dataclass(frozen=True)
class SlicePooling:
    """What a slice-pooling NAME decides: whether slices are encoded alone, and what then pools them.

    One record per name instead of a bool table plus an `== ATTEND` branch repeated in every module that
    builds a head. That branch had already been copied into `head_trial`, so a pooling added here would
    have been accepted by the trial's argument parser and then silently built as the maximum — this
    project's plumbing failure exactly: a flag that lands in the record and never reaches the consumer.
    """

    per_slice: bool
    aggregator: Callable[[int], nn.Module] | None = None
    """The module that pools the slice axis away, or None when a bare `amax` does it.

    It may replace that axis with a DIAGNOSIS axis rather than remove it outright — `labelwise` does —
    and `SlotHead.per_diagnosis` is the one place that reconciles the two shapes."""


SLICE_POOLINGS: dict[str, SlicePooling] = {
    CHANNELS: SlicePooling(per_slice=False),
    MAXIMUM: SlicePooling(per_slice=True),
    ATTEND: SlicePooling(per_slice=True, aggregator=SliceAttention),
    DEPTH: SlicePooling(per_slice=True, aggregator=DepthAttention),
    LABELWISE: SlicePooling(per_slice=True, aggregator=LabelwiseAttention),
}
"""The vocabulary, mapped to how the slice axis is read — the fact three call sites each decided.

An unknown name USED to mean channel stacking, because `per_slice` asked `in {"max", "attend"}` and
anything else fell through. `members.py` and `holdout_predictions.py` read this string out of a
manifest and hand it straight to `StudyDesign`, so a stale or misspelled name rebuilt a different
function on different pixels and loaded without complaint. The lookup now happens where the model is
BUILT, so it raises there instead."""
