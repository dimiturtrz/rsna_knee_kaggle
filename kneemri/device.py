"""Which device this process trains, encodes or scores on — decided in ONE place.

`torch.cuda.is_available()` IS THE WRONG QUESTION, and this project has already paid for asking it.
The card here is shared with another project, and the way it is handed over is `CUDA_VISIBLE_DEVICES=`
rather than uninstalling anything. Under that setting on this machine `is_available()` stays TRUE while
`device_count()` is 0, so every `"cuda" if torch.cuda.is_available() else "cpu"` in the tree selects a
device that owns no card, and the process dies on the first property read with

    AssertionError: Invalid device id

— sometimes immediately, sometimes several minutes in, after a decode has already been paid for.
`member_curve` knew this and worked around it with its own `--device` flag; nine other call sites
carried the same expression and none of the knowledge. `device_count()` asks what the caller meant:
is there a card I may use.
"""

import torch


class VisibleCards:
    """The cards this process may actually use, which is not the same as the cards that exist."""

    def device(self) -> torch.device:
        """cuda when a visible card exists, cpu otherwise — the one place that choice is made."""
        return torch.device("cuda" if torch.cuda.device_count() else "cpu")
