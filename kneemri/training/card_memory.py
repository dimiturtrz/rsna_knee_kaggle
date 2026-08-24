"""What the card had free when a run started, against what it has now."""

import logging
import shutil
import subprocess
from collections.abc import Callable

logger = logging.getLogger(__name__)

FOREIGN_MIB = 1024
"""How much free VRAM must vanish mid-run before it is another process rather than our own allocator.

Derived from both regimes this has seen, not chosen. Our own reserve grows in fragments — a cache page,
an eval batch, a checkpoint write — and stays inside a few hundred MiB of where it settled. A foreign
trainer is measured in gigabytes: 11.4 GB on 2026-08-21, 4.8 GB on 2026-08-24. A gigabyte sits an order
of magnitude from the pathology and above the jitter, so the constant does not need tuning to be right.
"""

SMI_TIMEOUT_SECONDS = 10.0
"""A driver query that has not answered in this long is a wedged card, and the caller may not block on it."""


class CardMemory:
    """Free VRAM at the start of a run, against free VRAM now.

    THE HEARTBEAT COULD ALREADY PROVE A COLLAPSE AND NOT ITS CAUSE. On 2026-08-24 it printed
    "94 img/s against this run's own peak of 1221" and then left the reader to run `nvidia-smi`
    themselves. The reading that settles it was seven minutes away in a terminal, and the two answers it
    distinguishes — a foreign job arrived, or this recipe simply does not fit — have opposite responses:
    one is a scheduling collision to negotiate, the other is a recipe to shrink.

    A BASELINE IS WHAT MAKES THE READING MEAN ANYTHING. Free VRAM at the moment of a stall is nearly
    uninterpretable alone: a healthy run of ours also leaves the card near-full, which is the whole reason
    `wait_for_free_card` gates on it before launching. What is diagnostic is the DIFFERENCE — VRAM that
    was there when we started and is not there now belongs to somebody who arrived after us.

    The read is injected so this can be unit-tested without a device, and so a machine with no driver
    degrades to saying nothing rather than to failing.
    """

    def __init__(self, read: Callable[[], int | None] | None = None) -> None:
        self.read = read if read is not None else self.free_mib
        self.at_start: int | None = None

    @staticmethod
    def free_mib() -> int | None:
        """Free VRAM as the driver reports it, or `None` when there is nothing to ask.

        `None` rather than a raise or a zero, because every caller here is on an already-exceptional path
        and none of them may turn a missing tool into a failed run. A zero would be worse than silence: it
        reads as "the card is full", which is the exact conclusion this is supposed to establish.
        """
        smi = shutil.which("nvidia-smi")
        if smi is None:
            return None
        probe = subprocess.run(  # noqa: S603 - fixed argv, no shell, binary resolved by shutil.which
            [smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=SMI_TIMEOUT_SECONDS,
        )
        answered = probe.stdout.strip().splitlines()
        readable = probe.returncode == 0 and bool(answered) and answered[0].strip().isdigit()
        return int(answered[0]) if readable else None

    def remember(self) -> None:
        """Record the free VRAM this run began with, once, and never overwrite it.

        Called from the first heartbeat rather than from construction so the reading is taken after the
        model is on the card — a baseline taken before allocation would make every healthy run look like
        it had lost twenty gigabytes to a stranger.
        """
        self.at_start = self.at_start if self.at_start is not None else self.read()

    def verdict(self) -> str:
        """Name whether VRAM disappeared after this run started, which is what a foreign job looks like."""
        at_start, now = self.at_start, self.read()
        if at_start is None or now is None:
            return "the card could not be read, so nothing here says whether another process arrived"
        lost = at_start - now
        if lost < FOREIGN_MIB:
            return (
                f"{now} MiB free against {at_start} at the start, so no foreign process arrived — this is "
                f"our own footprint against this card and the recipe is what has to shrink"
            )
        return (
            f"{lost} MiB vanished after this run started ({at_start} MiB free then, {now} now), which is "
            f"another process on the card rather than anything this run did"
        )
