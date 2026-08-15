"""Progress within an epoch, so a slow run can be told apart from a wedged one."""

import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 30.0
"""How often a run says where it is. Long enough to be quiet, short enough to catch a wedged step."""


class Heartbeat:
    """Progress WITHIN an epoch, because one line per epoch cannot tell slow apart from wedged.

    On 2026-08-11 a whole-stack arm ran fifteen minutes without printing an epoch, and nothing in the
    logs could say whether it was working. The answer had to be inferred from the card's power draw:
    145 W of 575 meant it was thrashing on a VRAM overflow rather than computing. A step counter would
    have said so immediately — zero steps completed is unambiguous in a way that silence is not.

    Paced by SECONDS rather than by steps, because step cost varies by 20x across arms: a three-slice
    window and a whole-stack read are both "one step". Every N steps would be a wall of noise for one
    and near-silence for the other.

    Not tqdm, deliberately. Every arm's output is redirected to a file, where a progress bar leaves a
    trail of carriage returns — the HuggingFace weight-loading bar already litters every log we have.
    A log line is the right instrument when the reader is a file.
    """

    def __init__(self, every_seconds: float = HEARTBEAT_SECONDS, clock: Callable[[], float] = time.monotonic) -> None:
        self.every_seconds = every_seconds
        self.clock = clock
        self._since = 0.0
        self._images = 0

    def tick(self, step: int, steps: int, loss: Callable[[], float], images: int) -> None:
        """Count one step's images, and report the rate if the interval has elapsed.

        THE LOSS ARRIVES AS A CALLABLE because reading it costs a device synchronisation on the caller's
        side, and this prints it about once in a thousand steps. Taking a float would have made every
        step pay for a number that thirty seconds of them never use.
        """
        self._images += images
        now = self.clock()
        if not self._since:
            self._since = now
            return
        elapsed = now - self._since
        if elapsed < self.every_seconds:
            return
        logger.info("  step %d/%d  loss %.4f  %.0f img/s", step, steps, loss(), self._images / elapsed)
        self._since = now
        self._images = 0
