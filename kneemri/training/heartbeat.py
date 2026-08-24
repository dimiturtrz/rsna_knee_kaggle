"""Progress within an epoch, so a slow run can be told apart from a wedged one."""

import logging
import time
from collections.abc import Callable

from kneemri.training.card_memory import CardMemory

logger = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 30.0
"""How often a run says where it is. Long enough to be quiet, short enough to catch a wedged step."""

STALL_FRACTION = 0.5
"""Below this share of the run's OWN peak rate, a report is a stall rather than a slow patch.

Derived from the two regimes, not chosen: within one arm the rate varies by a few tens of percent
(batch composition, an occasional checkpoint write), while the failure this catches — GPU memory paged
over PCIe when two processes overflow the card — runs at a tenth to a fifteenth of the healthy rate and
stays there. Half the peak sits an order of magnitude away from the pathology and well outside normal
variation, so the constant does not have to be tuned to be right.
"""

STALL_REPORTS = 3
"""Consecutive slow reports before saying so — 90 s at the default interval.

One slow report is an eval, a checkpoint write or a cold cache page. A minute and a half of them is not.
"""


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

    def __init__(
        self,
        every_seconds: float = HEARTBEAT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        card: CardMemory | None = None,
    ) -> None:
        self.every_seconds = every_seconds
        self.clock = clock
        self.card = card if card is not None else CardMemory()
        self._since = 0.0
        self._images = 0
        self._peak = 0.0
        self._slow = 0
        self._stalled = False

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
        rate = self._images / elapsed
        logger.info("  step %d/%d  loss %.4f  %.0f img/s", step, steps, loss(), rate)
        self.card.remember()
        self.watch(rate)
        self._since = now
        self._images = 0

    def watch(self, rate: float) -> None:
        """Say so when the run collapses against its own established rate.

        On 2026-08-19 this run fell from ~1090 img/s to 30-170 and stayed there for the rest of the arm.
        Nothing reported it: there was no error, no traceback, and `nvidia-smi` said 99% utilisation —
        at 183 W of a 575 W limit, because a second project's trainer had taken the card to 31949 MiB of
        32607 and Windows pages GPU memory over PCIe rather than failing. The card looked busy while
        doing almost nothing, and the only visible symptom was in a number nobody was comparing to
        itself.

        Compared against the run's own PEAK rather than an absolute threshold, because step cost varies
        by 20x across arms and any constant that fits a whole-stack read is silence for a windowed one.
        The peak is what this arm demonstrably achieved on this card with this recipe, which is the only
        honest reference available without a second run.

        Latching, so a long stall says it once and its recovery says so too. A warning per report would
        be the same wall of noise the heartbeat interval exists to avoid.
        """
        self._peak = max(self._peak, rate)
        if rate < self._peak * STALL_FRACTION:
            self._slow += 1
            if self._slow >= STALL_REPORTS and not self._stalled:
                self._stalled = True
                logger.warning(
                    "  STALLED: %.0f img/s against this run's own peak of %.0f — %s",
                    rate,
                    self._peak,
                    self.card.verdict(),
                )
            return
        if self._stalled:
            logger.info("  recovered: %.0f img/s, against a peak of %.0f", rate, self._peak)
        self._slow = 0
        self._stalled = False
