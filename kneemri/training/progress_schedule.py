"""A value over the run's PROGRESS, piecewise linear: start -> plateau -> end, as one shared schedule.

NEUTRAL OF ITS CONSUMER. Augmentation strength is the first thing read off it; example hardness is the
next (knee-hardness bead). Both are "how much of this regulariser at this point of the run", and two
schedule types would be two places for the ramp arithmetic to disagree.

THE DEFAULT IS THE CONSTANT 1.0, and it is exact: `value` returns `mid` untouched on the plateau, so a
consumer that scales by it multiplies by the float 1.0 at every step and every ledger number keeps its
meaning.

Progress runs 0 -> 1 over the run. `ramp` is where the plateau begins and `tail` how much of the end is
spent leaving it, so `ramp=0.3, tail=0.15` climbs over the first 30%, holds, and descends over the last
15%. A zero ramp with `start != mid` is a STEP: the ramp segment is empty and `mid` holds from progress 0.
"""

from dataclasses import astuple, dataclass

FULL = 1.0
"""The identity strength: every consumer at its configured value."""

NONE = 0.0
"""An empty ramp or tail — the segment does not exist."""

NEGATIVE = "a progress schedule's strengths must be >= 0; a negative width is not a weaker augmentation"
OVERLAP = (
    "a progress schedule needs 0 <= ramp, 0 <= tail and ramp + tail <= 1 — the plateau cannot end before it begins"
)

ROLES = {
    "start": "strength at progress 0",
    "mid": "strength on the plateau",
    "end": "strength at progress 1",
    "ramp": "share of the run climbing start -> mid",
    "tail": "share of the run descending mid -> end",
}


@dataclass(frozen=True)
class ProgressSchedule:
    """Five knobs: strength at progress 0, on the plateau, at progress 1, and where the plateau sits."""

    start: float = FULL
    mid: float = FULL
    end: float = FULL
    ramp: float = NONE
    tail: float = NONE

    def __post_init__(self) -> None:
        if min(self.start, self.mid, self.end) < NONE:
            raise ValueError(NEGATIVE)
        if min(self.ramp, self.tail) < NONE or self.ramp + self.tail > FULL:
            raise ValueError(OVERLAP)

    def value(self, progress: float) -> float:
        """The multiplier at this share of the run, clipped into [0, 1]."""
        progress = min(max(progress, NONE), FULL)
        closing = FULL - self.tail
        if progress < self.ramp:
            return self.start + (self.mid - self.start) * progress / self.ramp
        if progress > closing:
            return self.mid + (self.end - self.mid) * (progress - closing) / self.tail
        return self.mid

    @staticmethod
    def knob_help(knob: str, subject: str) -> str:
        """A knob's flag help: its role in the schedule, then what that schedule drives."""
        return f"{ROLES[knob]} of {subject}; defaults are the ledger"

    def recorded(self) -> tuple[float, ...]:
        """The five knobs in flag order — what a landed-check compares and a log line prints."""
        return astuple(self)
