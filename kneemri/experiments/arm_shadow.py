"""Which arm files may never be RUN, because the stub cannot shadow what they call.

THE ONE THING THAT MAKES `arm_check` A LINT IS THE STUB, AND THE STUB SHADOWS `uv` AND NOTHING ELSE.
That module RUNS every arm file to read its flags (see `ArmCheck.invocations`), which is only ever a
lint because `experiments/stub_lib.sh` puts a no-op `uv` ahead of the real one. An arm naming
`.venv/Scripts/python.exe` walks straight past that and the lint becomes a launcher.

MEASURED 2026-09-09, and it was the owner who noticed rather than any check: `arm_check --only
carve_adapt_read` put three `foreign_readout` processes on the card — 19.4 GB and 12.3 GB resident, on
`--device cuda` — sixteen seconds after it started, beside a live twelve-epoch training arm, with no
`card_check` consulted. Two of them survived the first `Stop-Process` and GREW before a second kill took
them. Eight arms had this shape. The rule was already written down in `arm_check` ("Check instead that
nothing under `arms/` reaches an interpreter without going through `uv`") and nothing enforced it, which
is the same hole knee-q4c3u was closed as no-defect over.

THE FIX IS TO REFUSE, NOT TO REWRITE THE ARMS. `experiments/lib.sh` exports
`UV_PROJECT_ENVIRONMENT=.venv-arm`, so `uv run` inside an arm resolves to a DIFFERENT environment from
the `.venv` those eight deliberately name — a blind rewrite would silently move them to an env that has
never held their dependencies (`sklearn` was missing from `.venv-arm` while present in `.venv`, and it
killed the same arm twice in one evening). So the launcher declines to launch them and says so.

It REPORTS rather than refusing: the eight are real arms, and blocking would break every invocation of
the lint over a hazard the lint itself has already stopped taking.
"""

import re
from dataclasses import dataclass
from pathlib import Path

INTERPRETER = re.compile(r"\.venv[\w-]*[/\\](\w+[/\\])?python")
"""An arm reaching a Python interpreter by PATH rather than through `uv`.

Matches `.venv` and any sibling environment (`.venv-arm`), on either slash, with or without the one
intervening segment a layout puts there (`Scripts/` on Windows, `bin/` elsewhere). Permissive on
purpose: this pattern is what stands between the lint and the card, so a spelling it does not know is a
hazard, while a false positive costs one arm a report line.
"""


@dataclass(frozen=True)
class ArmShadow:
    """The arms directory, asked which of its files the stub cannot stand in front of."""

    arms: Path

    def unshadowed(self) -> list[Path]:
        """Arm files that reach an interpreter directly, sorted by name.

        Read from the FILE rather than from what the arm printed, since an arm that reaches its
        interpreter directly may print nothing at all — it would have been on the card instead.

        COMMENTS ARE NOT CODE, and reading them as code costs an arm its check. Three arms in this tree
        name `.venv/Scripts/python.exe` only in prose — two of them in the very comment explaining why an
        arm must not do that — and matching those dropped both halves of the carve pair from the lint's
        pool while every gate stayed green. A stripped line is the whole fix: a shell comment cannot
        launch anything.
        """
        return [
            arm
            for arm in sorted(self.arms.glob("*.sh"))
            if any(
                INTERPRETER.search(line)
                for line in arm.read_text(encoding="utf-8", errors="ignore").splitlines()
                if not line.lstrip().startswith("#")
            )
        ]

    def reported(self) -> list[str]:
        """One line per unshadowed arm, plus what to do about them — empty when the tree is clean.

        Named rather than skipped: an arm silently dropped from the pool is the clean-pass-over-nothing
        failure, which is the thing this module exists to stop, one level up.
        """
        found = self.unshadowed()
        if not found:
            return []
        return [
            f"NOT CHECKED {arm.name} reaches an interpreter directly - the stub shadows uv alone" for arm in found
        ] + [
            f"{len(found)} arm(s) would be LAUNCHED by this lint, not checked by it. Route them through "
            "`uv run` (mind UV_PROJECT_ENVIRONMENT=.venv-arm) or move them out of arms/"
        ]
