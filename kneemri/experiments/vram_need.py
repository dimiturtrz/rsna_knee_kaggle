"""What one arm's flags will reserve, in MiB, for the shell that decides whether to start it.

`python -m kneemri.experiments.vram_need --slices 20 --batch 2 ...` — the same flags the arm passes to
the trainer, answered by the same cost model the trainer caps itself with.

WHY IT IS AN ENTRYPOINT AND NOT A CONSTANT. `lib.sh` will not start an arm until the card has room for
it, and it asked for a hard-coded 18432 MiB: a figure derived from a 12-slice arm and then handed to
every arm regardless of its slices, batch, encoder or trainable depth. That is the identical
fitted-on-one-configuration failure `DeviceBudget` has now recorded five times, one layer out in the
shell where the class could not see it. On 2026-08-21 it parked `encoder_adapt_thawed` — which reserves
9.6 GB at batch 2 — behind an 18 GB gate while 15.8 GB of the card sat free, and the wait bound is an
hour.

It prints ONE integer and nothing else, because the caller is `$(...)` in bash. A recipe whose slices
are unknown derives 0 (see `DeviceBudget.needed`), and 0 is a truthful answer the shell can fall back
on rather than a failure it has to parse.
"""

import math
import sys

from kneemri.training.options import TrainingOptions
from kneemri.training.run_setup import RunSetup

MIB_PER_GB = 1024
"""Both quantities are binary — `DeviceBudget` works in GiB and `nvidia-smi` reports MiB."""


def main() -> None:
    need = RunSetup(TrainingOptions.parse()).budget().needed()
    # CEILING, because this number is compared against free VRAM and the comparison is one-sided: a MiB
    # rounded down is a MiB the arm will ask the allocator for and the gate did not require.
    sys.stdout.write(f"{math.ceil(need * MIB_PER_GB)}\n")


if __name__ == "__main__":
    main()
