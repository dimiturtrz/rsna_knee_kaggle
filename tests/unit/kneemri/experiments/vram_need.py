import math
import sys

import pytest

from kneemri.experiments.vram_need import MIB_PER_GB, main
from kneemri.training.options import TrainingOptions
from kneemri.training.run_setup import RunSetup

THAWED = ["--slices", "20", "--batch", "2", "--unfreeze-last", "12"]
"""`encoder_adapt_thawed`'s own flags — the arm that parked behind the constant with the card free."""


def _printed(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], flags: list[str]) -> str:
    monkeypatch.setattr(sys, "argv", ["vram_need", "--labels", "labels.csv", *flags])
    main()
    return capsys.readouterr().out.strip()


def test_main(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """ONE integer on stdout, because the caller is `$(...)` in bash and cannot parse anything else.

    The gate it feeds asked for a hard-coded 18432 MiB — a 12-slice arm's figure handed to every arm,
    which overstates this batch-2 recipe by 8 GB and understates the shipped 20-slice one by 10.
    """
    thawed = _printed(monkeypatch, capsys, THAWED)
    assert thawed.isdigit(), "a shell substitution has to read it without parsing prose"

    monkeypatch.setattr(sys, "argv", ["vram_need", "--labels", "labels.csv", *THAWED])
    expected = RunSetup(TrainingOptions.parse()).budget().needed()
    assert int(thawed) == math.ceil(expected * MIB_PER_GB), (
        "the gate and the trainer must answer from the SAME cost model, or the constant is back"
    )

    shipped = int(_printed(monkeypatch, capsys, ["--slices", "20"]))
    assert shipped > int(thawed), "the shipped 20-slice recipe at the default batch reserves far more"
