"""One arm read against another — the undeclared control diff, and the outright duplicate."""

from pathlib import Path

from kneemri.experiments.arm_diff import ArmDiff
from kneemri.experiments.arm_header import ArmHeader


def test_values():
    """A flag with a value, a bare switch, and a flag whose value is the next flag rather than a value."""
    parsed = ArmDiff.values("--epochs 8 --one-cycle --out run.pt")

    assert parsed == {"--epochs": "8", "--one-cycle": "", "--out": "run.pt"}


def test_mismatched(tmp_path: Path):
    """The defect that cost 75 minutes of card, and the three ways an arm is allowed to differ anyway.

    Partitioned on what a flag does relative to the control it names: OMITTED (the real defect —
    `distill_pairs` dropped `--backbone-lr` and silently took the trainer's 8e-6 against a control at
    4e-5), ADDED without a declaration, DECLARED with a `# deliberate:` line, INCIDENTAL (`--out`, which
    differs between any two arms by construction), and EQUAL-BUT-SPELLED-DIFFERENTLY (`4e-5` against
    `4e-05`, the same learning rate).
    """
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "silent.sh").write_text("# control  base = 0.85\n", encoding="utf-8")
    (arms / "declaring.sh").write_text("# deliberate: --distill-weight\n# control  base = 0.85\n", encoding="utf-8")
    diff = ArmDiff(ArmHeader(arms))
    against = {"base": "--epochs 8 --backbone-lr 4e-5 --out base.pt"}

    found = diff.mismatched(
        {
            "silent": "--epochs 8 --distill-weight 0.3 --out silent.pt",
            "declaring": "--epochs 8 --backbone-lr 4e-05 --distill-weight 0.3 --out declaring.pt",
        },
        against,
    )

    assert found == {"silent": ["--backbone-lr: 4e-5 -> absent", "--distill-weight: absent -> 0.3"]}, (
        "a declared flag, an incidental one and the same rate spelled twice are not differences"
    )


def test_twinned(tmp_path: Path):
    """One recipe under two names is a duplicate; the same identity DECLARED, or from one shell, is not.

    MEASURED 2026-08-30: `recal_band_dense` was written, checked, committed and queued while
    `recal_dense` — the same flag line, already on the card — was the job it would have waited on.
    Partitioned on the three ways two arms can share an identity: undeclared across two shells (the
    defect), declared with `# twin-of:` (a before/after across a code change), and printed by ONE shell
    (two runs whose variable is the environment, which no flag line can carry).
    """
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "copy.sh").write_text("# bead: knee-yuqf\n", encoding="utf-8")
    (arms / "original.sh").write_text("# bead: knee-yuqf\n", encoding="utf-8")
    (arms / "recoded.sh").write_text("# twin-of: staged\n", encoding="utf-8")
    (arms / "staged.sh").write_text("# bead: knee-yuqf\n", encoding="utf-8")
    pair = tmp_path / "throughput.sh"
    pair.write_text("# bead: knee-yuqf\n", encoding="utf-8")
    headers = ArmHeader(arms)
    headers.record("bf16", pair)
    headers.record("compiled", pair)
    diff = ArmDiff(headers)
    duplicated = "--epochs 8 --backbone-lr 4e-5"
    recoded = "--epochs 8 --backbone-lr 1e-4"
    environmental = "--epochs 6 --backbone-lr 4e-5"
    against = {
        "original": f"{duplicated} --out original.pt",
        "copy": f"{duplicated} --out copy.pt",
        "staged": f"{recoded} --out staged.pt",
        "recoded": f"{recoded} --out recoded.pt",
        "bf16": f"{environmental} --out bf16.pt",
        "compiled": f"{environmental} --out compiled.pt",
    }

    found = diff.twinned(against, against)

    assert sorted(found) == ["copy", "original"], (
        "a declared twin and two runs from one shell are deliberate; only the undeclared pair spends a slot twice"
    )
    assert found["copy"].startswith("is original "), "and the sentence names the arm whose recipe it repeats"
