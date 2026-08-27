import json
from pathlib import Path

import torch

from kneemri.experiments.ledger_check import LedgerCheck
from kneemri.runs.run_metrics import RunMetrics


def test_best(tmp_path: Path):
    """The best scored epoch, and None for a run that died before scoring one.

    None rather than 0.0, because a run that never scored is not a run that scored badly — quoting a
    zero in a results ledger would be worse than omitting it.
    """
    scored = tmp_path / "a.metrics.jsonl"
    scored.write_text(
        "\n".join(
            [
                json.dumps({"recipe": {"epochs": "3"}}),
                *(json.dumps({"epoch": e, "auc": auc}) for e, auc in enumerate([0.81, 0.8473, 0.84])),
            ]
        ),
        encoding="utf-8",
    )
    empty = tmp_path / "b.metrics.jsonl"
    empty.write_text("", encoding="utf-8")
    unscored = tmp_path / "c.metrics.jsonl"
    unscored.write_text(json.dumps({"epoch": 0, "auc": float("nan")}), encoding="utf-8")

    assert LedgerCheck(tmp_path / "absent.md").best(scored) == 0.8473, "the recipe line is not an epoch"
    assert LedgerCheck(tmp_path / "absent.md").best(empty) is None
    assert LedgerCheck(tmp_path / "absent.md").best(unscored) is None, (
        "a final fit records NaN by design, and 'nan' quoted in a ledger is never found again"
    )


def test_finished(tmp_path: Path):
    """A run counts as finished on its checkpoint, and the metrics file refines the score.

    Scanning metrics files alone was a pass computed over 15 of 69 runs: 50 checkpoints had no metrics
    file, and 11 of those carried a score written down in no document at all.
    """
    torch.save({"holdout": 0.8143}, tmp_path / "checkpoint_only.pt")
    torch.save({"holdout": 0.81}, tmp_path / "both.pt")
    torch.save({"holdout": float("nan")}, tmp_path / "final_fit.pt")
    torch.save({"holdout": 0.77}, tmp_path / "throughput_bf16.pt")
    torch.save({"holdout": 0.6562}, tmp_path / "smoke_v3base.pt")
    torch.save({"holdout": 0.5639, "options": {"studies": "60"}}, tmp_path / "member_fold0.pt")
    (tmp_path / "both.metrics.jsonl").write_text(
        "\n".join(json.dumps({"epoch": e, "auc": auc}) for e, auc in enumerate([0.81, 0.8473])),
        encoding="utf-8",
    )

    finished = LedgerCheck(tmp_path / "absent.md").finished(tmp_path)

    assert finished["checkpoint_only"] == 0.8143, "a run with no metrics file still ran"
    assert finished["both"] == 0.8473, "the metrics file knows the BEST epoch, the checkpoint only its own"
    assert "final_fit" not in finished, "a final fit records NaN by design and never scored"
    assert "throughput_bf16" not in finished, "speed probes have no score worth quoting in a results ledger"
    assert "smoke_v3base" not in finished, "a smoke scores 12 holdout studies; quoting it would put that in the table"
    assert "member_fold0" not in finished, "the documented smoke writes the DEFAULT name and is caught by its recipe"


def test_truncated(tmp_path: Path):
    """A run that capped the corpus is a smoke by construction; one whose recipe is silent is real.

    The documented pre-flight — `--studies 60 --epochs 1` with no `--out` — writes `member_fold0.pt`,
    a name matching no exemption fragment, so the check exited 1 on a clean repo every time the
    pipeline was smoked. An unrecorded recipe must NOT be read as a smoke: 14 members predate options
    entirely and dropping them would shrink the denominator of the check that exists to police it.
    """
    torch.save({"holdout": 0.56, "options": {"studies": "60"}}, tmp_path / "member_fold0.pt")
    torch.save({"holdout": 0.84, "options": {"studies": "0"}}, tmp_path / "full.pt")
    torch.save({"holdout": 0.84}, tmp_path / "predates.pt")
    check = LedgerCheck(tmp_path / "absent.md")

    assert check.truncated(RunMetrics(tmp_path / "member_fold0.pt"))
    assert not check.truncated(RunMetrics(tmp_path / "full.pt")), "0 is the whole corpus"
    assert not check.truncated(RunMetrics(tmp_path / "predates.pt")), "an unrecorded recipe is not a smoke"


def test_reran(tmp_path: Path):
    """Rotations left beside a run count the EARLIER runs that wrote to its name.

    The checkpoint and the `.holdout.npz` are not rotated, so after a second run the ledger's row can
    quote a best whose artefacts have been replaced — which is what `depth_regions_span` did on
    2026-08-19 when it re-entered `run_train` mid-flight. The current record must not be counted, and a
    run that shares a PREFIX with another must not be counted either: `x` and `x_run1` sit side by side
    in this project precisely because a re-run was preserved by hand.
    """
    (tmp_path / "twice.metrics.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "twice.metrics.1.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "once.metrics.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "once_run1.metrics.jsonl").write_text("", encoding="utf-8")
    check = LedgerCheck(tmp_path / "absent.md")

    assert check.reran(tmp_path, "twice") == 1, "one rotation means one earlier run under this name"
    assert check.reran(tmp_path, "once") == 0, "the live record is not a rotation"
    assert check.reran(tmp_path, "absent") == 0, "a run that never wrote a metrics file never re-ran"


def test_plain(tmp_path: Path):
    """The one normalisation that makes the ledger's prose name and the file's output name equal."""
    check = LedgerCheck(tmp_path / "absent.md")

    assert check.plain("`whole_stack`") == check.plain("WholeStack"), "punctuation and case carry no meaning"
    assert check.plain("0.8473") == "08473", "a score is compared as digits, so its dot must go too"


def test_stem(tmp_path: Path):
    """The seed and fit-regime tail is dropped, and only a tail — a name that merely contains one is not."""
    check = LedgerCheck(tmp_path / "absent.md")

    assert check.stem("warm_calib_dense_s2026") == "warmcalibdense"
    assert check.stem("final_ws_cf") == "finalws"
    assert check.stem("band_dense") == "banddense", "a name with no suffix is only normalised"
    assert check.stem("s2026_probe") == "s2026probe", "the suffix is a tail, never a substring anywhere"


def test_trimmed(tmp_path: Path):
    """The suffix drop before normalisation, so the separators are still there to be counted."""
    check = LedgerCheck(tmp_path / "absent.md")

    assert check.trimmed("warm_calib_dense_s2026") == "warm_calib_dense"
    assert check.trimmed("floor_fold0") == "floor", "what is left is one segment, and that is the point"
    assert check.trimmed("band_dense") == "band_dense", "a name with no suffix is returned whole"


def test_attributed(tmp_path: Path):
    """Name and score on ONE line is attribution; the same two facts on two lines is not.

    This is the case the number-only match cannot see, and it is not hypothetical: six lines quote
    0.8562 and none of them is about `warm_calib_dense`, whose own mention carries no number.
    """
    ledger = tmp_path / "README.md"
    ledger.write_text(
        "\n".join(
            [
                "| `slice_attention` | attention over slices | 0.8473 | kept |",
                "the `warm_calib_dense` arm ran overnight and is discussed below",
                "3 members  0.8562  +0.0034",
            ]
        ),
        encoding="utf-8",
    )
    check = LedgerCheck(ledger)

    assert check.attributed("sliceattn_s2026", 0.8473) is False, "a file name the ledger never spells is not tied"
    assert check.attributed("slice_attention_s2026", 0.8473), "prose name plus its own score on one line"
    assert check.attributed("warm_calib_dense_s2026", 0.8562) is False, (
        "the name and the number are both in the record, and that is exactly what is not enough"
    )


def test_owns(tmp_path: Path):
    """Attribution plus the one extra condition: a one-word stem may not be accused of owning a score."""
    ledger = tmp_path / "README.md"
    ledger.write_text(
        "\n".join(
            [
                "| `band_dense` | twenty slices | 0.8556 | kept |",
                "| + weight floor at 0.25 | 0.8221 | +0.0019 | kept |",
            ]
        ),
        encoding="utf-8",
    )
    check = LedgerCheck(ledger)

    assert check.owns("band_dense_s2026", 0.8556), "two segments and its own number on its own row"
    assert check.attributed("floor_fold0", 0.8221), "the lenient test is satisfied, and wrongly"
    assert check.owns("floor_fold0", 0.8221) is False, (
        "`floor` is prose here, and a one-word stem cannot be told apart from it"
    )


def test_unattributed(tmp_path: Path):
    """Reports the loosely-recorded runs, and names the run that provably owns the number instead."""
    ledger = tmp_path / "README.md"
    ledger.write_text(
        "\n".join(
            [
                "| `band_dense` | twenty slices | 0.8556 | kept |",
                "| `slice_attention` | attention over slices | 0.8473 | kept |",
                "an ensemble of three read 0.8562 on the same studies",
            ]
        ),
        encoding="utf-8",
    )

    loose = LedgerCheck(ledger).unattributed(
        {"band_dense_s2026": 0.8556, "band140_s2026": 0.8556, "warm_calib_dense_s2026": 0.8562}
    )

    assert "band_dense_s2026" not in loose, "its own row states its own number"
    assert loose["band140_s2026"] == ["band_dense_s2026"], "the number vouching for it is provably another run's"
    assert loose["warm_calib_dense_s2026"] == [], "quoted by no run at all — loose, but nobody else's number"


def test_unrecorded(tmp_path: Path):
    """A score the ledger quotes is recorded; one it does not is reported.

    Matched on the NUMBER, not the arm name: the table names arms in prose (`whole_stack`) while the
    files are outputs (`wholestack_s2026`), and a mapping between the two is one more thing to get
    wrong. This is exactly the case that went unnoticed for two days — five arms sat in the ledger as
    "queued" while their metrics files held real scores.
    """
    ledger = tmp_path / "README.md"
    ledger.write_text("| `slice_attention` | attention over slices | 0.8473 | +0.0016 | kept |", encoding="utf-8")
    interpretation = tmp_path / "2026-08-11_what-actually-moved.md"
    interpretation.write_text("| + weight floor at 0.25 | 0.8221 | +0.0019 | kept |", encoding="utf-8")

    missing = LedgerCheck(ledger, interpretation, tmp_path / "absent.md").unrecorded(
        {"sliceattn_s2026": 0.8473, "floor_fold0": 0.8221, "extended_ws_s2026": 0.8459}
    )

    assert "sliceattn_s2026" not in missing, "a score quoted in the ledger is recorded"
    assert "floor_fold0" not in missing, "an interpretation is part of the written record, not a lesser one"
    assert missing == {"extended_ws_s2026": 0.8459}
