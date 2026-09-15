from pathlib import Path

import pandas as pd
import pytest

from core.data.corpus.crop_centre import Crop
from core.data.corpus.slot_reader import Reading
from core.data.corpus.study_cache import StudyCache
from core.data.labels.ruler import DEFAULT_LABELS, STUDY_COLUMN
from core.paths import COMPETITION
from kneemri.experiments.arm_check import ArmCheck
from tests.unit.conftest import write_series

GOOD = "--fold 0 --epochs 12 --one-cycle --slice-pooling max --whole-stack --labels /tmp/labels.csv"
SLICES = 4
SIZE = 28


def test_refused():
    """A real invocation passes, a typo'd flag is caught, and the parser's own complaint is reported.

    The pure half on purpose: `invocations` shells out to bash, which the unit gate forbids, while this
    is where the logic lives. Splitting them is what makes the check testable at all.
    """
    refused = ArmCheck().refused(
        {
            "good": GOOD,
            "typo": GOOD + " --slice-poolng max",
            "bad_value": "--fold 0 --slice-pooling nonsense",
        }
    )

    assert "good" not in refused, "a real arm's flags must survive the trainer's parser"
    assert "typo" in refused, "a misspelled flag is exactly what this exists to catch before a queue"
    assert "bad_value" in refused, "a value outside the choices is refused too"
    assert refused["typo"], "the parser's own complaint is reported, not a generic failure"


def test_carriage_returns(tmp_path: Path):
    """A CR on disk is reported with its count, and a clean tree is silent.

    Written as BYTES, because the defect only exists on disk: every text-mode write on this platform
    would put the CRs back and every git layer takes them out again, which is exactly why the check
    reads the file rather than the diff.
    """
    arms = tmp_path / "arms"
    arms.mkdir()
    (tmp_path / "lib.sh").write_bytes(b"echo one\n[ -f x ] || { echo two; exit 1; }\n")
    (arms / "windows.sh").write_bytes(b"echo one\r\n[ -f x ] || { echo two; exit 1; }\r\n")
    check = ArmCheck(arms, tmp_path / "stub_lib.sh", tmp_path)

    found = check.carriage_returns()

    assert set(found) == {(arms / "windows.sh").as_posix()}, (
        "only the file with CRs on disk is named, and the sourced lib is walked alongside the arms"
    )
    assert found[(arms / "windows.sh").as_posix()] == 2, "the count is reported, so a partial file is visible"

    (arms / "windows.sh").write_bytes(b"echo one\n")

    assert check.carriage_returns() == {}


def test_overstated():
    """A cap copied off a bigger recipe is reported; one the slices justify, and one absent, are not.

    `depth_regions_narrow` is the live case: `depth_regions` with the slices dropped to 12 and the
    20-slice cap of 28 carried along, ten GB above what a 12-slice step reserves. Free on an empty card
    and a wait-then-crawl once a co-tenant holds a third of it.
    """
    overstated = ArmCheck().overstated(
        {
            "stale_cap": GOOD + " --slices 12 --vram-gb 28",
            "derived": GOOD + " --slices 20 --vram-gb 28",
            "rounded_up": GOOD + " --slices 12 --vram-gb 18",
            "no_cap": GOOD + " --slices 12",
        }
    )

    assert round(overstated["stale_cap"]) == 10, "a 20-slice cap on a 12-slice recipe is what this catches"
    assert "derived" not in overstated, "asking UNDER the derived need is not an overstatement"
    assert "rounded_up" not in overstated, "a cap is a bound, so one slice of rounding is not a defect"
    assert "no_cap" not in overstated, "an arm that names no cap gets the derived one and says nothing"


def test_unstated_reads_a_recipe_file_as_stated(tmp_path: Path):
    """A `--config` file naming the operating point is a statement, not silence."""
    recipe = tmp_path / "r.toml"
    recipe.write_text("[data]\nslices = 20\nband_low = 0.1\n", encoding="utf-8")

    unstated = ArmCheck().unstated({"recipe": f"{GOOD} --config {recipe.as_posix()}"})

    assert unstated["recipe"] == ["--band-high"], "only the key the file left out"


def test_unstated():
    """An arm that names its operating point passes whatever the values are; one that is silent is named.

    The pre-settlement values are NOT what this reports — `narrow_on_purpose` asks for exactly the
    defaults and is clean, because it wrote them down. What it reports is silence, which is the only
    thing that makes a recorded delta unreadable against a later one.
    """
    unstated = ArmCheck().unstated(
        {
            "settled": GOOD + " --slices 20 --band-low 0.1 --band-high 0.9",
            "narrow_on_purpose": GOOD + " --slices 12 --band-low 0.2 --band-high 0.8",
            "half_stated": GOOD + " --slices 20",
            "silent": GOOD,
        }
    )

    assert "settled" not in unstated
    assert "narrow_on_purpose" not in unstated, "the old values, STATED, are a choice rather than a defect"
    assert unstated["half_stated"] == ["--band-low", "--band-high"], "only the flags it left silent"
    assert unstated["silent"] == ["--slices", "--band-low", "--band-high"]


def _label_table(path: Path, acl: list[float | None]) -> Path:
    """A minimal label table: one graded column with its two companions, one row per value."""
    rows = "\n".join(f"study{index},{'' if value is None else value},0.95,YES" for index, value in enumerate(acl))
    path.write_text(f"{STUDY_COLUMN},ACL,ACL__conf,ACL__verdict\n{rows}\n", encoding="utf-8")
    return path


def test_moving_ruler(tmp_path: Path):
    """An arm re-mapping its targets without pinning --eval-labels is named; the three exemptions are not.

    `recal_holdannot`'s shape verbatim (knee-8wa0): `--labels report_labels_v3.csv`, no `--eval-labels`,
    a printed 0.8964 against a 0.8487 control that is two rulers rather than a +0.0477.
    """
    ruler = _label_table(tmp_path / DEFAULT_LABELS, [0.82, 0.28])
    crossed = _label_table(tmp_path / "report_labels_v3.csv", [0.42, 0.28])
    base = "--fold 0 --epochs 12 --one-cycle --slice-pooling max --whole-stack"
    moved = ArmCheck().moving_ruler(
        {
            "recal_holdannot": f"{base} --labels {crossed.as_posix()}",
            "recal_dense": f"{base} --labels {crossed.as_posix()} --eval-labels {ruler.as_posix()}",
            "ordinary": f"{base} --labels {ruler.as_posix()}",
            "silent": base,
            "absent_tables": f"{base} --labels {(tmp_path / 'never_written.csv').as_posix()}",
        }
    )

    assert "report_labels_v3.csv" in moved["recal_holdannot"]
    assert "recal_dense" not in moved, "a pinned --eval-labels is what makes the re-map readable"
    assert "ordinary" not in moved
    assert "silent" not in moved, "an arm naming no table inherits the shared one and moves nothing"
    assert "absent_tables" not in moved, "gitignored tables are absent in CI; a lint must not fail on that"


def test_selected():
    """`--only` narrows to the named arms, passes everything through when unset, and refuses a typo.

    The refusal is the point: a name matching nothing would smoke zero arms and exit 0 saying "0 arms
    checked", which is the failure this module already carries two scars from.
    """
    invocations = {"resolution": GOOD, "slot_mean": GOOD, "final_fit": GOOD}

    assert ArmCheck.selected(invocations, ["slot_mean"]) == {"slot_mean": GOOD}
    assert ArmCheck.selected(invocations, None) == invocations, "naming nothing means every arm"

    with pytest.raises(ValueError, match="slot_men"):
        ArmCheck.selected(invocations, ["slot_men"])


def test_uncached(tmp_path: Path):
    """An arm reading a cache nobody has decoded is named, and one reading a decoded cache is not.

    The failure this catches costs the same as a flag typo: `main` selects the card before it maps the
    cache, so an undecoded reading holds the GPU and then raises on a memory map. It has to stay silent
    without a data root, because `paths.yaml` is gitignored and CI has none.
    """
    cache = StudyCache(tmp_path / "processed" / COMPETITION, slices=SLICES, image_size=SIZE)
    cache.directory.mkdir(parents=True)
    for path in (cache.images_path, cache.mask_path, cache.index_path):
        path.write_text("decoded", encoding="utf-8")
    paths = tmp_path / "paths.yaml"
    paths.write_text(f"data: {tmp_path.as_posix()}\n", encoding="utf-8")
    # POSIX separators: the flags go through `shlex.split`, which eats a Windows backslash as an escape
    # and hands the parser a path that resolves to nothing — under which this test passes vacuously.
    reading = f"{GOOD} --paths {paths.as_posix()} --slices {SLICES} --image-size {SIZE}"

    uncached = ArmCheck().uncached(
        {
            "decoded": reading,
            "other_crop": f"{reading} --crop-mm 80",
            "rootless": f"{GOOD} --paths {(tmp_path / 'absent.yaml').as_posix()}",
        }
    )

    assert "decoded" not in uncached, "the reading this arm asks for is on disk"
    tighter = StudyCache(cache.directory, slices=SLICES, image_size=SIZE, reading=Reading(crop=Crop(mm=80.0)))
    assert uncached["other_crop"] == tighter.key, (
        "a different physical crop is a different cache, and the key is what says which"
    )
    assert "rootless" not in uncached, "no data root resolves, so nothing can be claimed about it"


def _rooted(tmp_path: Path) -> str:
    """An arm's flags pointed at a real data root, at a resolution a test can afford to decode."""
    paths = tmp_path / "paths.yaml"
    paths.write_text(f"data: {tmp_path.as_posix()}\n", encoding="utf-8")
    # POSIX separators: the flags go through `shlex.split`, which eats a Windows backslash as an escape.
    return f"{GOOD} --paths {paths.as_posix()} --slices {SLICES} --image-size {SIZE}"


def test_readings(tmp_path: Path):
    """Every accepted arm's root and cache, resolved once — the parse both consumers would repeat."""
    rootless = f"{GOOD} --paths {(tmp_path / 'absent.yaml').as_posix()}"
    readings = ArmCheck().readings({"real": _rooted(tmp_path), "rootless": rootless, "typo": f"{GOOD} --fld 0"})

    assert set(readings) == {"real"}, "no data root and no valid flags are both silence, not a claim"
    assert readings["real"].key == StudyCache(tmp_path, slices=SLICES, image_size=SIZE).key


def test_key(tmp_path: Path):
    """A reading names the file it would read, which is the string an UNCACHED report prints."""
    reading = ArmCheck().readings({"real": _rooted(tmp_path)})["real"]

    assert reading.key == f"{SIZE}px_{SLICES}sl"


def test_decoded(tmp_path: Path):
    """A reading answers whether its cache is on disk, rather than three path checks at each caller."""
    reading = ArmCheck().readings({"real": _rooted(tmp_path)})["real"]
    assert not reading.decoded()

    cache = StudyCache(tmp_path / "processed" / COMPETITION, slices=SLICES, image_size=SIZE)
    cache.directory.mkdir(parents=True, exist_ok=True)
    for path in (cache.images_path, cache.mask_path, cache.index_path):
        path.write_text("decoded", encoding="utf-8")

    assert reading.decoded()


def test_truncated(tmp_path: Path):
    """The smoke's reading is the arm's own over the first N studies — a different file, by name."""
    reading = ArmCheck().readings({"real": _rooted(tmp_path)})["real"]

    smoked = reading.truncated(2)

    assert smoked.key == f"{SIZE}px_{SLICES}sl_first2", "the truncation is in the name, not only the contents"
    assert reading.key != smoked.key, "so a smoke decode can never land on the corpus it is a prefix of"


def _corpus(tmp_path: Path, studies: int) -> None:
    """A raw root with the two shipped tables and one series per study, as a decode expects to find it."""
    raw = tmp_path / "raw" / COMPETITION
    identifiers = [f"study{index}" for index in range(studies)]
    for identifier in identifiers:
        write_series(raw / "train_series" / identifier / "sag", slices=4, size=32)
    pd.DataFrame({"StudyInstanceUID": identifiers}).to_csv(raw / "train.csv", index=False)
    # Keyed by the series DIRECTORY, which is what `Study.from_directory` looks the plane up by — a
    # table naming anything else routes every series nowhere and the decode reports a corpus of zeros.
    pd.DataFrame({"SeriesInstanceUID": ["sag"], "Anatomical_Plane": ["Sagittal"]}).to_csv(
        raw / "train_series.csv", index=False
    )


def test_decode(tmp_path: Path):
    """A reading decodes ITS OWN cache out of raw, which is what makes an untried axis smokeable."""
    _corpus(tmp_path, studies=2)
    reading = ArmCheck().readings({"real": _rooted(tmp_path)})["real"].truncated(1)

    # ONE worker: the pool is what a decode costs, not the pixels. Eight interpreters re-importing
    # torch under Windows' spawn, for two studies, once cost a unit run seven and a half hours.
    reading.decode(workers=1)

    assert reading.decoded(), "all three files, so the smoke that follows has something to map"


def test_decode_smoke_caches(tmp_path: Path):
    """The truncated cache is built before the smoke, once per reading, and never for a refused arm."""
    _corpus(tmp_path, studies=2)
    flags = _rooted(tmp_path)
    check = ArmCheck()

    built = check.decode_smoke_caches(
        {"real": flags, "same": flags, "typo": f"{flags} --fld 0", "broken": flags},
        {"broken": "refused by the parser"},
        first=1,
        workers=1,
    )

    assert built == [f"{SIZE}px_{SLICES}sl_first1"], "two arms on one reading decode it once, and only once"
    assert check.readings({"real": flags})["real"].truncated(1).decoded()
    assert not check.readings({"real": flags})["real"].decoded(), "the FULL corpus is untouched by a smoke"


def test_invocations(tmp_path: Path):
    """An arm is SOURCED, so the shell finds what only the shell can — and its flags come back.

    Sourcing rather than reading the file is the point: an arm lost two hours to a quoting error once,
    and no regex finds that. Arms that never call run_train are absent rather than reported, because a
    lint that complains about files it cannot check gets switched off.
    """
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "real.sh").write_text(
        'source "${ARM_LIB}"\nrequire "$LABELS"\nrun_train "real" --fold 0 --epochs 8\n', encoding="utf-8"
    )
    (arms / "no_train.sh").write_text('source "${ARM_LIB}"\necho "this arm drives uv directly"\n', encoding="utf-8")
    # An arm its own gate refuted: still committed as the record of what was tried, never run again.
    (arms / "retired.sh").write_text(
        'source "${ARM_LIB}"\nretired "gated out"\nrun_train "retired" --fold 0 --epochs 8\n', encoding="utf-8"
    )

    found = ArmCheck(arms, Path("experiments/stub_lib.sh")).invocations()

    assert found == {"real": "'--fold' '0' '--epochs' '8'"}, (
        "a retired arm reaches no parser, so it is never reported UNCACHED for a cache nobody should build"
    )


def test_invocations_never_spawns_an_arm_the_stub_cannot_shadow(tmp_path: Path):
    """An arm naming an interpreter is NEVER SPAWNED — the stub shadows `uv` and nothing else.

    The file below would put a CUDA job on the card if this check ran it, which is exactly what happened
    on 2026-09-09: three `foreign_readout` processes, 19.4 and 12.3 GB resident, sixteen seconds after
    `arm_check --only` started, beside a live training arm. It is written to write a MARKER rather than
    to touch the device, so the assertion is that the marker is absent — proof the file was not executed,
    not merely that nothing was reported from it.
    """
    arms = tmp_path / "arms"
    arms.mkdir()
    ran = tmp_path / "it_ran"
    (arms / "direct.sh").write_text(
        f'source "${{ARM_LIB}}"\ntouch "{ran.as_posix()}"\n"$REPO/.venv/Scripts/python.exe" -m whatever\n'
        'run_train "direct" --fold 0 --epochs 8\n',
        encoding="utf-8",
    )
    (arms / "safe.sh").write_text('source "${ARM_LIB}"\nrun_train "safe" --fold 0 --epochs 8\n', encoding="utf-8")
    check = ArmCheck(arms, Path("experiments/stub_lib.sh"))

    assert check.invocations() == {"safe": "'--fold' '0' '--epochs' '8'"}
    assert not ran.exists(), "the guard has to stop the SPAWN, not merely drop what the spawn printed"


def test_queued(tmp_path: Path):
    """An arm carrying ARGUMENTS is checked on the flags those arguments produce, not on its defaults.

    `after.sh` queues an arm as one argv element with its own arguments, and every other check here runs
    the arm bare — so a non-default rung was queued on flags nothing had parsed. Entries are taken in
    that same spelling, so what is checked is the string that will be queued.
    """
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "rung.sh").write_text(
        'source "${ARM_LIB}"\nrun_train "rung_${1:-8}" --fold 0 --epochs "${1:-8}"\n', encoding="utf-8"
    )
    check = ArmCheck(arms, Path("experiments/stub_lib.sh"))

    assert check.queued([f"{(arms / 'rung.sh').as_posix()} 12"]) == {"rung_12": "'--fold' '0' '--epochs' '12'"}
    assert check.invocations() == {"rung_8": "'--fold' '0' '--epochs' '8'"}, "bare stays bare — the default rung"

    # An arm's NAME where a queue entry belongs: bash runs nothing, the invocation set empties, and the
    # complaint used to surface in `selected` as a missing arm — blaming `--only` for a `--as-queued` typo.
    with pytest.raises(ValueError, match="SCRIPT PATH"):
        check.queued(["rung"])


def test_unpriced():
    """The BOTH-MARKS boundary: a cost alone is exactly the argument that let an unpriced arm run.

    Partitioned on which of the two marks a bead's text carries — neither, cost only, odds only, both.
    Only the last is priced. `bd show` output is passed in as text rather than read here, so the detector
    stays pure and this test shells out to nothing.
    """
    priced = "PRICE: 40 min of card. P(clears 1.5%) ~ 0.2, so the expected score is an instrument."

    assert ArmCheck.unpriced(
        {
            "knee-both": priced,
            "knee-cost": "PRICE: 40 min of card, and it is the obvious next thing to run.",
            "knee-odds": "P(clears 1.5%) ~ 0.2 — worth it if the card is free.",
            "knee-neither": "Try the deeper mixer and see what happens.",
        }
    ) == ["knee-cost", "knee-neither", "knee-odds"], "a cost with no expected gain is not a price"

    assert ArmCheck.unpriced({"knee-both": priced}) == []


def test_prices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The JOIN of the two detectors either side of the tracker read: who named no bead, whose is unpriced.

    The tracker read is stubbed rather than run — a unit test may not shell out, and what is under test is
    the join, not `bd`. It returns verdicts rather than a refusal because whether this run is ARMING is not
    something this method can see; `main` composes that in one place.
    """
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "priced.sh").write_text("# bead: knee-yuqf\nrun_train priced\n", encoding="utf-8")
    (arms / "cheap.sh").write_text("# bead: knee-cost\nrun_train cheap\n", encoding="utf-8")
    (arms / "bare.sh").write_text("run_train bare\n", encoding="utf-8")
    monkeypatch.setattr(
        ArmCheck,
        "_descriptions",
        staticmethod(lambda beads: dict.fromkeys(beads, "PRICE: 40 min. P(clears 1.5%) ~ 0.2")),
    )
    check = ArmCheck(arms, tmp_path / "stub_lib.sh", tmp_path)
    all_three = {"priced": "--epochs 8", "cheap": "--epochs 8", "bare": "--epochs 8"}

    unbeaded, unpriced = check.prices(all_three)

    assert (unbeaded.label, unpriced.label) == ("UNBEADED", "UNPRICED")
    assert set(unbeaded.found) == {"bare"}, "an arm whose header names no bead"
    assert unpriced.found == {}, "every bead that IS named states both marks"

    monkeypatch.setattr(ArmCheck, "_descriptions", staticmethod(lambda beads: dict.fromkeys(beads, "just run it")))

    assert set(check.prices(all_three)[1].found) == {"priced", "cheap"}, "a bead with neither mark is unpriced"


def test_verdicts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Every check this tool runs, with the POLICY each one carries — the composition, not the detectors.

    Each detector is partitioned by its own test; what is under test here is that the ten are all
    present and that each is handed the refusal policy the tool's rules give it. That mapping is the part
    a refactor can silently invert — a REFUSED that stops refusing, or an OVERSTATED that starts, would
    leave every detector test green.
    """
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "silent.sh").write_text("run_train silent\n", encoding="utf-8")
    monkeypatch.setattr(ArmCheck, "_descriptions", staticmethod(lambda beads: dict.fromkeys(beads, "")))
    # An empty shell root, because `Path.write_text` translates newlines on Windows: the arm above lands
    # on disk as CRLF and CRLF is exactly what one of these eight checks is looking for.
    check = ArmCheck(arms, tmp_path / "stub_lib.sh", tmp_path / "shells")

    verdicts = check.verdicts({"silent": "--epochs 8"}, {}, {}, {}, require_caches=False)
    policy = {verdict.label: (verdict.refuses, verdict.arming_refuses) for verdict in verdicts}

    assert list(policy) == [
        "REFUSED",
        "UNCACHED",
        "CRLF",
        "OVERSTATED",
        "UNSTATED",
        "UNMET-GATE",
        "MISMATCHED",
        "MOVING-RULER",
        "TWIN",
        "UNBEADED",
        "UNPRICED",
    ]
    assert policy["REFUSED"] == (True, False) and policy["CRLF"] == (True, False), (
        "a flag typo and a destroyed completion marker each cost a slot on any run"
    )
    assert policy["OVERSTATED"] == (False, False) and policy["UNSTATED"] == (False, False), (
        "wall-clock only: neither mismatches an arm to its control nor invalidates a past number"
    )
    assert policy["MISMATCHED"] == policy["UNBEADED"] == policy["UNPRICED"] == (False, True), (
        "the advisory rules stop MOMENTUM, so they refuse the run that arms an arm and no other"
    )
    assert policy["MOVING-RULER"] == (True, False), (
        "a moved ruler does not merely mis-arm: it makes the run's own number unreadable, so it refuses "
        "on any run the way a flag typo does"
    )
    assert policy["TWIN"] == (True, False), (
        "one recipe under two names refuses on any run: the card it spends is spent whether or not the "
        "duplicate is the arm being armed"
    )
    assert policy["UNMET-GATE"] == (False, True), (
        "an unmet gate is a fact about ONE arm: it must not fail a sweep, and must stop that arm arming"
    )

    cached = check.verdicts({"silent": "--epochs 8"}, {}, {}, {}, require_caches=True)

    assert [verdict.refuses for verdict in cached if verdict.label == "UNCACHED"] == [True], (
        "--require-caches is the one policy a run's own flags decide: a queue that builds no cache"
    )
    assert {verdict.label for verdict in verdicts if verdict.found} == {"UNSTATED", "UNBEADED"}, (
        "the arm states no operating point and names no bead; nothing else has a finding to make"
    )
