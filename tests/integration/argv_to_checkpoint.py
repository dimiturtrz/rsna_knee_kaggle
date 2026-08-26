"""The command line to a checkpoint: `train.main` itself, on a synthetic corpus, on the CPU.

Every piece of the run's setup is unit-tested and `corpus_to_checkpoint` fits a member through the real
setup methods — but `main` is the function that WIRES them, and it was executed by nothing. That is the
half of this repo's failure class no unit test can see: nine times a value was recorded in a config
while its consumer read its own default, and the way it happens is a flag that reaches the options
object and then never reaches the thing it configures. `RunSetup.trainer` attaching a policy proves
nothing if `main` stops calling it.

So this drives the real entry point from argv: cache load, label pairing, fold assignment, model build,
trainer build, the fitting loop and the final save. The assertion is that the checkpoint records the
reading the COMMAND LINE asked for — the flags that select pixels, because a member served under a
reading it was not fitted under is the defect that reached the leaderboard.
"""

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import pandas as pd
import pytest
import torch

from core.data.corpus.crop_centre import CENTRES, DEFAULT_CENTRE, Crop
from core.data.corpus.folds import ReportFolds
from core.data.corpus.slot_reader import ADJACENT, IMAGE_SIZE, SPREAD, Band, Reading
from core.data.corpus.slots import CANONICAL_SCHEME, DETAIL_MM, LAST_MAXIMAL, MULTISCALE_SCHEME, SCHEMES, Routing
from core.data.corpus.study_cache import StudyCache
from core.findings import TARGETS
from core.paths import COMPETITION
from kneemri.inference.members import Member, MemberConfig
from kneemri.models.backbone import ENCODERS, SMALL
from kneemri.models.model import StudyDesign
from kneemri.models.slice_pooling import MAXIMUM
from kneemri.models.slot_head import AVERAGED, REPORT_DIMS
from kneemri.models.token_pooling import CLS_MEAN, CLS_MEAN_FOCAL, POOL_PARTS
from kneemri.training import train as train_module
from kneemri.training.device_budget import DeviceBudget
from kneemri.training.options import TrainingOptions

from ..unit.kneemri.models.model import DIM, _BlockedBackbone, _FakeBackbone

STUDIES = 8
SLICES = 4
SIZE = 28
"""A multiple of the patch grid, because `main` refuses a read size the encoder cannot tile."""
FOLDS = 2
SPLIT_SEED = 2026
"""Passed on the argv too, so the labels are balanced against the split the run actually uses."""
READ_SIZE = 14
"""One patch, so it tiles — and NOT `SIZE`, so a dropped `train_size` stamp cannot hide behind `img`."""
TIGHT_MM = 80.0
"""The tight crop an arm would actually ask for, which is a different cache than the historic 130 mm."""
TISSUE = next(name for name in sorted(CENTRES) if name != DEFAULT_CENTRE)
"""The centre a run must state to be served about it; taken from the table so a third one is noticed."""
WIDE_BAND = Band(0.10, 0.90)
"""A band away from the historic one, which is a different cache and a different set of slices."""


def _corpus(
    root: Path,
    routing: Routing | None = None,
    crop: Crop | None = None,
    band: Band | None = None,
    spacing: str = SPREAD,
    first: int = 0,
) -> list[str]:
    """Write the three files `StudyCache.load` reads, under the key the argv below selects.

    The routing, the crop, the band and the spacing are parameters because they are part of that key: an arm asking
    for the other tie rule, another physical extent or another slice band reads a DIFFERENT cache, which
    is the property that keeps two readings from overwriting each other.
    """
    identifiers = [f"s{index}" for index in range(STUDIES)]
    cache = StudyCache(
        root / "processed" / COMPETITION,
        slices=SLICES,
        image_size=SIZE,
        routing=routing or Routing(),
        reading=Reading(crop=crop or Crop(), band=band or Band(), spacing=spacing),
        first=first,
    )
    cache.directory.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(2026)
    np.save(cache.images_path, generator.integers(0, 256, cache.shape(STUDIES), dtype=np.uint8))
    np.save(cache.mask_path, np.ones((STUDIES, cache.slots), dtype=np.float32))
    cache.index_path.write_text(json.dumps(identifiers), encoding="utf-8")
    return identifiers


def _tables(root: Path, identifiers: list[str]) -> Path:
    """The report-derived labels and the shipped `train.csv` the folds are hashed from.

    The findings are laid out AGAINST the fold assignment rather than against the study order, so each
    fold holds both classes in every column. A holdout of one class scores NaN in every label, which
    ranks no epoch, which raises before the run reaches the save this test exists to check.
    """
    raw = root / "raw" / COMPETITION
    raw.mkdir(parents=True, exist_ok=True)
    # Distinct report text per study: two studies sharing a report share a fold by construction.
    table = raw / "train.csv"
    reports = pd.DataFrame({"StudyInstanceUID": identifiers, "Report": [f"report {name}" for name in identifiers]})
    reports.to_csv(table, index=False)

    assignments = ReportFolds(FOLDS, SPLIT_SEED).assign(str(table), identifiers)
    positive = np.zeros(len(identifiers), dtype=np.float32)
    for fold in range(FOLDS):
        members = np.flatnonzero(assignments == fold)
        positive[members[: len(members) // 2]] = 1.0
    labels = raw / "labels.csv"
    scores = dict.fromkeys(TARGETS, positive)
    pd.DataFrame({"StudyInstanceUID": identifiers, **scores}).to_csv(labels, index=False)
    return labels


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data root `main` can run against, with the weights and the card taken out of its way."""
    _tables(tmp_path, _corpus(tmp_path))
    paths = tmp_path / "paths.yaml"
    paths.write_text(f"data: {tmp_path.as_posix()}\n", encoding="utf-8")

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda repository: _FakeBackbone())
    # The stand-in backbone is DIM wide, and the head is built at the ENCODER's width: a mismatch here
    # is a shape error rather than a test of the wiring.
    monkeypatch.setattr(train_module, "ENCODERS", {SMALL: replace(ENCODERS[SMALL], dim=DIM)})
    # This suite never trains on the card, and `DeviceBudget` takes it whenever it is there.
    # The count, not availability: `VisibleCards` asks how many cards are VISIBLE, so patching
    # availability alone would leave the run on cuda while claiming to be on cpu.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    return tmp_path


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *flags: str) -> dict[str, object]:
    """Run `main` from an argv and read back the checkpoint it wrote."""
    destination = tmp_path / "member.pt"
    argv = [
        "train",
        "--paths",
        str(tmp_path / "paths.yaml"),
        "--labels",
        str(tmp_path / "raw" / COMPETITION / "labels.csv"),
        "--out",
        str(destination),
        "--epochs",
        "1",
        "--batch",
        "2",
        "--folds",
        str(FOLDS),
        "--split-seed",
        str(SPLIT_SEED),
        "--slices",
        str(SLICES),
        "--image-size",
        str(SIZE),
        *flags,
    ]
    monkeypatch.setattr("sys.argv", argv)
    train_module.main()
    return cast(dict[str, object], torch.load(destination, map_location="cpu", weights_only=False))


def _member(config: dict[str, object]) -> Member:
    """A member reading a config, with the fields no reading is derived from left empty."""
    return Member(
        identifier="x",
        file=Path("m_x.pt"),
        fold=0,
        config=cast(MemberConfig, config),
        fingerprint=np.zeros((1, 1), np.float32),
        holdout=0.0,
        annot=0.0,
    )


def _nested(payload: dict[str, object], key: str) -> dict[str, object]:
    """One of the checkpoint's nested tables, as the loaders read it."""
    return cast(dict[str, object], payload[key])


def test_the_command_line_reaches_the_checkpoint(corpus: Path, monkeypatch: pytest.MonkeyPatch):
    """The flags that decide WHICH PIXELS a member reads must arrive on the member.

    Each of these travels through a different part of `main`: `--whole-stack` through `RunSetup.trainer`,
    `--slice-pooling` through the `StudyDesign` built here, `--crop-mm`/`--tie` through the cache key
    and the record. A flag parsed and then dropped is this project's most expensive failure — it does
    not crash, it trains a recipe nobody asked for and reports a plausible number for it.
    """
    _corpus(corpus, Routing(tie=LAST_MAXIMAL))
    payload = _run(corpus, monkeypatch, "--whole-stack", "--slice-pooling", "max", "--tie", LAST_MAXIMAL)

    config = _nested(payload, "config")
    assert config["whole_stack"] is True, "a whole-stack member served in windows predicts from other pixels"
    assert config["slice_pooling"] == "max"
    assert config["tie"] == "last", "the router is what fills each slot, so serving cannot fall back to ours"
    assert config["slices"] == SLICES
    assert config["train_size"] == SIZE, "0 means the cache's own size, and the member must state which that was"
    assert payload["fingerprint"], "a member without one cannot be verified, so it cannot be submitted"
    assert _nested(payload, "fingerprint_config")["n_slot"] == len(SCHEMES[CANONICAL_SCHEME])


def test_the_pool_reaches_the_model_and_the_manifest(corpus: Path, monkeypatch: pytest.MonkeyPatch):
    """`--pool` must change the network AND the field the serve path rebuilds that network from.

    The manifest stamped the string `cls_mean` as a LITERAL, three lines under the comment explaining
    why `crop_mm` must be the run's own value — and `Member.pool` reads this field back to rebuild the
    design. The trainer had no `--pool` at all, so the focal pooling existed in the model and no arm
    could select it. Both halves are asserted here: the width the checkpoint carries is what the
    pooling produces, and the name travels with it.
    """
    _corpus(corpus)
    payload = _run(corpus, monkeypatch, "--pool", CLS_MEAN_FOCAL)

    assert _nested(payload, "config")["pool"] == CLS_MEAN_FOCAL, "a member is rebuilt from this field"
    state = _nested(payload, "model")
    width = cast(torch.Tensor, state["head.proj.0.weight"]).shape[0]
    assert width == DIM * POOL_PARTS[CLS_MEAN_FOCAL], "the head is built for the parts the forward pass concatenates"
    assert width != DIM * POOL_PARTS[CLS_MEAN], "a pooling that does not change the network is a flag that was dropped"


def test_every_design_switch_survives_argv_to_rebuild(corpus: Path, monkeypatch: pytest.MonkeyPatch):
    """Every field of `StudyDesign` must travel argv -> manifest -> `from_config` and come back itself.

    The ratchet the nine defects wanted. Each switch is asserted by ITERATING the dataclass rather than
    by naming the five that exist today: a field added to the design and not stamped by the record
    comes back as its default here, which is precisely the failure — a member rebuilt as a different
    function whose weights load without complaint — and it fails at the run that adds the field rather
    than at the arm that first uses it.

    Every value on this command line is chosen AWAY from its default for the same reason: a switch that
    happens to agree with the default proves nothing about whether it was carried.
    """
    _corpus(corpus)
    payload = _run(
        corpus,
        monkeypatch,
        *("--pool", CLS_MEAN_FOCAL),
        *("--slice-pooling", MAXIMUM),
        *("--slot-pooling", AVERAGED),
        *("--encoder-layer", "-2"),
        *("--slice-regions", "2"),
        "--anatomical-prior",
        "--through-plane",
    )

    rebuilt = StudyDesign.from_config(_nested(payload, "config"))
    assert rebuilt == StudyDesign(
        pool=CLS_MEAN_FOCAL,
        slice_pooling=MAXIMUM,
        slot_pooling=AVERAGED,
        anatomical_prior=True,
        layer=-2,
        slice_regions=2,
        through_plane=True,
    )
    for field in fields(StudyDesign):
        assert getattr(rebuilt, field.name) != field.default, (
            f"{field.name} came back as its default, so the run's own value never reached the manifest"
        )


FIXED_BY_THE_BUILD = {"group", "variant"}
"""Readings no command line can move, so a run cannot state them away from a legacy manifest's answer.

`group` is a module constant in the record — the encoder takes three channels — and there is no flag
for it. `variant` has one entry in the encoder table this suite patches, since a second would need a
second set of weights to download. `band` left this set on 2026-08-15, when it stopped being stamped
from `SLICE_BAND` and became the run's own value like every rule beside it. Anything else
that lands in this set is a switch losing its ratchet, which is what the set exists to make visible.
"""


def test_every_reading_survives_argv_to_the_member(corpus: Path, monkeypatch: pytest.MonkeyPatch):
    """Every reading `Member` exposes must come back as the run's own, not as a legacy manifest's default.

    The other half of the design ratchet, and the half that has actually cost submissions: eight of the
    nine defects were a READING — the crop, the depth, the tie rule, the precision, the whole-stack flag
    — recorded faithfully and then answered by a consumer's own default. Every one of those accessors
    carries a default for manifests that predate it, which is correct and is also exactly what hides a
    stamp that stopped being written.

    So this compares against a member built from a config carrying ONLY the keys with no default, and
    walks `Member`'s properties rather than naming them: each must differ from what the legacy manifest
    answers. A property added tomorrow whose value never reaches the record answers identically here and
    fails at that commit. The three that no argv can move are named, with the reason, in
    `FIXED_BY_THE_BUILD` — a switch that ends up there has to be put there deliberately.
    """
    # The blocked stand-in, because `--lora-rank` wraps the Linears INSIDE a block and the plain one
    # has none — an encoder shape, not a preference, and adaptation refuses a backbone it cannot adapt.
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda repository: _BlockedBackbone())
    _corpus(
        corpus,
        Routing(scheme=MULTISCALE_SCHEME, tie=LAST_MAXIMAL),
        Crop(mm=TIGHT_MM, centre=TISSUE),
        WIDE_BAND,
        ADJACENT,
        first=STUDIES,
    )
    payload = _run(
        corpus,
        monkeypatch,
        *("--scheme", MULTISCALE_SCHEME),
        *("--tie", LAST_MAXIMAL),
        *("--crop-mm", str(TIGHT_MM)),
        *("--centre", TISSUE),
        *("--band-low", str(WIDE_BAND.low)),
        *("--band-high", str(WIDE_BAND.high)),
        *("--spacing", ADJACENT),
        *("--train-size", str(READ_SIZE)),
        # A TRUNCATED cache, which is a different file and a different index — the axis a smoke run
        # turns, and one a member re-read by a local tool has to answer for or it reads other pixels.
        *("--cache-studies", str(STUDIES)),
        *("--pool", CLS_MEAN_FOCAL),
        *("--slice-pooling", MAXIMUM),
        *("--slot-pooling", AVERAGED),
        *("--encoder-layer", "-2"),
        *("--lora-rank", "4"),
        # No `--anatomical-prior`: the prior is defined over the canonical six slots and this run is
        # multiscale. It is not a reading either — `Member` exposes no accessor for it — so the design
        # ratchet above is where it belongs.
        "--whole-stack",
    )

    stated = _member(_nested(payload, "config"))
    legacy = _member(
        {
            "img": IMAGE_SIZE,
            "group": stated.config["group"],
            "slots": [slot.name for slot in SCHEMES[CANONICAL_SCHEME]],
        }
    )
    # The walk below compares against a LEGACY manifest, so a reading that would differ from it through
    # some OTHER key is invisible to it: `read_size` falls back to `img`, and both are away from 336
    # here, so dropping the `train_size` stamp passes the walk. Mutation-checked, and this is the line
    # that catches it — the two sizes must be each other's, not merely not the legacy one.
    assert (stated.image_size, stated.read_size) == (SIZE, READ_SIZE), "a resampled member reads its OWN size"

    readings = [name for name, value in vars(Member).items() if isinstance(value, property)]
    assert set(FIXED_BY_THE_BUILD) < set(readings), "a name here that is not a reading is a stale exemption"
    for name in readings:
        if name in FIXED_BY_THE_BUILD:
            continue
        assert getattr(stated, name) != getattr(legacy, name), (
            f"{name} answers what a manifest predating it means, so the run's own value never arrived"
        )


def test_the_scheme_reaches_the_model_and_the_member(corpus: Path, monkeypatch: pytest.MonkeyPatch):
    """`--scheme` decides three things at once, and each is a separate chance to drop it.

    It selects the CACHE (the key carries a routing suffix), the SLOT COUNT the model is built with,
    and the slot NAMES recorded on the member — which is what `Member.scheme` matches back to find the
    cache at serving time. A run whose model kept six slots while its member claimed twelve loads
    without complaint and reads half the pixels it says it does.

    Multiscale is the case worth driving because it is the newest and the only one where the slot axis
    carries two physical extents rather than two sequences.
    """
    _corpus(corpus, Routing(scheme=MULTISCALE_SCHEME))
    payload = _run(corpus, monkeypatch, "--scheme", MULTISCALE_SCHEME)

    expected = [slot.name for slot in SCHEMES[MULTISCALE_SCHEME]]
    assert _nested(payload, "fingerprint_config")["n_slot"] == len(expected), "the MODEL's own slot axis"
    assert _nested(payload, "config")["slots"] == expected
    assert [name for name in expected if name.endswith(f"@{DETAIL_MM:g}")], "the tight half is what it is for"


def test_the_run_keeps_an_epoch_and_saves_it(corpus: Path, monkeypatch: pytest.MonkeyPatch):
    """The final save is the KEPT epoch's, with the score and the witness that ranked it.

    A run once left its best epoch as weights with no fingerprint and no recorded reading, and the
    witness is what `witness_check` re-derives before a submission — an empty one is a member that
    cannot be checked against the cache a kernel builds.
    """
    payload = _run(corpus, monkeypatch, "--whole-stack", "--slice-pooling", "max")

    assert payload["epoch"] == 0
    assert isinstance(payload["holdout"], float)
    witness = _nested(_nested(payload, "config"), "witness")
    studies = cast(list[str], witness["studies"])
    assert studies, "the witness names studies, so it travels to another process"
    assert len(cast(list[object], witness["predictions"])) == len(studies)
    assert payload["model"], "the weights the kept epoch produced"


def test_an_undecoded_reading_fails_before_the_card_is_taken(corpus: Path, monkeypatch: pytest.MonkeyPatch):
    """A cache that does not exist must not cost the GPU to discover.

    `--crop-mm` selects a different cache file, and nothing has been decoded at that extent here — the
    same position `crop80_tissue` is in on disk. The device used to be selected first, so an arm queued
    with an undecoded reading held the card and then raised on a memory map.
    """

    def refuse() -> torch.device:
        raise AssertionError("the card was taken before the cache was mapped")

    monkeypatch.setattr(DeviceBudget, "select", lambda self: refuse())

    with pytest.raises(FileNotFoundError):
        _run(corpus, monkeypatch, "--crop-mm", "80")


def test_a_full_fit_holds_nothing_back(corpus: Path, monkeypatch: pytest.MonkeyPatch):
    """`--fold -1` is the final-fit mode: no fold is ever assigned -1, so the holdout is empty.

    It cannot report a score by construction, which is the one path through `Fitting` where the kept
    epoch is the LAST rather than the best — and a run that raised here would be discovered after the
    card had already been held for an hour.

    It DOES still witness how it must be read. This assertion used to demand the opposite — an empty
    witness, "because a run with no holdout has nothing to witness" — and that sentence is knee-52mu:
    five all-data members reached a live board entry unverifiable at serve time, because a replay
    fingerprint was assumed to need unseen studies when it never did.
    """
    payload = _run(corpus, monkeypatch, "--fold", "-1")

    witness = cast("dict[str, object]", _nested(payload, "config")["witness"])
    assert witness["studies"], "named training studies, replayed through the training path"
    assert np.isnan(cast("float", payload["holdout"])), "and still no score, because it held nothing back"


class _WatchedOptions(TrainingOptions):
    """`TrainingOptions` that records which fields the run actually READ.

    Not which it stamped. `metrics.recipe()` walks `vars(options)`, which reads `__dict__` directly and
    so never reaches here — exactly the discrimination this needs, because a field that appears in every
    ledger row and reaches no consumer is the shape of nine defects this repo has already paid for.
    """

    reads: ClassVar[set[str]] = set()

    def __getattribute__(self, name: str) -> object:
        if not name.startswith("_"):
            _WatchedOptions.reads.add(name)
        return super().__getattribute__(name)


STAMPED_ONLY: set[str] = set()
"""Fields whose whole job is to travel to the record; no consumer reads them, by design."""

ADAPT_DEPTH = 2
"""Non-default, so the depth is a value the run acts on rather than a default it could ignore."""


def _stock_adaptation(into: Path) -> Path:
    """A pretraining output carrying the stand-in encoder's own weights, keyed as `Pretrainer.save` keys them.

    The stand-in and not a real adaptation because the fixture has already replaced the backbone with
    it: an adaptation of anything else would be a shape error rather than a test of the wiring. Its
    twelve `encoder.layer.N` blocks are what give the depth something to actually cut.

    The values are beside the point — what a depth keeps is unit-tested in `pretrained_encoder`. What
    this buys is the WIRING, and it is the second instance of the same shape as report supervision:
    `encoder_adapt_depth` is consulted only inside the branch `encoder_weights` opens, so an argv
    giving neither reports the depth as a field nothing reads. Exempting it would have been a lie —
    `STAMPED_ONLY` is for fields with no consumer, and this one has one.
    """
    path = into / "adapted.pt"
    torch.save({"encoder": _FakeBackbone().state_dict(), "variant": SMALL, "config": {}}, path)
    return path


def test_every_flag_a_run_can_pass_is_read_by_something(corpus: Path, monkeypatch: pytest.MonkeyPatch):
    """A parsed field nothing ever reads is a switch that changes the ledger row and not the run.

    The sibling of `options_to_policies`, one level down: that walks the policy METHODS and asks each is
    attached, this walks the FIELDS and asks each is consulted. A flag can pass that test by being
    stamped — the recipe string is built from `vars()` — and still configure nothing at all.
    """
    identifiers = _corpus(corpus)
    # Two terms are read only when a SECOND flag turns them on, so the run has to ask for both — an argv
    # that leaves either off would report its dependent field as dead: report supervision's vectors, and
    # the depth that scopes an encoder adaptation.
    vectors = corpus / "reports.npz"
    np.savez(vectors, identifiers=np.array(identifiers), vectors=np.zeros((len(identifiers), REPORT_DIMS), np.float32))
    _WatchedOptions.reads.clear()
    monkeypatch.setattr(train_module, "TrainingOptions", _WatchedOptions)

    _run(
        corpus,
        monkeypatch,
        "--whole-stack",
        "--slice-pooling",
        "max",
        "--report-weight",
        "1.0",
        "--report-vectors",
        str(vectors),
        "--encoder-weights",
        str(_stock_adaptation(corpus)),
        "--encoder-adapt-depth",
        str(ADAPT_DEPTH),
    )

    declared = {field.name for field in fields(TrainingOptions)}
    assert declared > STAMPED_ONLY, "a name exempted here is not a field any more — stale exemption"
    unread = declared - _WatchedOptions.reads - STAMPED_ONLY
    assert not unread, f"{sorted(unread)} is parsed and recorded and read by nothing"
