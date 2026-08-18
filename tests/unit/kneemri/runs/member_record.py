from dataclasses import replace
from pathlib import Path

import torch

from kneemri.corpus.crop_centre import CROP_MM
from kneemri.corpus.slots import LAST_MAXIMAL
from kneemri.models.backbone import Encoder
from kneemri.runs.member_record import NOT_IN_CONFIG, RECORDED_AS, MemberRecord, StampedRun
from kneemri.training.options import TrainingOptions

from ..models.model import SLOTS, _FakeBackbone  # the stand-in encoder this suite already trusts

DIM = 8
OPTIONS = TrainingOptions(slice_pooling="max", whole_stack=True, backbone_lr=4e-5)
"""The REAL options object, because a stand-in copy of it is a copy that drifts.

This was 19 hand-written fields mirroring `TrainingOptions`. A field added to the run and read by
`MemberRecord` would have been recorded by every real run and by nothing here — and what this file
asserts is precisely that the checkpoint stamps the values the RUN used.
"""


def _record(options: TrainingOptions = OPTIONS) -> MemberRecord:
    from kneemri.models.model import StudyModel  # noqa: PLC0415 - keeps the fake backbone import local

    torch.manual_seed(2026)
    model = StudyModel(_FakeBackbone(), DIM, SLOTS)
    encoder = Encoder(variant="small", repository="facebook/dinov2-small", dim=DIM)
    return MemberRecord(options, encoder, model, torch.device("cpu"), [f"S{i}" for i in range(SLOTS)])


def _config(options: TrainingOptions) -> dict[str, object]:
    """The config block a run with these options stamps into its checkpoint."""
    config = _record(options).payload({"w": torch.zeros(1)}, holdout=0.0, epoch=0)["config"]
    assert isinstance(config, dict)
    return config


def test_recipe():
    """Both the checkpoint's `options` block and the metrics file read this, so it has one home.

    Iterated from the options object rather than listed: a field added to a run reaches the recipe
    without anyone remembering to add it here, which is the failure this project keeps paying for.
    """
    recipe = _record().recipe

    assert recipe.keys() == vars(OPTIONS).keys(), "every option, not a chosen few"
    assert recipe["backbone_lr"] == "4e-05", "as strings, because a recipe is read and diffed, not run"
    assert recipe["whole_stack"] == "True"


def test_payload():
    """A checkpoint carries the reading it was fitted under, not just its weights.

    The two saves used to disagree — the final one wrote config and fingerprint, the periodic one
    wrote neither — so a run killed at its best epoch left weights MemberPackage refuses. Every field
    asserted here is one a loader reads back to decide how the member must be run.
    """
    record = _record()
    payload = record.payload({"w": torch.zeros(1)}, holdout=0.8413, epoch=6)

    assert payload["holdout"] == 0.8413
    assert payload["epoch"] == 6
    config = payload["config"]
    assert config["slice_pooling"] == "max", "a member run under the other aggregation predicts from wrong pixels"
    assert config["whole_stack"] is True
    assert config["backbone_lr"] == 4e-5, "the value the run USED, never a module constant"
    # The objective's own fields, because a member fitted on cross-step pairs and one fitted on
    # batch-local pairs are otherwise indistinguishable on disk — and an arm that cannot prove which
    # loss produced it cannot be read against a control.
    assert config["rank_weight"] == OPTIONS.rank_weight
    assert config["pair_memory"] == OPTIONS.pair_memory
    assert payload["fingerprint"], "without this the member cannot be verified, so it cannot be submitted"
    assert payload["fingerprint_config"]["n_slot"] == SLOTS


ALTERNATES = {
    "centre": "tissue",
    "tie": LAST_MAXIMAL,
    "pool": "cls_mean_focal",
    "slice_pooling": "attend",
    "slot_pooling": "mean",
}
"""A second legal value for each categorical option, taken from its own table rather than invented.

Categorical because a probe cannot be derived for one: `centre + 0.5` is meaningless and `centre + "x"`
would stamp a value no consumer can rebuild from, which is a worse test than none.
"""

DERIVED_ELSEWHERE = {"variant"}
"""`config["variant"]` comes from the ENCODER, which is the authority on which one it is; the option of
that name only chooses which encoder `main` builds. Excluded from the walk with the reason stated, not
quietly, so a future crossing here is a deliberate decision rather than an omission."""


def _probe(default: object, name: str) -> object:
    """A second legal value for a field, so a stamp that did not follow its option is visible."""
    if name in ALTERNATES:
        return ALTERNATES[name]
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return default + 7
    return float(default) + 0.5  # type: ignore[arg-type]


def test_payload_takes_each_config_field_from_the_option_of_its_own_name():
    """Move ONE option; exactly the config field of that name must follow.

    THE NINE DEFECTS THIS CLASS'S OWN DOCSTRINGS COUNT are all a value recorded here and a consumer
    reading a default. This guards the other end of the same wire — a value recorded here from the
    WRONG option. `"ema": options.anatomical_prior` type-checks, runs, and passes every assertion in
    `test_payload`, because both are False in the recipe those assertions were written against; the
    member would then advertise a training rule it was not fitted under and be rebuilt to match.

    Spot-checks cannot see that. Only moving one option at a time and demanding that exactly one
    stamped field follow can.
    """
    shared = {name for name in _config(OPTIONS) if name in vars(OPTIONS)} - DERIVED_ELSEWHERE

    assert len(shared) == 21, "pinned, so a renamed option cannot drop silently out of the walk"

    plain = _config(OPTIONS)
    for name in sorted(shared):
        probe = _probe(getattr(OPTIONS, name), name)
        stamped = _config(replace(OPTIONS, **{name: probe}))

        assert stamped[name] == probe, f"--{name} does not reach the config field of its own name"
        followed = {other for other in shared if stamped[other] != plain[other]}
        assert followed == {name}, f"--{name} also moved {followed - {name}}"


def test_every_option_is_recorded_or_written_off():
    """The walk above is an INTERSECTION, so an option recorded NOWHERE escapes it. This closes that.

    Three gates guard the same wire and each sees a different end of it. `Landed.NOT_A_SWITCH` demands
    every option be compared or written off on the TRAIN side; `members.CONSUMED` demands every
    RECORDED field be read or declared provenance-only on the SERVE side. Both walk what already
    exists. Neither can see a knob that changes the rebuilt architecture and was simply never stamped —
    it is absent from the config, so the serve gate has nothing to hold, and the name walk above
    intersects the two sets and drops it.

    Made total here: a new field on `TrainingOptions` lands in exactly one of three places — stamped
    under its own name, stamped under another (`RECORDED_AS`), or written off with a reason
    (`NOT_IN_CONFIG`) — or this fails. Nine defects in this project are a value recorded here and a
    consumer reading a default, and the fingerprint cannot catch that class because it is checked
    against the member's OWN reading: a member rebuilt as the wrong function verifies perfectly.
    """
    options = set(vars(OPTIONS))
    recorded = set(_config(OPTIONS))
    by_own_name = options & recorded

    unaccounted = options - by_own_name - set(RECORDED_AS) - set(NOT_IN_CONFIG)
    assert not unaccounted, f"{sorted(unaccounted)}: stamp it into the config, or write it off with WHY"

    assert not set(RECORDED_AS) - options, "a rename entry for a field that no longer exists"
    assert not set(NOT_IN_CONFIG) - options, "a write-off for a field that no longer exists"
    assert not by_own_name & (set(RECORDED_AS) | set(NOT_IN_CONFIG)), "recorded AND declared unrecorded"
    assert set(RECORDED_AS.values()) <= recorded, "a rename pointing at a config key nothing stamps"


def test_read_size():
    """The size the model SAW, which is the cache's unless the run resampled off it.

    Both classes matter and they fail in opposite directions. Zero must mean the cache's own size, or
    every member fitted before resampling gets a fingerprint taken at the wrong resolution. A set
    value must WIN over the cache's, or a member trained at 392 from a 448 cache advertises 448 and is
    rebuilt as a different function on different pixels - which is exactly what slice_pooling did.
    """
    from dataclasses import replace  # noqa: PLC0415 - local to the one test that needs it

    record = _record()
    assert record.read_size == 336, "train_size 0 means the model read the cache as decoded"
    assert record.payload({}, 0.8, 0)["config"]["train_size"] == 336

    resampled = replace(record, options=replace(OPTIONS, train_size=392))
    assert resampled.read_size == 392
    assert resampled.payload({}, 0.8, 0)["fingerprint_config"]["img"] == 392


def test_witness():
    """The witness names STUDIES, not corpus positions, and carries what training predicted for them.

    A position is meaningless without the same corpus index, and the entire point is to be checkable
    from another process against a cache built later — so the identifiers have to travel.
    """
    import numpy as np  # noqa: PLC0415 - only this test needs arrays

    record = _record()
    identifiers = [f"study-{index}" for index in range(20)]
    holdout = np.array([5, 7, 11], dtype=np.int64)
    predictions = np.arange(3 * 12, dtype=np.float64).reshape(3, 12) / 100.0

    witness = record.witness(identifiers, holdout, predictions)
    assert witness["studies"] == ["study-5", "study-7", "study-11"]
    assert len(witness["predictions"]) == 3
    assert record.payload({}, 0.8, 0, witness)["config"]["witness"] == witness
    assert record.payload({}, 0.8, 0)["config"]["witness"] == {}, "a run without a holdout has nothing to witness"


def test_payload_records_the_runs_own_crop():
    """The member stamps the crop the RUN used, not the module constant.

    Stamping the constant would record 130 mm about the image centre for an arm trained at 80 mm about
    the tissue — and the serve path reads this field, so the member would then be fed pixels it was
    never fitted on, with every fingerprint still passing against its own reading.
    """
    record = replace(_record(), options=replace(OPTIONS, crop_mm=80.0, centre="tissue"))

    config = record.payload({}, holdout=0.8, epoch=1)["config"]

    assert config["crop_mm"] == 80.0
    assert config["centre"] == "tissue"
    assert _record().payload({}, holdout=0.8, epoch=1)["config"]["crop_mm"] == CROP_MM


def test_payload_records_the_runs_own_router():
    """The member stamps the ROUTER the run used, so serving cannot fall back to ours.

    A graft-matching arm fitted under last-maximal and served under first would be fed a different
    series in every contested slot - fitted on one set of images and shown another, with the fingerprint
    still passing because it is checked against the member's own reading.
    """
    record = replace(_record(), options=replace(OPTIONS, tie="last"))

    assert record.payload({}, holdout=0.8, epoch=1)["config"]["tie"] == "last"
    assert _record().payload({}, holdout=0.8, epoch=1)["config"]["tie"] == "first"


def test_of(tmp_path: Path):
    """The stamp read back from a checkpoint, round-tripped through the assembler that writes it.

    Round-tripped rather than hand-written, because a hand-written payload only proves the copy agrees
    with itself — and what this reader exists for is the OLDER half of the ledger, whose only account
    of its own split is this block.
    """
    record = _record()
    checkpoint = tmp_path / "arm.pt"
    torch.save(record.payload({"w": torch.zeros(1)}, holdout=0.83, epoch=6), checkpoint)

    stamped = StampedRun.of(checkpoint)

    assert stamped is not None
    assert stamped.options["whole_stack"] == "True", "the run's flags, as the recipe stores them"
    assert stamped.kept_epoch == 6


def test_of_refuses_to_derive_a_kept_epoch_a_checkpoint_never_wrote(tmp_path: Path):
    """`epochs_done` was read as the kept epoch plus one. On the older half it is the run's LENGTH.

    That read stamped `wsfold0_e6`'s re-derived predictions as epoch 11 while its weights are the
    epoch-6 best, and all three pre-`epoch` checkpoints inspected carry `epochs_done == options.epochs`.
    Under patience those are different epochs, so the derivation is a guess that looks like a fact —
    and a stamp nothing can distrust is worse than one that says it does not know.
    """
    record = _record()
    payload = record.payload({"w": torch.zeros(1)}, holdout=0.83, epoch=6)
    del payload["epoch"]
    payload["epochs_done"] = 12
    checkpoint = tmp_path / "older.pt"
    torch.save(payload, checkpoint)

    stamped = StampedRun.of(checkpoint)

    assert stamped is not None
    assert stamped.options["whole_stack"] == "True", "the split it DOES record is still readable"
    assert stamped.kept_epoch is None, "11 was the run's length wearing a save's name"


def test_of_returns_nothing_for_a_checkpoint_with_no_stamp(tmp_path: Path):
    """14 members predate options entirely: name and score is all they have, and that must be said."""
    checkpoint = tmp_path / "ancient.pt"
    torch.save({"model": {"w": torch.zeros(1)}, "holdout": 0.81}, checkpoint)

    assert StampedRun.of(checkpoint) is None
    assert StampedRun.of(tmp_path / "absent.pt") is None
