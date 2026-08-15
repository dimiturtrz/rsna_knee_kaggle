import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import override

import pandas as pd
import pytest
import torch
from torch import Tensor

from kneemri.corpus.crop_centre import CROP_MM, Crop, ImageCentre, TissueCentre
from kneemri.corpus.slot_reader import Band, SlotReader
from kneemri.corpus.study import SlotPipeline
from kneemri.inference.budget import Budget
from kneemri.inference.ensemble import Ensemble, Read
from kneemri.inference.members import MemberPackage
from kneemri.kaggle.submission import Split, Submission
from kneemri.models.model import TARGETS, StudyModel

from ..conftest import write_series

SEED = 2026
DIM = 8
PATCHES = 16
IMAGE = 32
"""The size the fixture members record as their cache — matched to their config img."""
SLOT_NAMES = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]


@dataclass
class _Output:
    last_hidden_state: Tensor


class _FakeBackbone(torch.nn.Module):
    """A stand-in encoder whose tokens vary PER DIMENSION with the input.

    An earlier version returned `base + mean(pixels)` — the same scalar added to every feature. The
    head begins with a LayerNorm, which removes exactly that, so the stand-in made the model's output
    independent of its pixels and no test could detect a change in pooling. A useful double has to
    perturb the feature vector's SHAPE, not just its level.
    """

    @override
    def forward(self, *, pixel_values: Tensor) -> _Output:
        n = pixel_values.shape[0]
        rows = pixel_values.mean(dim=(1, 3))[:, :DIM] if pixel_values.shape[2] >= DIM else None
        if rows is None:
            rows = pixel_values.mean(dim=(1, 3)).repeat(1, (DIM // pixel_values.shape[2]) + 1)[:, :DIM]
        base = torch.arange(PATCHES + 1, dtype=torch.float32).reshape(1, -1, 1) * 0.01
        return _Output(last_hidden_state=base + rows.reshape(n, 1, DIM))


def _competition(root: Path) -> Path:
    """A miniature competition mount: two studies, one series each, and the three CSVs."""
    root.mkdir(parents=True, exist_ok=True)
    identifiers = ["study_a", "study_b"]
    series_rows = []
    # A DIFFERENT SEED PER STUDY. They shared one, so both studies decoded to identical pixels and the
    # whole table came out constant — which `Submission.check` now refuses, correctly: two identical
    # studies carry no ordering, and ordering is the only thing this metric scores.
    for seed, identifier in enumerate(identifiers):
        write_series(root / "test_series" / identifier / "sag", slices=4, size=32, seed=SEED + seed)
        series_rows.append({"StudyInstanceUID": identifier, "SeriesInstanceUID": "sag", "Anatomical_Plane": "Sagittal"})
    pd.DataFrame({"StudyInstanceUID": identifiers}).to_csv(root / "test.csv", index=False)
    pd.DataFrame(series_rows).to_csv(root / "test_series.csv", index=False)
    sample = {"StudyInstanceUID": identifiers} | {target: [0.5, 0.5] for target in TARGETS}
    pd.DataFrame(sample).to_csv(root / "sample_submission.csv", index=False)
    return root


def _submission(tmp_path: Path, *, depths: tuple[int, ...] = (3,), budget: Budget | None = None) -> Submission:
    """A miniature submission over `depths` — one member per entry, members of equal depth sharing a read.

    Depth rather than a member count, because the slice depth is a key `Ensemble.reads` groups by: two
    members at the same depth are one decode and one give-up decision, and a member at another depth is a
    second read group. That is the only knob a test needs to build either shape.
    """
    device = torch.device("cpu")
    torch.manual_seed(SEED)
    model = StudyModel(_FakeBackbone(), DIM, len(SLOT_NAMES))
    package_root = tmp_path / "weights"
    package_root.mkdir()
    torch.save(model.state_dict(), package_root / "m.pt")
    identifiers = [f"m{index}" for index in range(len(depths))]
    members = [
        {
            "id": identifiers[index],
            "file": "m.pt",
            "fold": 0,
            "config": {
                "img": 32,
                "slices": depth,
                "group": 3,
                "crop_mm": 130.0,
                "band": [0.2, 0.8],
                "slots": SLOT_NAMES,
                "variant": "small",
                "pool": "cls_mean",
            },
            "fingerprint": model.fingerprint(32, 3, device).tolist(),
            "holdout": 0.8 - 0.01 * index,
            "annot": 0.8,
        }
        for index, depth in enumerate(depths)
    ]
    (package_root / "manifest.json").write_text(json.dumps({"members": members}), encoding="utf-8")
    package = MemberPackage(package_root)
    ensemble = Ensemble(package, dict.fromkeys(identifiers, model), device, budget=budget)
    working = tmp_path / "working"
    working.mkdir()
    return Submission(ensemble, _competition(tmp_path / "comp"), working)


def test_check(tmp_path: Path):
    """A table that cannot be a real answer is refused BEFORE it is written and scored.

    Both refused shapes score around chance and neither looks wrong: a table missing studies, and a
    table whose columns are constant — which is what a rank-average degenerates to when every member
    returns the same thing, and what the old fallback wrote on purpose. That fallback scored 0.500 on
    2026-08-12 and spent a submission slot to say nothing.
    """
    frame = pd.DataFrame({"StudyInstanceUID": ["a", "b"], **{target: [0.1, 0.9] for target in TARGETS}})
    Submission.check(frame, ["a", "b"])

    with pytest.raises(ValueError, match="1 rows for 2 studies"):
        Submission.check(frame.head(1), ["a", "b"])

    # ONE flat column is allowed: predictions tie on small splits, and the visible test set inside a
    # kernel is three studies — a per-column rule would fail the run that must succeed before anything
    # can be submitted at all.
    one_flat = frame.copy()
    one_flat["MCL"] = 0.5
    Submission.check(one_flat, ["a", "b"])

    flat = pd.DataFrame({"StudyInstanceUID": ["a", "b"], **{target: [0.5, 0.5] for target in TARGETS}})
    with pytest.raises(ValueError, match="scores as chance"):
        Submission.check(flat, ["a", "b"])


def test_decode(tmp_path: Path):
    """A chunk decoded ONCE comes back at every size, each identical to a read at that size alone.

    The identity is the point: resolution is the one axis a single read can serve many of, so members
    fitted at different sizes can share a decode without any of them being fed pixels it never saw.
    """
    submission = _submission(tmp_path)
    members = submission.ensemble.package.members
    scheme = submission.ensemble.decoded_scheme(members)

    sized = submission.decode(["study_a", "study_b"], {"sag": "Sagittal"}, Read(3, scheme, (IMAGE, IMAGE // 2)))

    assert sorted(sized) == sorted((IMAGE, IMAGE // 2))
    images, mask = sized[IMAGE]
    assert images.shape[0] == 2
    assert mask.shape == (2, len(SLOT_NAMES))
    assert mask[:, 0].tolist() == [1.0, 1.0], "the sagittal fluid slot is filled for both"

    alone = submission.decode(["study_a", "study_b"], {"sag": "Sagittal"}, Read(3, scheme, (IMAGE // 2,)))
    assert torch.equal(sized[IMAGE // 2][0], alone[IMAGE // 2][0]), "sharing a read must change nothing"


def test_predict(tmp_path: Path):
    """Every study gets one probability per target, in [0, 1]."""
    submission = _submission(tmp_path)
    combined = submission.predict(["study_a", "study_b"], {"sag": "Sagittal"})
    assert combined.shape == (2, len(TARGETS))
    assert combined.min() >= 0.0
    assert combined.max() <= 1.0


def test_run(tmp_path: Path):
    """The whole path writes a submission whose rows are the test studies, in order.

    `run()` takes no slice count: it reads the depth from the members. This test used to pass
    `slices=3` explicitly, which meant it exercised the caller's opinion rather than the member's, and
    so could not have noticed that every kernel was decoding a hardcoded twelve.
    """
    submission = _submission(tmp_path)
    frame = submission.run()
    written = pd.read_csv(submission.working / "submission.csv")
    assert written["StudyInstanceUID"].tolist() == ["study_a", "study_b"]
    assert list(written.columns) == ["StudyInstanceUID", *TARGETS]
    assert len(frame) == 2


def test_chunks(tmp_path: Path):
    """On the ensemble's device, and PIXEL-IDENTICAL to decoding serially.

    THE EQUALITY ORACLE FOR THE DECODE OPTIMISATION. `chunks` runs a spawned process pool across the
    whole read and submits the next chunk before awaiting this one, so decode overlaps the forward
    pass. None of that may change a single pixel, and nothing downstream would notice if it did — a
    submission built from subtly different pixels has the right shape, the right length, and scores
    like a coin.
    """
    submission = _submission(tmp_path)
    scheme = submission.ensemble.decoded_scheme(submission.ensemble.package.members)
    read = Read(3, scheme, (IMAGE,))
    produced = list(submission.chunks(["study_a", "study_b"], {"sag": "Sagittal"}, read))

    assert len(produced) == 1, "two studies fit in one chunk"
    images, mask = produced[0][IMAGE]
    assert images.device.type == submission.device.type
    assert images.shape[0] == 2

    serial = submission.decode(["study_a", "study_b"], {"sag": "Sagittal"}, read)
    assert torch.equal(images.cpu(), serial[IMAGE][0]), "the pooled path must be pixel-identical"
    assert torch.equal(mask.cpu(), serial[IMAGE][1]), "and mask-identical"


def test_predict_gives_up_a_member_when_the_wall_clock_runs_out(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """The deadline mitigation, end to end — the WIRING, not the arithmetic.

    Every part of this was unit-tested and none of it was tested joined up, which is this project's
    dominant failure: a value recorded in one place while its consumer reads a constant. The budget only
    works if the kernel hands one to the ensemble, the ensemble reports its read groups into it, the
    decode loop reports its chunks into it, and `afford` reads the same object back. Any one of those
    four missing leaves every unit test green and the run dies at nine hours with no submission.

    Three members at two depths: the shallow pair share a read, so by the time that group is entered a
    whole group has finished and the projection is measurable — which is exactly when a real run first
    learns it is overrunning.
    """
    budget = Budget(seconds=1e-6, margin=1.0)
    submission = _submission(tmp_path, depths=(4, 3, 3), budget=budget)

    with caplog.at_level(logging.WARNING):
        combined = submission.predict(["study_a", "study_b"], {"sag": "Sagittal"})

    assert combined.shape == (2, len(TARGETS))
    assert budget.measured(), "the read groups reached the budget the kernel built"
    assert budget.overrunning(), "and a microsecond ceiling is projected past"
    # WHICH of the pair goes is not asserted: `worth` divides by seconds MEASURED in this run, and two
    # members a hundredth of holdout apart are separated by timing jitter on a fixture this small. What
    # must hold is that a member of the shared group leaves and the group still predicts.
    assert "giving up m" in caplog.text and ", 1 members left" in caplog.text
    assert set(submission.ensemble.spent) == {"m0", "m1", "m2"}, "and every member was timed while it ran"


def test_decoded(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """A study that cannot be opened contributes zeros and an empty mask, not an exception.

    `pool.map` re-raises, so before this a single unreadable study took down the whole run — and a run
    that fails writes no submission by design, so the entry is lost. Three visible studies never fire
    it; a hidden test set of thousands almost certainly does.
    """
    submission = _submission(tmp_path)
    members = submission.ensemble.package.members
    pipeline = SlotPipeline(
        scheme=submission.ensemble.decoded_scheme(members),
        reader=SlotReader(image_size=members[0].image_size),
    )

    with caplog.at_level(logging.ERROR):
        sized = submission.decoded("no-such-study", {"sag": "Sagittal"}, pipeline, Read(3, pipeline.scheme, (IMAGE,)))
    bag, mask = sized[IMAGE]

    assert bag.shape == (len(SLOT_NAMES), 3, IMAGE, IMAGE)
    assert bag.dtype == torch.uint8
    assert float(mask.sum()) == 0.0, "an unreadable study must claim nothing was acquired"
    assert "no-such-study" in caplog.text, "and it must say so, or a silent zero row reads as a real one"


def test_series(tmp_path: Path):
    """Pixels sit beside the tables by default, and elsewhere when a dry run says so.

    The override is what lets the SHIPPED submission path run over labelled studies: a scratch
    directory holds a subset of the train tables while the DICOMs stay where they are. Without it the
    only way to find out whether this path produces sane predictions was to spend an entry.
    """
    assert Split().series(tmp_path) == tmp_path / "test_series"
    assert Split("train").series(tmp_path) == tmp_path / "train_series"
    assert Split("train", tmp_path / "elsewhere").series(tmp_path) == tmp_path / "elsewhere"


def test_decoded_at(tmp_path: Path):
    """One study read from its directory, and an unreadable one costing a row rather than the entry."""
    submission = _submission(tmp_path)
    scheme = submission.ensemble.decoded_scheme(submission.ensemble.package.members)
    read = Read(3, scheme, (IMAGE,))
    pipeline = SlotPipeline(scheme=scheme, reader=SlotReader(image_size=IMAGE))
    root = submission.split.series(submission.competition)

    images, mask = Submission.decoded_at(root / "study_a", {"sag": "Sagittal"}, pipeline, read)[IMAGE]
    assert mask[0] == 1.0, "the sagittal fluid slot is filled"

    missing, empty = Submission.decoded_at(root / "no_such_study", {"sag": "Sagittal"}, pipeline, read)[IMAGE]
    assert not missing.any(), "an unreadable study contributes zeros"
    assert not empty.any(), "and an empty mask, so the head renormalises over what is really there"


def test_batched(tmp_path: Path):
    """Per-study bags stacked into one batch per size, in the order given."""
    submission = _submission(tmp_path)
    scheme = submission.ensemble.decoded_scheme(submission.ensemble.package.members)
    read = Read(3, scheme, (IMAGE,))
    pipeline = SlotPipeline(scheme=scheme, reader=SlotReader(image_size=IMAGE))
    root = submission.split.series(submission.competition)
    bags = [Submission.decoded_at(root / name, {"sag": "Sagittal"}, pipeline, read) for name in ("study_a", "study_b")]

    images, mask = submission.batched(bags, read)[IMAGE]

    assert images.shape[0] == 2
    assert torch.equal(images[0], bags[0][IMAGE][0]), "row order follows the bags given"


def test_pool(tmp_path: Path):
    """A usable executor either way, and a small run does NOT pay to spawn one.

    The floor is not cosmetic: the Kaggle run of 2026-08-14 decoded three visible studies in 6s against
    3s serially, because four spawned interpreters cost more than three studies of decoding save.
    """
    submission = _submission(tmp_path)
    with submission.pool(studies=1000) as pool:
        assert pool.submit(len, "abc").result() == 3
    with submission.pool(studies=3) as small:
        assert isinstance(small, ThreadPoolExecutor), "a three-study run must not spawn interpreters"
        assert small.submit(len, "abc").result() == 3


def test_pipeline(tmp_path: Path):
    """The decode reader is built from what the members RECORDED, not from the module constant.

    A member fitted on an 80mm crop and served under 130mm is a different function, and nothing
    downstream would say so: every fingerprint passes, because each is checked against its own reading.
    """
    submission = _submission(tmp_path)
    scheme = submission.ensemble.decoded_scheme(submission.ensemble.package.members)

    default = submission.pipeline(Read(3, scheme, (IMAGE,)))
    assert default.reader.crop_mm == CROP_MM

    tighter = submission.pipeline(Read(3, scheme, (IMAGE,), crop=Crop(mm=80.0)))
    assert tighter.reader.crop_mm == 80.0, "the crop the member recorded reaches the reader"
    assert tighter.reader.image_size == IMAGE, "and the size is still the largest the group asked for"


def test_pipeline_honours_the_recorded_band(tmp_path: Path):
    """The last recorded reading rule the decode ignored: which slices exist to be sampled at all."""
    submission = _submission(tmp_path)
    scheme = submission.ensemble.decoded_scheme(submission.ensemble.package.members)

    default = submission.pipeline(Read(3, scheme, (IMAGE,)))
    assert default.reader.band == Band()

    wider = submission.pipeline(Read(3, scheme, (IMAGE,), band=Band(0.1, 0.9)))
    assert wider.reader.band == Band(0.1, 0.9), "the band the member recorded reaches the reader"


def test_pipeline_honours_the_recorded_centre(tmp_path: Path):
    """A member cropped about the tissue must not be served cropped about the image."""
    submission = _submission(tmp_path)
    scheme = submission.ensemble.decoded_scheme(submission.ensemble.package.members)

    default = submission.pipeline(Read(3, scheme, (IMAGE,)))
    assert isinstance(default.reader.centre, ImageCentre)

    located = submission.pipeline(Read(3, scheme, (IMAGE,), crop=Crop(centre="tissue")))
    assert isinstance(located.reader.centre, TissueCentre), "the recorded centre reaches the reader"


def test_peak_vram(tmp_path: Path):
    """Off the GPU the reading is zero rather than a raised CUDA error, so the chunk line still logs."""
    submission = _submission(tmp_path)

    assert submission.peak_vram() == 0.0
