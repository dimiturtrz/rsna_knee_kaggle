import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch
from torch import Tensor

from kneemri.backbone import ENCODERS
from kneemri.ensemble import BUDGET_WIDTH, IMAGE_BUDGET, Ensemble, Read
from kneemri.members import MemberConfig, MemberPackage, Window
from kneemri.model import TARGETS, StudyDesign, StudyModel

SEED = 2026
DIM = 8
PATCHES = 16
SLOTS = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]


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


def _ensemble(tmp_path: Path, *, whole_stack: bool = False, slice_pooling: str = "channels") -> Ensemble:
    device = torch.device("cpu")
    torch.manual_seed(SEED)
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = StudyModel(_FakeBackbone(), DIM, len(SLOTS), StudyDesign(slice_pooling=slice_pooling))
    config = {
        "img": 14,
        "whole_stack": whole_stack,
        "slice_pooling": slice_pooling,
        "slices": 6,
        "group": 3,
        "crop_mm": 130.0,
        "band": [0.2, 0.8],
        "slots": SLOTS,
        "variant": "small",
        "pool": "cls_mean",
    }
    torch.save(model.state_dict(), tmp_path / "m.pt")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "members": [
                    {
                        "id": "m",
                        "file": "m.pt",
                        "fold": 0,
                        "config": config,
                        "fingerprint": model.fingerprint(14, 3, device).tolist(),
                        "holdout": 0.8,
                        "annot": 0.8,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    package = MemberPackage(tmp_path)
    return Ensemble(package, {"m": model}, device)


def test_predict_member_bounds_the_forward_by_images(tmp_path: Path):
    """Peak work per forward is fixed by IMAGES, so reading deeper cannot multiply it.

    A chunk is studies; what reaches the encoder is studies x slots x window width. A whole-stack
    member reads four times the depth of a windowed one, and sending the whole chunk regardless is what
    OOM'd the kernel's T4 on 2026-08-12 — the run wrote no submission and the deleted fallback scored
    0.500. The answer must be identical whether it took one forward or six; only the peak changes.

    Watched with a forward hook rather than by swapping the model out: the hook sees exactly what the
    encoder is asked for, and leaves the object the ensemble holds untouched.
    """
    ensemble = _ensemble(tmp_path / "budget", whole_stack=True, slice_pooling="max")
    member = ensemble.package.members[0]
    model = ensemble.models[member.identifier]
    widest = 0

    def watch(_module: object, inputs: tuple[torch.Tensor, ...]) -> None:
        nonlocal widest
        images = inputs[0]
        widest = max(widest, int(images.shape[0]) * int(images.shape[1]) * int(images.shape[2]))

    handle = model.register_forward_pre_hook(watch)
    studies = IMAGE_BUDGET // (len(SLOTS) * 6) + 3
    images = torch.randint(0, 256, (studies, len(SLOTS), 6, 14, 14), dtype=torch.uint8)
    mask = torch.ones(studies, len(SLOTS))
    split = ensemble.predict_member(member, images, mask, overlap=False)
    handle.remove()

    assert widest <= IMAGE_BUDGET, f"a forward sent {widest} images, over the T4's known-good {IMAGE_BUDGET}"
    assert studies * len(SLOTS) * 6 > IMAGE_BUDGET, "the fixture must actually exceed the budget"
    model.eval()
    with torch.no_grad():
        whole = torch.sigmoid(model(images, mask, member.read_size).float()).numpy()
    assert np.allclose(split, whole), "splitting the forward must not change the answer"


def test_predict_member_shrinks_the_budget_for_a_wider_encoder(tmp_path: Path):
    """A 1024-wide member costs 2.4x a 384-wide one per image, so one flat budget OOMs the wide one.

    MEASURED on this card at 336 px, no-grad, autocast, activation per image beyond the weights: small
    0.0099 GB, base 0.0187, large 0.0235. At 864 images that is 8.6 GB, 16.5 GB and 21 GB against a
    T4's ~15 GB — so the flat number that was safe for every scored entry so far would have OOM'd the
    first one carrying `large_deep_rank24`, which is in our pool at 0.8450.
    """
    ensemble = _ensemble(tmp_path / "wide", whole_stack=True, slice_pooling="max")
    member = ensemble.package.members[0]
    assert ENCODERS[member.variant].dim == BUDGET_WIDTH, "this fixture is the width the budget was measured at"

    for variant, expected in (("small", 864), ("base", 432), ("large", 324)):
        assert IMAGE_BUDGET * BUDGET_WIDTH // ENCODERS[variant].dim == expected


def test_predict_member(tmp_path: Path):
    """A member's probabilities are averaged over its windows and land in [0, 1]."""
    ensemble = _ensemble(tmp_path)
    member = ensemble.package.members[0]
    images = torch.randint(0, 256, (2, len(SLOTS), 6, 14, 14), dtype=torch.uint8)
    mask = torch.ones(2, len(SLOTS))
    predictions = ensemble.predict_member(member, images, mask)
    assert predictions.shape == (2, len(TARGETS))
    assert float(predictions.min()) >= 0.0
    assert float(predictions.max()) <= 1.0


def test_combine():
    """Members combine by RANK per label, because rank order is all the metric reads."""
    ensemble = Ensemble.__new__(Ensemble)
    first = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    # A wildly over-confident second member that AGREES on the ordering. Averaging probabilities would
    # let its 0.999 swamp the first member's 0.9; averaging ranks keeps the ordering both agree on.
    second = np.array([[0.999, 0.001], [0.998, 0.002], [0.997, 0.003]])
    combined = ensemble.combine([first, second])
    assert combined.shape == first.shape
    assert combined[:, 0].argsort().tolist() == [2, 1, 0], "the shared ordering of column 0 survives"
    assert combined.max() <= 1.0, "ranks are normalised, so no member's scale leaks through"


def test_combine_returns_a_single_member_untouched():
    """One member has nothing to combine with, and ranking it would discard its values for no gain."""
    ensemble = Ensemble.__new__(Ensemble)
    only = np.array([[0.4, 0.6]])
    assert np.array_equal(ensemble.combine([only]), only)


def test_submission():
    """The table is the study id followed by the twelve targets, in the order the metric expects."""
    ensemble = Ensemble.__new__(Ensemble)
    table = ensemble.submission(["a", "b"], np.zeros((2, len(TARGETS))))
    assert list(table)[0] == "StudyInstanceUID"
    assert list(table)[1:] == list(TARGETS)
    assert len(table["ACL"]) == 2


def test_slices(tmp_path: Path):
    """The decode depth is the deepest stack any member was fitted on, never a default.

    A member trained on twenty slices and fed twelve raises nothing - the window loop just reads a
    shallower stack than the weights ever saw, and the fingerprint covers image size and group rather
    than depth. So the number has to come from the members, which is what this pins.
    """
    ensemble = _ensemble(tmp_path)
    assert ensemble.slices() == 6, "the fixture member declares six, so six is what gets decoded"

    ensemble.package.members[0].config["slices"] = 20
    assert ensemble.slices() == 20, "a deeper member must raise the decode depth, not be truncated to it"


def test_predict_member_reads_a_whole_stack_member_in_one_pass(tmp_path: Path):
    """A whole-stack member sees its ENTIRE stack at once, not `group` slices at a time.

    Its slot embedding is the elementwise maximum over the full stack, so windowing it by three would
    hand it maxima over a quarter of the stack — a different function of the same pixels. Nothing
    raises when that happens and the holdout cannot expose it, because holdout scoring reads the same
    window training did. The leaderboard did: two members at 0.8481 holdout scored 0.882 while five at
    0.8269 scored 0.888.

    The stack here is deeper than the member wants, which is the real case — a chunk is decoded to the
    deepest depth ANY member needs, so a member fitted on six must be handed the middle six of eight,
    exactly as `Trainer.predict` takes the middle window.
    """
    ensemble = _ensemble(tmp_path / "whole", whole_stack=True, slice_pooling="max")
    member = ensemble.package.members[0]
    images = torch.randint(0, 256, (2, len(SLOTS), 8, 14, 14), dtype=torch.uint8)
    mask = torch.ones(2, len(SLOTS))

    predictions = ensemble.predict_member(member, images, mask, overlap=True)

    model = ensemble.models[member.identifier]
    model.eval()
    with torch.no_grad():
        expected = torch.sigmoid(model(images[:, :, 1:7], mask, member.read_size).float()).numpy()
    assert np.allclose(predictions, expected), "one pass over the middle six slices, not ten windows of three"


def test_predict_member_still_windows_a_member_that_is_not_whole_stack(tmp_path: Path):
    """The published members were fitted windowed and must stay windowed — the two paths differ."""
    windowed = _ensemble(tmp_path / "win", whole_stack=False, slice_pooling="max")
    member = windowed.package.members[0]
    images = torch.randint(0, 256, (2, len(SLOTS), 8, 14, 14), dtype=torch.uint8)
    mask = torch.ones(2, len(SLOTS))

    model = windowed.models[member.identifier]
    model.eval()
    with torch.no_grad():
        single = torch.sigmoid(model(images[:, :, 1:7], mask, member.read_size).float()).numpy()
    assert not np.allclose(windowed.predict_member(member, images, mask, overlap=True), single), (
        "a windowed member averaged over its windows is not one pass over the stack"
    )


def test_decoded_size(tmp_path: Path):
    """The decode resolution comes from the members, and disagreement is refused rather than resolved.

    A single value, not a maximum: depth can be windowed down at inference but resolution cannot, so
    two members wanting different sizes genuinely need two decodes. Silently picking one would hand the
    other an image it never saw — 336 px upsampled to 392 where training downsampled from 448.
    """
    ensemble = _ensemble(tmp_path / "one")
    assert ensemble.decoded_size() == 14

    member = ensemble.package.members[0]
    coarser: MemberConfig = {**member.config, "img": 28}
    ensemble.package.members = [member, replace(member, identifier="other", config=coarser)]
    with pytest.raises(ValueError, match="two decodes"):
        ensemble.decoded_size()


def test_decoded_scheme(tmp_path: Path):
    """The slot list comes from the members, and an unknown or conflicting one is refused.

    Inference built SlotPipeline() with no scheme, so a nine-slot member would have been routed and
    bagged under the canonical six — the head expects nine positions and the three it never received
    are exactly the ones the extended scheme exists to add.
    """
    ensemble = _ensemble(tmp_path / "canonical")
    assert [slot.name for slot in ensemble.decoded_scheme().slots] == SLOTS

    member = ensemble.package.members[0]
    wider: MemberConfig = {**member.config, "slots": [*SLOTS, "COR_FLUID_NOFS"]}
    ensemble.package.members = [replace(member, identifier="wide", config=wider)]
    assert len(ensemble.decoded_scheme().slots) == len(SLOTS) + 1

    ensemble.package.members = [member, replace(member, identifier="other", config=wider)]
    with pytest.raises(ValueError, match="two decodes"):
        ensemble.decoded_scheme()

    unknown: MemberConfig = {**member.config, "slots": ["NOT_A_SLOT"]}
    ensemble.package.members = [replace(member, identifier="bad", config=unknown)]
    with pytest.raises(ValueError, match="does not define"):
        ensemble.decoded_scheme()


def test_groups(tmp_path: Path):
    """Members needing identical pixels share a group; a different resolution is a different decode."""
    ensemble = _ensemble(tmp_path / "grouped")
    member = ensemble.package.members[0]
    assert len(ensemble.groups()) == 1

    coarser: MemberConfig = {**member.config, "img": 28}
    ensemble.package.members = [member, replace(member, identifier="big", config=coarser)]
    grouped = ensemble.groups()
    assert len(grouped) == 2, "336 and 448 cannot share one decode"
    assert [len(members) for members in grouped] == [1, 1]

    twin = replace(member, identifier="twin")
    ensemble.package.members = [member, twin]
    assert len(ensemble.groups()) == 1, "same pixels, one decode"


def test_predict_members(tmp_path: Path):
    """One array per member, uncombined — groups must be rank-averaged together, not group by group."""
    ensemble = _ensemble(tmp_path / "members")
    member = ensemble.package.members[0]
    images = torch.randint(0, 256, (2, len(SLOTS), 6, 14, 14), dtype=torch.uint8)
    mask = torch.ones(2, len(SLOTS))

    arrays = ensemble.predict_members([member], [{member.image_size: (images, mask)}], overlap=False)
    assert len(arrays) == 1
    assert arrays[0].shape == (2, len(TARGETS))


def test_predict_groups(tmp_path: Path):
    """Members at DIFFERENT resolutions share ONE read, and every member still votes in one combine.

    This used to assert the opposite — one decode per pixel group — and that was the cost: our best
    member reads a 448 px cache while the rest read 336, so an entry carrying both decoded every DICOM
    twice. Resolution is the one axis a single read can serve many of, because the decode and the crop
    do not depend on the grid, so the unit of decoding is (scheme, slices) and the sizes ride along.

    The decode is injected, so this records exactly what was asked for.
    """
    ensemble = _ensemble(tmp_path / "two-groups")
    member = ensemble.package.members[0]
    coarser: MemberConfig = {**member.config, "img": 28}
    ensemble.package.members = [member, replace(member, identifier="big", config=coarser)]
    ensemble.models["big"] = ensemble.models[member.identifier]

    asked: list[tuple[int, tuple[int, ...]]] = []

    def chunks_for(read: Read):
        asked.append((read.slices, read.sizes))
        return [
            {
                size: (
                    torch.randint(0, 256, (2, len(SLOTS), read.slices, 14, 14), dtype=torch.uint8),
                    torch.ones(2, len(SLOTS)),
                )
                for size in read.sizes
            }
        ]

    combined = ensemble.predict_groups(chunks_for, overlap=False)

    assert asked == [(6, (14, 28))], "ONE read serving both resolutions, not one decode per group"
    assert combined.shape == (2, len(TARGETS))


def test_forwarded():
    """Chunks the forward, and halves the rows rather than dying when the card runs out.

    A budget is a PREDICTION about a card we cannot measure from here, and predicting high costs the
    whole entry: the kernel writes no submission by design, so Kaggle fails it and the run is gone.
    Predicting low costs a slower pass. The double raises once and then succeeds, which is the case
    that decides between those two outcomes.
    """
    calls: list[int] = []

    class _Greedy(torch.nn.Module):
        """Refuses any batch above two rows, exactly as a T4 refuses a budget set too high."""

        def forward(self, images: torch.Tensor, mask: torch.Tensor, size: int) -> torch.Tensor:
            calls.append(images.shape[0])
            if images.shape[0] > 2:
                raise torch.OutOfMemoryError
            return torch.zeros(images.shape[0], len(TARGETS))

    read = torch.zeros(6, 2, 3, 8, 8, dtype=torch.uint8)
    logits = Ensemble.forwarded(
        Ensemble.__new__(Ensemble), _Greedy(), read, torch.ones(6, 2), Window(starts=[0], width=3, size=8), rows=6
    )

    assert logits.shape == (6, len(TARGETS))
    assert calls[0] == 6, "the budget is tried first, so a card that can take it pays nothing"
    assert calls[-1] <= 2, "and it comes down to a size the card accepts rather than raising"
    assert calls == [6, 3, 1, 1, 1, 1, 1, 1], "halving, and each accepted size then runs every chunk"


def test_arithmetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A bf16 member is served in fp16 on a card that cannot do bf16, and unchanged on one that can.

    A T4 is sm_75 and bf16 autocast needs sm_80, so this is not a preference — it is the difference
    between an entry that runs and an unhandled error. It cost three submissions, and it could not be
    seen from this machine, whose card supports bf16.
    """
    ensemble = _ensemble(tmp_path / "arith")
    member = ensemble.package.members[0]
    fitted_bf16 = replace(member, config={**member.config, "bf16": True})
    fitted_fp16 = replace(member, config={**member.config, "bf16": False})
    object.__setattr__(ensemble, "device", torch.device("cuda"))

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    assert ensemble.arithmetic(fitted_bf16) is torch.float16, "emulated bf16 is not what the member was fitted under"
    assert ensemble.arithmetic(fitted_fp16) is torch.float16

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    assert ensemble.arithmetic(fitted_bf16) is torch.bfloat16, "a card that can run it gets the real arithmetic"

    ensemble.device = torch.device("cpu")
    assert ensemble.arithmetic(fitted_bf16) is torch.bfloat16, "off cuda there is no autocast to constrain"


def test_reads(tmp_path: Path):
    """Resolution shares a read; scheme and depth do not.

    The distinction is the whole saving. A decode cannot serve two slot SCHEMES (different series are
    routed) or two DEPTHS (different slices are sampled), but resolution is only the grid the same
    cropped stack is resized onto — so members at 336 and 448 belong in one read, and grouping them by
    pixels charged a full decode of the corpus for the difference.
    """
    ensemble = _ensemble(tmp_path / "reads")
    member = ensemble.package.members[0]
    coarser: MemberConfig = {**member.config, "img": 28}
    deeper: MemberConfig = {**member.config, "slices": member.slices * 2}
    ensemble.package.members = [
        member,
        replace(member, identifier="coarse", config=coarser),
        replace(member, identifier="deep", config=deeper),
    ]

    reads = ensemble.reads()

    assert len(reads) == 2, "two resolutions share one read; a different depth cannot"
    shared = next(group for group in reads if len(group) > 1)
    assert {m.identifier for m in shared} == {member.identifier, "coarse"}
    assert {m.image_size for m in shared} == {member.image_size, 28}


def test_reads_separates_bands(tmp_path: Path):
    """A different band is a different set of slices, so it cannot share a decode with the default."""
    ensemble = _ensemble(tmp_path / "bands")
    member = ensemble.package.members[0]
    wider: MemberConfig = {**member.config, "band": [0.1, 0.9]}
    ensemble.package.members = [member, replace(member, identifier="wide", config=wider)]

    reads = ensemble.reads()

    assert len(reads) == 2, "the band decides WHICH slices exist, which one decode cannot serve twice"


class _BrokenModel(StudyModel):
    """A member that dies the way a card out of memory dies: mid-prediction, not at load."""

    @override
    def forward(self, imgs: torch.Tensor, mask: torch.Tensor, img_size: int | None = None) -> torch.Tensor:
        message = "CUDA out of memory"
        raise RuntimeError(message)


def _broken() -> StudyModel:
    return _BrokenModel(_FakeBackbone(), DIM, len(SLOTS), StudyDesign())


def test_predict_members_drops_a_member_that_fails_instead_of_the_entry(tmp_path: Path):
    """One member failing mid-prediction costs that member, not the submission.

    Three entries died because any single failure took the whole run down, and a run that fails writes
    no submission at all — so the slot buys nothing. An ensemble is the one structure that does not need
    every part: losing a member costs a little accuracy, losing the entry costs a slot and a day.
    """
    ensemble = _ensemble(tmp_path / "resilient")
    good = ensemble.package.members[0]
    bad = replace(good, identifier="doomed")
    ensemble.package.members = [good, bad]

    ensemble.models["doomed"] = _broken()
    images = torch.randint(0, 256, (2, len(SLOTS), 6, 14, 14), dtype=torch.uint8)
    chunk = {good.image_size: (images, torch.ones(2, len(SLOTS)))}

    arrays = ensemble.predict_members([good, bad], [chunk], overlap=False)

    assert len(arrays) == 1, "the working member still votes"
    assert arrays[0].shape == (2, len(TARGETS))


def test_predict_members_raises_when_every_member_fails(tmp_path: Path):
    """All of them failing is a broken run, and a broken run must write NO submission.

    The difference matters: dropping some members is a real answer from the rest, while dropping all of
    them and writing a table anyway is the constant-value submission that scored 0.500 twice.
    """
    ensemble = _ensemble(tmp_path / "doomed")
    member = ensemble.package.members[0]

    ensemble.models[member.identifier] = _broken()
    chunk = {member.image_size: (torch.zeros(2, len(SLOTS), 6, 14, 14, dtype=torch.uint8), torch.ones(2, len(SLOTS)))}

    with pytest.raises(RuntimeError, match="every member"):
        ensemble.predict_members([member], [chunk], overlap=False)


def test_predict_member_redoes_a_non_finite_pass(tmp_path: Path):
    """An inf logit is redone rather than averaged, because a rank-average over inf is a wrong number.

    fp16 gives up range, not precision — it has more mantissa than bf16 — so overflow is its only real
    failure mode. Measured, these members peak at 3% of the fp16 ceiling, so this is a belt: it should
    never fire, and if a hidden study is stranger than any we hold it costs a slower pass rather than
    the meaning of the entry.
    """
    ensemble = _ensemble(tmp_path / "overflow")
    member = ensemble.package.members[0]
    calls: list[bool] = []

    class _Overflowing(StudyModel):
        """Returns inf the first time, exactly as an activation past 65504 does under fp16."""

        @override
        def forward(self, imgs: torch.Tensor, mask: torch.Tensor, img_size: int | None = None) -> torch.Tensor:
            calls.append(torch.is_autocast_enabled())
            rows = imgs.shape[0]
            if len(calls) == 1:
                return torch.full((rows, len(TARGETS)), float("inf"))
            return torch.zeros(rows, len(TARGETS))

    ensemble.models[member.identifier] = _Overflowing(_FakeBackbone(), DIM, len(SLOTS), StudyDesign())
    images = torch.zeros(2, len(SLOTS), 6, 14, 14, dtype=torch.uint8)

    scored = ensemble.predict_member(member, images, torch.ones(2, len(SLOTS)), overlap=False)

    assert len(calls) == 3, "two windows, and the one that overflowed is redone — not kept"
    assert np.isfinite(scored).all(), "and what comes back is a number"


def test_decoded_scheme_refuses_mixed_tie_rules(tmp_path: Path):
    """Members fitted under different routers cannot share one decode, and must say so.

    The published members reproduce their own predictions better under last-maximal and ours were fitted
    under first, so a fused entry really does hold both. Resolving that silently would feed one side
    images it never trained on, with every fingerprint still passing against its own reading.
    """
    ensemble = _ensemble(tmp_path)
    ours = ensemble.package.members[0]
    theirs = replace(ours, config={**ours.config, "tie": "last"})

    assert ensemble.decoded_scheme([ours]).tie == "first"
    assert ensemble.decoded_scheme([theirs]).tie == "last"
    with pytest.raises(ValueError, match="tie rule"):
        ensemble.decoded_scheme([ours, theirs])
