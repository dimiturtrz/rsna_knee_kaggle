import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import Tensor

from core.findings import TARGETS
from kneemri.inference.budget import KERNEL_SECONDS, SECONDS_PER_HOUR, Budget
from kneemri.inference.ensemble import BUDGET_WIDTH, IMAGE_BUDGET, Ensemble, Read, Serving
from kneemri.inference.member_package import MemberPackage
from kneemri.inference.members import Member, MemberConfig, Window
from kneemri.inference.read_groups import SharedRead
from kneemri.inference.test_time import SpatialViews, WindowPooling
from kneemri.inference.vote import LogitAverage
from kneemri.models.backbone import ENCODERS
from kneemri.models.model import StudyDesign, StudyModel

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

    with pytest.raises(ValueError, match="empty"):
        ensemble.slices([])


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

    ensemble.package.members = [member]
    with pytest.raises(ValueError, match="0 distinct"):
        ensemble.decoded_scheme([])
    assert [slot.name for slot in ensemble.decoded_scheme(None).slots] == SLOTS, (
        "an EMPTY subset is a caller's bug and refuses; only the ABSENT one means the whole package"
    )


def test_agreed(tmp_path: Path):
    """One shared value passes; two are refused by name, whatever field disagreed."""
    ensemble = _ensemble(tmp_path / "agreed")
    member = ensemble.package.members[0]
    other = replace(member, identifier="other")

    assert ensemble.agreed([member, other], "depth", lambda each: each.slices) == member.slices

    with pytest.raises(ValueError, match="depth"):
        ensemble.agreed(
            [member, replace(other, config={**other.config, "slices": member.slices + 1})],
            "depth",
            lambda each: each.slices,
        )


def test_absorb(tmp_path: Path):
    """One chunk through a group: a survivor accumulates, a member already failed is not retried."""
    ensemble = _ensemble(tmp_path / "absorb")
    member = ensemble.package.members[0]
    other = replace(member, identifier="dead")
    ensemble.models["dead"] = ensemble.models[member.identifier]
    sized = {
        member.image_size: (
            torch.randint(0, 256, (2, len(SLOTS), 6, 14, 14), dtype=torch.uint8),
            torch.ones(2, len(SLOTS)),
        )
    }
    per_member: dict[str, list[NDArray[np.float64]]] = {member.identifier: [], "dead": []}

    ensemble.absorb([member, other], sized, per_member, {"dead"}, overlap=False)

    assert len(per_member[member.identifier]) == 1
    assert per_member["dead"] == [], "a member already given up is skipped, not attempted again"


def test_survived(tmp_path: Path):
    """A member counts only if it BOTH survived and predicted something — the two are not the same."""
    ensemble = _ensemble(tmp_path / "survived")
    member = ensemble.package.members[0]
    empty = replace(member, identifier="empty")
    gone = replace(member, identifier="gone")
    predicted = np.zeros((2, len(TARGETS)))
    per_member = {member.identifier: [predicted], "empty": [], "gone": [predicted]}

    arrays = ensemble.survived([member, empty, gone], per_member, {"gone"})

    assert len(arrays) == 1, "the failed one and the one that never predicted are both out"
    assert arrays[0].shape == (2, len(TARGETS))


def test_read_of(tmp_path: Path):
    """The group's read is ASKED of the group, and a group that disagrees is refused, not served."""
    ensemble = _ensemble(tmp_path / "read-of")
    member = ensemble.package.members[0]
    cropped: MemberConfig = {**member.config, "crop_mm": member.crop.mm - 10.0}

    read = ensemble.read_of([member])
    assert read.reading == member.reading
    assert read.sizes == (member.image_size,)

    with pytest.raises(ValueError, match="reading"):
        ensemble.read_of([member, replace(member, identifier="tight", config=cropped)])


def test_budget(tmp_path: Path):
    """The run budget reaches `afford` whatever the entry's read grouping.

    It was off for more than one group for exactly one commit, between the traversal moving and the
    projection being repriced (knee-fma1r). A chunk of studies is now the unit of progress, and that is
    the same fraction of the run for one read group and for five.
    """
    ensemble = _ensemble(tmp_path / "budget")
    ensemble.serving = replace(ensemble.serving, budget=Budget(seconds=1.0, margin=1.0))
    member = ensemble.package.members[0]
    assert ensemble.budget() is not None, "one read group keeps the net it shipped with"

    deeper = replace(member, identifier="deep")
    ensemble.reads = lambda: [[member], [deeper]]
    assert ensemble.budget() is not None, "and so does more than one"


def test_attempt(tmp_path: Path):
    """A pass that lands is banked as seconds; one that raises returns None and is charged nothing."""
    ensemble = _ensemble(tmp_path / "attempt")
    member = ensemble.package.members[0]
    images = torch.randint(0, 256, (2, len(SLOTS), 6, 14, 14), dtype=torch.uint8)
    sized = {member.image_size: (images, torch.ones(2, len(SLOTS)))}

    predicted = ensemble.attempt(member, sized, overlap=False)
    assert predicted is not None
    assert predicted.shape == (2, len(TARGETS))
    assert ensemble.spent[member.identifier] > 0

    banked = ensemble.spent[member.identifier]
    del ensemble.models[member.identifier]
    assert ensemble.attempt(member, sized, overlap=False) is None, "a failed member drops, it does not raise"
    assert ensemble.spent[member.identifier] == banked, "and pays for no prediction it did not make"


def _chunk(read: Read) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """One decoded chunk for one read, every size the read asks for."""
    return {
        size: (
            torch.randint(0, 256, (2, len(SLOTS), read.slices, 14, 14), dtype=torch.uint8),
            torch.ones(2, len(SLOTS)),
        )
        for size in read.sizes
    }


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

    def chunks_for(reads: list[Read]):
        asked.extend((read.slices, read.sizes) for read in reads)
        return [[_chunk(read) for read in reads]]

    combined = ensemble.predict_groups(chunks_for, overlap=False)

    assert asked == [(6, (14, 28))], "ONE read serving both resolutions, not one decode per group"
    assert combined.shape == (2, len(TARGETS))


def test_predict_groups_refuses_a_group_that_does_not_share_its_pixels(tmp_path: Path):
    """A group whose members crop differently is refused, not served under the first one's crop.

    `reads` groups by the reading, so the read built from a group agrees with every member of it — by
    construction, which is the guarantee an edit to that grouping key drops without a word. The group is
    forced here rather than built, because that is precisely the state such an edit would produce: the
    members' own recorded readings differ and only the key stopped noticing. Under `members[0]`, the
    second member is then served pixels it was never fitted on, and every fingerprint still passes.
    """
    ensemble = _ensemble(tmp_path / "mixed")
    member = ensemble.package.members[0]
    cropped: MemberConfig = {**member.config, "crop_mm": member.crop.mm - 10.0}
    mixed = [member, replace(member, identifier="tight", config=cropped)]
    ensemble.models["tight"] = ensemble.models[member.identifier]
    ensemble.reads = lambda: [mixed]

    with pytest.raises(ValueError, match="reading"):
        ensemble.predict_groups(lambda reads: [], overlap=False)


def test_predict_groups_combines_by_the_rule_the_serving_names(tmp_path: Path):
    """The vote is a served knob, and a submission built under `LogitAverage` must not be rank-averaged.

    `RankAverage` was called directly here, so the whole family was reachable from `member_curve` and
    from nothing that writes a submission — the offline ruler could vary an axis the board could never
    be asked about. Asserted as a DIFFERENCE against the shipped rule rather than against fixed values,
    because what matters is that the declared rule is the one that ran.
    """
    ensemble = _ensemble(tmp_path / "voted")
    member = ensemble.package.members[0]
    ensemble.package.members = [member, replace(member, identifier="second")]
    ensemble.models["second"] = ensemble.models[member.identifier]

    def chunks_for(reads: list[Read]):
        torch.manual_seed(SEED)
        return [[_chunk(read) for read in reads]]

    ranked = ensemble.predict_groups(chunks_for, overlap=False)
    ensemble.serving = replace(ensemble.serving, vote=LogitAverage())
    logits = ensemble.predict_groups(chunks_for, overlap=False)

    assert ranked.shape == logits.shape
    assert not np.allclose(ranked, logits), "the declared rule never reached the combine"


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


def test_forwarded_stops_halving_at_one_row():
    """A card that cannot take a SINGLE study is a real failure, and must not be retried forever.

    The halving has to bottom out somewhere, and one row is where retrying stops being a smaller ask —
    the next halve is the same size. Raising here is what lets `attempt` drop the member and the run
    continue with the rest; a loop that kept retrying would hang the kernel until Kaggle killed it.
    """

    class _Refuses(torch.nn.Module):
        def forward(self, images: torch.Tensor, mask: torch.Tensor, size: int) -> torch.Tensor:
            raise torch.OutOfMemoryError

    with pytest.raises(torch.OutOfMemoryError):
        Ensemble.forwarded(
            Ensemble.__new__(Ensemble),
            _Refuses(),
            torch.zeros(2, 2, 3, 8, 8, dtype=torch.uint8),
            torch.ones(2, 2),
            Window(starts=[0], width=3, size=8),
            rows=2,
        )


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


def test_a_forced_precision_overrides_the_member_but_not_the_card(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The knob a precision MEASUREMENT turns, and the guard it must not turn off.

    The loadcheck compared three precisions by wrapping `predict_member` in an outer autocast, which
    the inner explicit-dtype one overrides — so every arm ran the member's own arithmetic and reported
    max|diff| of exactly 0.0 twelve times. That is the shape of a null result from a disconnected knob,
    and only a test that the forced dtype actually ARRIVES can tell it from agreement.

    The downgrade still applies on top, because a forced bf16 that silently ran emulated on a T4 would
    make two arms identical again by a different route.
    """
    ensemble = _ensemble(tmp_path / "forced")
    member = replace(ensemble.package.members[0], config={**ensemble.package.members[0].config, "bf16": False})
    object.__setattr__(ensemble, "device", torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))

    assert ensemble.arithmetic(member) is torch.float16, "unforced, a member is served as it was fitted"
    ensemble.serving = Serving(precision=torch.bfloat16)
    assert ensemble.arithmetic(member) is torch.bfloat16, "the forced arithmetic is the one that reaches autocast"
    ensemble.serving = Serving(precision=torch.float32)
    assert ensemble.arithmetic(member) is torch.float32, "fp32 is the reference arm a comparison needs"

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    ensemble.serving = Serving(precision=torch.bfloat16)
    assert ensemble.arithmetic(member) is torch.float16, "forcing an arithmetic the card emulates measures that"


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


def test_predict_groups_drops_a_member_that_fails_instead_of_the_entry(tmp_path: Path):
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

    combined = ensemble.predict_groups(lambda _: [[chunk]], overlap=False)

    assert combined.shape == (2, len(TARGETS)), "the working member still votes"


def test_predict_groups_does_not_retry_a_member_that_already_failed(tmp_path: Path):
    """A member that failed on one chunk is not asked about the next, or the failure costs the whole run.

    Dropping is per RUN, not per chunk. A card that ran out of memory on chunk one will do it again on
    every chunk, and the corpus is thousands of studies: retrying is a fixed cost per chunk against a
    member that can no longer vote, which is how a nine-hour kernel limit gets spent on nothing.
    """
    ensemble = _ensemble(tmp_path / "once")
    good = ensemble.package.members[0]
    bad = replace(good, identifier="doomed")
    ensemble.package.members = [good, bad]

    attempts: list[str] = []
    original = ensemble.attempt

    def counted(
        member: Member, sized: dict[int, tuple[Tensor, Tensor]], *, overlap: bool
    ) -> NDArray[np.float64] | None:
        attempts.append(member.identifier)
        return original(member, sized, overlap=overlap)

    ensemble.attempt = counted  # type: ignore[method-assign]
    ensemble.models["doomed"] = _broken()
    images = torch.randint(0, 256, (2, len(SLOTS), 6, 14, 14), dtype=torch.uint8)
    chunk = {good.image_size: (images, torch.ones(2, len(SLOTS)))}

    ensemble.predict_groups(lambda _: [[chunk]] * 3, overlap=False)

    assert attempts.count("doomed") == 1, "a member that failed is asked once, not once per chunk"
    assert attempts.count(good.identifier) == 3, "and the survivors still see every chunk"


def test_predict_groups_raises_when_every_member_fails(tmp_path: Path):
    """All of them failing is a broken run, and a broken run must write NO submission.

    The difference matters: dropping some members is a real answer from the rest, while dropping all of
    them and writing a table anyway is the constant-value submission that scored 0.500 twice.
    """
    ensemble = _ensemble(tmp_path / "doomed")
    member = ensemble.package.members[0]

    ensemble.models[member.identifier] = _broken()
    chunk = {member.image_size: (torch.zeros(2, len(SLOTS), 6, 14, 14, dtype=torch.uint8), torch.ones(2, len(SLOTS)))}

    with pytest.raises(RuntimeError, match="every member"):
        ensemble.predict_groups(lambda _: [[chunk]], overlap=False)


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


def _timed(ensemble: Ensemble, seconds: float, ceiling: float) -> None:
    """Put the ensemble on a budget that has already burned `seconds` of `ceiling`, one whole chunk done."""
    budget = Budget(seconds=ceiling, margin=1.0)
    ensemble.serving = Serving(budget=budget)
    budget.started -= seconds
    budget.chunk(1, 1)


def test_afford(tmp_path: Path):
    """An overrun costs the worst score-per-second, one per chunk, never the last one — and only an overrun.

    Twenty-five members that do not finish score nothing while twenty that do score most of what
    twenty-five would have. What is being bought is TIME, so a slightly better member that costs four
    times as much per study is the one to give up: shedding cheap members saves the least wall clock per
    vote lost, and a run can shed three of them and hit the ceiling anyway.
    """
    ensemble = _ensemble(tmp_path)
    quick = replace(ensemble.package.members[0], identifier="quick", holdout=0.84)
    slow = replace(quick, identifier="slow", holdout=0.85)
    ensemble.spent = {"quick": 10.0, "slow": 100.0}
    failed: set[str] = set()

    ensemble.afford([quick, slow], failed)
    assert not failed, "no budget means no ceiling — the dry run and the witness are not timed"

    _timed(ensemble, 1.0, KERNEL_SECONDS)
    ensemble.afford([quick, slow], failed)
    assert not failed, "a run projected well inside nine hours keeps all of them"

    _timed(ensemble, 3600, 1.0)
    ensemble.afford([quick, slow], failed)
    assert failed == {"slow"}, "the better member goes first when it costs ten times as much per second"

    ensemble.afford([quick, slow], failed)
    assert failed == {"slow"}, "and the last survivor is kept even while overrunning — a run needs a vote"

    # VIEWS BEFORE MEMBERS. `kneemri-views` scored blank on 2026-08-17 — two members under rot3, the
    # only entry tripling the forward, and the one cost this mitigation could not reach. Shedding a
    # member of two saves a factor of two and makes it a different entry; shedding the views saves the
    # whole multiplier and leaves the pair that already scored 0.919.
    served = _ensemble(tmp_path)
    served.views = SpatialViews()
    served.spent = {"quick": 10.0, "slow": 100.0}
    _timed(served, 3600, 1.0)
    dropped: set[str] = set()

    served.afford([quick, slow], dropped)
    assert served.views is None, "the unmeasured multiplier goes before any fitted member"
    assert not dropped, "and no member is given up in the same chunk — one give-up, then re-measure"

    served.afford([quick, slow], dropped)
    assert dropped == {"slow"}, "still overrunning with no views left falls back to the member order"


def test_worth(tmp_path: Path):
    """A member is worth its holdout ABOVE CHANCE per second spent, and an untimed one is not free."""
    ensemble = _ensemble(tmp_path)
    member = replace(ensemble.package.members[0], identifier="timed", holdout=0.9)
    ensemble.spent = {"timed": 20.0}

    assert ensemble.worth(member) == pytest.approx(0.4 / 20.0), "chance is subtracted, not the whole holdout"

    unreached = replace(member, identifier="unreached", holdout=0.9)
    assert ensemble.worth(unreached) == pytest.approx(0.4 / 20.0), "an unreached member costs the pool's mean"

    ensemble.spent = {}
    assert ensemble.worth(member) == float("inf"), "before any member is timed there is no cost to divide by"


def test_mean_cost(tmp_path: Path):
    """The stand-in cost for a member the loop has not reached is what the reached ones have cost."""
    ensemble = _ensemble(tmp_path)

    assert ensemble.mean_cost() == 0.0, "nothing timed yet"

    ensemble.spent = {"a": 10.0, "b": 30.0, "c": 0.0}
    assert ensemble.mean_cost() == pytest.approx(20.0), "a zero is untimed, not a free member"


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


def test_read(tmp_path: Path):
    """A forced serving size moves the size the ENCODER runs at, and only that.

    `img` names the decode cache every member of a group shares, so forcing it would feed resampled
    pixels to an encoder still at the old resolution and price nothing (knee-d2fc). The default is the
    member untouched, because a submission serves each member the way it was fitted.
    """
    member = _ensemble(tmp_path).package.members[0]

    served = Serving(read_size=280).read(member)

    assert served.read_size == 280, "the encoder runs at the size the entry forces"
    assert served.image_size == member.image_size, "and the group still shares one decode"
    assert Serving().read(member) is member, "an entry that forces nothing serves the member as fitted"


def test_served(tmp_path: Path):
    """The forced reading is applied BEFORE the grouping, or the group it exists to collapse survives.

    `reads` keys on each member's own depth and band, so a re-group applied only at predict time would
    decode every group and then predict from one reading — paying the whole bill the group removes.
    """
    ensemble = _ensemble(tmp_path / "served")
    member = ensemble.package.members[0]
    deeper: MemberConfig = {**member.config, "slices": member.slices * 2}
    ensemble.package.members = [member, replace(member, identifier="deep", config=deeper)]
    ensemble.serving = Serving(group=SharedRead(member.slices, (0.1, 0.9)))

    assert len(ensemble.reads()) == 1, "two depths served as one group are ONE decode"
    assert {served.slices for served in ensemble.served()} == {member.slices}


def _paced(ensemble: Ensemble, elapsed: list[float], chunks: list[list[dict[int, tuple[Tensor, Tensor]]]]):
    """A decode that takes a NAMED number of seconds to reach each chunk, driving the budget as it goes.

    The existing drop tests set `spent` by hand and call `afford` directly, which proves the ordering
    rule and nothing about when the rule fires. What ships is a loop: the decode reports progress, the
    budget projects from it, and `afford` runs once per chunk. A simulated pace is the only way to ask
    whether the give-up lands on the chunk the projection actually turns on — and whether a run that
    keeps pace gives up nothing at all, which is the failure mode that costs votes for free.
    """
    budget = Budget(seconds=KERNEL_SECONDS, margin=1.0)
    ensemble.serving = replace(ensemble.serving, budget=budget)
    started = budget.started

    def chunks_for(_reads: list[Read]):
        for index, (seconds, chunk) in enumerate(zip(elapsed, chunks, strict=True), start=1):
            budget.started = started - seconds
            budget.chunk(index, len(chunks))
            yield chunk

    return chunks_for


def _charged(ensemble: Ensemble, member: str) -> None:
    """Charge a forward a FIXED second, so `worth` is decided by the holdout and not by the wall clock.

    `attempt` times itself off `time.time()`, and the two members here are the same fake backbone —
    their true costs are equal, so which one measures cheaper is decided by whatever else the machine
    is doing. That made this test read the ORDERING of two microsecond timings: it passed alone and gave
    up the STRONG member under a full suite on a loaded box, blocking the push. Charged here rather than
    by patching the clock, because `Budget` reads the same module and a faked clock stops the projection
    that the give-up is triggered by.
    """
    ensemble.spent[member] = ensemble.spent.get(member, 0.0) + 1.0


def _forwarded(ensemble: Ensemble, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every member the loop actually forwards, in order — the record `failed` cannot give afterwards."""
    seen: list[str] = []
    attempt = ensemble.attempt

    def spy(member: Member, sized: dict[int, tuple[Tensor, Tensor]], *, overlap: bool = True):
        seen.append(member.identifier)
        predicted = attempt(member, sized, overlap=overlap)
        _charged(ensemble, member.identifier)
        return predicted

    monkeypatch.setattr(ensemble, "attempt", spy)
    return seen


def _two_members(ensemble: Ensemble) -> None:
    """A cheap strong member and a cheap weak one, so worth-per-second is decided by the holdout."""
    member = ensemble.package.members[0]
    ensemble.package.members = [
        replace(member, identifier="strong", holdout=0.87),
        replace(member, identifier="weak", holdout=0.76),
    ]
    ensemble.models["strong"] = ensemble.models[member.identifier]
    ensemble.models["weak"] = ensemble.models[member.identifier]


def test_predict_groups_gives_up_the_weakest_at_the_chunk_the_projection_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A decode that falls behind mid-run costs the weakest member — from the chunk it falls behind, not before.

    Four chunks of a nine-hour ceiling. One hour spent at chunk 1 projects four hours and is affordable;
    six hours at chunk 2 projects twelve and is not. So `weak` must be forwarded on chunks 1 and 2 and
    on neither of the rest, and `strong` on all four. A give-up that fired at chunk 1 would throw away a
    vote a run that was on course did not need to lose; one that never fired would be killed at nine
    hours with nothing written, which is what three unscored entries actually did.
    """
    ensemble = _ensemble(tmp_path / "slow")
    _two_members(ensemble)
    reads = [ensemble.read_of(members) for members in ensemble.reads()]
    chunks = [[_chunk(read) for read in reads] for _ in range(4)]
    hour = float(SECONDS_PER_HOUR)
    chunks_for = _paced(ensemble, [hour, 6 * hour, 6.1 * hour, 6.2 * hour], chunks)
    seen = _forwarded(ensemble, monkeypatch)

    combined = ensemble.predict_groups(chunks_for, overlap=False)

    assert seen.count("strong") == 4, "the survivor votes on every chunk"
    assert seen.count("weak") == 2, "given up when the projection turned, not at the first chunk"
    assert combined.shape == (8, len(TARGETS)), "and the entry is still written, from what survived"


def test_predict_groups_gives_up_nothing_when_the_decode_keeps_pace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A run inside its ceiling keeps every member — the drop is a mitigation, not a tax.

    The same four chunks and the same two members, paced so the projection never leaves four hours.
    Worth the assertion because `afford` runs once per chunk whatever the pace: a projection that read
    an unmeasured budget as overrunning, or a margin applied twice, would shed votes on a run that had
    hours to spare and no log anybody reads would look wrong.
    """
    ensemble = _ensemble(tmp_path / "quick")
    _two_members(ensemble)
    reads = [ensemble.read_of(members) for members in ensemble.reads()]
    chunks = [[_chunk(read) for read in reads] for _ in range(4)]
    hour = float(SECONDS_PER_HOUR)
    chunks_for = _paced(ensemble, [hour, 2 * hour, 3 * hour, 4 * hour], chunks)
    seen = _forwarded(ensemble, monkeypatch)

    combined = ensemble.predict_groups(chunks_for, overlap=False)

    assert seen.count("strong") == 4, "no member is given up on a run that keeps pace"
    assert seen.count("weak") == 4, "least of all the weakest one"
    assert combined.shape == (8, len(TARGETS))


def test_pooled() -> None:
    """The serving answers with its own window policy, so no caller reaches through it to the pooling.

    THE DEFAULT ARM IS AN EQUALITY ORACLE: every array this project has served was a mean over windows,
    and a serving that pooled any other way by default would silently re-price the whole roster.
    """
    windows = torch.tensor([[[0.2] * len(TARGETS), [0.8] * len(TARGETS)]])
    assert torch.allclose(Serving().pooled(windows), windows.mean(dim=1))
    forced = Serving(pooling=WindowPooling.uniform("max")).pooled(windows)
    assert torch.allclose(forced, torch.full((1, len(TARGETS)), 0.8))
