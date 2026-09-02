"""The card a run may hold, and what a recipe reserves — every term of it fitted or derived.

Four arms have died to a term of this model being fitted on one configuration and handed to another,
so each test here names the incident it stands for rather than merely the property it checks.
"""

import logging

import pytest
import torch

from kneemri.device import VisibleCards
from kneemri.models.backbone import CONVNEXT_BASE, ENCODERS, SMALL, V3_BASE, V3_SMALL, Encoder
from kneemri.training.device_budget import (
    CAP_HEADROOM,
    CONTENTION_POLL_SECONDS,
    CONTENTION_WAIT_SECONDS,
    DEFAULT_SHARE,
    FITTED_BATCH,
    FITTED_DEPTH,
    FITTED_PARAMS_M,
    FITTED_RETAINED,
    FITTED_TRAINABLE,
    FITTED_UNITS,
    GB_PER_SLICE,
    OPTIMISER_GB_PER_MPARAM,
    RESIDENT_GB,
    RETAINED_WEIGHT,
    SMALLEST_RUN_GB,
    TRAINABLE_FLOOR,
    DeviceBudget,
)


def _priced(encoder: Encoder, size: int, *, slices: int = 12, batch: int = FITTED_BATCH) -> DeviceBudget:
    """A budget for a trunk at a size with the FITTED number of blocks open.

    Both activation terms come from the same encoder, because that is what `RunSetup.budget` does — a
    caller that scales the trunk and leaves `retained` at the fitted point is describing a recipe
    nobody runs.
    """
    return DeviceBudget(
        slices=slices,
        units=encoder.activation_units(size),
        batch=batch,
        retained=encoder.retained_units(size, FITTED_TRAINABLE),
    )


def test_select():
    """An uncapped budget still yields a device; the cap is applied only where there is a card.

    The unit suite runs without a GPU, so this pins the contract that matters here — `select` is the
    one place a run's device is decided, and it must work with no CUDA present rather than being a
    CUDA-only path that silently never runs.
    """
    device = DeviceBudget().select()

    assert device.type in {"cuda", "cpu"}
    assert DeviceBudget(vram_gb=8.0).select().type == device.type, "a budget must not change WHICH device"


def test_floor():
    """The refusal threshold is the cheapest run on record, or the run's own budget when that is lower.

    IT IS DELIBERATELY NOT THE CAP. The cap defaults to 0.8 of the card, and refusing on that would
    refuse whenever the co-tenant holds a fifth of it — common, and harmless to a 5.5 GB smoke.
    """
    assert DeviceBudget().floor() == SMALLEST_RUN_GB, "silence is refused only below our cheapest run"
    assert DeviceBudget(vram_gb=26.0).floor() == SMALLEST_RUN_GB, "a big budget does not raise the floor"
    assert DeviceBudget(vram_gb=3.0).floor() == 3.0, "a run that claims to fit in less is taken at its word"


def test_select_refuses_a_card_that_stayed_held(monkeypatch: pytest.MonkeyPatch):
    """Past the wait's bound, a card short of the cheapest run is refused rather than started into.

    THE FAILURE THIS CLOSES cost both projects on 2026-08-17: starting into a held card does not raise
    on Windows, it pages over PCIe and crawls, and the pair of us then drew enough to reset the driver.
    A refusal is loud, immediate and re-runnable; the crawl was none of those.
    """
    monkeypatch.setattr(VisibleCards, "device", lambda _: torch.device("cuda", 0))
    monkeypatch.setattr(DeviceBudget, "await_headroom", lambda _: (1.0, 32.0))

    with pytest.raises(RuntimeError, match="Refusing rather than paging"):
        DeviceBudget().select()


def test_headroom():
    """Free VRAM is readable without a card, and free never exceeds the total.

    The suite runs on CPU, where the honest answer is (0, 0) — the point is that asking costs nothing
    and returns a pair rather than raising, since `select` reads it before every run on every machine.
    """
    free, whole = DeviceBudget().headroom()

    assert free <= whole, "free VRAM cannot exceed the card"
    assert (free, whole) == (0.0, 0.0) or whole > 0, "a visible card reports a positive total"


def test_await_headroom(monkeypatch: pytest.MonkeyPatch):
    """A held card is waited out, and a card held by something stuck is waited out only so far.

    THE CO-TENANT IS PERMANENT: a second project of the same owner trains on this same card, so
    contention is a condition to schedule around rather than an incident to report. Waiting costs a
    queued arm minutes; starting anyway cost, on 2026-08-17, three driver resets that killed BOTH
    projects' processes.

    The sleep is injected because this must not spend the wall clock it is describing.
    """
    readings = iter([(4.0, 32.0), (6.0, 32.0), (30.0, 32.0)])
    monkeypatch.setattr(DeviceBudget, "headroom", lambda _: next(readings))
    slept: list[float] = []

    free, whole = DeviceBudget(vram_gb=22.0).await_headroom(slept.append)

    assert (free, whole) == (30.0, 32.0), "it returns the headroom the run actually starts on"
    assert slept == [CONTENTION_POLL_SECONDS] * 2, "and polls only while the card is short"

    held = iter([(1.0, 32.0)] * 1000)
    monkeypatch.setattr(DeviceBudget, "headroom", lambda _: next(held))
    stalled: list[float] = []

    assert DeviceBudget(vram_gb=22.0).await_headroom(stalled.append) == (1.0, 32.0), "it gives up and says so"
    assert sum(stalled) == CONTENTION_WAIT_SECONDS, "a co-tenant that hung cannot hold the queue all night"

    idle: list[float] = []
    monkeypatch.setattr(DeviceBudget, "headroom", lambda _: (31.0, 32.0))

    assert DeviceBudget().await_headroom(idle.append) == (31.0, 32.0)
    assert not idle, "a run that named nothing has no need to wait ON, and the whole card is never free"

    oversized: list[float] = []
    monkeypatch.setattr(DeviceBudget, "headroom", lambda _: (31.0, 32.0))

    assert DeviceBudget(vram_gb=33.0).await_headroom(oversized.append) == (31.0, 32.0)
    assert not oversized, (
        "a need larger than the CARD cannot be waiting for a co-tenant: on 2026-09-02 "
        "ortho_fresh_lowlr wanted 31.9 GB of a 31.8 GiB card and polled the full half hour with the "
        "card idle at 1229 MiB, mrnet_warm parked behind it, the log reading like contention throughout"
    )


def test_headroom_without_a_card(monkeypatch: pytest.MonkeyPatch):
    """No visible card answers (0, 0) rather than asking the driver, which would raise.

    THE MACHINE THAT RUNS THIS SUITE HAS A CARD, so the branch that keeps a CPU box working is the one
    nothing here exercises — and `select` reads it before every run, on every machine.
    """
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    assert DeviceBudget().headroom() == (0.0, 0.0), "asking costs nothing and returns a pair"


def test_select_warns_when_it_starts_into_a_busy_card(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Past the wait's bound, a card that is short but not FATAL is started into — and says so.

    The symptom is not a failure and cannot be read off the throughput afterwards: Windows pages over
    PCIe and the run looks like a slow recipe. So the contention is named before the first batch,
    which is the whole difference between this and the crawl that cost both projects on 2026-08-17.
    """
    monkeypatch.setattr(VisibleCards, "device", lambda _: torch.device("cuda", 0))
    monkeypatch.setattr(DeviceBudget, "await_headroom", lambda _: (10.0, 32.0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    fractions: list[float] = []
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction, _: fractions.append(fraction))

    with caplog.at_level(logging.WARNING):
        device = DeviceBudget().select()

    assert device.type == "cuda", "short of the cap is not short of the floor, so the run starts"
    assert "another process holds the rest" in caplog.text, "named before the first batch, not inferred after it"
    assert fractions == [pytest.approx(0.8)], "and the cap it starts under is still applied"


def test_capped():
    """A run that named a budget gets it; one that named nothing still gets a cap.

    THE SECOND HALF IS THE POINT. Uncapped was the default until now, which made this class's own
    worst failure — Windows paging VRAM over PCIe, so an over-committed run crawls at a reported 100%
    utilisation instead of raising — the behaviour a run got by SAYING NOTHING. Every accepted arm
    names its budget, so the default was never what a measured run did, only what a smoke or a
    one-off got.
    """
    assert DeviceBudget(vram_gb=26.0).capped(31.8) == 26.0, "an explicit budget is used as asked"
    assert DeviceBudget().capped(31.8) == pytest.approx(DEFAULT_SHARE * 31.8), "and silence still caps"
    assert DeviceBudget().capped(31.8) < 31.8, "the driver context and the desktop live outside the allocator"
    assert DeviceBudget(vram_gb=28.0, slices=12).capped(31.8) == 28.0, "a named number still overrides it"
    # THE SHARE MUST NOT CLAMP A DERIVED NEED. 20 slices reserve 26.92 GB measured and the share is
    # 25.44, so the first version of `capped` — a `min` against the share — would have capped every
    # 20-slice arm below its own measured demand. Those arms RAN, at a hand-typed 28. A cap under a
    # measured need does not prevent the over-commit, it schedules it.
    assert DeviceBudget(slices=20).capped(31.8) > DEFAULT_SHARE * 31.8, "a measured need outranks the blind default"
    assert DeviceBudget(slices=20).capped(31.8) == pytest.approx(CAP_HEADROOM * (26.92 + 1.335), abs=0.02)
    # AND IT MUST ONLY EVER RAISE THE CEILING (knee-l93q). Handing `needed()` to the allocator as a
    # ceiling made a recipe that matched its own prediction OOM: `dinov3_base_lora`, which this repo ran
    # to 0.8268, died at "15.14 GiB is allocated by PyTorch ... 16.27 GiB allowed", and so did the arm
    # under test beside it. Re-smoked at a 1.10 margin it read "17.90 GiB allowed ... 1.93 GiB is
    # reserved by PyTorch but unallocated" — peak ALLOCATION 16.15, under the 16.27 derived. The fit is
    # right and the cap governs a different quantity, so a 12-slice recipe must get the co-tenant share
    # rather than a ceiling fitted to itself.
    assert DeviceBudget(slices=12).capped(31.8) == pytest.approx(DEFAULT_SHARE * 31.8), (
        "a derivation below the share must not tighten the ceiling under it"
    )
    assert DeviceBudget(slices=12).capped(31.8) > DeviceBudget(slices=12).needed() * 1.127, (
        "and it must clear the measured reserved-to-allocated ratio, not sit on the prediction"
    )
    assert DeviceBudget(vram_gb=26.0, slices=12).capped(31.8) == 26.0, "a MEASURED number gets no margin"


def test_needed():
    """The fitted line, checked at the point it was NOT fitted on.

    12 and 20 slices (16.24 and 26.92 GB reserved) give the coefficients; 16 was held out and measured
    21.57. The margin is one slice, so `needed` sits a slice above each measured peak rather than on it.
    """
    held_out_measured = 21.57
    one_slice = 1.335
    assert DeviceBudget(slices=16).needed() == pytest.approx(held_out_measured + one_slice, abs=0.02)
    assert DeviceBudget(slices=20).needed() > DeviceBudget(slices=12).needed(), "more slices, more memory"
    assert DeviceBudget().needed() == 0.0, "an unstated recipe predicts nothing rather than guessing"

    fitted = ENCODERS[SMALL]
    at_336 = DeviceBudget(slices=12).needed()
    assert _priced(fitted, 336).needed() == at_336, "the fit's own encoder at the fit's own size is the default"
    at_420 = _priced(fitted, 420).needed()
    assert at_420 == pytest.approx(at_336 * (420 / 336) ** 2, abs=0.3), (
        "a slice is a patch grid, so it costs the square of the side"
    )
    assert at_420 > 23.42, "the cap must clear the rung MEASURED at 420 px, or it forbids a recipe that fits"


def test_needed_reads_the_encoder():
    """The cap is a function of the ENCODER too, and every arm behind the fit must be untouched by it.

    `v3base_band` died in seconds on 2026-08-20 with `Tried to allocate 848.00 MiB ... 12.68 GiB
    allowed` while 17.51 GiB of the card was free: a `small`-fitted cap handed to a 768-wide encoder.
    The first assertion is the one that protects the ledger — the whole DINOv2 corpus is patch 14 and
    384 wide, so the new terms must reduce to exactly the old number for every arm already recorded.
    """
    v2 = ENCODERS[SMALL]
    for size in (224, 336, 392, 420):
        scale = (size / 336) ** 2
        legacy = DeviceBudget(slices=12, units=FITTED_UNITS * scale, retained=FITTED_RETAINED * scale).needed()
        assert _priced(v2, size).needed() == pytest.approx(legacy), (
            "naming small's own structure cannot move a number the arms were fitted with"
        )

    small, base = ENCODERS[V3_SMALL], ENCODERS[V3_BASE]
    narrow = _priced(small, 224, slices=20).needed()
    wide = _priced(base, 224, slices=20).needed()
    assert wide == pytest.approx(2 * (narrow - RESIDENT_GB) + RESIDENT_GB, abs=0.02), (
        "activation bytes go linearly with width, so 768 costs twice 384 above the resident intercept"
    )
    assert wide > 12.68, "the derivation must clear the cap that OOM'd, or the fix forbids the same recipe"

    pixel_only = FITTED_DEPTH * (224 / 14) ** 2 * base.dim
    patch14 = DeviceBudget(slices=20, units=pixel_only, retained=pixel_only / 2).needed()
    assert patch14 > wide, "224 px is 256 tokens at patch 14 and 196 at 16, so a pixel-only cap overstates v3"


def test_needed_reads_a_trunk_that_is_not_a_transformer():
    """The final grid is not the bill for a hierarchical trunk, and reading it as one killed an arm.

    `cnx_dense`'s smoke died with `Tried to allocate 1.77 GiB ... 2.84 GiB allowed` while 28.03 GiB of
    the card sat free. The cap had priced a ConvNeXtV2 by its OUTPUT grid — 121 tokens at patch 32 —
    when three quarters of its blocks run at 16x that grid or finer. This is the assertion that says
    the model now sees them, and it is stated against the MEASURED peak rather than against itself.
    """
    convnext, fitted = ENCODERS[CONVNEXT_BASE], ENCODERS[SMALL]
    output_grid_only = FITTED_DEPTH * (352 / convnext.patch) ** 2 * convnext.dim
    assert convnext.activation_units(352) > 7 * output_grid_only, (
        "the early stages dominate: the real bill is 7.75x what the output grid alone accounts for"
    )

    measured_peak_gb = 10.49
    derived = _priced(convnext, 352, slices=20, batch=2).needed()
    assert derived > measured_peak_gb, "a cap under the measured demand is the one failure this class exists to stop"
    assert DeviceBudget(slices=20, units=output_grid_only, batch=2).needed() < measured_peak_gb, (
        "and the old form is BELOW it, which is why the smoke OOM'd — the test would pass on the bug otherwise"
    )

    assert convnext.activation_units(352) > fitted.activation_units(352), (
        "36 blocks of convolution cost more than 12 of attention at the same read size"
    )


def test_needed_reads_the_batch():
    """The batch is a factor of the ask, and every arm at the settled batch must be untouched by it.

    MEASURED 2026-08-20: `base_dense` derived 56.3 GB against 30.3 free and parked for the whole
    contention bound without starting a step. Lowering `--batch` to 6 derived 56.3 AGAIN — the one lever
    a caller has against an overstated ask moved nothing, because batch was baked into `GB_PER_SLICE`
    as a constant. That did not make the larger-encoder axis expensive, it made it unreachable.
    """
    settled = DeviceBudget(slices=20).needed()
    assert DeviceBudget(slices=20, batch=FITTED_BATCH).needed() == settled, "the fit's own batch is the default"

    halved = DeviceBudget(slices=20, batch=FITTED_BATCH // 2).needed()
    assert halved == pytest.approx((settled - RESIDENT_GB) / 2 + RESIDENT_GB, abs=0.02), (
        "activation bytes go linearly with batch above the intercept, where the weights are held once"
    )

    units = ENCODERS[V3_BASE].activation_units(336)
    wide = DeviceBudget(slices=20, units=units, batch=FITTED_BATCH).needed()
    narrower_step = DeviceBudget(slices=20, units=units, batch=FITTED_BATCH // 2).needed()
    assert narrower_step < wide, "a wider encoder must be reachable by asking for fewer studies at once"


def test_needed_reads_the_parameter_count():
    """Optimiser state is 16 bytes a parameter that no slice loop multiplies, and the cap held it fixed.

    `v3base_dense` was refused outright on 2026-08-20 — `9.17 GiB is free; 21.68 GiB allowed` — with
    nine gigabytes of the card unused, because the intercept was DINOv2-small's 0.22 GB handed to a
    trunk four times its size (knee-bj90). The fitted point must not move: every recorded arm is a 22M
    trunk, and an intercept that changed there would change every cap in the ledger.
    """
    fitted = DeviceBudget(slices=20).needed()
    assert DeviceBudget(slices=20, params_m=FITTED_PARAMS_M).needed() == fitted, "the fitted trunk is the default"
    assert fitted == pytest.approx(26.92 + GB_PER_SLICE, abs=0.01), "and it still derives the ledger's own peak"

    bigger = DeviceBudget(slices=20, params_m=FITTED_PARAMS_M + 64).needed()
    assert bigger == pytest.approx(fitted + 64 * OPTIMISER_GB_PER_MPARAM, abs=1e-6), (
        "4 bytes of master weight, 4 of gradient and 8 of Adam's two moments, per parameter"
    )
    assert bigger - fitted == pytest.approx(1.024, abs=0.01), "a whole gigabyte the refused arm could not see"

    smaller = DeviceBudget(slices=20, params_m=FITTED_PARAMS_M / 2).needed()
    assert smaller == fitted, (
        "below the fitted point the intercept is MEASURED and holds — it carries the allocator's own "
        "blocks, which scale with nothing, so extrapolating it downwards would understate a real cost"
    )


def test_needed_reads_how_many_blocks_supervision_opened():
    """Retention goes with the TRAINABLE blocks, and holding it at six refused a run at its own number.

    `encoder_adapt_thawed` asks for `--unfreeze-last 12`, derived 9.56 GB at batch 4 and died with
    `Tried to allocate 204.00 MiB ... 9.56 GiB allowed` while 9.55 GiB of the card sat free. A cap that
    had already scaled itself down correctly for the batch and then refused the run anyway is the
    signature of a MISSING term rather than a mis-fitted one: a frozen block's activations are freed as
    soon as the next block consumes them, a trainable block's must survive to its backward.

    The term is AFFINE, not a ratio (knee-pf8b). A pure ratio priced a frozen block at zero, doubled the
    ask at twelve open where the measurement doubles it by 1.27, and refused `unfreeze4_s2026` at 11.79
    GB against its own measured 14.79. It is carried in UNITS rather than in blocks (knee-rwjs), which
    on this uniform-block trunk is the same arithmetic said differently.
    """
    fitted = DeviceBudget(slices=20).needed()
    assert DeviceBudget(slices=20, retained=FITTED_RETAINED).needed() == fitted, (
        "the default `unfreeze_last` is the fitted point, so every arm in the ledger is untouched"
    )

    opened = TRAINABLE_FLOOR * FITTED_UNITS + RETAINED_WEIGHT * FITTED_UNITS
    ratio = opened / (TRAINABLE_FLOOR * FITTED_UNITS + RETAINED_WEIGHT * FITTED_RETAINED)
    thawed = DeviceBudget(slices=20, retained=FITTED_UNITS).needed()
    assert thawed == pytest.approx(ratio * (fitted - RESIDENT_GB) + RESIDENT_GB, abs=1e-6), (
        "twelve open costs 1.27x six open, not 2x — `encoder_adapt_thawed` measured 6.13 GB against 9.12 derived"
    )

    frozen = DeviceBudget(slices=20, retained=0.0).needed()
    assert frozen > 0.7 * fitted, (
        "a trunk with NOTHING open still runs every forward and still holds what the blocks above it "
        "read — only gradients and Adam state go away, so the cost of opening blocks has a floor"
    )

    assert DeviceBudget(slices=20, batch=FITTED_BATCH // 3, retained=FITTED_UNITS).needed() < fitted, (
        "and the batch stays the lever against it — thirding the studies must outweigh doubling the depth"
    )


def test_needed_prices_the_open_suffix_at_its_own_width_not_the_trunk_average():
    """A block count cannot price a trunk whose blocks differ, and two of ours do (knee-rwjs).

    Two trunks with the SAME total activation bill and the same number of blocks open, differing only
    in whether the open ones are the trunk's cheap end or its average, must not derive the same cap.
    Under the block-count form they did — the scalar multiplied a whole-trunk sum, so where the open
    blocks sat in the trunk was invisible. `cnxbase` opens its last six into 9.7% of its bill.
    """
    average = DeviceBudget(slices=20, units=4 * FITTED_UNITS, retained=4 * FITTED_RETAINED).needed()
    cheap_end = DeviceBudget(slices=20, units=4 * FITTED_UNITS, retained=0.1 * FITTED_RETAINED).needed()
    assert cheap_end < average, "opening the narrow end of a hierarchical trunk retains less, and must cost less"
