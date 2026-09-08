import math
import random
from types import SimpleNamespace
from typing import override

import pytest
import torch
from torch import Tensor, nn

from kneemri.training.attention_health import RETENTION_FLOOR, AttentionHealth, AttentionProfile

TOKENS = 16
PREFIX = 1
KEYS = PREFIX + TOKENS
HEADS = 2
BATCH = 2
BLOCKS = 12
UNIFORM = math.log(KEYS)

random.seed(0)
torch.manual_seed(0)

D252_KEYS = 257
D252_UNIFORM = math.log(D252_KEYS)
D252_STOCK = (3.388, 3.201, 3.358, 2.593, 3.239, 3.677, 4.273, 3.794, 3.991, 4.004, 3.067, 4.294)


class _Attentive(nn.Module):
    """A backbone returning attention maps of a chosen shape, and nothing else.

    Chosen rather than learned because this gate's whole job is a comparison between two readings, and
    the three cases that decide it — ordinary, uniform, and locked onto one key — are exactly the three
    an untrained real encoder will not reliably hand you.
    """

    def __init__(self, *, peak: float = 3.5, sink: tuple[int, ...] = (), maps: bool = True) -> None:
        super().__init__()
        self.peak = peak
        self.sink = sink
        self.maps = maps

    def _block(self, index: int) -> Tensor:
        logits = torch.zeros(BATCH, HEADS, KEYS, KEYS)
        if index in self.sink:
            logits[..., 0] += 30.0
            return logits.softmax(dim=-1)
        return (logits + torch.eye(KEYS) * self.peak).softmax(dim=-1)

    @override
    def forward(self, pixel_values: Tensor, output_attentions: bool = False) -> object:
        blocks = tuple(self._block(index) for index in range(BLOCKS))
        return type("Output", (), {"attentions": blocks if self.maps else ()})


class _Switchable(nn.Module):
    """A backbone that records which attention kernel it was asked for, as a real one does on its config."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="sdpa")

    def set_attn_implementation(self, implementation: str) -> None:
        self.config._attn_implementation = implementation


def _images() -> Tensor:
    return torch.zeros(BATCH, 3, 8, 8)


def _health(model: nn.Module | None = None) -> AttentionHealth:
    return AttentionHealth.baseline(_Attentive() if model is None else model, _images(), TOKENS)


def _d252(entropy: tuple[float, ...]) -> tuple[AttentionHealth, AttentionProfile]:
    """The real stock profile as a reference, and one reading against it, at D252's 257 keys."""
    reference = AttentionProfile(D252_STOCK, (0.0,) * len(D252_STOCK), D252_KEYS)
    health = AttentionHealth(tokens=256, reference=reference)
    return health, AttentionProfile(entropy, (0.0,) * len(entropy), D252_KEYS)


def test_baseline():
    """The reference is MEASURED off the encoder as it starts, not pasted from a letter.

    Measured because a constant array is wrong the moment the image size, the patch or the trunk changes
    — the same defect the first version's `ln(N)` reference had one layer down.
    """
    health = _health()
    assert len(health.reference.entropy) == BLOCKS
    assert health.reference.keys == KEYS
    assert health.probe is not None, "the reference's own images are what later readings are compared on"
    assert not health.degenerate(health.reference), "an encoder must never be degenerate against itself"

    wider = AttentionHealth.baseline(_Attentive(peak=0.0), _images(), TOKENS)
    assert wider.reference.entropy[0] > health.reference.entropy[0], "a flatter encoder must read flatter"


def test_uniform():
    """Reported, not used as the threshold — three of four degenerate checkpoints sit at the far end."""
    assert _health().uniform() == pytest.approx(UNIFORM)
    assert AttentionHealth(tokens=TOKENS, reference=AttentionProfile((), ())).uniform() == 0.0


def test_span():
    """How wide stock itself is — the number that broke the absolute margin this gate replaced."""
    health, _ = _d252(D252_STOCK)
    assert health.span() == pytest.approx(4.294 - 2.593)
    assert AttentionHealth(tokens=TOKENS, reference=AttentionProfile((), ())).span() == 0.0


def test_maps():
    """One map per block, a REFUSAL when the backend served none, and a REFUSAL with no probe batch.

    The first refusal is the point of the gate: a fused-attention backend returns no maps rather than
    failing, which turns the check into a silent pass — the exact shape of failure it was built to end.
    The second is what makes a reading falsifiable: with the batch held, an `--lr 0` run reads +0.00 on
    every block, so any deviation the gate reports is the weights and not the sample.
    """
    health = _health()
    assert len(health.maps(_Attentive())) == BLOCKS

    with pytest.raises(RuntimeError, match="no attention maps"):
        health.maps(_Attentive(maps=False))

    with pytest.raises(RuntimeError, match="not comparable"):
        AttentionHealth(tokens=TOKENS, reference=health.reference).maps(_Attentive())


def test_profile():
    """Both statistics, per block, so a reading names WHICH degeneracy arrived and not only that one did."""
    profile = _health().profile(_Attentive(sink=(11,)))
    assert len(profile.entropy) == len(profile.prefix_mass) == BLOCKS
    assert profile.keys == KEYS
    assert profile.prefix_mass[11] > 0.99, "an attention sink puts nearly everything on the prefix"
    assert profile.prefix_mass[0] < 0.5


def test_entropy():
    """The RAW softmax row over ALL keys, from patch queries — nothing dropped, nothing renormalised.

    D251 dropped the prefix column and renormalised, and on a block where the class token has taken the
    mass that divides ~0 by ~0: it reported an entropy of 0.000 with a max weight of 0.000, which is
    impossible for a distribution. The sink block below is exactly that case, and it must read LOW.
    """
    health = _health()
    flat = torch.full((1, 1, KEYS, KEYS), 1.0 / KEYS)
    assert health.entropy(flat) == pytest.approx(UNIFORM, abs=1.0e-4)

    logits = torch.zeros(1, 1, KEYS, KEYS)
    logits[..., 0] += 30.0
    assert health.entropy(logits.softmax(dim=-1)) < 0.01, "all the mass on one key is the far end from uniform"


def test_prefix_mass():
    """How much of a patch query's mass lands on the prefix — the statistic that names an attention sink."""
    health = _health()
    logits = torch.zeros(1, 1, KEYS, KEYS)
    logits[..., 0] += 30.0
    assert health.prefix_mass(logits.softmax(dim=-1)) > 0.99

    flat = torch.full((1, 1, KEYS, KEYS), 1.0 / KEYS)
    assert health.prefix_mass(flat) == pytest.approx(PREFIX / KEYS, abs=1.0e-4)


def test_class_mass():
    """The class token's own share — the half of the prefix that is a defect rather than a mechanism.

    The two cases below are the whole reason the field exists. `prefix_mass` scores ~1.0 on BOTH: a
    trunk whose patch queries have collapsed onto the class token, and one whose queries have parked
    on its four registers exactly as Darcet et al. intend. Summed they are one number and the second
    reads as the first; split, the register case leaves `class_mass` near zero.
    """
    health = _health()
    registers = 4
    wide = KEYS + registers

    onto_class = torch.zeros(1, 1, wide, wide)
    onto_class[..., 0] += 30.0
    onto_class = onto_class.softmax(dim=-1)
    assert health.prefix_mass(onto_class) > 0.99
    assert health.class_mass(onto_class) > 0.99

    onto_registers = torch.zeros(1, 1, wide, wide)
    onto_registers[..., 1 : 1 + registers] += 30.0
    onto_registers = onto_registers.softmax(dim=-1)
    assert health.prefix_mass(onto_registers) > 0.99
    assert health.class_mass(onto_registers) < 0.01


def test_retention():
    """The share of stock's own room a block still holds, whichever end it is closer to losing.

    A ratio and not a number of nats, because the previous version was a number of nats: 0.8, fitted
    where stock spanned 1.7, which refused a healthy encoder on the first configuration it met at 2.48.
    """
    health, _ = _d252(D252_STOCK)
    assert health.retention(health.reference) == pytest.approx((1.0,) * len(D252_STOCK))

    # `adapt_small_e8_unitfix` blocks 8-11: the failure TOWARD uniform, in headroom terms.
    toward = health.retention(_d252((*D252_STOCK[:8], 5.372, 5.507, 5.532, 5.512))[1])
    assert toward[8] == pytest.approx((D252_UNIFORM - 5.372) / (D252_UNIFORM - 3.991), abs=1.0e-3)
    assert max(toward[8:]) < 0.12, "every one of the four sits far below the floor"

    # `adapt_small_e6` block 11 at 0.000: the failure AWAY from it, in entropy terms.
    away = health.retention(_d252((*D252_STOCK[:11], 0.0))[1])
    assert away[11] == pytest.approx(0.0)
    assert min(away[:11]) == pytest.approx(1.0), "the other eleven blocks were healthy and must read so"


def test_retention_with_no_room():
    """A reference already at an end kept no room to lose, so a reading there is not a loss."""
    reference = AttentionProfile((0.0,), (0.0,), KEYS)
    health = AttentionHealth(tokens=TOKENS, reference=reference)
    assert health.retention(AttentionProfile((0.0,), (0.0,), KEYS))[0] == pytest.approx(1.0)


def test_degenerate():
    """TWO-SIDED, because the collapse has two opposite shapes and a one-sided gate misses three of four.

    `adapt_small_e8_unitfix` failed toward uniform; `adapt_small_e6`, `adapt_small_e6_lr1e5` and
    `pretrain_small_matched_e8` failed the other way, at 0.000 / 0.035 / 0.023 nats. A margin-to-`ln(N)`
    test reads those three as healthier than stock and passes the worst checkpoint most comfortably.
    """
    health = _health()
    assert not health.degenerate(health.reference)

    uniform = AttentionProfile((UNIFORM,) * BLOCKS, health.reference.prefix_mass, KEYS)
    assert health.degenerate(uniform), "the failure the first version was built for"

    silent = AttentionProfile((0.0,) * BLOCKS, health.reference.prefix_mass, KEYS)
    assert health.degenerate(silent), "the failure it passed, three times out of four"

    # ANY block, not the top ones together: D252's fronts start as low as block 5, inside the half
    # `unfreeze_last: 6` freezes, where the fine-tune has no gradient to repair the damage.
    deep = list(health.reference.entropy)
    deep[5] = 0.0
    assert health.degenerate(AttentionProfile(tuple(deep), health.reference.prefix_mass, KEYS))

    with pytest.raises(RuntimeError, match="block count"):
        health.degenerate(AttentionProfile(health.reference.entropy[:3], health.reference.prefix_mass[:3], KEYS))


def test_floor_separates_the_d252_populations():
    """The floor is a MEASUREMENT on five checkpoints, so the arithmetic that sized it is asserted here.

    Every degenerate cell D252 reported keeps under 0.115 of stock, stock keeps 1.0, and the floor is
    inside that band with room on both sides. If a later edit moves the floor out of it, this fails.
    """
    health, _ = _d252(D252_STOCK)
    degenerate = (5.372, 5.507, 5.532, 5.512, 0.0, 0.035, 0.023)
    worst = max(min(health.retention(_d252((value,) * len(D252_STOCK))[1])) for value in degenerate)
    assert worst < RETENTION_FLOOR < 1.0
    assert worst < 0.115, "the degenerate population sits below this; the floor must clear it"


def test_attention_health_report():
    """The reading beside its own reference, so a log line carries the comparison it is judged on."""
    health = _health()
    line = health.report(health.reference)
    assert "prefix" in line
    assert "vs stock" in line
    assert "keep" in line
    assert f"{RETENTION_FLOOR:.2f}" in line


def test_attention_profile_report():
    """`AttentionProfile.report` prints both statistics, one entry per block, for a log line to carry."""
    line = AttentionProfile((1.0, 2.0), (0.1, 0.2), KEYS).report()
    assert line == "H 1.00  2.00 | prefix 0.10  0.20"


def test_readable():
    """The read runs in EAGER attention and in eval, and BOTH are restored — the run trains under neither.

    Eager because transformers 5.x defaults to `sdpa`, whose fused kernel never materialises the attention
    matrix: `output_attentions=True` then returns a tuple of `None` and the whole gate reads as a clean
    pass. Eval because stochastic depth and dropout are drawn fresh per forward, so two readings of ONE
    encoder would differ by their draws rather than by their weights.
    """
    model = _Switchable()
    model.train()
    health = AttentionHealth(tokens=TOKENS, reference=AttentionProfile((), ()))
    with health.readable(model):
        assert model.config._attn_implementation == "eager"
        assert not model.training
    assert model.config._attn_implementation == "sdpa"
    assert model.training


def test_register_mass():
    """The benign half of a sink, which is the whole reason the two masses are reported as a pair.

    Both blocks below carry a large prefix mass and one of them is a defect. The first is dinov2's
    shape — a prefix of one token, so every unit of it is on the class token and the difference is
    exactly zero. The second is what a register trunk is supposed to look like when it works: the same
    mass arrives, and almost none of it lands on the class token.
    """
    profile = AttentionProfile(entropy=(1.0, 1.0), prefix_mass=(0.75, 0.83), class_mass=(0.75, 0.04))

    masses = profile.register_mass()

    assert masses[0] == 0.0
    assert masses[1] > masses[0]


def test_patch_entropy():
    """A sink cannot lower this by leaving, which is the whole reason it is measured beside `entropy`.

    One block, two readings. As served, a row that hands a prefix key almost all of its mass has very
    little entropy left over the patches, and the raw statistic reports that collapse. Renormalised, the
    patches are still spread exactly as evenly as they were, so the number is the uniform one — which is
    the true statement about what a downstream head reads through this block.
    """
    logits = torch.zeros(BATCH, HEADS, KEYS, KEYS)
    logits[..., 0] += 30.0
    sunk = logits.softmax(dim=-1)
    health = _health()

    assert health.entropy(sunk) < math.log(TOKENS) / 2
    assert health.patch_entropy(sunk) == pytest.approx(math.log(TOKENS), abs=1e-3)


def test_patch_entropy_reported():
    """The reading carries it, so a log line is judged on the statistic the sink does not move."""
    profile = AttentionProfile(entropy=(1.0,), prefix_mass=(0.9,), class_mass=(0.0,), patch_entropy=(2.5,))

    assert "Hpatch 2.50" in profile.report()


def test_axis():
    """A register sink stops being a refusal, and a class-token sink does not.

    Both profiles below collapse on the raw statistic. The first is the register case: the mass left
    the patches, and what remains is spread. The second is the class-token lock, where renormalising
    recovers nothing because there was nothing behind the sink. The gate must separate them, and the
    axis is what does it -- the floor is untouched.
    """
    health = _health()
    stock = AttentionProfile(entropy=(5.0,), prefix_mass=(0.0,), class_mass=(0.0,), patch_entropy=(5.0,))
    benign = AttentionProfile(entropy=(0.5,), prefix_mass=(0.9,), class_mass=(0.0,), patch_entropy=(4.8,))
    locked = AttentionProfile(entropy=(0.5,), prefix_mass=(0.9,), class_mass=(0.9,), patch_entropy=(0.1,))
    gate = AttentionHealth(tokens=health.tokens, reference=stock, probe=health.probe)

    uniform, values, reference = gate.axis(benign)

    assert values == benign.patch_entropy
    assert reference == stock.patch_entropy
    assert uniform == pytest.approx(math.log(TOKENS))
    assert not gate.degenerate(benign)
    assert gate.degenerate(locked)


def test_axis_falls_back():
    """A reading taken before the statistic existed is judged on the ruler it was taken with."""
    health = _health()
    stock = AttentionProfile(entropy=(5.0,), prefix_mass=(0.0,), keys=KEYS)
    older = AttentionProfile(entropy=(0.5,), prefix_mass=(0.9,))
    gate = AttentionHealth(tokens=health.tokens, reference=stock, probe=health.probe)

    uniform, values, _ = gate.axis(older)

    assert values == older.entropy
    assert uniform == pytest.approx(math.log(KEYS))
