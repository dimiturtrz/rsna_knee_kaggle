# Seven second terms against one control: the calibration corner is a local optimum

2026-08-27. A read of `experiments/README.md`, not a run. Zero card, zero submission slots.

## What was asked

`knee-orjq` (P0) is the bead that carries the whole remaining plan for 0.950. Its argument: no single arm
in 110 reaches the +0.010 offline that first place costs, so the route has to be a STACK of directional
gains — and in 110 arms this project had never put two measured-positive flags in one run. `stackfit`
was to be that test, pairing `--calibrated-targets` (+0.0061) with `--ema`.

`--ema` came in at +0.0010 against a pre-registration of ≥0.8640, so the arm was cancelled as
uninformative and the bead's next step was written as *find a second term whose single measured over the
floor, and pair the calib term with THAT one*. This is the read that was supposed to shortlist it.

## The shortlist does not exist, and the reason is the answer

`calib_dense_hold` — 336 px, 20 slices, band 0.1–0.9, `--calibrated-targets --hold-annotations`, fold 0,
seed 2026, twelve epochs, one-cycle — reads **0.8634** and is the control of every arm below. Each adds
exactly one further mechanism at that corner.

| second term | arm | holdout | delta |
|---|---|---|---|
| `--ema` | `stackfit` | 0.8631 | −0.0003 |
| `--slice-pooling labelwise` | `labelwise_dense` | 0.8613 | −0.0021 |
| `--finding-calibration` | offsets arm | 0.8595 | −0.0039 |
| warm start from a converged parent | `warm_calib_dense` | 0.8562 | −0.0072 |
| `--contrast-weight 1.0` | `contrast_dense` | 0.8547 | −0.0087 |
| `--through-plane` + `attend` + `--pool sided` | `fuse_attend` | 0.8509 | −0.0125 |
| `--auc-margin 1.0 --rank-weight 0` | `margin_calib` | 0.8235 | −0.0399 |

**Seven of seven below the control.** Three sit inside the 0.0062 paired floor and are FLAT rather than
negative-with-confidence — the individual magnitudes are not the claim. The SIGN is. Under a
null-additive model, where a mechanism-sound second term is as likely to land above the control as
below, seven of seven on one side is p = 2⁻⁷ = **0.008**. Every one of those arms was already on disk
and paid for; the reading cost twenty minutes.

## The ordering is monotone in how much the term changes

A weight-averaging trick costs 0.0003. A pooling swap, 0.0021. A per-finding offset, 0.0039. A warm
start, 0.0072. An auxiliary loss on the shared trunk, 0.0087. Three pooling changes at once, 0.0125. A
competing ranking objective, 0.0399.

That is not seven unrelated bad bets. It is the shape of a **local optimum**: the further a term moves
the recipe off this corner, the more it costs, with no threshold at which a term starts paying. Two of
the seven come with their own mechanism already worked out and pointing the same way — `fuse_attend`'s
three flags all reduce depth dilution and are "three spellings of one", and `margin_calib`'s margin term
optimises a boundary `--calibrated-targets` was chosen to blur. Both were read as arm-specific failures.
They are also two instances of one pattern.

## What this does not say

It does not say directional gains never compose. `2026-08-27_aggregation-beats-input` stands: fourteen
arms split cleanly by whether they added input or changed aggregation, four positives agreeing on a
sign. The project's own running total — 0.8272 → 0.8457 → 0.8523 — is a real chain of increments, and
CLAUDE.md's "a chain of them that sums past it IS a result" is not refuted here.

What is measured is narrower and load-bearing: **composition has failed every test made AT THIS CORNER**,
where the first term is already `--calibrated-targets`. A stack built on a different base is untried and
unaddressed. And the ceiling this establishes is on the SOLO member, not on the ensemble — the seat ruler
is a separate instrument with a separate answer.

## Consequence for the 0.950 plan

The remaining lane for a better solo is not another flag on our recipe. Seven flags at three levels of
the network have now been asked and none of them answered. The under-searched axis is a genuinely
different representation, which is exactly what `2026-08-27_the-pool-is-82-percent-one-number.md`
concluded from the other side: 81.7% of what separates our 105 members is one quality scalar, so the
pool has almost nothing left to trade.

That points at `knee-9g28` — `kneemri.evaluation.foreign_member`, built and gates-clean, **never run**,
needing only the card. It serves a foreign CoAtNet through its own decode, and the number it exists to
recover is the −0.0348 that our cache costs a checkpoint that reads 0.9167 through theirs. `knee-tlba`
(five more foreign checkpoints) queues behind it.

`knee-orjq`'s arm stays cancelled. What changed today is that it is cancelled by a measurement of the
ROUTE rather than by one term's null.

Beads: `knee-orjq`, `knee-9g28`, `knee-tlba`.
