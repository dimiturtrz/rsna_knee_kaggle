# Three trunk families were closed on a silenced objective, and the term is being priced for free

**Date:** 2026-09-04 · **Beads:** knee-tj88y (open), knee-if3ze (closed) · **Status:** census complete, price in flight

## The mechanism

`experiments/recipe.sh:23` defines the house recipe as
`--fold 0 --one-cycle --weight-floor --rank-weight 1.0`. Every arm in the ledger carries it.

`RankedObjective.ranking` forms pairs WITHIN the micro-batch (`pairs(batch, batch)`), and self-pairs
drop out. At `--batch 1` no pair forms, so the ranking term contributes **exactly zero**.
`--accumulate` cannot repair it: it sums the gradients of micro-batch losses each computed alone.

`Objective.settings()` reports the CONSTRUCTOR's weight, so the recipe line records
`rank_weight: 1.0` either way. The run looks correct in every artefact a human reads.

`knee-if3ze` found this on `ortho_stack` and fixed that one arm. **The census was never taken.**

## The census

Sixteen arms run `--batch 1` and none of them carries `--pair-memory`:

```
cnx_dense_448   cnx_mid_rate    cnx_warm_anneal   cnx_top_rate
cnx_own_recipe  cnx_distill_warm
raptor_feat_fit raptor_feat_pre
swin_toprate    swin_supervised swin_open_half    swin_wholetop
swin_distill_pre swin_distill_fit swin_calib_dense
radnet_family
```

That is the whole ConvNeXt family, the whole Swin family, the RadImageNet arm and the raptor
feature-distillation pair. Every one was compared against a control trained at batch 12 with the term
ON, so **every one moved two variables** — the trunk and the objective — and every family was closed
on that comparison.

`radnet_family` 0.6716 is the number recorded in `comp/members.declines` closing the
trunk-**architecture**-family lane, which the standing policy lists as CLOSED.

(`--batch 2` arms — `band_dense_llrd`, `cnx_dense`, `encoder_adapt_thawed`, `vit_in21k`,
`radiodino_corpus` — form C(2,2) = 1 pair a label against the control's 66. Weakened by a factor of
66, not silenced. A separate and lesser question.)

## The term is priced, free, in flight

The two `ortho_stack` runs differ by exactly one flag, `--pair-memory 66`. Same trunk, same seed,
same data order, same everything else:

| epoch | term OFF (void run) | term ON | delta |
|---|---|---|---|
| 0 | 0.7389 | 0.7389 | +0.0000 |
| 1 | 0.7778 | 0.7837 | +0.0059 |
| 2 | 0.8010 | 0.8090 | +0.0080 |
| 3 | 0.8146 | 0.8234 | +0.0088 |

Growing with epoch. Epoch 0 is identical to four decimals, which is what a term that shapes ranking
rather than magnitude should do before the model has any ranking to shape.

## The pair cannot converge, and the price is readable anyway

Written 2026-09-04 01:05Z, correcting the sentence that stood here.

That sentence read "**Nothing is re-run until this pair converges.**" **The pair can never converge.**
The term-OFF half was KILLED at epoch 3 — it exists as four rows in
`ortho_stack_s2026.metrics.1.jsonl` and no process will ever add a fifth. Gating a P0 on a
convergence that cannot arrive is a gate that never opens, and it sat here for a day.

The matched-epoch price is readable from the four epochs that DO exist, and the shape settles it.
The per-epoch INCREMENT is **+0.0059, +0.0021, +0.0008** — the delta is not growing linearly, it is
saturating, and an asymptote near **+0.009 to +0.010** is what that curve extrapolates to. It is
above the 0.0062 report floor, so the term is real; and it is bounded well under +0.015, which is
what the decision below needs.

This is a PARTIAL read on ONE trunk (ViT-L, batch 1, `--pair-memory 66`), not a converged number,
and nothing here says the same size transfers to Swin or ConvNeXt. But the SIGN and the ORDER of
magnitude are what the decision turns on, and both are settled.

## The decision the price licenses: do not re-run the sixteen

At a generous **+0.015** — half again the extrapolated asymptote — against `band_dense` 0.8578 and
the 0.8459 seat bar:

```
swin_toprate     0.8393 -> 0.8543    under the incumbent by 0.0035
swin_wholetop    0.8331 -> 0.8481    at the seat bar, inside the floor of it
swin_supervised  0.8192 -> 0.8342    under the seat bar
cnx_own_recipe   0.8186 -> 0.8336    under the seat bar
radnet_family    0.6716 -> 0.6866    nowhere
```

**Not one of them reaches the incumbent even at the generous price.** So re-running sixteen arms
buys a corrected number for members that still cannot be shipped — sixteen arms at ~5 h each is
about eighty hours of card for zero seats. That ratio is not close.

What the price does NOT retire is the paragraph below on the DECISION to stop a family, and that is
where the recoverable value sits. A Swin family whose best corrected arm reads 0.854 against
0.8578 is a family that was closed at a 0.0035 gap, not the 0.0185 gap on file. That is a family
one arm away, not a dead one — and the pool it would join is DINOv2-monoculture, where
`ortho_dense` (the LOWEST offline of all 37 board entries, 0.8452) served alone and read 0.907,
about +0.02 above the DINOv2-fitted slope. So the argument for a corrected Swin arm is SEAT
DIVERSITY, not solo quality, and it must be priced as such: ~5 h for one member landing ~0.85 solo
whose whole case is that it is not a DINOv2.

**Revised gate: re-run none of the sixteen. ONE corrected arm on the strongest family
(`swin_toprate`'s flags plus a derived `--pair-memory`) is the only defensible spend, and it is
gated behind `ortho_stack` and `v3large_stack`, which ask the same diversity question on a trunk
that already has a board point.**

## What the arithmetic already says, before the price lands

Known void scores, against `band_dense` 0.8578 and the 0.8459 seat bar:

```
swin_toprate     0.8393        radnet_family    0.6716
swin_wholetop    0.8331        cnx_own_recipe   0.8186
swin_supervised  0.8192
```

If the term converges worth +0.010–0.015, **only `swin_toprate` plausibly reaches the seat bar**, and
none reaches the incumbent. So the likeliest outcome is a correctness debt, not a rescued member.

**But the families were abandoned on these numbers.** What this re-opens is not the members — it is
the DECISION to stop. A family whose first two arms read 0.819 and 0.833 gets no third arm; a family
whose first two arms read 0.833 and 0.847 does. That is the cost, and it is not recoverable by
re-running the sixteen.

## What this does not say

It does not say the ranking term is worth +0.008 — that is an epoch-2 reading on one trunk and the
converged number is what counts. It does not say any of the sixteen would clear a bar. And it does
not touch any arm at `--batch 6` or above, which is the whole DINOv2 fleet, the champion, and every
member currently on the roster.

The guard is already in place: `run_setup._check_pairs_can_form` raises before the card is touched
and names all four silenced flags, so none of the sixteen can re-run as written. The fix is to add a
derived `--pair-memory` to each header, never to relax the guard.
