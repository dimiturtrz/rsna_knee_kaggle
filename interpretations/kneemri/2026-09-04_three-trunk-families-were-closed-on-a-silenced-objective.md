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

Growing with epoch. Epoch 0 is identical to four decimals, which is what a term that shapes ranking
rather than magnitude should do before the model has any ranking to shape.

**Nothing is re-run until this pair converges.** It licenses or kills all sixteen at once and costs
zero card, because both halves were already going to be run.

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
