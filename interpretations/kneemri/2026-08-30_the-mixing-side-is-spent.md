# The mixing side is spent: three vote rules, two seat rules, and a flat curve over 77 honest members

2026-08-30. Offline, no card and no slot. `record_mix` over the **77 records whose checkpoint is named in
`comp/members.roster`**, 909 shared studies, fold 0. The roster is the filter and that choice is load-bearing —
see "the pool has to be built by provenance" below.

## The numbers

| seat rule | vote rule | honest peak | at seats | in-sample peak |
|---|---|---|---|---|
| greedy (honest) | logit | **0.8763** | 20 | 0.8777 @ 13 |
| greedy (honest) | median | 0.8756 | 13 | 0.8775 @ 21 |
| greedy (honest) | rank | 0.8745 | 23 | 0.8767 @ 11 |
| diverse | logit | 0.8727 | 43 | — |

Best honest solo in the pool: `calib_dense_hold_s2026`, 0.8634.

The paired holdout floor is **0.0062**. Every spread in that table is under it:

- **vote rule spans 0.0018** across logit / median / rank.
- **seat count is flat from six to twenty-three**: 0.8734 at six, 0.8763 at twenty, 0.8745 at twenty-three.
  The whole curve past the third seat lives in a 0.003 band.
- **the seat rule costs 0.0036** when it stops looking at the score — `diverse` peaks 0.8727 against greedy's
  0.8763, and needs 43 seats to get there.

And the pool grew 37% for nothing. The ledger's reading was 0.8754 at thirteen seats over 104 records; twenty-three
more honest members move that to 0.8763. **+0.0009.** That is the result: not that the peak is 0.8763, but that
adding two dozen trained members to an already-large pool is indistinguishable from adding none.

## Why decorrelation does not rescue it

`diverse` seats the member LEAST correlated with those already seated, so it is the direct test of the
diversity thesis with the score removed. Its trace prints the nearest correlation at each seat, and the
shape of that column is the mechanism:

```
diverse 2: 0.8613  +focus_mcl_s2026        (nearest 0.0445)
diverse 3: 0.8579  +mae_matched_s2026      (nearest 0.4342)
diverse 4: 0.8616  +vjepa_warm_s2026       (nearest 0.8119)
...
diverse 43: 0.8727 +calib_span_s2026       (nearest 0.9014)
diverse 77: 0.8720 +distill_pairs_s2026    (nearest 0.9404)
```

Genuinely decorrelated members EXIST — `focus_mcl` at 0.04, `mae_matched` at 0.43 — and seating them
*costs* score: the curve falls to 0.8579 by the third seat and does not recover past the honest greedy peak
at any count. By the fortieth seat the nearest neighbour is already 0.90 and rising to 0.94, so what remains
is near-copies. The pool is a small number of distinct opinions wearing seventy-seven names, which is the same
thing [the pool is 82 percent one number](2026-08-27_the-pool-is-82-percent-one-number.md) measured from the
other direction.

So the two ends are both closed. Decorrelated-and-weak does not pay, and strong-and-correlated is already
saturated. There is no third pile to draw from in this pool.

## What this licenses, and what it forbids

**Forbidden:** spending a card, a slot, or an hour on the mixing side. Seat count, vote rule and seat rule are
all read now, on an honest pool, and all three are under the floor. A future "try N seats" or "try median"
is re-measuring a settled flat. Quote this file, not a fresh scan.

**Licensed:** a gain has to come from a member that is genuinely a different opinion AND not weak — which is
an argument for spending the card on mechanism (`contrast_pool` and the representation lane, the topology
work in knee-2sbk) rather than on more members of the kind already here.

**CORRECTION 2026-08-30 05:40Z: two different pass-throughs were welded into one sentence.** This section
first read "pass-through is ~23%, so even a real +0.003 offline is +0.0007 on the board". The 23% is real
and it is not this. It comes from `experiments/README.md` line 4341 — "`calib_dense_hold` is +0.0056 solo
over `band_dense` and +0.0013 in a seat, a 23% pass-through" — which is **SOLO into SEAT**, a member's own
gain diluted by the ensemble it joins. The offline-to-BOARD ratio is a separate number and it was never
measured at 23%. `python -m kneemri.evaluation.board_transfer`, refit on the completed record:

```
compression: median +1.23 board points per holdout point over the WHOLE record
  among entries at or above 0.8670: 16 resolved pairs, median +0.98 [+0.94, +1.08]
  among entries at or above 0.8692:  4 resolved pairs, median +0.80 [+0.70, +1.19]
inversions: 9/392 pairs the board ordered the other way, 0/252 of the pairs the holdout RESOLVED
```

`BoardPair.ratio()` is board delta over holdout delta in raw AUC on both sides, so **+0.98 is one-for-one**:
where the entries being compared already sit at the top of our record, a holdout point buys a board point.
An offline +0.003 is +0.003 on the board, which is twice the board's own 0.0015 resolution rather than half
of it. This repo has now quoted this slope wrong three times — once at +4.6 (`2026-08-17_the_ensemble_axes`
line 99, "+0.0005 offline is +0.0023 board"), once at +0.23 here, both from prose. Run the module.

**The chain, stated in full so it is not welded again.** A member gain reaches the board through TWO
stages, and only the second is one-for-one:

```
member solo gain --(x0.23, solo->seat dilution)--> ENTRY offline gain --(x0.98, board_transfer)--> BOARD
```

So a +0.007 gain on ONE member is ~+0.0016 on the board — right at the board's 0.0015 resolution, which is
why every single-seat swap this project has shipped read flat (knee-xuj5's nine entries inside 0.002). The
0.23 is the dilution, not the transfer, and it is a property of changing ONE seat out of six.

**What that licenses, and it is the "bank the increments" branch `knee-m69u` names as one of its two closing
conditions.** The mechanism-sound increments measured and then discarded for sitting under the 0.0062 floor
— distillation +0.0035, label-table swap +0.0034, specialist columns +0.0043, supervision +0.0034 — have
never been STACKED into one recipe. Two moves follow, in order, and the second is the one that pays:

1. **Prove the stack on one member.** Additivity is an assumption, not a measurement: the terms may share
   mechanism (distillation and the supervision policy both move ORDERING; the label swap and the policy both
   act on the target table), in which case the stack is worth one term wearing four names. Two terms compose
   to +0.0069 offline solo, which clears the 0.0062 floor and is therefore READABLE — the first version of
   this plan that our own ruler can resolve.
2. **If it lands, retrain the seats, not one seat.** The 0.23 dilution applies to changing one member. When
   EVERY seat moves under the new recipe the entry's offline score moves with them, and THAT gain transfers
   one-for-one. `calibsix_v1` is the precedent: five members retrained under a different supervision policy,
   singles WORSE (0.923 vs 0.925), six-seat entry +0.002.

The 0/252 inversion count is the other half of the licence: where our holdout resolves a pair at all, the
board has never once ordered it the other way. The offline ruler is trustworthy exactly where it speaks, so
the job is to make it speak — build a stack big enough to clear the floor instead of measuring the terms one
at a time in a band the ruler cannot see.

**Unchanged:** the champion. `raptorweight_v1` (0.944, rank 27 of 2672 on the 2026-08-30T03:53Z read) is still
buildable and still the board's best row. Nothing here beats it and nothing here is shippable on its own.

## The pool has to be built by provenance, not by a glob

The first run of this scan globbed every `*.holdout.npz` in the processed root — 142 files, 127 mixable —
and returned **0.8994 at six seats**. That is +0.024 over the ledger, four times the floor, and it is false.
The four top solos in that pool were the foreign raptor checkpoints (`raptor_v4_swa` 0.8938, `raptor_v4`
0.8915, `raptor_v9` 0.8769, `raptor_effv2l480` 0.8703), every one of them above the best solo we own, and
`raptor_v4_swa`'s 0.8938 is the exact cell `board_check` voided on 2026-08-29 — 896 of our 909 fold-0 holdout
studies sit inside raptor's published label table, 98.6% in-sample. Three seats already read 0.8977.

`--largest-pool` cannot catch this. Its drop rule is "shares a split AND a label table", and a foreign record
decoded onto our studies under our table satisfies it — being mixable and being contaminated are the same
property here. `foreign_overlap` cannot catch it either: it takes a foreign label TABLE, so it guards where a
foreign model is priced and not where one is seated. The roster catches it only because the roster is
hand-maintained provenance (`raptor_distill_s2026` is on it because we trained it; the four `raptor_v*` are
not because they are theirs). That is a convention, not a guard. `knee-e4hq` is the fix: provenance in the
record, refusal by default.

**The failure mode to remember is not "a wrong number".** It is that the wrong number was plausible, in range,
arrived with a full curve and a monotone bias column, and was better than everything we own — which is exactly
what a real breakthrough looks like from the outside.
