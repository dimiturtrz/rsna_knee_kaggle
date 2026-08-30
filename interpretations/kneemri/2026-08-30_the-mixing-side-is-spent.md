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
work in knee-2sbk) rather than on more members of the kind already here. The board agrees from its own side:
pass-through is ~23%, so even a real +0.003 offline is +0.0007 on the board.

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
