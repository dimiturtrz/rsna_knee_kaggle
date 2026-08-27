# Seat count is flat within a fit regime — and the champion cannot be scored offline at all

**Read the corrections, not just the conclusion.** This document was drafted, then substantially
rewritten the same day after two of its three arguments turned out to use the wrong ruler and the wrong
comparison. What survives is the conclusion (seat count is not a lever) and one structural fact that is
worth more than it (the offline pool and the shipped roster are different populations). The withdrawn
arguments are left in place, marked, because the way they failed is the reusable part.

2026-08-27. Offline: `record_mix --largest-pool --seats honest --vote logit`, 42 packaged members that
carry a holdout record, narrowed to the largest agreeing pool, 909 shared studies. Board: `comp/board.jsonl`,
51 scored ensemble rows.

## What was asked

Whether adding seats to the champion roster is a lever worth a submission slot. The champion `calibsix`
serves **six** members and scores **0.941**. A twelve-seat roster is buildable from members already inside
the shipped `kneemri-members` package, so it costs no training and no upload — the cheapest thing on the
board, if it works.

## What the honest curve says

Greedy forward selection refitted on half the studies and scored on the other half, both directions —
the only rule this tool permits quoting, because a greedy order chosen and scored on the same studies is
in-sample by construction.

```
   1 members: held-out 0.8598   in-sample 0.8634   bias +0.0036
   5 members: held-out 0.8721   in-sample 0.8748   bias +0.0027
   6 members: held-out 0.8725   in-sample 0.8756   bias +0.0031
   9 members: held-out 0.8741   in-sample 0.8763   bias +0.0022
  12 members: held-out 0.8755   in-sample 0.8765   bias +0.0010   <- honest peak
  16 members: held-out 0.8749   in-sample 0.8763   bias +0.0014
  20 members: held-out 0.8741   in-sample 0.8759   bias +0.0018
  24 members: held-out 0.8740   in-sample 0.8753   bias +0.0014
```

Read naively this says twelve seats beats six by **+0.0030**, which at the fitted `board_transfer` slope of
+2.14 board points per holdout point predicts **+0.0064** — 0.941 to about 0.947, for free.

**That reading is wrong twice over, and neither error is the one I reached for first.** My first draft
killed it with the 0.0062 noise floor — the curve spans 0.0030, half the floor, so nothing here is a
finding. That argument does not hold: 0.0062 bounds an arm against an arm and carries training variance
these frozen predictions do not have (see below), and under the right ruler 0.0030 is nominally readable.

What actually kills it is the slope. `board_transfer`'s +2.14 was fitted on changes in member QUALITY —
a better member measured solo, then seated — and seat count is a different axis. A fit does not carry
across axes, so multiplying a count delta by it produces a number that answers no question. The
prediction of +0.0064 was never entitled to be made.

What survives from the table is weaker and honest: the peak at twelve is not a peak. There is no
monotone trend, the curve rises and falls inside three thousandths, and twelve is simply where the
scatter happened to be highest.

The in-sample column makes the same point from the other side: it is flat to four decimal places from
eleven seats to fifteen (0.8765, 0.8765, 0.8765, 0.8764), and the only thing moving across the table is the
**bias**, which shrinks from +0.0036 at one seat to +0.0010 at twelve. More seats does not buy accuracy here;
it buys *stability of the estimate*. Those are different goods and only the first one is a score.

## The board ran a related experiment, and it is NOT a count experiment

The twelve-seat roster the greedy order picks is `honest12`, and it was submitted, so there is a board
number for it. But it does not price seat count against `calibsix`, and saying it did was this document's
own first error. The two rosters share **one member of eighteen** (`merged_blend_cf`), and they differ in
FIT REGIME as much as in size: `calibsix` is six `_cf`/`_ff` members, `honest12` is ten `_s2026` plus two.
Count, membership and regime all move at once, so the −0.004 between them is not attributable to any of
the three.

The regime half of that is the half the ledger has already priced, and priced better than this
comparison could: every `_ff` member is its `_s2026` twin plus `--fold -1` **and** `--ema`, the `ema_dense`
arm separated the pair at fold 0 and read EMA at +0.0010 (a sixth of the floor), so `allfit`'s +0.004 is a
**data** result — training on all folds rather than holding one out. See `experiments/README.md`,
`ema_dense`. A board-wide sort by regime agrees in direction (pure-`_s2026` entries top out at 0.933;
`_cf`/`_ff`-dominated ones bottom out at 0.936) but it is confounded with date — `_cf`/`_ff` adoption
begins 2026-08-24 and the board rises with the calendar regardless — so that sort corroborates the
controlled arm and must not be quoted over it.

The members, for the record:

```
calib_dense_hold_s2026, merged_blend_cf, finding_calib_dense_s2026, blendhold_s2026,
margin_seat_s2026, calib_span_s2026, through_dense_s2026, small_batch6_s2026,
mae_matched_s2026, base224_s2026, calib_span_hold_s2026, merged_dense_cf
```

It scored **0.937** against `calibsix`'s **0.941** — but per the section heading, read that as a regime
result with count riding along, not the reverse.

**The offline curve cannot see the champion at all, and that is the more useful fact.** Five of
`calibsix`'s six members are `--fold -1`: they have seen every holdout study, so they carry no
`.holdout.npz` and no offline number exists for them, ever. The 42-member pool this curve was computed
over is the `_s2026` family. So every offline reading in this project — this curve, the greedy order, the
seat values — describes a family whose best board entry is **0.933**, while the family that ships sits at
**0.941**. The ledger records this per-arm in several places (`experiments/README.md` lines 412, 2802,
4149, 4301); the consequence worth stating once is that the offline ruler and the board are not measuring
the same population, which is a standing discount on every offline→board transfer this project quotes.

Within a single family the count axis is flat, which is the claim this document can actually support:

```
pure _s2026:   4 seats 0.930   5 seats 0.931   6 seats 0.931   7 seats 0.933   8 seats 0.932
_cf/_ff-led:   6 seats 0.941   7 seats 0.937   8 seats 0.940  11 seats 0.941
```

Each family spans about 0.003 across its whole count range with no monotone trend — the `_s2026` line
drifts up to seven seats and back down at eight, the `_cf`/`_ff` line is unordered in count entirely.
That is at the board's own resolution of roughly 0.001 ticks. **Stratified by regime, both rulers agree
the count axis is flat**, and the earlier version of this table — which pooled the families and so let
the regime gap masquerade as a count effect — is withdrawn.

The floor those spreads should be read against is **not** the 0.0062 quoted above. That number bounds an
ARM against an ARM and carries training variance; two rosters built from the same frozen member
predictions carry none. `MemberCurve.swap` already uses the right order of magnitude — `BOARD_RESOLVABLE`
= 0.0015, fitted on six paired offline/board entries — and a paired bootstrap on the competition metric
at this scale independently reads a 95% band of **0.00134** for two near-identical arms against 0.01663
for two independent ones. Two unrelated routes to the same resolution. `LabelScores.paired_mean_interval`
now measures it per comparison instead of borrowing a constant (knee-oned).

Under a 0.0015 ruler a 0.003 span is nominally resolvable, so "flat" here rests on the absence of a
monotone trend and on the regime stratification, not on the span being unreadable.

## What this costs and what it buys

It costs the cheapest submission idea on the table. A zero-upload twelve-seat roster looked like +0.006 for
nothing; it is a repeat of a submission already scored 0.004 *below* the champion. Finding that out cost one
CPU-only run of a tool that already existed, and it protects the last slot of the day.

What it buys is a negative that is worth more than the arm it killed: **seat COUNT is not a lever, and the
`board_transfer` slope must not be applied to it.** That slope was fitted on changes in member QUALITY —
a better member measured solo, then seated. Seat count is a different axis, and the fit does not carry
across axes. Quoting +2.14 against a count delta is the same class of error as the compression constant
being quoted from prose twice (`board_check`), and it nearly bought a submission.

## What is still open, and it is the same thing it was this morning

Selection, not count. Within the `_cf`/`_ff` family eleven seats ties six at 0.941 and twenty-five
collapses to 0.894, so the roster is made or lost by WHICH members sit, not how many.

I closed the first draft by saying the honest curve cannot answer that — its range sits inside its noise
floor, so it cannot rank two rosters of the same size. **That was the wrong ruler, and the claim is
withdrawn.** At a 0.0015 resolution, and with `paired_mean_interval` able to bound one named pair
directly, roster-vs-roster differences of a few thousandths ARE readable. The lane is open, not closed.

Two things bound what that buys. The comparison is only honest for a pair named IN ADVANCE — read on the
winner of a search over rosters, a tight band is selection-optimistic in exactly the way `LabelCombine`
describes, and a searched roster still owes the split-half treatment. And the pool it can rank is the
`_s2026` family, whose board ceiling is 0.933; the roster that actually ships is drawn from members that
carry no holdout record at all. So a better ruler over this pool sharpens a question about the family we
do not submit. Closing that gap — pricing `_cf`/`_ff` members against each other — needs a held-out fold
they were deliberately trained without, which is why nothing offline has ever measured the champion.

The one member-level result from the same run that is NOT flat: `contrast_dense_s2026`, which lost **0.0087**
solo (outside the floor, a real loss), takes **greedy seat five** on the full 104-record pool. A member can be
worse alone and still earn an early seat — the owner's standing point about diversity over averaging, measured
on our own pool rather than asserted. That is a statement about a member, on an axis where the deltas clear
the floor; the seat-count table above is not. See `experiments/README.md`, `contrast_dense`.

Related: `knee-9g28` (price a SWAP rather than a marginal addition), `knee-unmb` (a new member costs a 7 GB
package push), and the board seat-count rows in `comp/board.jsonl`.
