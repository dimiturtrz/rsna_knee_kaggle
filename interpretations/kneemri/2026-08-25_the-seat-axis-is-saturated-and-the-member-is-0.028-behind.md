# The seat axis is saturated, and our member is 0.028 behind the field's

**2026-08-25, from `comp/board.jsonl` — 53 scored submissions, our own record, no forum prose.**

> **CORRECTION 2026-09-01 (knee-macc, knee-3uje). Two of this document's three load-bearing claims have since been withdrawn or
> superseded. Read the corrections before the body.**
>
> **1. The title's `0.028` is WITHDRAWN — it is a two-ruler artefact** (knee-3uje, 2026-08-27; already
> carried in knee-xuj5's notes and never written here). It compared our BOARD single (`soloseat_v1`,
> Kaggle's hidden test) against rivals' FORUM self-reports. On the 58 radiologist-annotated studies both
> sides hold out, `calib_span_hold` reads 0.9152 on the gold 58 against `raptor_ft_coatnet_v4_full`'s 0.9167 — a delta of
> 0.0015 against a macro standard error of about 0.012 at n=58. **Our member is level with theirs.** The
> 0.028 is the distance between the rulers.
>
> **2. The best-by-count table below is stale, and the "+0.024 then turns over" shape does not survive.**
> `board_check` completed the record on 2026-08-25 (it held 34 of 58 submissions; fifteen scored all-ours
> entries were missing) and 34 further submissions have scored since. Recomputed 2026-09-01 over all 87
> rows, excluding `scope: excluded`:
>
> | members | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 11 | 12 | 25 |
> |---|---|---|---|---|---|---|---|---|---|---|---|
> | n | 5 | 5 | 4 | 12 | 5 | 12 | 19 | 5 | 1 | 1 | 1 |
> | best | 0.925 | 0.936 | 0.938 | 0.937 | 0.931 | 0.943 | 0.944 | 0.943 | 0.941 | 0.937 | 0.894 |
>
> Best single is **0.925**, not 0.914; a TWO-member entry reads 0.936 and a THREE-member entry 0.938,
> where this document recorded 0.921 and 0.913. So the span from one seat to the best entry we own is
> **+0.019**, not +0.024, and most of it is bought by the second and third seat — everything from four
> seats up sits inside 0.937–0.944.
>
> **What that does NOT license.** Count and member quality are confounded across these rows: the
> high-reading small entries are late ones built from better members, and no matched count sweep exists.
> So this is not "count is worth less than we thought" measured — it is the count table no longer
> supporting the claim it was quoted for. The conclusion the body draws (**slots do not go to seat
> permutations**) is untouched and is independently measured: nine 6/7-member entries span 0.002, and
> every single-seat operation at the top since — `holdswap_v1`, `vjepaseat_v1`, `spanseat_v1` — read flat.
>
> **3. The `+1.82` transfer constant used throughout the body has the same cause** (knee-8pp8). It was
> fitted on the incomplete record; the completed one reads +1.55 whole-record, and in the regime this
> document argues about (offline ≥ 0.867) `board_transfer` reports **+0.99 [+0.77, +1.18]**. Every
> board→holdout conversion below is therefore roughly 1.8× too small: "0.002 board is 0.0011 holdout"
> is really ~0.002, and "0.028 board is 0.0154 holdout" would be ~0.028 — except that the 0.028 is
> withdrawn anyway (correction 1). The document's own section "How firm is that 0.0154 — less than it
> reads" was already circling this defect from the inside; it is now named.

## The table that settles it

Best board score at each member count, over every submission we have ever scored:

| members | n  | best  | entry           | worst |
|---------|----|-------|-----------------|-------|
| 1       | 2  | 0.914 | `solo_v4`       | 0.868 |
| 2       | 4  | 0.921 | `ratepair_v1`   | 0.882 |
| 3       | 3  | 0.913 | `trio_v1`       | 0.882 |
| 4       | 11 | 0.930 | `greedy4_v1`    | 0.918 |
| 5       | 5  | 0.931 | `basedense_v1`  | 0.868 |
| 6       | 6  | 0.938 | `votefit_v1`    | 0.925 |
| 7       | 13 | 0.937 | `allfit_v1`     | 0.908 |
| 8       | 1  | 0.932 | `peak8_v1`      | 0.932 |
| 25      | 1  | 0.894 | `fusion25_v2`   | 0.894 |

Two facts, both ours:

**Ensembling works and is finished.** One member reads 0.914 and six read 0.938 — combination buys
**+0.024**, which is the largest single effect on this board and larger than anything the training
recipe has ever bought. It then STOPS: seven is 0.937, eight is 0.932, twenty-five is 0.894. The curve
does not plateau, it turns over.

**The seat is exhausted to below our own floor.** Nineteen entries sit at six or seven members. The top
NINE of them — `votefit`, `allfit`, `lastseat`, `dark6`, `dark7`, `farseat`, `augseat`, `v3seat`,
`finalfit` — span **0.936 to 0.938**. That is nine distinct member sets, chosen by nine different
arguments, landing inside **0.002**. The board's own transfer constant is +1.82 points per holdout
point and our holdout floor is 0.0062, so 0.002 board is 0.0011 holdout: roughly a fifth of the
smallest difference we can measure. Those nine entries are one number reported nine times.

## What that costs us against the field

The competition's discussion 735304 reports single models at 0.915, 0.92, 0.926, 0.929, 0.934, 0.938,
and **0.942 with tweaks at 224 px** — and carries the remark "highly doubt the top results are
ensembles, look at the efficiency lb". Set beside our table:

- their best reported SINGLE model, 0.942, beats our best SIX-MEMBER ensemble, 0.938;
- it beats our best single member, 0.914, by **0.028**.

### The thread does not use "single model" the way I read it

Read verbatim at 23:35Z, the thread's own wording splits: two of the strong entries say
**"single(5-fold) model"** (0.929 @ 224 px, 0.926 @ 336 px), and a 5-fold submission is five
checkpoints averaged — an ensemble of five by our own table's definition, where our own best five reads
0.931. Those two are not evidence about a single checkpoint at all, and against our 5-member row they
are FLAT. The 0.942 entry ("Single model 0.934 @ 224px Update: 0.942 with some tweaks") does not say
which it is.

What survives unambiguously is **"I'm at 0.938 with a single model, single fold"** and "single model
0.915, DinoV2 based". The first still beats our best single, 0.914, by **0.024**, and still matches our
best six. So the ordering — their member beats our ensemble — holds on the one post that states its
own terms. The deficit is real; the specific 0.028 was read off a post that may be describing a five.

**"with some tweaks" names nothing.** No augmentation, schedule, backbone, or pooling detail appears
anywhere in the thread. There is no method to replicate here, only a target — which is why tonight's
card goes to a mechanism we can price ourselves rather than to a reconstruction of somebody's hint.

Divided by the transfer constant, 0.028 board is **0.0154 holdout** — which is, to two decimals, the
1.5% significance bar this project set for itself before any of this was measured. The field's member
is exactly one real result ahead of ours, and our +0.024 of ensembling is what has been hiding it.

### How firm is that 0.0154 — less than it reads

The transfer constant was fitted on ENSEMBLE submissions, and there is exactly one single-model pair in
the record to check it against: `solo_v1`, offline **0.8518**, board **0.868**. `solo_v4` is the better
single at board 0.914 and carries no offline number at all. Run the constant across those two: 0.046
board between them implies 0.0253 holdout, putting `solo_v4`'s member near **0.877** — higher than any
member holdout we have ever recorded (the best is `band_dense_long`'s 0.8587 at epoch 6).

That inconsistency is the finding, not a rounding error. Either the single-member mapping is steeper
than +1.82, or `solo_v4` gained board score from something other than its member's holdout (serving,
TTA, overlap), or one of the two entries is not what its row says. **So 0.0154 is an estimate carrying
one confirmed anchor and one contradiction, and it should be quoted as "about one bar's worth", never
as a measured target.** What survives the uncertainty is the ORDERING, which needs no constant at all:
their single beats our six, so the deficit is in the member.

The leader sits at 0.952 against our 0.938 (rank 50 of 2404, read live at 2026-08-25T23:15:59Z). If
they sit on the same transfer curve — an assumption, not a measurement, since that constant was fitted
on OUR submissions — the gap is 0.0077 holdout, about 1.24× our noise floor. Small in holdout terms.
Unreachable in seat terms.

## What follows, and what does not

**A slot spent on another seat permutation is a slot spent measuring 0.002.** Nine entries already say
so. This is the same conclusion `knee-uco3` reached from the joint-selection side and the forum reached
from the outside, now stated from our own board record with a number attached.

**It does not say ensembling was a mistake.** +0.024 is the biggest thing on the board and it is banked.
It says the axis is CLOSED, not that it was wrong — and the top nine's tightness is itself the evidence
that the combination machinery is well built, since nine different member sets extract the same value
from it.

**It does not close the external-data swap.** All nine flat entries permute same-family members. The
one member trained on external expert IMAGE reads — `merged_blend_s2026`, from Rijeka — has never been
on the board inside a six. `kneemri-blendseat` is that test and is the one seat question still open;
it is a different KIND of swap, and on this project's own rule (`merged_corpus` moved Lateral Meniscus
+0.0196 on a corpus whose labels never carried that finding) the gain from foreign data has come
through the ENCODER rather than through the supervision.

**Everything else goes to the member.** The regulariser lane running tonight — `mixup224` at α=0.2 and
`mixup224_strong` at α=0.4 against `native224` = 0.8502 — is the only new member mechanism in flight,
and this table is why it is where the card is pointed. The prize it must reach is 0.0154 holdout, and
nothing smaller than that changes our position, because the 0.024 that ensembling would have added on
top is already spent.

Related: [`2026-08-25_effusion-threshold-ruler.md`](2026-08-25_effusion-threshold-ruler.md) ·
beads `knee-uco3`, `knee-x580`, `knee-jk3`, `knee-ombx`.
