# The seat axis is saturated, and our member is 0.028 behind the field's

**2026-08-25, from `comp/board.jsonl` — 53 scored submissions, our own record, no forum prose.**

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
