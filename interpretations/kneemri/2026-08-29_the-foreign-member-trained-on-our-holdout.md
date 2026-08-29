# The foreign member's +0.030 is contamination: it trained on 896 of our 909 holdout studies

2026-08-29, 17:44Z. No card, no slot. One overlap query against two files already on disk.

## The anomaly that started it

Two readings of ONE checkpoint, `raptor_v4_swa`, disagree by about 0.05 and both are in the repo:

- `knee-9g28` (closed 2026-08-27): **0.8938 on our 909-study fold-0 holdout**, against our best member
  `calib_dense_hold_s2026` at 0.8634. `+0.0304`, and `0.0%` fallback, so it is over the whole holdout.
- `knee-rh4t` (note, 2026-08-28): the same checkpoint submitted solo scored **0.923 on the board**,
  *exactly level* with our own best single (`calibsolo`, 0.923).

`board_transfer` fits at +1.72 to +1.82×, so a genuine +0.0304 offline predicts roughly +0.05 on the
board. It delivered 0.000. Nothing in this project's transfer record compresses a five-floor offline
lead to nothing, and the two numbers were never put beside each other until now.

## The measurement

Their published label table `D:/data/external/raptor/labels/labels_llm_soft.csv` against our own fold
assignment (`ReportFolds`, 5 folds, seed 2026):

```
corpus 4407 | fold0 holdout 909 | their labels 4349
holdout in their label file: 896 / 909
gold in their label file:      0 / 58
holdout that is gold:         13
```

The arithmetic closes exactly: `909 = 896 + 13`. Their table is the corpus **minus the 58 gold studies
and nothing else**, and our holdout is **98.6% inside it**. The 13 of our holdout studies they do not
carry are precisely the gold studies that fall in fold 0.

That row-set is the signature of *train on everything, hold out the oracle only* — and it is
corroborated from their side: the metric their checkpoints self-report is `gold_auc` on the 58, which
is only a validation metric if the gold is the only thing they withheld.

## What it does to the three rulers

| ruler | theirs | ours | held out for BOTH? | verdict |
|---|---|---|---|---|
| gold 58 | 0.9167 | 0.9152 | yes | **null** |
| our fold-0 909 | 0.8938 | 0.8634 | **no — theirs is 98.6% in-sample** | +0.0304 |
| board hidden test | 0.923 | 0.923 | yes | **null** |

Two clean rulers read null. The single contaminated ruler carries the entire +0.030. That is not three
noisy points around a small positive; it is two agreements and one artefact, and the artefact is the
one with a mechanism.

## The correction this owes

`interpretations/kneemri/2026-08-27_the-feed-was-the-whole-shortfall.md` reads the same pattern
backwards. Its closing section observes that going from the 58 to the 909 our member falls 0.0587 while
theirs falls only 0.0252, and concludes *"the small set flatters ours."* The direction is the other
way: on the small set both models are honest and they tie, and on the large set theirs stops being
honest. **The large set flatters theirs.** Everything else on that page — the feed diagnosis, the
0.0% fallback, the per-finding table — stands; only the cross-set comparison inverts.

## What it kills, and what it does not

**KILLS: the member gap, as a quantity.** Every remaining support for "rivals hold a 0.013–0.017 better
single model" is now gone:

- the forum self-report of 0.938–0.942 — a single unreproduced number, already flagged on `knee-rh4t`;
- the +0.030 offline lead — contaminated, measured above;
- the board — a null;
- the gold 58 — a null.

On every ruler where both models are honest, the best published member of this field and our best
member are **level**. Six open P0s rest on member quality being the free term; that premise now has no
measurement behind it at all.

**DOES NOT KILL: the foreign member as a seat.** Seat value was priced offline on the contaminated
ruler too, so `raptor_v4_swa`'s seating numbers on the 909 are unreadable in both directions — the
0.8928-solo and the champion-six-plus-raptor 0.8910 are equally affected. This says the pricing is
void, not that the seat is bad. Re-pricing needs a ruler it did not train on, which on our side is the
58 gold studies (too small to seat on) or the board (a slot).

**DOES NOT TOUCH:** the 224px throughput argument (`knee-s9cr`) rests on our own ledger. The
`--calibrated-targets` result, the aggregation chain, and every purely-internal comparison are
unaffected — this is a contamination in ONE foreign checkpoint family, not in our fold.

## Where it points

`knee-fuhq`. Two independently-built models, different architecture, different decode, different
training corpus, different label reader, landing on the same 0.923 on a hidden test is what a **shared
supervision ceiling** looks like, and it is now the only reading left standing. Today's own
`supervision_gap` run over six `--hold-annotations` members is independent evidence for it from our
side: `labels - model: -0.0498 [-0.0742, -0.0255]`, the model beating its own labels on ten of twelve
findings, on models we fitted rather than on a stranger's.

The standing caution on that lane holds and is not weakened by any of this: `blendlabels` measured the
label lever at `+0.0034` on the metric and `+0.0070` on the six findings a reader can reach, and
`table_blend` brackets every re-weighting of the readers already on disk inside ±0.008 on the 58. The
lane is confirmed live and confirmed small. What this page adds is that it is also the only one left.

## Rule this earns

**A foreign checkpoint may not be scored on any split of the corpus it was trained on, and the default
assumption is that it was trained on all of it.** Ours is the only fold assignment that respects our
folds; a competitor's model has seen our holdout unless their own published artefacts prove otherwise.
The check is one set intersection against whatever row set they publish, it costs no card and no slot,
and it runs BEFORE the number is read — not two days after it has driven six P0s.

See `knee-9g28` (closed on the contaminated number), `knee-rh4t`, `knee-fuhq`, `knee-49pp`.
