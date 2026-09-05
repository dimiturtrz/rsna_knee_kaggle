# The CNN axis is maximally distinct and still worthless — the re-opener fired and it closed instead

**Date** 2026-09-05 · **Beads** knee-cj40f, knee-iyad · **Cost** zero card; both holdout records were
already on disk.

## What the ledger was owed

`radnet_pinned`'s row (`experiments/README.md`) closed the convolutional-trunk lane on its solo number
(0.7505, against a 0.8521 seat line) and then said explicitly what would re-open it:

> the CNN axis is not closed on its merits ... What it does license, card-free, is the per-case error
> correlation against the mean ViT-incumbent, which is the only measurement that could re-open it.

That measurement had not been taken. It is what `python -m kneemri.kaggle.seat_value` prints, and it
costs seconds. It was taken today against `honest12` (0.937 on the board, 12 members, all scoreable —
the champion `raptorweight_v1` cannot be priced against, because 5 of its 6 members are `--fold -1`
and have seen every holdout study).

## What it says

```
roster of 12: 0.8750
  +radnet_family_s2026.pt: 0.8746  (-0.0004)  [-0.0015, +0.0006]
  +radnet_pinned_s2026.pt: 0.8741  (-0.0009)  [-0.0017, -0.0001]
best seat each could take (displacing mae_matched): -0.0001 for both

member axes: roster of 12, boundaries rank 0.8956  js 0.0150  maha 4.319
  DISTINCT  radnet_family_s2026.pt: rank 0.4394  js 0.0251  maha 4.965
  DISTINCT  radnet_pinned_s2026.pt: rank 0.6415  js 0.0261  maha 4.871
```

Both CNNs read **DISTINCT on all three axes at once** — rank correlation 0.44 and 0.64 against a
roster boundary of 0.8956, Jensen-Shannon divergence 0.025-0.026 against 0.0150, Mahalanobis 4.87-4.97
against 4.319. `radnet_family` is the most distinct member this project has ever measured: its ordering
agrees with the seated pool less than half as much as the pool agrees with itself.

And the seat is **negative anyway**: −0.0004 and −0.0009 added, −0.0001 as a swap, 0 of 2 clearing the
+0.0040 line under which this ruler has never picked a board tick. The board charges a negative seat at
about 3x (knee-c0u9), so the honest price on the best of them is roughly −0.0002.

## Why this closes the axis rather than re-opening it

The re-opener was built on a real mechanism: a rank-average vote cannot charge a near-twin for
contributing one ordering twice, so `seat_value` alone is blind to a stranger, and `member_distance`
exists to cover that blindness (it was written after `sixfresh`/`sixstrong` split the board on exactly
that failure). The hypothesis was that the CNNs read as worthless BECAUSE the quality ruler is the only
one that had looked.

Both rulers have now looked, and they agree. A member can be maximally far from the pool on every
distance this project can compute and still be worth less than nothing to seat. That is the strongest
form the argument against the convolutional lane can take, and it is a measurement rather than an
inference from a solo score.

It also lands on the same side as NVIDIA's IJCAI-ECAI 2022 diversity result that `radnet_pinned`
pre-registered: a branch more than ~5 points off the incumbents measures as diverse and ensembles
worse, because diversity metrics track how differently models classify, not the mix's potential. These
two sit 10-19 points off. They are not diverse members; they are worse models that happen to be wrong
elsewhere.

**So the supervised-ImageNet VGG-16 / ResNet-50 arm stays unqueued**, which is what `radnet_pinned`'s
own gate said at < ~0.80, and the CNN lane now costs no further card on either of its two arguments.

## What it does NOT rule out

Nothing here says a convolutional trunk cannot work — it says these two, at these solo scores, cannot
buy a seat in this pool by being different. A CNN that reached the incumbents' 0.85 band would be a
different question, and the axis would be re-opened by producing one, never by re-quoting these
distances.

It also says nothing about `maha`, which is still behind its adoption gate (knee-iyad: the Mahalanobis
column must reproduce board-validated seats before it is read as a decision). It agreed with the other
two here, which is one data point toward that gate and not the gate itself.
