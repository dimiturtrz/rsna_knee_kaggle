# The board names its own price: 0.946 is +0.0047 on one member

2026-08-31. A re-read of everything the project has recorded — `comp/board.jsonl` (82 submissions, 73
scored), the 145 `*.metrics.jsonl` runs on disk, the interpretation set and `research/`. The point was to
find out whether what we have already tried says anything we have not read out of it. It says three
things, all of them arithmetic over records that were already there.

## 1. Every board entry above 0.937 carries no honest offline number

Sorted by board score, the top of our 73 scored submissions is:

| board | entry | offline |
|---|---|---|
| 0.944 | `raptorweight` | none — in-sample |
| 0.944 | `holdswap` | none — in-sample |
| 0.943 | `raptorseat`, `vjepaseat` | none — in-sample |
| 0.942 | `fleettwo` | none — in-sample |
| 0.941 | `calibsix`, `calibmix` | none — in-sample |
| 0.937 | `honest12` | **0.8751** |

`honest12` is the last entry above which any member could be priced on data it had not seen. Everything
above it is built out of `--fold -1 --ema` members that have read every study we own.

So the climb from 0.937 to 0.944 was bought by **how members were fit, not by how good they are**. That
is worth stating plainly because the repo had been quoting a "~0.008 blind-vs-fold-0 board gap" that
traces to `experiments/README.md:6553` as prose with no run behind it. This is the same claim with a
witness: seven board ticks separate the best honest entry from the best blind one, and no member-quality
change in that window accounts for them.

It also bounds the lane. Five of the champion's six seats are already blind; `stackseat` refits the
sixth. **Route B has exactly one move left and it is in flight.** Every slot after it is paid out of
member quality.

## 2. The transfer, fitted rather than quoted

The 36 scored entries carrying both numbers give

```
board = 1.37 x offline - 0.2598        (n = 36, least squares)
```

The slope is consistent with `2026-08-16_board_transfer.md`'s +1.55 and with the +1.82 that a complete
board file produced — it is a curve, not a constant (`2026-08-26_the-transfer-constant-is-a-curve.md`),
and this fit spans the honest entries only. Read as a target rather than a law:

```
board 0.946  ->  ensemble offline 0.8815
best ensemble offline ever assembled  0.8751   (honest12)
                                      ------
                                      +0.0064
```

+0.0064 of ensemble offline, against a paired floor of `STRONG_OF_BEST = 0.0062`. The top ten is one
*detectable* step away — the smallest step this project's instruments can resolve at all, and no smaller.

The ensemble has tracked best-single + ~0.015 for a fortnight (knee-urvf), unchanged through every
seating operation. Carrying the gap back through that relation:

```
best clean fold-0 member   0.8618   (merged_span_cf, over the 97 runs sharing its ruler)
needed                     0.8665
                           ------
                           +0.0047 on ONE member
```

That is the number every remaining arm should be priced against. It is above the noise bar the owner set
for high scores and below the 1.5% that the general rule asks for — which is the honest position at
0.944: the remaining distance is small, real, and not cheap.

## 3. Per-label routing is closed, measured over the whole pool

Taking the best per-label AUC achieved by *any* of the 97 clean fold-0 full-corpus runs — a routing
oracle no real system could reach, since it picks the winner per finding after the fact:

| | mean AUC |
|---|---|
| best single member (`merged_span_cf`) | 0.8618 |
| per-label oracle over 97 runs | 0.8735 |
| **oracle headroom** | **+0.0117** |
| what the ensemble already earns | +0.015 |

The unreachable ceiling of per-label routing is **below what averaging already gives us**. This is the
same conclusion as `2026-08-17_the-oracle-ceiling-is-the-average.md` and
`2026-08-23_the_per_label_axis_is_worth_0.004.md`, now measured over the whole pool rather than a
handful of members. Per-label selection, routing and label-specialist members are closed. Per-label
*reading* stays open — it is how a mechanism that moves two findings is detected — but it is a
diagnostic, never a lever.

### A denominator caution, recorded because it nearly went the other way

The first pass over all 145 metrics files put `recal_holdannot_s2026` on top at 0.8964 and gave two
labels an oracle of exactly 1.0000. Both are artefacts: `--hold-annotations` scores on a different
holdout, and the 1.0000s come from `--smoke` runs holding out twelve studies. Filtering to
`fold == 0`, full corpus, no held annotations drops 145 runs to 97 and moves the top from 0.8964 to
0.8618. Every number above is on the filtered set. A pool ranking is a fact about a ruler first.

## What this leaves armed

- `stackseat` — route B's last move, gated clean (`run_check`: 6 of 6 members loaded), waiting on a slot.
- `stack224` — prices slice coverage on the mean; at epoch 10 it reads 0.8476 against `native224`'s
  0.8502. Its per-label read is the gate on `stack_cover_ff` (knee-ujd3), because the coverage mechanism
  names the thin findings and never claimed the mean.
- The KIND axis — `knee-ombx` (MRNet) and a same-modality encoder — unchanged, still the only lane whose
  plausible size exceeds +0.0047, and still blocked on the archive.
