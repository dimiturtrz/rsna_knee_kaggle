# The encoder rate carried what resolution was credited for, and the board paid for it

2026-08-18. Two measurements, one on the fold-0 holdout and one on the leaderboard, that turn out to be
about the same thing: which member goes in, and what tells us.

## What was measured

The encoder learning rate, held at every other variable of the control recipe — 336 px, 12 epochs,
whole-stack reading, max slice-pooling. Only `--backbone-lr` moves.

| rate | holdout | vs control |
|---|---|---|
| 8e-6 (the inherited 125:1 head ratio) | 0.8457 | control |
| **4e-5** | **0.8525** | **+0.0068** |
| 1e-4 | 0.8490 | +0.0033 |
| 2e-4 | 0.7391 | −0.1066 |

4e-5 clears the 0.0053 noise floor. 1e-4 does not separate from it — 0.0035 apart, inside one floor
width — so the honest reading is a **plateau spanning 4e-5..1e-4**, not a peak at 4e-5. 2e-4 reproduced
the failure signature written into the arm header *before* it ran: loss rises from epoch 0 to epoch 1
(1.2296 → 1.2739) and flattens at 1.07 where healthy arms reach 0.70. That is catastrophic forgetting of
the pretrained features, i.e. "past the edge", and it is why the axis is now spent — a rung between the
two surviving rates would land inside the floor.

## It dissolved a confound we had been building on

`res392_ws_s2026` scored +0.0066 and the gain was credited to reading at 392 px. It changed two things at
once: the resolution AND the encoder rate. Run at 392 px with the control's own 8e-6 (knee-a8x), the
resolution is worth **+0.0002**. Nothing.

So `res392` and `lr4e-5` carry the same lever through different pixels, and only one of them needs a
64 GB 448 px cache to do it. Every member trained since was reasoning about resolution.

## The board then priced the member directly

| entry | members | LB |
|---|---|---|
| `kneemri-chosen` | res392 + intensity392 + calibrated + **sliceattn** | 0.918 |
| `kneemri-rate` | res392 + intensity392 + calibrated + **lr4e-5** | **0.924** |

Same count, same kit, same `overlap=False`, same `RankAverage`. One member swapped, **+0.006** — the
first clean single-member A/B this competition has given us, and a new best over the 0.919 that had
stood since 2026-08-16.

## The part that changes how entries get picked

The board moved +0.006 for a member whose SOLO holdout gain was +0.0068. The joint-subset search — the
thing that actually chose the set — ranked this four only +0.0013 above `chosen`. It was right about the
direction and wrong about the size by a factor of five.

That matters because the two rulers disagree about *which* member belongs. The pool's solo scores are

    lr4e-5 0.8525 · res392 0.8523 · calibrated 0.8493 · lr1e-4 0.8490
    sliceattn 0.8473 · extended 0.8462 · intensity392 0.8459

and the joint search puts `intensity392` — last on its own — inside the winning four, on the grounds
that it decorrelates. That is a real mechanism and it may well be right; it has simply never been tested
against the board while a solo-ranked alternative existed. `kneemri-solotop` is that test: the same four
with `intensity392` replaced by `lr1e-4`, which makes the set exactly the top four by solo score and the
set the joint search ranks BELOW the one that scored 0.924.

**Do not read this as "solo scores are the right ruler" yet.** One transfer is one point. What it does
establish is that the joint search's *magnitudes* are not to be trusted for slot decisions, since it
underpriced the only member swap the board has ever priced for us by 5×.

## What it does not say

- Nothing about seeds or folds. Every arm here is one recipe with one variable moved.
- Nothing about resolution being useless — 392 px at a matched rate is +0.0002 on THIS holdout, which
  bounds it near zero for whole-stack reading and says nothing about a windowed member or a finer axis.
- Nothing about the vote. That axis was measured 2026-08-17 and is flat, `QualityWeightedRank` included,
  which is worth restating here because "weight the members by how good they are" is the obvious next
  thought after this page and it has already been answered.

Ledger: `experiments/README.md`, arms `rate_wholestack` and `rate_sweep`. Board: `comp/SUBMISSIONS.md`.
Beads: knee-wnkv (the rate), knee-a8x (the resolution nullity).
