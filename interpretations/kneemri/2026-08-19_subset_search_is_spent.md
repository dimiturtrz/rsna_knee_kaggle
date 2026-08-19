# The subset search is spent; the member axis is not

`kneemri-band` scored **0.928**, the first entry past the 0.925 that four different member sets had all
returned. It is one named swap off that four — `res392_ws` (0.8523 solo) out, `band_dense` (0.8578) in,
with count, decode groups, vote and views all held — so member quality is the only thing that moved:
**+0.0055 solo, +0.0036 joint offline, +0.003 on the board.** Pre-registered 0.929, landed 0.928.

That corrects a reading that was forming around the plateau: that member quality does not transfer and
only decode diversity does. It transfers. The earlier evidence was taken on swaps that traded roughly
equal members, so it was a statement about those swaps, not about the board.

## What re-subsetting the pool is worth: nothing

Every same-split member that carries a holdout record was searched exhaustively over the six that
matter, card-free from the recorded per-study probabilities (`kneemri.evaluation.record_mix`, 909 shared
studies):

| four | offline |
|---|---|
| `band_dense` + `lr1e-4` + `calibrated` + `lr4e-5` (the 0.928 entry) | 0.8648 |
| `band_dense` + `band_wide` + `calibrated` + `lr4e-5` | 0.8654 |
| `band_dense` + `depth_regions` + `calibrated` + `lr4e-5` | 0.8655 |
| `band_dense` + `band_wide` + `depth_regions` + `calibrated` (the greedy four) | **0.8661** |

And the curve past four: best-5 0.8663, best-6 0.8663. **The whole reachable spread is 0.0015 on a
0.0062 noise floor.** The greedy four is the top of it and is +0.0013 over the entry that already
scored — a coin flip offline, bought at the price of a third decode group, a roster edit, a dataset
push and a kernel. It does not get a slot. Neither does any other subset of this pool.

Note also that the greedy/best numbers are an **argmax over the holdout** while 0.8648 was one named
swap. Comparing them already flatters the greedy four, and it still does not clear the floor.

The oracle line says the same from the other side: per-label selection scores 0.8612–0.8626 against
plain averaging's 0.8654–0.8663, i.e. headroom of **−0.004 to −0.005**. Selecting per label is worse
than not selecting, on every subset tried. Per-label weighting stays dead.

## What is not spent

`band_dense` did not win because it was another member. It won because it came from an axis the pool did
not contain — the sampled **band** — while every other member differed by learning rate, read size or
pooling. The board paid +0.003 for one new axis and 0.000 for four rearrangements of the old ones.

So the next board move is a NEW member on a productive axis, and the band axis is the one with a number
behind it. `depth_regions_narrow` is the band-axis follow-up already queued, and
[`2026-08-19_depth_regions.md`](2026-08-19_depth_regions.md) states what it must reproduce for the
region-pooling story to be anything other than one unexplained label.

## The gap this exposed in the tooling

Three roster members — `res392_ws`, `sliceattn`, `lora_r8_v3small_224` — have **no holdout record** and
were skipped, so the card-free search runs over 10 of 13 same-split checkpoints. One of them is a member
of the previous champion four. The tool that chooses what to submit cannot currently see it.
`kneemri/inference/holdout_backfill.py` fixes this at one GPU pass per member (knee bead filed).

## The gap is closed, and doubling the pool changed nothing (2026-08-19, later)

The backfill ran. All three members now carry records, and the same-split pool the search can see is
**22 checkpoints**, not 10 — the backfill plus every arm that has finished since. Re-running the greedy
over all 22:

| | best offline |
|---|---|
| the shipped 0.928 four | 0.8648 |
| greedy at 4 | 0.8661 |
| **greedy at 6–7 (the peak)** | **0.8665** |
| greedy at 22 (everything) | 0.8589 |

**Doubling the pool moved the reachable ceiling by +0.0004** — a fifteenth of the 0.0062 floor — and the
peak is still +0.0017 over what already shipped, which is still inside it. The conclusion this document
was written to state does not merely survive the larger pool; it gets stronger, because the twelve extra
members were the test of it. **The subset search is spent, now measured over 22 rather than 10.**

Two things the larger search says that the smaller one could not.

**The first three greedy picks are all band-axis members** — `band_dense`, then `band_wide`, then
`depth_regions` — and together they are worth +0.0070 before a single non-band member is chosen
(0.8578 → 0.8648). Every other axis in the pool competes for the fourth seat onward at a few ten-thousandths
each. That is an independent confirmation of the reading above, arrived at by an argmax that knows nothing
about which axis a member came from: **the board paid for a new axis, and the offline search picks the same
axis first.**

**The three backfilled members do not earn a seat.** This was the whole point of the tooling gap, so it is
worth stating plainly: `res392_ws` is picked ninth and is the first pick that does not raise the ensemble,
`sliceattn` tenth at −0.0003, and `lora_r8_v3small_224` sixteenth, deep in the declining tail. `res392_ws`
was a member of the 0.925 champion four. Now that the search can finally see it, it does not choose it.

That is not an argument for removing it from anything that already shipped — the 0.928 entry is a measured
board result and the offline ruler cannot arbitrate a 0.0017 difference. It is an argument that the seat
`res392_ws` holds was never earned on this evidence, and that the pool's remaining value is in the band
family rather than in any rearrangement of the rest.
