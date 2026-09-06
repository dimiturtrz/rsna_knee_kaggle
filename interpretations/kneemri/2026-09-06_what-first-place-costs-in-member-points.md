# What first place costs, converted from board points into member points

*2026-09-06, while `chainfour` held the card. Zero card, zero slots — every number below was read
from `comp/board.jsonl` and `board_transfer` run live.*

The target has been carried as "we need 0.953, we are at 0.944" and, when it came to arms, as "a
single model needs +0.015". Those two numbers were never reconciled against each other. They can be,
and the answer is smaller and sharper than +0.015.

## The four constants the board record already contains

| step | evidence | size |
|---|---|---|
| best solo member → board | `holdsolo_v1`, offline 0.8634 → board 0.928 | — |
| ensembling over that member | `honest12` offline 0.8751 vs 0.8634 | **+0.0117 offline** |
| that ensemble → board | `honest12` board 0.937 vs `holdsolo_v1` 0.928 | +0.009 board |
| full-fit dose on top of an ensemble | champion 0.944 vs `honest12` 0.937 | **+0.007 board** |
| offline → board, top regime | `board_transfer` live: at or above 0.8678, **+0.98 [+0.94, +1.08]**, 24 resolved pairs | ≈1:1 |

The last row is the one that was being got wrong. The whole-record compression is +1.12 and the prose
constant is +1.55, but compression **saturates**: +1.12 over the record, +0.98 above 0.8678, +0.82
above 0.8695 (4 pairs, direction only). Every remaining offline gain must be priced at **1:1 at
best**, never amplified.

## The arithmetic

Leader 0.954, ours 0.944, gap **0.010 board**.

Hold the structure fixed — same ensembling, same full-fit dose — and ask what the ensemble's offline
number has to be:

```
target board                      0.954
minus full-fit dose               0.007   -> ensemble must read 0.947 on the board
honest12 is                       0.937   -> needs +0.010 board
at the top-regime transfer 0.9-1.0        -> needs +0.011 offline
ensemble offline must reach       0.8862  (from 0.8751)
minus what ensembling itself adds 0.0117
BEST SINGLE MEMBER MUST REACH     0.8745
```

The best member this project owns is `calib_dense_hold` at **0.8634**.

> **First place costs +0.011 on the best single member, not +0.015.**

## Why this is worth more than the 0.004 it shaves off the target

It changes which arms are worth running. A +0.015 target rules out every composition result this
project has produced — tonight's three-term chain read +0.0089 at the 224 corner and +0.0055 at the
dense corner, both dismissible against +0.015. Against **+0.011** they are 50-80% of the distance,
which makes composition a partial route rather than a dead one, and makes stacking two independent
+0.005 terms a strategy rather than noise-chasing.

It also prices tonight's proxy result honestly. Three new members add +0.0027 offline to a
champion-shaped proxy, which at 0.98 transfer is **+0.0026 board — 26% of the entire gap to first
place**, from members that already exist. That is not a knob. It is a quarter of the remaining
distance sitting unshipped.

## The three assumptions, stated so they can be attacked

1. **Ensembling adds a constant +0.0117.** It does not — it saturates. `distinct14` (14 members,
   0.8749) reads *below* `honest12` (12 members, 0.8751), and `peak8_v1` at 8 members reads 0.8731.
   The +0.0117 is what ensembling delivers *at its own ceiling*, so treating it as constant while
   the member improves is optimistic. If a better member ensembles less well, the required member
   score rises above 0.8745.
2. **Full-fit dose adds a constant +0.007 board.** This rests on one comparison, champion versus
   `honest12`, and those two differ in membership as well as in dose. It is the weakest number here.
3. **Transfer is 0.9-1.0 in the regime we would be operating in.** The 0.8695 band already reads
   +0.82 on four pairs. If compression keeps falling, the offline requirement grows — at 0.82 the
   member target moves from 0.8745 to about 0.8767, another +0.002.

All three push the same way: **+0.011 is a floor on what first place costs, not a ceiling.** The
honest statement is that a member at 0.875 is *necessary* and might not be sufficient.

## What it does not say

It does not say composition is the route — `chainfour` is on the card testing whether terms sum onto
a saturated base at all, and if they substitute rather than add, the chain route closes regardless of
what the target is. And it does not license shipping the +0.0026: that entry is worth spending a slot
on as an **instrument**, because it tests whether marginal contributions measured on held twins
transfer to full-fit originals, and that question is worth more than the 0.0026.
