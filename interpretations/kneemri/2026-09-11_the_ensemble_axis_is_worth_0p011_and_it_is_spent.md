# Why we are not going beyond 0.944: the ensemble axis is worth +0.011 and it is fully spent

2026-09-11. Source: `comp/board.jsonl`, all 119 scored submissions. Read with `board_check --live`
(124 recorded, 100 live). No new slot was spent to produce this.

## The table

Board score by member count, over every scored submission the project has made:

    members    n     max      median
      1       14    0.933     0.922
      2        5    0.936     0.919
      3        6    0.939     0.925
      4       14    0.937     0.928
      5       11    0.941     0.931
      6       23    0.944     0.938
      7       24    0.944     0.937
      8        8    0.944     0.940
      9        1    0.936     0.936
     10        1    0.928     0.928
     11        1    0.941     0.941
     12        1    0.937     0.937
     14        1    0.937     0.937
     25        1    0.894     0.894

## Three readings, and the third is the one that matters

**(a) The ensemble axis is worth +0.011, total.** Best solo 0.933, best ensemble 0.944. Everything this
project has ever gained from combining members — every seating, every weighting, every diversity
argument — is eleven thousandths.

**(b) It saturates at six and then degrades.** 6, 7 and 8 members all cap at exactly 0.944; nine or more
has never once reached it, and 25 members reads 0.894. Ten DISTINCT entries sit at 0.944 — different
kernels, different families, 6/7/8 seats, spread over two weeks, `cnnseat_v1` (a CNN family the board
had never held) and `raptorweight_v1` (a single float changed) among them. The cap is invariant to
what is seated in it. This is the D337 closure of the member axis, measured on the board rather than
argued.

**(c) THE GAP TO THE LEADER IS THE SAME SIZE AS THE ENTIRE ENSEMBLE CONTRIBUTION.** Leader 0.954,
us 0.944: 0.010, against the ensemble's whole lifetime yield of 0.011. There is no seating that closes
it, because seating has already given everything it has. **The only remaining route is a better MEMBER
— the solo ceiling of 0.933 has to rise.** Raise best-solo to ~0.943 and the same six-seat machinery
that turned 0.933 into 0.944 lands ~0.954.

## What this licenses and what it forbids

FORBIDS: further seating instruments, weighting fits, member-count sweeps, and diversity arguments
priced as score. The axis is closed at +0.011 and 119 submissions say so. A slot spent there buys a
number we can already predict.

LICENSES: everything that makes one member better. The label-correction chain in flight
(v5blend → v6tears → v7nontear) is aimed at exactly this — it does not reseat anything, it changes what
a member learns. So is the encoder-pretraining route on unlabelled pools. These are now the only two
things that can move the board, and they should get the card.

REFRAMES THE READING: an arm should be judged by its SOLO number, not by whether an ensemble containing
it improves. The ensemble is a fixed +0.011 multiplier on top of whatever the best member is, and
measuring arms through it has been measuring a saturated function.

## The caveat this rests on

`board` in `comp/board.jsonl` is three decimal places, and ties at 3dp are common in the file (eight
rows at 0.943, eleven at 0.937). So "ten entries at exactly 0.944" is not by itself evidence that they
produce identical predictions — it is evidence that they land within a thousandth of each other. The
claim that survives either way is the one that matters: **nothing in 119 submissions has ever exceeded
0.944, across every member count from 1 to 25.** The ceiling is the finding; the exact-tie is not
needed to establish it.
