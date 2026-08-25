# The seventh seat is inert, and four board slots were spent varying it

`dark6` (55756018) reads **0.937**. It is six `--fold -1` members and nothing else — no seventh
member, no fold-0 seat, no other change. `allfit` (0.937) and `lastseat` (0.937) are those same six
plus a seventh. The seventh member is worth **zero board points**.

## What this voids

Every board entry since `finalfit` has held count at seven and varied WHICH member holds seat four.
That was the project's main search axis for two days. It was searching a coordinate with no value:

| entry | seat four | board |
|---|---|---|
| dark6 | *(no seventh member)* | 0.937 |
| dark7 | `small_batch6_ff` (all-data) | 0.937 |
| allfit | `band_dense_long_s2026` (fold-0) | 0.937 |
| lastseat | `small_batch6_s2026` (same family) | 0.937 |
| augseat | `aug_long_s2026` (regularised) | 0.936 |
| v3seat | `v3base_dense_s2026` (DINOv3-base) | pending |

Three slots spent, a fourth in flight. The pre-registrations were all written against 0.937 as *"the
champion"*, when 0.937 is really *"what six members already read"*. So `augseat`'s 0.936 is not a
reading about regularisation, and `v3seat`'s expected 0.936-0.939 will not be a reading about encoder
family. They are readings about noise around a null.

This is the harness failure mode named in the project rules — **faulty TEST**, a ruler that cannot see
the axis — and it went undetected because the control that exposes it (drop the seat entirely) had
never been run. It took one slot to find, which is the cheapest thing bought on this board all week.

## What it does NOT say

It does not say members are worthless. `weakseven` — seven members chosen for LOW quality — read
**0.908**, twenty-nine points down. Member quality moves this board hard. What is inert is the
**marginal seventh member**, added on top of six that already agree.

That is a saturation reading, and it has a named mechanism already written down:
`2026-08-24_architecture_diversity_is_closed_and_the_empty_cell_is_strong_and_distant.md` closes the
encoder axis on the observation that slice sampling, the **2D-slice-then-pool architecture**, the
five-level targets, the BCE objective and the head are held fixed across every arm — so a trunk swap
changes how features are *spelled*, not what is optimised, and the errors an ensemble needs
uncorrelated are made downstream of all of it. `dark6` is that note's board confirmation, and it is a
much stronger instrument than any seat swap: an entire extra member of the saturated family buys
nothing at all.

## dark7 landed FLAT, and it closes the dose axis too

`dark7` (55756369) reads **0.937** — dark6's six plus `small_batch6_ff`, the seventh all-data member,
and the highest-scoring member this project has ever trained. Its own pre-registration named that band:
*"within 0.002 = the dose is spent and the full-fit axis closes on a measurement rather than an
extrapolation."* So the all-data dose is **saturated at six**, and the seventh seat is inert whether it
is filled by a fold-0 member or by a full-fit one.

**Four board entries now read exactly 0.937**, and between them the member COUNT (six vs seven), the
member IDENTITY, and the seventh member's FOLD STATUS all vary:

    dark6      6 dark                          0.937
    dark7      6 dark + 1 dark                 0.937
    allfit     6 dark + band_dense_long (f0)   0.937
    lastseat   6 dark + small_batch6 (f0)      0.937

That is a plateau, not scatter, and it is the strongest statement this board has made. It also decides
a card question for free: `small_batch6_ff.sh` pre-registered *"queue the second conversion
immediately"* only on `>= 0.939`, so **`band_dense_long_ff` is not converted** and the ~3 h that would
have cost is not spent.

## What is still live

Three strictly nested entries, one variable each, and the third has never been run:

    dark6    six all-data members                    0.937
    dark7    + a seventh of the SAME family          0.937
    farseat  + a seventh of the ONE OTHER family     running

`farseat`'s tuple is exactly `dark6`'s six with `vjepa_warm_s2026` appended (verified member-for-member).
`vjepa_warm_s2026` is the only line in the roster that breaks the slice-then-pool term of the closure
mechanism: a V-JEPA2-large tubelet transformer, two slices per token, handed the stack, where every
other member reads slices and reconstructs depth afterwards. Its nearest-neighbour agreement is 0.8210
against a seated band of 0.8801-0.9143.

It had been unmounted since 2026-08-24 and was remounted 2026-08-25 01:14Z by walking the roster's own
written way back. Its kernel had been gated on `augseat >= 0.940`; that gate is void by the argument
above, and has been rewritten against `dark6`.

With `dark7` now flat, `farseat` is the whole remaining question, and its attribution is clean by
construction: the same-family control has been run and returned nothing, so a move at `farseat` can only
be FAMILY. It is the last variable at seat seven that has not been measured.

Beyond that triple, the axes never varied on any board entry are the **vote** (`RankAverage` on every
single submission this project has made; measured flat offline at +0.0002, but the offline ruler is
separately measured anti-correlated at the top of this range — `peak8` 0.8731 -> 0.932) and the
**serving/overlap path**.

## The transfer number

`board_transfer` reads **+1.82** board points per holdout point, fitted entirely on entries <= 0.933,
so it is a floor and not an estimate. Any prose in this tree quoting 4.67, 2.14 or 1.92 is stale.
