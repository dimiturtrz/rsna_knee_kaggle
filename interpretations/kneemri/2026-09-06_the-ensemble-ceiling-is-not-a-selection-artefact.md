# The ensemble ceiling is not a selection artefact, and the optimism constant was wrong by 10x

*2026-09-06 02:53Z, while `chainfour_ff` held the card. Zero card, zero slots.*

Two numbers changed tonight. One of them was a project constant that had been quoted, in a module
docstring and in planning prose, for three weeks.

## The optimism constant: +0.0201 -> +0.0019

`priceable_blend --optimism` re-runs the greedy forward selection inside k folds, selecting on the
studies of k-1 folds and pricing on the one it never saw. Its default is **ten** folds, which is the
regime every offline number in `comp/board.jsonl` was taken in: the search reads 90% of the 909
shared holdout studies.

At ten folds over the 206-member pool:

```
optimism +0.0019 over 10 folds, spread -0.0199 to +0.0254
```

The number this project has been carrying is **+0.0201**, and it is a **two**-fold reading — search on
455 studies, price on the 454 it never saw, taken once. That answers a different question (what a
search overfits on half the corpus) and it is an upper bracket. It was being cited as the optimism of
a normal run.

The correction matters because +0.0201 is larger than the entire +0.0121 that seven seats buy over the
best single member, so under the old constant **every searched blend was worthless by construction** —
its whole margin was assumed to be search noise. Under +0.0019 a searched blend is very nearly an
unbiased estimate of itself.

Read the spread honestly: each fold prices on 91 studies, so -0.0199 to +0.0254 is what one small
pricing set does, not a range of plausible biases. Ten folds put the bias near a fifth of a resolvable
board point.

## What that licenses, and what it does not

It re-opens **ensemble selection over the priceable pool** as a legitimate operation. That axis was on
the closed list, and the reason it was closed was this constant.

It does NOT license shipping a searched blend as an improvement, because of the second number.

## The second number: the ceiling did not move

The pool grew from 195 to 206 scoreable members, and 31 members joined the roster in the 2026-08-29
sweep. Greedy forward selection over that grown pool:

```
 1. chainfour_s2026            -> 0.8648
 2. calib_dense_hold_s2026     -> 0.8712
 3. stack_increments_s2026     -> 0.8733
 4. blendhold_s2026            -> 0.8741
 5. labelwise_dense_s2026      -> 0.8749
 6. margin_seat_s2026          -> 0.8758
 7. finding_calib_dense_s2026  -> 0.8764
```

**0.8764 searched, 0.8745 after subtracting the corrected optimism.**

Against what this project already owns:

| set | members | offline |
|---|---|---|
| `honest12` | 12 | 0.8751 |
| `distinct14` | 14 | 0.8749 |
| greedy over 206, optimism-corrected | 7 | **0.8745** |
| `peak8_v1` | 8 | 0.8731 |

The best search this project can run, over its largest pool ever, with the selection bias correctly
priced, lands **0.0006 below** a hand-built twelve-seat entry from weeks ago. That is inside any noise
this ruler can resolve.

> **The ensemble ceiling sits at about 0.875 offline, and it is reached by twelve hand-picked members,
> by fourteen, by eight, and by seven chosen optimally. It is a property of the POOL, not of the
> selection method.**

Three independent routes to the same ceiling is the strongest form this project has for that claim.
The previous readings could always be answered with "you picked badly"; this one picked optimally.

## Why this is the useful finding and not a disappointment

It removes an entire class of work. Any plan of the form "search the pool harder / weight the seats /
pick a better subset" is now answered before it is run: the optimum is measured, and it is where we
already are. Seven prior board flats on seating operations get a mechanism instead of an
accumulation of nulls.

It also prices the members added since 2026-08-29 at approximately zero **in ensemble**. They were
worth adding — `stack_increments`, `margin_seat`, `finding_calib_dense`, `labelwise_dense` and
`blendhold` all take seats in the optimal set — but they took seats FROM members that were already
there rather than raising the total.

## What it points the card at

Every route that stays inside this pool is capped at ~0.875 offline, and first place needs the
ensemble at 0.8862 (`interpretations/kneemri/2026-09-06_what-first-place-costs-in-member-points.md`).
The gap is **+0.0117**, which is the same size as the entire contribution ensembling makes over one
member. No re-selection, re-weighting or re-lensing of these 206 members can produce it.

Two things can, and only two:

1. **Full-fit dose**, worth about +0.007 board and applying to whatever the entry ends up being. This
   is why `chainfour_ff` is on the card. It is a production step, not a discovery.
2. **A member from outside the pool's error space** — which, given that four RadImageNet/medical-SSL
   trunks and ~25 foreign trunks are already refuted here, most plausibly means DATA rather than
   architecture.

The interesting corollary: `chainfour` is seat ONE of the optimal seven despite adding **+0.0000** to
the specific five-seat proxy tuple measured earlier tonight. A member can be redundant beside three
particular members and still be the best base to build a whole set on. Marginal contribution is a
statement about a SET, never about a member — which is exactly why the greedy answer and the nested
answer disagree, and why neither is wrong.
