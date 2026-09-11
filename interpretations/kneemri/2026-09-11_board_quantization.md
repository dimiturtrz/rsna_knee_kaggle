# The board reports to 0.001, and the "0.9440 ceiling" was a rounding artifact

Measured 2026-09-11T20:15Z off `comp/board.jsonl`.

## The measurement

`comp/board.jsonl` holds 129 rows, **120 with a numeric `board` value**.

- **43 distinct score values**
- **smallest gap between adjacent distinct values: exactly 0.001**
- every value carries three decimal places

The competition metric, as this project receives it, is **quantized to 0.001**. There is no fourth
decimal.

Distribution at the top (population: all 120 scored rows, `board` field, no cut applies -- this is the
board's own number):

| score | rows |
|---|---|
| 0.937 | 11 |
| 0.944 | 10 |
| 0.943 | 8 |
| 0.936 | 7 |

- **69 of 120 rows** sit at or above 0.930 -- more than half the project's entire submission history
  lives inside 14 ticks.
- **364 of the 7,140 row-pairs differ by exactly one step (0.001).**

## What this retracts

M374 told the researcher, and the owner was told, that **"eight separate entries score exactly 0.9440
-- different seats, different families, same number to four decimals"** and that this was a CEILING.

That is retracted. There are ten rows at 0.944, and they agree to three decimals because three decimals
is all the board has. They land in the same BIN. Ten entries at the mode of a tight, quantized,
saturated distribution over 120 submissions needs no ceiling to explain it.

The framing was built on a rounding artifact and then used to justify shipping seat operations against
it. That is the error, and it is mine.

## What survives, with a better mechanism

**Reseating the existing pool cannot move this board -- because a reseating delta is smaller than the
board's resolution.** A seat swap worth +0.0004 is rounded away and arrives as a null.

This is a statement about the RULER, not about the pool, and it is strictly stronger than the ceiling
claim it replaces. The earlier interpretation that seat swapping would not work was NOT wrong.

## What it reprices

Every flat result in the record of the form "seating X and seating Y both read 0.944" is evidence only
that the two differ by **less than 0.001**. The SIGN of every one of those differences is unknown. The
364 one-step pairs are the same hazard on the other side: each is a comparison that may have been drawn
across a rounding boundary.

Axes closed on a board flat should be re-read with this in mind before they are treated as closed.

## The number that should now gate submissions

The quantity nobody has computed: **the minimum member effect the board can detect at all**, given the
0.001 step, n=120 rows, and their overlap structure. Asked of the researcher as M376 ask 4'. If it
returns something like 0.004, then no single seat swap in this project's history has ever been
measurable on the board.

## Bearing on the offline/board disagreement

D367 found the offline proxy's gain ranking and its decorrelation ranking near-anti-aligned, and the
accepted reading was *"the board is the only ruler left that can see them."* That reading needs
qualifying: the board can see them only if the effect exceeds 0.001. The proxy's marginal adds under
discussion were `+0.0012` to `-0.0029` -- **all of them below the board's step size.** The board cannot
adjudicate D367's disagreement at the magnitudes D367 is arguing about.

-- main, 2026-09-11
