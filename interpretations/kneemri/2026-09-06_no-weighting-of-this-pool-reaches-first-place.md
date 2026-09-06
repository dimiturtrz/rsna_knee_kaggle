# No weighting of this pool reaches first place — the bound selection could not give

*2026-09-06 04:18Z, on CPU while the card ran a dose arm. Zero card, zero slots.*

Two readings already said the pool is spent, and both were about SELECTION.
`priceable_blend` walks subsets at equal weight and stops at 0.8764. `seat_requirement`
(`2026-09-06_one-more-member-cannot-close-the-gap.md`) prices every unseated member against that set and
finds the best eighth seat worth +0.0002 against a +0.0117 requirement. Neither touches the WEIGHTS, so
neither could answer the obvious objection: maybe the members are fine and the flat rank-average is what
is costing us.

`kneemri/kaggle/pool_ceiling.py` answers it. Every priceable member on the corpus ruler is a feature; a
per-label logistic stack is fitted on nine tenths of the studies and read on the tenth, stitched into one
out-of-fold column per label, scored under the same mean the ledger uses.

```
188 members, 909 shared studies, 10-fold cross-fitted
flat rank-average over all of them: 0.8663
  stack at C=0.01   out-of-fold 0.8700
  stack at C=0.1    out-of-fold 0.8765
  stack at C=1      out-of-fold 0.8686
  stack at C=10     out-of-fold 0.8361
best stack: 0.8765 at C=0.1 — chosen on the number it reports
first place needs 0.8862: the best weighting of this pool is -0.0097
```

**A fitted stack over all 188 members lands at 0.8765, and greedy selection at equal weight lands at
0.8764.** One ten-thousandth apart. The penalty was chosen by looking at the out-of-fold scores it
reports, which biases that number UP — deliberately, because a number biased upward that still misses by
0.0097 is the decisive direction.

## What this closes

**The combiner is not the constraint, and neither is selection.** Those were the two remaining cheap
routes: reweight what is owned, or choose a better subset of it. They land on the same ceiling from
opposite sides, which is what a ceiling looks like when it belongs to the POOL's information rather than
to the rule used to read it.

This is the fourth independent route to ~0.875 (`honest12` 0.8751, `distinct14` 0.8749, greedy-7 0.8764,
best stack 0.8765) and the first one that is not a selection at all.

## What it does not say

- It is a bound on LINEAR combinations of member outputs. A stack over member FEATURES rather than
  member predictions is a different object and is not measured here; nothing in the project owns the
  features to try it with.
- The C grid is coarse and the fit is per label on ~818 training studies with 188 features. A finer grid
  would move the number by less than the 0.0062 floor — the shape across the grid (0.8700 / 0.8765 /
  0.8686 / 0.8361) is a single interior optimum, not a truncated ramp.
- Board transfer is not modelled. The 0.8862 target is the offline score first place costs under the
  compression constant already fitted; the argument here is about the offline side only.

## What it leaves

The same single route the seat-requirement reading left, now with one fewer alternative beside it: a
member that breaks the distance/quality frontier, which means **external data continuing the trunk**
rather than another recipe over the same corpus. Everything that can be done by combining what is
already owned has now been measured twice, from both directions, and it stops 0.0097 short.
