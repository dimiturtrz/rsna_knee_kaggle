# The label-table axis is exhausted, and the ruler that says so is our own

*2026-08-26, on CPU, while `blendsix` held the card.*

## What was believed, and on what

`core/data/labels/table_blend.py` has carried this table since it was written:

```
mean(ours, steven, lixin)         0.8912
best single foreign table         0.8927
```

and its own conclusion: *"the blend does not beat its best input — which is what averaging independent
readers is supposed to do."* It also said, correctly, that the difference had **no computed interval**
and must not be quoted as separated. That caveat was written and then ignored — by me. `blendsix` is
spending seven hours dosing that blend onto six seats, and a P1 (`knee-9vgx`) was filed to spend two
card arms deciding whether the blend or its best input is better supervision. Both rest on a
fifteen-ten-thousandths gap on 58 studies.

## The measurement

`kneemri/evaluation/table_separation.py` scores any set of label tables against the radiologist
annotations and brackets every **pairwise** difference by bootstrap over **studies** — paired, because
both tables see the same 58 studies and a study that is hard to read is hard for both. 1000 draws,
`StudyBootstrap`, the same protocol every other interval in this repo is read under.

**The readers separate:**

| table | mean AUC vs the 58 |
|---|---|
| `hybrid_v4` (flight0234) | **0.8991** |
| `steven_v4blend` | 0.8927 |
| `blend_three` (what `blendsix` doses) | 0.8912 |
| `steven_v2` | 0.8873 |
| **ours** (`report_labels_v2`) | 0.8658 |
| `lixin_gpt56` | 0.8352 |

- `blend_three − ours` = **+0.0254 [+0.0098, +0.0428]** — separated.
- `hybrid_v4 − ours` = **+0.0333 [+0.0156, +0.0528]** — separated.
- `lixin_gpt56 − blend_three` = **−0.0560 [−0.0791, −0.0343]** — separated, and `lixin` is one of the
  three readers *inside* our blend.

**Their blends do not:**

| blend | mean | vs `blend_three` |
|---|---|---|
| `blend_three` = ours + steven + lixin | 0.8912 | — |
| ours + steven_v4 + hybrid (lixin dropped) | 0.8894 | −0.0019 [−0.0072, +0.0020] |
| ours + steven_v4 + hybrid + lixin | 0.8909 | −0.0003 [−0.0035, +0.0019] |
| ours + hybrid | 0.8931 | +0.0019 [−0.0055, +0.0091] |

Every interval covers zero. **Dropping the resolvably-worst reader does not help. Adding the
resolvably-best one does not help.**

## The one thing that reads

The best single reader is separated **above two of the blends that contain it**:

- `hybrid_v4 − blend_nolixin` = **+0.0098 [+0.0016, +0.0202]**
- `hybrid_v4 − blend_four` = **+0.0082 [+0.0005, +0.0183]**
- `hybrid_v4 − blend_three` = +0.0079 [−0.0008, +0.0180] — a hair off.

Averaging our coarser table into the strongest reader **dilutes** it. That is the opposite of the
noise-cancelling story the blend was built on, and it is the only directional signal in the whole set.
It does not license switching tables — a single reader is a reader we cannot audit, and `LEAKED` exists
because three public tables turned out to have read the ruler. It does say the blend is not buying what
it was assumed to buy.

## What this settles, and what it does not

**Settles:** recomposing the blend is *not a lever*. On 58 studies this ruler resolves about ±0.008 and
every recipe on hand lives inside that. A better label table has to come from a better **reader**, not
from a new weighting of the readers we already have.

**Also settles:** `blendsix`'s premise. `blend_three` is separated above our `v2` at +0.0254, so the arm
is dosing a table that is genuinely better than the one the champion's seats were trained on. That was
the outcome `knee-9vgx` named as its own closing condition, and it closed on it — without a card.

**Does not settle:** whether these tables differ as **supervision**. A table trains on ~4,400 studies
and is judged by a board score; a reader can be better there and unresolvable on 58 gold rows.
*Unresolved is not equal.* But we cannot buy board reads for label tables — the slots are pinned to the
`blendsix` harvest — and the arm-level effect is already measured: `blendlabels` reads **+0.0034** mean,
below the 0.0062 floor. The lane is live, small, and now bounded on both rulers we own.

## The habit this is about

The bare pair of numbers sat in a module docstring for a week, with a sentence beside it saying it must
not be quoted as separated, while two separate pieces of work were planned on top of it as if it had
been. Computing the interval took four minutes of CPU and killed a 45-minute card plan. The rule is not
"compute more intervals"; it is that **a number a docstring refuses to stand behind is not evidence**,
and the moment it starts steering a decision is the moment it has to be bracketed.

One bug fell out of building the ruler: `SupervisionGap.reported` returned only the columns a table
happened to carry, while `against` names its columns **by position**. A foreign table missing one
finding would have scored every later finding under its neighbour's name. Ours carries all twelve, so it
never fired — the first foreign table would have.

Beads: `knee-ym3z` (the finding), `knee-9vgx` (closed on it).
