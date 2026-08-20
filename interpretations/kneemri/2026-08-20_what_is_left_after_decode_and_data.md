# What is left after decode and data

Four lines of measurement, taken for different reasons over 2026-08-17..20, arrive at the same short
list. None of them was run to answer this question; the convergence is what makes the reading worth
writing down, because any one of them alone is an arm that did not pay.

## The four lines

**Decode closed on its own axes.** `attend_dense` replaced the pooling OPERATOR (elementwise max → a
learned 64-wide gate): **−0.0009**. `focal_dense` held the operator and widened the pooled vector:
**−0.0044**. Both are single-variable pairs against `band_dense` (0.8587), both verified in the
checkpoint config and not only in the recorded recipe, and a second ruler that costs no card time —
ordinal concordance — agrees with the binary one on both
(`2026-08-20_pooling-is-closed-on-the-mean-but-not-on-lateral-meniscus.md`). The one thing that survives
there is a per-label opening on Lateral Meniscus, not a mean lever.

**The data levers are individually spent.** The audit of knee-jk3 priced each of its four: whole-stack
**+0.0147** (paid, and long since banked into the shipped recipe), `extended_slots` **+0.0002** (83,297
previously-unseen slices bought nothing), `band_density` **+0.0014** at held span (below the 0.0062
holdout noise floor, and a trade of centre for edge rather than a mean gain —
`2026-08-19_density-inside-the-band-trades-centre-for-edge.md`), 336 px refuted. The band widening that
did pay (+0.0039) is already in the recipe every arm inherits.

**The ensemble ruler saturated.** Six entries now pair an offline joint-holdout score with a board
score. The chain rises across the first ~0.003 of offline spread and then stops flat:

| entry | offline | board |
|---|---|---|
| band_v1 | 0.8648 | 0.928 |
| swapdrn | 0.8653 | 0.928 |
| oneswap | 0.8661 | 0.927 |
| calibspan_v1 | 0.8670 | 0.929 |
| greedy4_v1 | 0.8679 | 0.930 |
| longseat | 0.8691 | 0.930 |

`longseat` carried **+0.0012** of seat value — the largest single move in a sweep of all 56 same-split
checkpoints, six times the runner-up — and tied the board rather than moving it. That is what
`BOARD_RESOLVABLE = 0.0015` in `MemberCurve` records: not a threshold anything is known to cross, but a
floor on what has ever been *demonstrated* to move a board tick. Re-run over the mounted pool it prints
`0 of 34 clear +0.0015`. Nothing currently on disk is a submission.

**The pool's diversity is recipe diversity.** Seventeen mounted members and sixteen are DINOv2. The two
exceptions argue that the other kind is what pays: `deep24` is the **worst** member in the file (0.8213,
ranking term off) and greedy forward selection takes it anyway, because its errors are not shaped by the
objective the other sixteen share.

## What the four agree on

The pooled representation is not the binding constraint, and neither is which pixels reach it. Each
remaining candidate changes *what the errors look like* rather than how many there are:

1. **The encoder family** (knee-22mt) — queued as `v3base_band`, and deliberately framed as the
   transfer question rather than "is DINOv3 better", which is already answered no. If the +0.012 of
   recipe this ledger bought is a fact about the DATA it must reproduce on a different patch grid,
   position encoding and pretraining corpus; if it does not, the whole decode chapter is a fact about
   DINOv2's feature geometry and has to be re-read with that scope written on it.
2. **Supervision** (knee-mxj) — the LLM pass over the 4,407 reports. **HELD pending an explicit
   go-ahead.**
3. **Self-supervision** (knee-c5n) — **HELD pending an explicit go-ahead.**

Two of the three are blocked on a decision, not on evidence. That is the actual state of the queue and
it is why this note exists: the ledger's own conclusion — written independently at the bottom of the
`focal_dense` entry, "the bottleneck moves to supervision or to the encoder, which are different beads
and different arms" — is now reachable from three other directions, and only one of the three named
axes can currently take the card.

## What this does NOT say

It does not say decode is exhausted in general — it says the cells this ledger opened came back empty,
and the Lateral Meniscus opening above is a live per-label counter-example on the same axis. It does not
say a fifth data lever cannot exist; it says the four that were listed are priced. And it does not
license retiring the ensemble search: the ruler saturating bounds what a *positive reading* is worth, it
does not make the ordering wrong. `MemberCurve.swap` still answers "which of these is the best move"
reliably, and only "is the best move worth a slot" not at all.

The refutation condition is unchanged from knee-825t: a future entry whose offline gain is under 0.0015
and which moves the board a tick.

## Addendum, same day: the head axis closed too, and it closed by pointing at one of the held two

`aprior_dense` ran after this note was written and came back **−0.0021** against its matched control
(0.8557 vs `band_dense`'s 0.8578, both at epoch 8 of the same one-cycle). It was the cheapest possible
version of the localisation argument — initialise `SlotHead.prior` from the 12x6 plane-by-finding table
instead of zeros, no new parameters, no new compute, 368 s an epoch identical to the control.

**It did not fail quietly, and the training loss is why that matters.** The two runs' losses track to the
fourth decimal at every epoch, with the ARM slightly lower. So the prior was absorbed and then written
over by gradients that had already been finding the same structure. That is a stronger statement than
"the flag did nothing": the head had ALREADY recovered the mapping from 4,407 study-level label vectors
carrying no localisation at all.

**The consequence was pre-registered, so it is owed rather than argued after the fact:** "supervision
never says where" is wrong about this head, and the localisation argument has to buy a real localisation
signal rather than a warm start. That is knee-mxj — one of the two axes this note records as HELD.

So the count above changes shape. It is no longer three candidates of which one can take the card; it is
**four levers on the decode/head subsystem all returning under a seventh of the floor** —

| arm | what it moved | delta |
|---|---|---|
| `attend_dense` | pooling operator | −0.0009 |
| `focal_dense` | pooled vector width | −0.0044 |
| `band_density` | slice count at fixed span | +0.0014 |
| `aprior_dense` | head initialisation | −0.0021 |

— and a fifth axis that a refutation clause now names explicitly. Four independent levers on one
subsystem, none clearing a seventh of the floor, is evidence about the SUBSYSTEM rather than about any
of the four.

## And the held axis is smaller than this note assumed

knee-mxj is described above as "the LLM pass over the 4,407 reports". An audit of the data on disk the
same day shrinks that considerably, and the shrink is the useful part.

`report_labels_v2.csv` carries a **third column per target** that `Targets.load` has never read:
`<target>__verdict`. Cross-tabulating all 52,872 populated cells gives a perfect bijection —

    verdict  severity  value  conf    cells
    NO         0        0.08  0.85   22,298
    UNK        0        0.28  0.05   14,445
    YES        1        0.68  0.95    6,461
    YES        2        0.82  0.95    6,427
    YES        3        0.94  0.95    3,241

The three YES levels are **severity grades 1/2/3, not certainty**; certainty is `__conf`, flat at 0.95
across all three, and `__conf` is already consumed as the loss weight. knee-mxj's central design — split
the scalar into an ordinal GRADE and a separate CERTAINTY — **is therefore already shipped**, and its
motivating example does not occur: the incumbent rulebook sends anything explicitly uncertain to UNK
(value 0.28, weight 0.05, i.e. near-off), so a confident low grade and an uncertain high one were never
colliding. The unread `__verdict` column is redundant with `value`, not a lost signal.

Its other half is refuted rather than untried: `Supervision.applied()` already implements the level
remap, `--calibrated-targets` is shipped, and the arm has RUN TWICE — +0.0038 at the old operating point
and **−0.0020 as `calib_span` on the recipe we actually ship**, with every pre-registered clause
inverting. Stale operating point, which this ledger already has a name for.

**What is left on the supervision axis is not the label REPRESENTATION. It is label ACCURACY** — are the
verdicts right? — which nothing here has measured and which no decode knob could ever have reached. That
is a much narrower question than a full re-read, and it is measurable on a sample before it is worth a
pass over 4,407 reports.

Found the same day and it is what makes a re-read even askable: the incumbent's own labeler ships inside
pilkwang's CC0 dataset as `api_labeler.py` (mtime identical to `report_labels_v2.csv`, in no repo file
and no git history here). It names the model (`claude-opus-5`, `effort="low"`), the constrained JSON
schema (verdict enum plus severity enum), and a verbatim rulebook that lists the languages and instructs
the reader to work in the original. It cannot run as-is — it imports from an `llm_labeler` module that
exists nowhere — but any contest of its labels has to be judged against **that exact rulebook**, or the
disagreements measured are definitional drift rather than errors.
