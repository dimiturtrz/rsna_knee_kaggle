# Teacher exposure is a fourth honesty filter, and attribute marginals saturate at about a third

2026-09-11. Two findings from one night, measured by the researcher (D376-D380) and read here. They are
written together because they were found together and because each one changes how the other is read.

All numbers are macro AUC at `holdout909`, midpoint-binarised v2 truth, unless a line says otherwise. A
number in this repo is midpoint unless it says otherwise.

---

## 1. The fold-0 leak transfers through distillation at +0.0051

The project ran three honesty filters on every member before it was allowed into a pool: `epoch >= 0`,
the `studies` vector identical to the canonical holdout909, and the binarised `truth` identical
cell-for-cell. None of them reads `--teacher-targets`. A student distilled from a teacher that was
itself fitted on fold 0 passes all three and is not honest.

The size of that was measured on a matched pair, which is the only correct measurement in the file that
had been arguing about it. `oof_folds.sh` and `teacher_clean.sh` differ in exactly one flag,
`--exclude-fold 0`; everything else is identical (224px, 8 epochs, 12 slices, band 0.2-0.8, folds 1-4,
seed 2026). Their students `oof_teacher.sh:63-67` and `seat_clean_distill.sh:44-48` are flag-identical
apart from run name, teacher file and output path.

    oof_teacher_s2026      0.8685   leaky teacher
    seat_clean_distill     0.8634   clean teacher
    transferred leak       +0.0051

That is the entire margin by which `oof_teacher` led the pool. **The honest pool maximum is 0.8649
(`clean_calib_s2026`), not 0.8685.** `comp/board.jsonl:118` was amended and knee-0c3vc closed on this.

The cost of the clean teacher is documented in its own header and is the mechanism: each clean block
fits ~2,600 studies against ~3,500. So the clean student matched a leaky twin out of teachers that are
weaker by construction, which is what makes the difference a treatment effect rather than pool noise.

### Exposure classes on disk

- **100%** -- `teacher_oof_corpus`, `teacher_oof5`, `teacher_oof_full` (ours), and `raptor_teacher.npz`
  (fold -1 published weights, the 98.6%-in-sample provenance that voided `raptor_solo`'s 0.8938).
- **~80%** -- `teacher_foreign_corpus`. Each of the five foreign blocks was fitted on 4/5 of ALL
  studies, so the block predicting any fold-1-to-4 study saw about 80% of fold 0.
- **0%** -- `teacher_clean`, `teacher_clean_full`. The only true zeros on disk.

Twelve members of the 133-member packaged honest pool are exposed: 6 raptor-teacher, 3 oof-line, 3
foreign-teacher. **All twelve are distillates**, which is why all twelve sat inside the n=22 that
produced the distill attribute marginal. One of them, `foreign_dissent_s2026`, was caught by a name map
and not by the attribute tagger, because its recipe spells the term `dissent_weight` rather than
`distill_weight` -- a tag-only audit would have missed it.

### What it does NOT cost

At ENSEMBLE level the term is small. By substitution on the no-search top six:

    no-search top six            0.8737
    0%-exposure-only top six     0.8733
    foreign-kept variant         0.8738

**The total ensemble-level exposure term is ~0.0004** -- a fifth of the board's own noise between two
offline-equal solos (`stackfit` 0.8631 -> board 0.933 against `holdsolo` 0.8634 -> board 0.928), and a
fifth of the board blur (median |board| 0.0020 over pairs the holdout cannot separate). That is why the
entries shipped tonight ship with the term STATED and bounded rather than rebuilt: rebuilding to chase
0.0004 would spend a slot on a delta neither ruler can resolve.

The filter is now a P1 bead. It belongs in code beside the other three, because the audit that found it
was done by hand.

---

## 2. Attribute marginals saturate, and stacking realises about a third of face value

The per-attribute marginal table is fitted as median-with minus median-without over members carrying
arbitrary other attributes, so a member with five positive attributes feeds five marginals and the table
cannot say whether it earned five terms or one. Refitted on the 0%-exposure sub-pool (n=290), the terms
are:

    labelwise head    +0.0138 (n=11)     merged-rijeka   +0.0117 (n=11)
    distill           +0.0125 (n=11)     hold-annot      +0.0112 (n=37)
    blend-three       +0.0109 (n=6)      ema             +0.0105 (n=10)
    calib-tgt         +0.0109 (n=37)     dinov2-small    +0.0246 (n=218)

The exposure cut moved `distill` from +0.0161 to +0.0125 -- the leak carried 23% of it, not the term --
and dropped it from #2 to #3 behind `labelwise`. Every attribute whose n was untouched moved by less
than 0.0016, so the estimator is not reading pool composition. The other real mover was `blend-three`
(-0.0043), which lost 3 of its 9 members to the cut: its previous marginal was materially
exposure-carried through the stack family.

Those terms sum to +0.0593 over a base of roughly 0.84, which is absurd on its face. Binning the
0%-exposure pool by attribute count k and taking the residual against strict additivity says where it
breaks:

| k | n | median solo | residual vs additive |
|---|---|---|---|
| 0 | 222 | 0.8439 | +0.0000 |
| 1 | 31 | 0.8545 | -0.0004 |
| 2 | 23 | 0.8575 | -0.0085 |
| 3 | 11 | 0.8607 | -0.0191 |
| 4 | 2 | 0.8609 | -0.0293 |
| 5 | 1 | 0.8648 | -0.0364 |

The first attribute transfers whole (+0.0106 median). Each further one realises about +0.003 against a
promise of about +0.011. **Pass-through at k=5 is 0.0209 realised of 0.0573 promised, about 36%, and it
does not flatten through k=5.** Coarsely, k<=1 residual -0.0001 (n=253) against k>=3 residual -0.0197
(n=14).

So the table is neither a sum nor a max. It is a budget with an exchange rate of roughly **0.35 realised
per unit of additive promise past the first term**.

### The prediction, and the arm that tested it

This was written as a pre-registered band while `distill_stack_max` was still on the card -- a member
stacking labelwise, distill, blend-three, calib-tgt and hold-annot at once, on an ~80%-exposure teacher.

    strict additivity, five terms   0.9032
    the saturation curve            0.865-0.867
    honest band incl. exposure      0.865-0.872
    THE ARM                         0.8634, CONVERGED, peak epoch 8 of 12
    honest pool max                 0.8649
    the other k=5 member            0.8648 (chainfour_s2026)

It landed 0.0398 under the additive promise and 0.0016 under the curve's floor, and it does not reach
the pool maximum despite carrying more positive attributes than the member that does. The curve's
qualitative call -- "lands near the pool's top rather than above it" -- is confirmed. The quantitative
miss is small and is not worth defending in either direction: the k=4 and k=5 bins were two members and
one member deep, so the floor was extrapolated from a tail of one. The arm is now the second member in
the k=5 bin, and refitting the curve with it in is the cheapest improvement that bin will ever get.

### What follows, and what does not

**Follows:** additive pricing of a stack arm is dead. An arm stacking N positive attributes states its
pre-registered branches against `base + 0.35 x (promise past the largest term)`, never against the sum.
Every stack arm this project has run was implicitly betting on the flat-residual hypothesis, and it has
now been tested once and refuted.

**Does not follow:** that spreading beats stacking. Saturation says the terms do not COMPOUND; it does
not say a one-attribute member is worth more than a five-attribute one. The k=1 band sits at 0.8545 and
the k=5 members at 0.8634-0.8648, so stacked members still top the pool by about ten ticks -- they
simply do not compound. A spread member is a weaker member whose only possible value is ensemble
marginal, and the seat lever is closed from three directions (routing +0.0008 +/- 0.0033, coverage-greedy
-0.0033, diversity addition -0.0004 +/- 0.0007, the last losing 28/40 precisely because
least-correlated-with-the-six is a proxy for weakest). Two closed results point the same way.

The reading taken here: keep building stacked members, stop expecting the stack to compound, and vary
what the attribute table does not index -- teacher identity, label table, fold geometry -- so members
decorrelate instead of summing.

The open question that would settle it is not in either document: every marginal above is a SOLO
marginal, and the only lever left is a member that improves the six. Solo value and seat value are
different quantities this project has never separated, and D377's in-sample note that ANY seventh seat
on the no-search six is negative says the gap is real. That refit -- per-attribute half-split seat
delta over the fixed six -- is asked and outstanding.

---

## Population note

The packaged honest pool is **133 of 155 unique rostered members**: `members.roster` 137 names plus
`members2.roster` 20, minus 2 cross-file duplicates (`calib_span_df`, `final_ws_df` -- a packaging
defect by members2.roster's own header, since a duplicate is a vote loading weights nobody chose), of
which 21 have no npz and 1 is an `epoch < 0` probe.

`greedy_halfsplit` and `diversity_seventh` both ran on 117, a single-shelf walk missing the second
package's 16 members. **Neither may be quoted until rebuilt on 133.** Their conclusions were null or
near-null, and a null on a subset stays a null on the superset only if the missing members are not
systematically different -- which is the kind of sentence that becomes a fact by being repeated.
