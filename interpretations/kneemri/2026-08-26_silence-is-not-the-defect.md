# The reader's silence is not the defect — its confidence is

**2026-08-26 22:35Z.** Reprices `knee-pju4`, which had been this project's P0 since it was filed earlier
today, on a measurement that cost no card, no slot and about three minutes.

## What the bead assumed

`knee-pju4` rests on one number: **27.3% of all 52,872 (study, finding) pairs carry an `UNK` verdict** and
are filled with a base-rate prior (0.28) instead of a read. It concluded that filling them with a local
multilingual reader was "the only lever found that is plausibly sized" for the remaining gap.

That number is a **coverage** statistic. It says how much the reader did not say. It does not say how much
of what it did not say was POSITIVE — and only the second number bounds what filling the silence can buy.
Nobody had asked.

## What the annotations say

`core.data.labels.reader_agreement` was already the acceptance-test module for any replacement reader; it
just never asked this question. It does now (`silence`, `silence_report`). Over the **684 annotated cells,
57 studies**:

```
177 silent cells of 684 (25.9%), the radiologist calls 13.6% of them positive
                                 against 41.6% where the reader spoke
  a perfect silence-filler fixes at most 24 cells; over-calling costs 108
```

**Silence correlates with absence.** That is the mechanism the bead's premise did not have. The lexicon
goes quiet mostly where the finding is not there, so the 0.28 prior on those cells is not merely
uninformative — it is **too high**, against a true positive rate of 0.136.

The annotated subset is representative on the axis that matters here: 25.9% silent against the corpus's
27.3%, so this is not a hole that happens to be measured on unrepresentative studies.

| what a perfect reader could fix | cells | share of 684 |
|---|---|---|
| every silent cell hiding a positive | **24** | 3.5% |
| every confident false positive | **108** | 15.8% |

And 14 of those 24 are **Synovitis alone**.

## The defect that was already in the file

`errors`' docstring has said this since it was written, without the silence census to size it against:

> *All 108 false positives are confident YES, while 24 of the 32 misses are UNK: the reader is
> confidently wrong far more often than it is uncertain, which is why its constant per-verdict `__conf`
> cannot be used to downweight them.*

Per finding, net over-calling: Contusion **+18**, Effusion **+16**, ACL +9, MCL +9, Medial OA +7, Lateral
Meniscus +7. Synovitis is the sole inversion at **−11**.

`__conf` takes exactly one value per verdict level — standard deviation 0.0 across all 52,872 cells — so
it is a deterministic function of the verdict it accompanies and can never downweight a confident error.
This is the same defect `knee-b8g9` names from the forum side: **an LLM's self-reported confidence
measures its certainty in applying a heuristic to text, not the uncertainty in the diagnosis.**

## What this changes

`knee-pju4` is not killed; its **sizing argument** is. A better reader is still worth building — it is now
`knee-4xww`, and it is built for a different thing:

1. **Precision on the cells the lexicon calls YES**, not coverage on the ones it left blank. That is the
   108, and `reader_agreement` is already the before/after gate — it also refuses tables it cannot read.
2. **A targeted Synovitis pass.** 84.2% UNK corpus-wide, 41 silent of 58 annotated, 14 hidden positives.
   The one finding where silence genuinely is the larger failure, and the finding whose labels predict the
   radiologist worst (0.6870, `supervision_gap`).
3. **Then** the general blank-filling, now priced at ≤3.5% of cells.

**One constraint worth stating before any of it is built.** Our member already scores 0.8932 against the
radiologist while its teacher scores 0.8658 ([our model is past its
teacher](2026-08-26_our-model-is-past-its-teacher.md)) — so the model has *already* learned to discount
some of the teacher's over-calls. A better table must beat what the model has learned to ignore, which is
a strictly higher bar than beating the table.

## The denominator, said out loud

**684 cells, 57 studies, 1.3% of the corpus.** 108 and 24 are small integers with no intervals attached.
Read the **ratio** between them — 4.5× — and never project either as a rate onto the 14,445 blank pairs.
What survives the discount is the ordering and its mechanism: silence tracks absence, confidence does not
track correctness, and the second is the larger hole.

This is the third time in two days a number was repriced by asking what its denominator was. The first
two are in `CLAUDE.md` for the same reason.
