# The three YES levels rank held-out studies above chance on every label

`python -m kneemri.evaluation.ordinal_signal --checkpoint wsfold0_e6.holdout.npz` — 909 held-out
studies, epoch 11, no GPU, seconds to run.

The report reader emits three graded values inside YES — 0.68, 0.82, 0.94 — and the open question was
whether they are SEVERITY (a worse tear scores higher) or only the reader's CERTAINTY (a
confidently-worded report scores higher). If it is certainty, a stronger reader has to supply real
grades before ordinal supervision means anything; if it is severity, the gradation is a signal already
in the corpus.

Concordance is the tie-correct Mann-Whitney statistic over pairs of studies at DIFFERENT graded
levels, so 0.5 is what a model that only learned present-vs-absent scores. Annotated rows are excluded
— `Targets.load` overwrites derived values with hard 0.0/1.0 where a human annotation exists, so a 1.0
there is a radiologist saying "present", not a reader saying "severe".

| label | concordance | pairs |
|---|---|---|
| Effusion | 0.826 | 79,925 |
| Baker's | 0.812 | 13,909 |
| Medial Meniscus | 0.799 | 71,647 |
| ACL | 0.770 | 19,008 |
| Medial OA | 0.744 | 35,310 |
| Contusion | 0.718 | 18,523 |
| Lateral OA | 0.674 | 17,095 |
| Synovitis | 0.674 | 3,311 |
| MCL | 0.669 | 6,786 |
| Lateral Meniscus | 0.667 | 15,296 |
| PF OA | 0.646 | 53,010 |
| Fracture | 0.565 | 1,122 |

Twelve of twelve above chance, none marginal except Fracture (0.565 over 1,122 pairs — the thinnest
band here, and the one label whose graded levels may genuinely be hedged wording).

**What this rules out.** "The levels are the LLM hedging" does not survive: hedging in a report is not
visible in the pixels of a study the model never saw, so a model that only learned certainty would sit
at 0.5. The ordering generalises out of sample, which means the level tracks something in the image.

**What it does NOT establish.** The model was FITTED on these same graded values as soft targets
(`Targets.load` → `objective`, unbinarised), so this measures that the graded target is *learnable*,
not that it matches clinical grade. A 0.94 effusion is reliably bigger than a 0.68 effusion in a sense
the pixels support; whether that sense is the radiologist's grading scale is a separate question that
needs a reader emitting real grades (meniscal 1/2/3, partial vs full-thickness) to answer.

**What it changes.** The (b) branch of knee-mxj — spend LLM budget on a reader that emits ordinal
grades — is now worth its cost, because the cheap prior it was gated on came back positive. The
counterfactual mattered: had these been certainty, the reader work would have been building supervision
on top of noise.

**The denominator.** 909 studies, pair counts as tabled — the concordance for a label is over pairs
drawn from the graded band ONLY, which is 6,461 + 6,427 + 3,241 rows corpus-wide before the
annotated-row exclusion, not over the whole corpus. This says nothing about how the model ranks a
graded study against a NO study; that is the ordinary AUC the ledger already carries.
