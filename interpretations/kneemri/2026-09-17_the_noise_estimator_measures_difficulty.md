# Corpus OOF disagreement measures column DIFFICULTY, not label noise

*2026-09-17. ~0.4 h CPU, no card. The gate knee-mmxew, which refused the ~6 h arm knee-dz8ip before it
was queued.*

## What was being gated

`2026-09-17_the_scaling_class_is_closed.md` ended by naming the one surviving lever: the board metric is
an unweighted mean over twelve findings, the reader's own error rate across those findings runs from
9.7% FP (Synovitis) to 69.6% (Effusion), so equal-weighted BCE spends its largest gradients on the
noisiest column. A per-column reliability weight changes the RELATIVE gradient in a way no monotone rung
remap and no amount of capacity can.

The blocker was never the weighting — it was where the weights come from. Any weight fitted on the 58
gold cells is unreadable on the 58 gold cells, which is the `raptor_solo` mechanism under a new name.
The proposed way past it: estimate per-column label noise from the disagreement between the report label
and a member's own OUT-OF-FOLD prediction over the full corpus, which is held out by construction and
never touches the 58.

That estimator already exists. `kneemri/evaluation/label_errors.py` runs confident learning
(`cleanlab.filter.find_label_issues`, twelve independent binary problems so the CELL stays the unit) over
`PublishedRuler` — twenty published checkpoints, four seeds × five folds, `oof` id-sets partitioning all
4,407 studies. The gate asks one question of it: **does its per-column flag rate recover the gold
ordering of reader error?**

## It does not

```
Spearman rho = 0.1194   (p = 0.7116, n = 12 findings)

corpus order  PF OA · Effusion · Medial Meniscus · Medial OA · Lateral OA · Contusion ·
              Lateral Meniscus · ACL · Synovitis · Baker's · MCL · Fracture
gold   order  Contusion · Synovitis · Effusion · Lateral Meniscus · PF OA · Fracture ·
              Medial OA · MCL · Lateral OA · ACL · Medial Meniscus · Baker's
```

```
finding             corpus flag rate        gold reader error
Fracture              0.0297  (12th)          0.1754  (6th)
Medial Meniscus       0.1248  ( 3rd)          0.1404  (11th)
PF OA                 0.1870  ( 1st)          0.2105  ( 5th)
Contusion             0.0976  ( 6th)          0.3860  ( 1st)
Synovitis             0.0958  ( 9th)          0.2982  ( 2nd)
Effusion              0.1369  ( 2nd)          0.2807  ( 3rd)
```

Effusion is the only column the two orderings agree on. The rest disagree in one consistent direction:
the corpus order is topped by PF OA, Effusion and the three OA/meniscus columns — the common, spatially
ambiguous findings — and floored by Fracture, which is rare and unambiguous when present. That is a
ranking of **how hard the column is to predict**, not of how often the label is wrong. Confident learning
flags a cell when the model's out-of-fold probability contradicts the label, and a model contradicts the
label most where it is weakest, whoever is right.

The two columns where the reader is measurably worst — Contusion 51.3% FP, Synovitis 51.9% FN — sit 6th
and 9th in the corpus flags. Weighting by this estimator would have moved gradient TOWARD PF OA and the
OA columns and AWAY from Contusion and Synovitis: the exact inversion of the intent.

`label_errors.py`'s own docstring already names this failure shape for a different input — *"A weight
that rises with the reader's confidence therefore UP-WEIGHTS the reader's dominant error by
construction."* The corpus estimator fails the same way through difficulty rather than through
confidence.

## Power, stated so this is not read as stronger than it is

Each gold rate is 6 to 22 errors out of 57 cells, so its own sampling sd is roughly 0.05 against rates
spanning 0.105 to 0.386, and a rank correlation over n=12 has weak power. This is **no evidence that the
estimator recovers the ordering**, not proof that it is orthogonal to it. What makes it a refusal rather
than a shrug is that the point estimate is ~0 AND the named disagreements all go the way difficulty
predicts — the pre-registered condition.

## What is closed and what is not

**Closed:** this route to the weights. knee-dz8ip is closed unarmed; ~6 h of card refused for ~0.4 h of
CPU, which is the whole reason the gate was filed separately.

**Not closed:** per-column reliability weighting itself. The mechanism is untouched — it is still the one
change that could move the relative gradient across all twelve columns. What it lacks is an estimator
that is neither fitted on the 58 gold cells nor confounded with difficulty, and this page rules out the
only candidate the project had. A second independent reader over a larger sample would be one; there is
no budget for it.

The gate was also not mechanised into a `--per-column` flag on `label_errors.py`. A code path that exists
only to re-derive a closed lever's refusal is dead code, and this page is the artefact.

## Denominators

Corpus flag rate: 4,406 studies per column (the 4,407-study OOF union minus one row absent from the label
table), masked to cells the reader actually read. Gold reader error: 58 annotated studies, 57 cells per
column, 696 cells total, 34.5% radiologist-positive. Confident learning over `PublishedRuler`, twenty
published checkpoints, four seeds × five folds.
