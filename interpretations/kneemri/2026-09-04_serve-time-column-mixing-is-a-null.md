# Mixing the twelve prediction columns into each other does not pay, and the ceiling is taken

**What was measured.** Seven fold-0 members (`band_dense`, `band_span`, `base_dense`, `calib_span`,
`depth_regions_narrow`, `grade_ladder`, `through_dense`) rank-averaged into one twelve-column table over
the 909-study fold-0 holdout, then each column refitted as a ridge combination of all twelve.
`kneemri/evaluation/column_mixing.py`, run 2026-09-04. No card, no submission slot.

## Why this operation and not another

The board metric is rank-based per column — measured, not assumed: `kneemri-monotone` shipped a monotone
transform of the champion's table and read the champion's exact 0.944. So calibration, smoothing,
prevalence matching and every other within-column rescaling are arithmetically dead here. A CROSS-column
blend is the one cheap serve-time operation left that can change within-column order, and there was a
concrete reason to think it might pay: `core.data.labels.table_correction`'s `fracture_contusion` rule
(`Fracture := rescale(Fracture + 0.25 * Contusion)`) reads +0.0632 on the gold Fracture column at
p = 0.0022. That rule is a TRAINING-LABEL correction. This asked whether the same structure survives to
serve time with all 144 coefficients learned instead of one guessed.

## The result

| penalty | mean delta | worst column | best column |
| ---: | ---: | ---: | ---: |
| 0.1 | −0.0072 | −0.0256 | +0.0082 |
| 1 | −0.0055 | −0.0219 | +0.0092 |
| 10 | −0.0044 | −0.0210 | +0.0079 |
| 100 | −0.0058 | −0.0170 | +0.0023 |
| 1e3 | −0.0022 | −0.0056 | +0.0022 |
| 1e5 | **+0.0001** | −0.0001 | +0.0004 |

### And again with the estimator the literature actually uses

The table above is DENSE L2, and on its own it bounds dense mixing and nothing else. Kimi's Q-LD answer
named the published form: Srivastava et al.'s Label-Graph Refinement, `z' = z + a * sigma(z) * A`, `A`
zero-diagonal and **L1**-regularised, evaluated on macro AUC — which at serve time reduces to keeping the
own column at weight one and lassoing the eleven cross terms. That is `SparseGraph` in
`kneemri/evaluation/mixing_rules.py`, and it is the rule with the prior that matches the situation: a
couple of finding pairs genuinely coupled, the other hundred-odd noise. L2 cannot express that prior; L1
can.

| L1 penalty | mean delta | worst column | best column |
| ---: | ---: | ---: | ---: |
| 1e-4 | −0.0088 | −0.0552 | +0.0067 |
| 1e-3 | −0.0064 | −0.0479 | +0.0060 |
| 3e-3 | −0.0041 | −0.0376 | +0.0079 |
| 1e-2 | −0.0019 | −0.0252 | +0.0062 |
| 3e-2 | −0.0024 | −0.0157 | +0.0000 |
| 0.1 | **+0.0000** | +0.0000 | +0.0000 |

Same shape, same verdict. The 0.1 row is every cross term driven to exactly zero — the shipped table,
reproduced to the digit, so the control fires here too. The sparse ceiling is **−0.0019** against the
dense **−0.0022**: the estimator with the right prior buys three ten-thousandths, a third of the noise
floor. **That is what closes the mechanism rather than the method.** Had only the L2 number existed, this
would have been a statement about ridges dressed as a statement about cross-column structure.

The 1e5 row is the identity — the shipped table — and it lands where it must, which is what makes the
rest of the column readable. The **ceiling** of the method is −0.0022: the best penalty picked with
hindsight, a choice no honest procedure is entitled to make, and it is still below the table we already
ship. There is no branch of this that pays. Nothing to re-run with a better selector.

The mechanism is the one the module was written expecting as its null: seven members reading twelve
findings off the same study have already shared what they know. By the time a member writes Fracture it
has looked at the same pixels its Contusion column looked at. The training-label rule works because it
changes what the model is FITTED on; there is no leftover cross-column information at the output.

## The bug this nearly shipped as a result, and how it announced itself

The first version fitted a plain ridge from all twelve columns onto truth, on the argument that "the
identity matrix is the table we already ship, so the baseline is inside the model." **That is false.** A
ridge shrinks toward ZERO, and as the penalty grows `solve` walks the weights toward a correlation-
weighted average of all twelve columns — a table with the target column's own signal diluted. The
identity is nowhere on that path, so the control could never have been chosen even where it was right.

It announced itself in one line: sweeping the penalty up seven decades made the score monotonically
worse (−0.007 at 0.1 → −0.133 at 1e7) instead of converging on the baseline. A control that cannot fire
looks exactly like a method that fails, and the only thing separating them was running the sweep past
where any real fit would sit. **Sweep a regulariser to its degenerate limit and check the number lands on
the control.** That is now the cheapest available test that a shrinkage-based comparison is wired
correctly, and it cost about forty seconds.

The fix is to fit the RESIDUAL — regress `truth − own_column` on all twelve and add 1 back to the target
column's coefficient — which puts the shipped table at the infinite-penalty end of the path by
construction.

## A second reading, about the procedure rather than the method

The honest nested penalty choice (an inner split inside the fitting half, `stack_fit`'s discipline)
returns **−0.0186 — worse than every fixed penalty in its own grid, including both extremes.** An inner
half of ~227 studies cannot separate six decades of shrinkage: it picks a loose penalty on noise and pays
for it out of sample.

That is not a fact about column mixing. It is a fact about nested hyperparameter selection on a fold of
this corpus, and it should be carried forward: at 909 holdout studies, a nested split has enough data to
FIT a small model and not enough to CHOOSE one. Anything here that wants to select a hyperparameter
honestly needs either a fixed value argued from outside the data or a much larger inner set.

## What this closes and does not close

**Closes:** learned serve-time mixing of the twelve output columns, on a rank-averaged pool, under BOTH
the dense L2 estimator and the sparse zero-diagonal L1 one the published work uses. Ceiling taken under
each, negative under each. A third estimator is not owed a run: the two bracket the shrinkage prior from
both ends and land within 0.0003 of each other.

**Does not close:** the training-label rule, which is measured positive and unaffected by this;
cross-label structure used INSIDE a model (a shared head, a label-correlation term in the loss), which
operates before the columns are written and is a different mechanism entirely; and label-dependency
methods evaluated on joint metrics, which this says nothing about because our metric is twelve
independent per-column AUCs.

Related: `kneemri/evaluation/column_mixing.py` (the numbers), `core.data.labels.table_correction` (the
positive training-label version), `interpretations/kneemri/2026-09-04_transduction-is-circular-not-miscalibrated.md`
(the other card-free null this day), Q-LD to Kimi (whether the literature reports macro-AUC gains from
output-dependency models at all).
