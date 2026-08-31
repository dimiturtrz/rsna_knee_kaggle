# Full slice coverage is not an aggregator bug — it is not a member lever at all

`cover_labelwise` (knee-tlya), read 2026-08-31 against both of its controls, per label, with
`arm_read` and `run_comparison`. Converged: 12 epochs asked, 12 run, best at epoch 8;
`convergence_check` does not list it, so the score is a score and not a floor.

## What was pre-registered, and which branch fired

`stack224` (knee-qphd) read the whole 36-slice stack under `--slice-pooling max` and finished at
0.8476 against `native224`'s 0.8502. Its two thinnest findings collapsed — Lateral Meniscus -0.0263,
Fracture -0.0180 — inside a mean that merely looked flat. knee-tlya read that shape as an AGGREGATOR
failure: the expected maximum of a set of noisy per-slice scores grows with the size of the set, so
coverage and max pooling are in direct conflict, and a finding living on two slices out of
thirty-six is exactly the case a spurious maximum drowns. The test moved one flag,
`--slice-pooling labelwise`, and pre-registered three branches.

Measured, against `native224` (0.8502):

| finding | stack224 | cover_labelwise | moved |
|---|---|---|---|
| Lateral Meniscus | -0.0263 | -0.0150 | +0.0113 |
| Fracture | -0.0180 | -0.0198 | -0.0018 |
| MCL | -0.0012 | **+0.0247*** | +0.0259 |
| Medial Meniscus | +0.0003 | **+0.0125*** | +0.0122 |
| MEAN | -0.0026 | -0.0007 | +0.0019 |

`*` marks the pair's own paired floor; half-widths reach ±0.0372.

**Branch three fired.** Both thin findings are still below control, the mean is still below control,
and the branch's text is unambiguous: coverage itself is refuted independent of the aggregator.
knee-qphd closes and knee-ujd3 (`stack_cover_ff`) closes with it.

## The part that is not in the branches, said rather than dropped

The aggregator DID do what the mechanism predicted on one of the two labels it named: Lateral
Meniscus recovered +0.0113 of its -0.0263 collapse, and it recovered it by swapping a shared
extremum for a per-finding attention map, which is the exact substitution the argument called for.
That is a mechanism holding, and it is also inside the ±0.0372 half-width, which means this run
cannot tell it from scatter. Fracture did not move at all. So the honest statement is that the
aggregator argument survives as a mechanism and dies as a lever: it is worth at most a third of one
label's collapse, on a run whose ruler cannot resolve a third of a label's collapse.

What that closes is COVERAGE, not `labelwise` pooling, which was already proven at 20 slices
(`labelwise_dense_s2026`, 0.8613) and is untouched by this reading.

## The one real gain, and why it does not buy anything

MCL +0.0247 clears the paired floor against BOTH controls — the only starred move in either
comparison, and the largest single-label gain any 224 px arm has produced here. Medial Meniscus
+0.0125 is starred against `native224` too. Neither survives into the mean, because six other labels
give it back, and the mean is what a member is seated on.

Priced as a member against `honest12` (0.937, twelve members, fully scoreable): seat value
**-0.0001**, band [-0.0006, +0.0004], best available seat -0.0000. Zero of one clears +0.0015. Under
knee-c0u9 a negative offline reading is charged by the board at roughly 3x, so this is not a free
look.

## First real reading from `member_axes`, and what it does NOT say

`cover_labelwise` reads **DISTINCT** on the three-ruler join — rank 0.8587 against a roster boundary
of 0.8956, Mahalanobis 4.627 against a boundary of 4.436, JS 0.0173. It is genuinely off the
roster's span, not merely noisy along an axis the pool is already wide in.

And its seat value is -0.0001. That is not the join contradicting the seat ruler; DISTINCT is what
the join says is NECESSARY for a seat, never what is sufficient. The pair is worth recording exactly
because it is the shape the join was built to make visible: a member can be off the pool's span and
still have nothing to sell, and a rank ruler alone would have called this one merely uncorrelated.

The knee-iyad adoption gate is untouched by this. One candidate that both rulers decline is
agreement, not reproduction of a board-validated seat decision.

## What is now closed

- knee-qphd — full slice coverage. Refuted as a member lever, on its own pre-registered branch.
- knee-ujd3 — `stack_cover_ff`, which was gated on knee-qphd surviving.
- knee-tlya — this arm. The aggregator hypothesis is answered: real mechanism, sub-resolution size.

## What is not

Whether per-finding attention buys anything at 20 slices where coverage is not in play — that is
`labelwise_dense`'s question and it already reads 0.8613. And whether MCL's +0.0247 is a finding
about MCL or about this run: one starred label out of twelve at a ±0.0372 half-width is what a
single arm can say, and nothing here licenses a per-finding architecture on it.
