# Warm-starting is FLAT at two rates, and the midpoint ruler invented the refutation I was about to build on

2026-09-17. Beads knee-wbwq, knee-rkegi. No card spent: `ledger75` and `convergence_check` over run
records already on disk, plus a read of `recipe_pairs --key encoder_weights`.

This answers the owner's question — *"you are warm starting right? ... the reason it didnt progress might
be the training parameters, not that the model as a whole is saturated"* — and it corrects a file written
three hours earlier on the same day, which claimed the corrected arm had never run. It had. It ran on
2026-08-31, and a second one on 2026-09-14.

**The answer, stated first: warm-starting the best solo we owned was tried at 4e-5 AND at 1e-5, both runs
converged, and both land INSIDE the 0.0062 paired floor of their cold parent. The rate was not the
problem, because fixing the rate changed nothing. What moved the warm base was the operator pair
(`--slot-dropout 0.40`, `--epochs 24`), and it moved it by the same amount it moves cold bases — so the
warm start contributes about zero, and no warm member has ever been the best solo this project owns.**

## The four numbers, one ruler, all converged

`ledger75 --holdouts D:/models/rsna_knee_abnormality --only …`, population `66c6f231`, 909 studies.
`convergence_check` flags none of these four, so every epoch quoted is a real peak and not a killed
process.

| run | start | backbone-lr | operators | ledger75 | peak epoch | vs its own control |
|---|---|---|---|---|---|---|
| `calib_dense_hold_s2026` | cold | 4e-5 | — | **0.8949** | 6 | (the parent) |
| `warm_calib_dense_s2026` | warm from the parent | 4e-5 | — | **0.8916** | 6 | −0.0033 |
| `warm_calib_lowlr_s2026` | warm from the parent | **1e-5** | — | **0.8910** | 4 | −0.0039 |
| `warm_calib_lowlr_drop40_long_s2026` | warm from the parent | 1e-5 | drop40 + 24ep | **0.8994** | 8 | +0.0084 over `warm_calib_lowlr` |
| `labelwise_depth24_drop40_long_s2026` | cold, better base | 4e-5 | drop40 + 24ep | **0.9053** | 18 | the shipping corner |

## Three findings, in order of what they cost me

**1. The midpoint ruler manufactured a refutation.** `warm_calib_dense` reads **−0.0072 at the midpoint
and −0.0033 on ledger75** against the same parent. The paired floor is 0.0062. So on the midpoint the
warm start is a clear loss worth explaining, and on the good ruler it is FLAT and explains nothing. Every
word of the schedule story below was built on the midpoint number. This is the `nn_corr@pool251` defect
again — **a scalar read off the pool carries its CUT, and −0.0072 without one was never a fact.**

**2. The schedule hypothesis is refuted by its own arm.** `warm_calib_dense`'s header pre-registered the
cause of any large loss: *"one_cycle ramps the backbone rate above 4e-5 before annealing, and a converged
trunk can be walked off its optimum by the ramp. That outcome sends the next arm to a lower
--backbone-lr."* That arm exists and ran — `warm_calib_lowlr`, 4e-5 → 1e-5, single variable. It read
**0.8910, i.e. 0.0006 WORSE than the arm it was sent to rescue.** Quartering the rate moved nothing. The
ramp was not the mechanism, because removing the ramp's bite did not move the number.

**3. What actually moved the warm base is base-independent.** `warm_calib_lowlr_drop40_long` adds
`--slot-dropout 0.40` and `--epochs 24` to the warm 1e-5 line and gains **+0.0084** over its matched warm
control. That is the same operator pair, at the same size, that `band_dense` measured on a cold base
(+0.0041) and that carries the cold corner. The warm start is not what produced the gain; the operators
are, and they were available cold. And the result still sits **0.0059 below the cold corner's 0.9053**.

## What this closes, and what it does not

**Closed: warm-starting as a QUALITY lever.** Three warm runs against a matched cold parent, two of them
single-variable, both inside the floor, and the one mechanism the corpus proposed for the shortfall
tested and null. There is no unarmed arm left on this axis that the existing runs do not already answer,
which is the opposite of what this repo believed at 14:00Z today. **The ~1.6 h / P~0.15 warm-start-at-the-
corner arm is withdrawn** — it would re-run, on a better base, a comparison that has been made twice.

**Still standing: the TIME claim.** `warm_calib_dense` epoch 0 reads 0.8503 at midpoint where the cold
run needs four epochs to get there — one warm epoch is worth about five cold ones. Warm-starting remains
the right way to get a cheap member, and the wrong way to get a better one.

**Not closed: the owner's actual hypothesis.** *Is the corner in a local optimum?* Warm-starting was only
ever one probe of it, and a probe that reads flat says the trunk neither gains nor loses from being
re-annealed — not that no better basin exists. What it does say is that **a fresh schedule over good
weights is not the instrument that will find one**, so that hypothesis needs a different arm entirely
(a different trunk, a different objective, or different data), not another rate.

## What this does NOT license

- It does not say `recipe_pairs` is wrong. It says a pair table answers only about runs with matched
  single-key siblings, and the three warm starts that matter have none — so **an axis can read "6 pairs,
  0 clearing" while its decisive runs sit outside the table.** Check the beads AND the checkpoint
  directory for an axis before pricing it off a pair count.
- It does not license quoting any of these deltas at the midpoint. All five rows are ledger75 on
  population `66c6f231`; the midpoint disagrees with the sign of the conclusion on row two.
- **The process failure is the finding worth keeping.** A file was written this morning asserting an arm
  had never run, from a grep of `experiments/arms/*.sh` for a FLAG and a read of a three-week-old bead.
  The arm was on disk under a name the grep did not predict, and `ls $MODELS | grep warm_calib` would
  have found it in one second. **Before writing that something was never tried, list the checkpoints.**
