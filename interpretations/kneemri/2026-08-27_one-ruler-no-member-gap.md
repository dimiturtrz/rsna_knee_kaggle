# One ruler, and the member gap is not there

*2026-08-27, 14:41Z. Card busy with `fuse_regions`; slot 5 held for ~23:30Z. No slot spent on this.*

## The thing that was wrong for two weeks

`knee-rh4t` has priced a **~0.017 member gap** since 2026-08-25: rivals self-report 0.938–0.942 for a
single model in forum topic 735304, our best board single `soloseat_v1` reads 0.925, and the difference
was read as a fact about their model. `knee-urvf` built on it — the ensemble tracks best-single + ~0.015,
that bonus has been flat fourteen days, so member quality is the only free term. Both are P0. Both have
shaped what got queued.

The two numbers were never on the same ruler. Ours is a **board** score on the hidden Kaggle test set;
theirs is a **forum claim** about a run we cannot see, on a split we cannot see. Nothing joined them.

Kaggle user `dreaddevelopment` published six fine-tuned twelve-finding checkpoints, and each one carries
its own `aucs` dict — macro-AUC per finding on the **58 radiologist-annotated studies**. Their training
corpus is 4,349 studies, and 4,407 − 58 = 4,349, so those 58 are held out of it.

Our six cached `annotated_predictions` are the `*_hold` family — members trained with
`--hold-annotations`, which holds exactly those 58 studies out.

**Both sides hold out the same 58 studies, and both report per-finding AUC against the same
radiologists.** That is one ruler, and it had been sitting in the repo unjoined.

## What it says

| finding | n+ | ours `calib_span_hold` | theirs `coatnet_v4_full` | delta |
| --- | --: | --: | --: | --: |
| PF OA | 21 | **0.9305** | 0.8288 | −0.1017 |
| Lateral Meniscus | 23 | **0.9366** | 0.8807 | −0.0559 |
| ACL | 24 | **0.9914** | 0.9669 | −0.0245 |
| Medial Meniscus | 26 | **0.9688** | 0.9483 | −0.0204 |
| Synovitis | 27 | 0.7694 | 0.7575 | −0.0119 |
| Fracture | 18 | 0.9236 | 0.9125 | −0.0111 |
| Contusion | 19 | 0.9001 | **0.9150** | +0.0148 |
| MCL | 9 | 0.9524 | **0.9705** | +0.0181 |
| Baker's | 12 | 0.9493 | **0.9873** | +0.0380 |
| Effusion | 35 | 0.9366 | **0.9789** | +0.0422 |
| Lateral OA | 11 | 0.8259 | **0.8685** | +0.0426 |
| Medial OA | 15 | 0.8977 | **0.9860** | +0.0884 |
| **MACRO** | | **0.9152** | 0.9167 | **+0.0016** |

**+0.0016 against a macro standard error of ~0.012. The member gap is a null.**

That does not make the board numbers wrong — 0.925 and 0.941 are what Kaggle returned. It makes
`knee-rh4t`'s *mechanism sentence* unsupported. Whatever separates a rival's 0.942 from our 0.925 is not
that their member is a better model of the twelve findings; on the only matched measurement either side
has, the two members are the same strength. The residual has to be explained by something else — their
inference protocol (42 windows per study against our single view), their test-time ensembling, or the
forum claim being a different kind of number than it reads as.

## What it does NOT say, stated before it gets quoted as more

n = 58. Hanley–McNeil standard error per label runs **0.031 to 0.072** across these positive counts, so
**not one per-label delta clears its own noise** — not the +0.0884 on Medial OA, not the −0.1017 on PF OA.

The per-label oracle — take whichever model wins each finding — reads **0.9355, +0.0188 over the better
single**. That number is worthless, and the arithmetic that shows it is worth stating properly, because
my first pass at it was wrong in a way that happened to reach the right verdict.

I estimated the selection inflation by hand as `SE/√π ≈ 0.023` and called the oracle "smaller than its
own null". That over-states the null: routing's baseline is the *better* single, which is itself selected
on the same noise, so part of the inflation cancels. `foreign_ruler` simulates it properly — draw two
models both at the pair's mean, perturb each finding by its own Hanley–McNeil SE, route — and the null
comes out at **+0.0163**, *below* the observed +0.0188. On a mean-threshold test that reads as "exceeds".

The mean is the wrong test. The null has scatter of its own, and **33.9% of its draws reach +0.0188 or
better**. So the honest statement is unchanged in verdict and much better founded: the oracle is what
routing on noise buys, one time in three. The module now prints that share rather than a threshold verdict
— which flipped four of the six foreign checkpoints from "exceeds its own null" to noise.

So the complementarity pattern — they win the diffuse/fluid group (Medial OA, Lateral OA, Effusion,
Baker's), we win the focal structural group (PF OA, both menisci, ACL) — is a **hypothesis**. It is an
attractive one, because it recapitulates the focal-vs-regional structure scale this ledger has been
reporting independently for two weeks, and because a 6/6 split that is *anatomically coherent* is a
different object from a 6/6 split that is scattered. But coherence is not significance, and the honest
next step is `knee-9g28`: score their checkpoints on our 909-study fold-0 holdout, fifteen times the
studies, which is a ruler that can actually settle it.

## The second table, which is mostly a rediscovery — and the correction I owe it

The same six cached sets, ranked:

```
calib_span_hold       0.9152     <- --calibrated-targets
calib_dense_hold      0.9117     <- --calibrated-targets
finding_calib_dense   0.9098     <- --calibrated-targets
band_span_hold        0.9023
band_dense_hold       0.8956
holdannot             0.8932
```

Two **paired single-flag** comparisons — same recipe, same studies, one flag moved:

- span corner: `band_span_hold` 0.9023 → `calib_span_hold` 0.9152, **+0.0129**
- dense corner: `band_dense_hold` 0.8956 → `calib_dense_hold` 0.9117, **+0.0161**

**Most of that is already in the ledger, and I nearly filed it as a discovery.** `knee-bs4l` recorded the
span corner at **+0.0072 proxy / +0.0129 annotated** on 2026-08-26, and the dense-hold entry that same day
states flatly that *"`--calibrated-targets` is the only measured-positive term this project currently
holds."* `calib_dense` was not killed unrun — it ran, as `calib_dense_hold`, and at 0.8634 it is this
project's best offline single member. `knee-ep74` has nothing to reopen. The −0.0020 story below is about
the *older* `calib_span` arm, which was revived a day before I re-found it. This is the second time this
repo has produced a P0 against something it had already measured better, and the rule that catches it —
read the ledger, not a grep about the ledger — is written down precisely because grepping is what I did.

**And these numbers are not independent.** `annotated_readings.py`'s own docstring says so: `Supervision.
MEASURED_AGREEMENT` and `FINDING_OFFSETS` are hardcoded constants **fitted on these same 696 cells**, and
`--hold-annotations` drops the cells from the *training table* without refitting the constants. So a
calibrated arm's annotated read is scored on a ruler that shares an origin with its own targets. It is a
consistency check, never a confirmation. (`level_agreement.stability` refits on disjoint halves and puts
four of five levels under 2σ, which is what makes the number quotable at all — it does not make it
independent.)

What is genuinely new here is thin, and worth stating at its real size:

- **The dense corner's annotated number, +0.0161.** The ledger holds that corner on the offline proxy
  only. Third corner-ruler cell, same sign as the other two, subject to the circularity above.
- **`finding_calib_dense` 0.9098 sits *below* pooled `calib_dense_hold` 0.9117.** The *per-finding*
  offsets do not beat the *pooled* constants where both are read against the radiologists. −0.0019 is far
  inside n=58 noise, so this is "no evidence per-finding helps", not "per-finding hurts" — but it is the
  first read of that pair on this ruler, and it bears on `knee-p9oc`.
- The six `*_hold` members ranked on one ruler, which had not been tabulated.

The mechanism that explains why the *offline* ruler undercounts the lever is not subtle, and it was
written down before any of this was measured. `core/data/labels/finding_calibration.py`'s own docstring:

> It cannot check itself: a holdout AUC is scored against the REPORT labels, which carry the very defect
> these offsets correct, so the offline ruler is biased AGAINST any table that stops agreeing with the
> reader.

The offline holdout is made of the reader. A lever whose entire purpose is to move the target *away* from
the reader will score negative on it by construction. The −0.0020 was never a measurement of the lever;
it was a measurement of how far the lever moved.

**Three rulers, three agreements, one mechanism:**

| ruler | reading |
| --- | --- |
| annotated 58 (paired, single flag) | **+0.0129** span corner, **+0.0161** dense corner |
| board, six seats | `resseat` 0.938 → `calibsix` **0.941** (+0.003) |
| board, eleven–twelve seats | `honest12` 0.937 → `calibmix` **0.941** (+0.004) |
| offline holdout (report labels) | −0.0020 — the outlier, and the one built from the defect |

Two board entries at 0.941 from *different member sets at different seat counts* is a replication. The
board and the annotated ruler agree with each other; the holdout ruler is the odd one out, and there is a
stated mechanism for why.

Honest size: +0.013/+0.016 is 1.1–1.3σ against the unpaired macro SE of ~0.012, and the pairing cuts that
SE by an amount this document does not quantify. On its own ruler it is **directional** — and it is
*circular*, per the correction above. **The board rows are the only non-circular support calib has**, and
they are what makes it an asset: two entries at 0.941, different member sets, different seat counts,
nothing shared but the flag. That is the part of this table worth acting on, and it is the part that was
already true before I opened the file.

## The general hazard

This is not a fact about calibration. It is a fact about **every label-side lever this project has ever
measured**. Any change that moves the training target away from the report is scored by a ruler made of
the report, and will be undercounted by construction — the more the lever works, the worse it scores.
`calibrated_targets`, `confident224`, `blendlabels`, `grade_ladder`, the whole family.

The instrument already exists (`kneemri.evaluation.annotated_readings`, and the six cached prediction
sets that made this document possible in four minutes with no card and no slot). It was never made a
routine second reading. It should be: **no label-side lever's sign is trusted from the holdout alone.**

## The instrument

`kneemri/evaluation/foreign_ruler.py`, run as:

```
python -m kneemri.evaluation.foreign_ruler \
  --cache <processed>/annotated_predictions --foreign <external>/raptor --train <raw>/train.csv
```

It ranks our cached annotated members, reads the best one against every foreign checkpoint that publishes
per-finding AUCs, and prints each oracle beside its simulated null and that null's quantile. Three things
are built in rather than remembered: `weights_only=True` so a foreign archive's numbers are read without
running its code, a `sha256` check that the cached block's study order is the one being scored against
(a mismatched order scores at chance, and chance is a number the module would otherwise *print*), and the
quantile test above.

It is not a verdict machine. Its output chooses what to measure on the 909-study holdout.

## Beads

- `knee-3uje` — the matched read; `knee-rh4t`'s mechanism sentence is unsupported.
- `knee-m4ei` — the calib table, **carrying its own correction in `--notes`**: mostly a rediscovery of
  `knee-bs4l`, circular against constants fitted on the same cells, and `knee-ep74` does *not* reopen.
  What survives is the dense corner's annotated cell and pooled-beats-per-finding.
- `knee-9g28` — score their six checkpoints on our 909-study holdout. Needs the card.
