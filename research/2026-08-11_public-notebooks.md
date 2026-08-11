# What the public frontier for this competition actually does

*Field synthesis — theirs, not ours. Our own measurements live in
`interpretations/kneemri/`. Sources are Kaggle notebooks, pulled with
`kaggle kernels pull <ref>` and read as source rather than as rendered pages (every Kaggle page is
JavaScript-only to a fetcher; the CLI is the way in).*

## The notebooks that matter

| ref | votes | what it is |
|---|---|---|
| `pilkwang/rsna-knee-baseline-v1` | 268 | the baseline everyone forks; also publishes the LLM label table |
| `prvsiyan/rsna-knee-read-the-report-then-the-knee` | 138 | the multilingual rule reader, argued at length |
| `ryanholbrook/rsna-knee-abnormalities-efficiency-lb` | 110 | the host's Efficiency leaderboard, published as a notebook output |
| `fleongg/rsna-knee-dinov2-base-physical-scale-soup` | 8 | DINOv2-**base**, 130 mm crop, EMA, ranking loss, checkpoint soup |
| `aadigupta7686/0-899-let-me-cook` | 77 | 0.899 public, and mostly a report reader |

## The single most useful observation

**Their model design and ours have converged; their remaining differentiator is the label reader.**

The 0.899 notebook independently arrives at: a report-hash split so template reports cannot straddle it,
members that carry the preprocessing they were fitted under, a seeded fingerprint recomputed before use
so a mismatched member stops the run, weights shipped as a dataset so the scored run does not retrain,
and an annotation check that is reported but never allowed to arbitrate. That is our architecture, point
for point, and it is not the interesting part of their notebook — it is preamble.

What they actually spend their lines on is a **multilingual rule-based reader** over the radiology
reports: negation scoped by direction rather than by clause, Turkish and Croatian negators that govern
what precedes them, `sin alteraciones` / `geen afwijkingen` read as explicit normality rather than as
negation, numeric grades interpreted per structure (meniscus grade 3 is a tear, ligament grade 1 is a
stretch), and subchondral oedema under a cartilage defect read as osteoarthritis rather than as a bruise.

Two facts make that a coherent bet rather than fashion:

- **52% of the corpus reports contain no common English word** (Spanish, Dutch, French, Greek, Cyrillic).
- The metric is a rank metric, so a structure a radiologist looked at and called intact must rank *below*
  one the report never mentions — not equal to it. A reader that throws explicit normals away collapses
  that distinction.

## What we have already tested against this

| their idea | our result | note |
|---|---|---|
| 32-slice inference | −0.009 to −0.010 | refuted, under BOTH poolings — the mechanism we proposed was wrong too |
| per-slice max pooling | +0.0018 | inside our 0.006 noise floor |
| loss normalised by label count | −0.0022 | refuted |
| weight floor at 0.25 | +0.0019 | inside noise |
| label fusion across readers | 0.8966 vs 0.8982 best single | refuted; ours are all derived from the same reports |
| their label table vs another | −0.0008 | underpowered, could not have detected anything |

The pattern is worth stating plainly: **almost nothing transferred, and the two things that did were our
own bugs** (augmentation never called, +0.0110; a fold split derived from the training seed).

## Untested and still plausible

- **DINOv2-base.** They run base; we have only ever run small. 86M parameters against 22M. Running.
- **EMA.** Universal in their notebooks; damps exactly the run-to-run wobble we measured at 0.0062.
  Wired 2026-08-11 (`--ema`), and measuring.
- **Checkpoint soup — subsumed, not skipped.** A soup averages weights along one training trajectory,
  which is what an EMA already computes online. Souping across our *seeds* would be a different and
  worse thing: `--seed` reseeds the head's initialisation as well as the data order, so two seeds start
  their heads in different basins and averaging them is not a soup, it is a collision. The shared
  DINOv2 init means only the ENCODER halves would be legitimately soupable, and a head matched to a
  soup of encoders it never saw is a new model, not an average. So EMA is our test of this mechanism.
- **Final full-data fit.** Available as `--fold -1`. The place EMA should matter MOST, and for a reason
  separate from variance: with no holdout, the run keeps its last epoch by construction, so the answer
  depends entirely on where training happened to stop. An average removes that arbitrariness.
- **The report reader itself.** Expensive. Gated behind a cheap power test first (`knee-5od`): if
  striking the current reader's own `UNK` calls — 27% of all pairs — cannot move our score by more than
  the noise floor, then label content is not our lever whatever they spend their lines on.

## What their leaderboard file gives away

`kaggle kernels output ryanholbrook/rsna-knee-abnormalities-efficiency-lb` returns the full efficiency
leaderboard as a csv. From it: all 905 teams are ranked automatically, the best accuracy anywhere on the
board (0.940) sits at **efficiency rank 638**, and the teams scoring exactly our 0.888 sit at ranks
**29, 33 and 50**. Runtime is weighted hard, and it is not a re-sort of the accuracy board.

## What the host says, from the primary source

`rsna.org` and the competition page, rather than a notebook's summary of them:

- **The evaluation set is annotated by expert radiologists.** Our training labels are read out of
  reports; the thing being scored is not. That is the ruler gap, now confirmed rather than inferred —
  see `interpretations/kneemri/2026-08-11_what-actually-moved-the-number.md`.
- **Twelve languages, 16 sites, five continents**, over 5,000 exams. We had recorded *nine* languages,
  which was a count of what we detected rather than of what is there. Corrected in `docs/PLAN.md`.
- **The competition runs to 22 October 2026**, winners announced in November. So there are no published
  winning solutions to gather yet: everything in this document is a mid-competition public notebook, and
  the frontier it describes is the *shared* frontier, not the best one.
- **$77,000 total, including a first-time efficiency award** — which is the second leaderboard our
  efficiency kernel targets.

The practical consequence of the third point: "SOTA gather → replicate → merge → build upon" has no SOTA
to gather here. The public notebooks ARE the floor, we have replicated them, and the remaining distance
to 0.940 is somebody's unshared work. That argues for spending effort on the axes our own measurements
endorse rather than waiting for a write-up that will not exist until December.

Sources: <https://www.rsna.org/news/2026/august/ai-challenge-knee-mri>,
<https://www.kaggle.com/competitions/rsna-knee-abnormality-detection>
