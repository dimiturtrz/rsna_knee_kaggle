# What label correction is actually worth — 2026-09-02

Six corrections were built and measured against the 57 radiologist-annotated studies in one session.
**One survives.** This file records the five that do not so they are not re-proposed, and states the
ceiling the survivor runs into.

## The ruler, stated before any number

`train.csv`'s twelve finding columns are blank on 4,349 of 4,407 rows. They are a **gold annotation set
of 58 studies, not training labels**; training labels are report-derived and live under
`processed/.../labels/`. 57 of the 58 appear in the blend tables, so every gold number below rests on 57
studies, and on Fracture that is 18 positives against 39 negatives.

This ruler matters because `test.csv` carries ONLY `StudyInstanceUID` — the hidden test has **no Report
column**, so its truth cannot be report-derived, and `docs/PLAN.md:45` already records that RSNA states
the evaluation set is radiologist-annotated. Correcting toward the radiologist points at the real ruler.
That is the whole case for this lane.

The gold set is also **internally inconsistent**: it counts contusional marrow oedema IN as Fracture
while denying three reports that explicitly name a subcortical, impaction or osteochondral fracture.
Every number here inherits that.

## The one that survives

**Fracture := monotone_rescale(Fracture + 0.25 · Contusion)**, written to `blend_hybrid_fx.csv`.

| table | mean gold AUC | Fracture |
|---|---|---|
| `blend_three_v1.csv` — what every arm to date trains on | 0.8955 | 0.8147 |
| `blend_hybrid.csv` | 0.8978 | 0.8713 |
| `blend_hybrid_fx.csv` | **0.9030** | **0.9346** |

Pre-registered before measurement. Permutation p = 0.0022 on `report_labels_v2`, p = 0.0010 on
`blend_hybrid`; bootstrap CI excludes zero both times; the placebo is coherent — all four helping
partners are the acute-trauma set (ACL, MCL, Contusion, Effusion), all seven hurting ones degenerative.
Verified byte-wise to move the Fracture column and nothing else (eleven columns at exactly +0.0000).
The rescale is order-preserving to 0.999984 — the residual is 4-dp rounding on write, not the transform.

**The mechanism.** Report-derived Fracture misses bone-contusion impaction injury. Of the four gold
studies where the reader said UNK and the radiologist said fracture, **all four reports describe a bone
bruise and none uses a fracture word**. Contusion is the report's name for what the radiologist scored.
The reader's NO is never wrong on Fracture (12/12); its error is entirely over-calling (6 wrong YES
against 4 misses), which is the corpus-wide 108-vs-32 shape in miniature.

**The ceiling.** Fracture is 1/12 of a mean-of-AUCs, so +0.0632 on the column is **+0.0053 on the
score**. That is the same wall the per-label programme hit at +0.0043. Real, below the bar, directional.

**A second, free fact.** `blend_three_v1` is DOMINATED and we have been training on it: it degrades
Fracture to 0.8147 against the plain report label's 0.8713, **−0.057 self-inflicted on our second-weakest
column**. `report_labels_v3` is identical to `v2` on every gold column.

## The five that do not survive

1. **Global severity correction — REFUTED.** Permutation p = 0.3575. Built, tested and killed in one
   call; the raw +0.0116 it showed was a search artifact and is not reported as anything else.
2. **Synovitis += w · Effusion — detectable and nil.** +0.0019, identical at w = 0.25, 0.50 and 0.75,
   permutation p = 0.0462. The Fracture rule does **not** generalise to another column, which is what
   the permutation null predicted. Note `blend_hybrid` had already fixed Synovitis on its own (0.8142
   against `report_labels_v2`'s 0.6938), so the headroom was spent before the correction arrived.
3. **Per-column pairwise search generally.** With the global severity null at p = 0.36, the gains the
   search produced are artifacts of the search. Only a pre-registered rule counts here.
4. **Co-occurrence structure in the tail — NULL**, +0.017 ± 0.021, 0.8σ. This one is recorded mainly
   because the first attempt at it produced a "lift" of 85.5 from parsing `train.csv`'s blank finding
   columns as labels. Twelve findings cannot each occur in 0.5% of a knee-MRI corpus. The absurd number
   caught it; care did not.
5. **Per-study centering — monotone worse on three members.** Not re-proposable.

## The negation pass is UNTESTED, not refuted, and the distinction is the denominator

`core.data.labels.overcall_audit` run over all 16,129 YES cells flags **876 `missed_negation` cells,
5.43% of positives** — reproducing the go/no-go's figure exactly. Applied as a soft correction toward
the negative value (P(positive | flagged) ≈ 0.20 from the 12-of-15 audit, so no cell is hard-flipped)
it reads **−0.0024** on the gold ruler.

**That number cannot arbitrate anything: only 15 of the 876 flagged cells fall inside the 57 gold
studies.** The ruler is judging a corpus-wide correction on 1.7% of the cells it touches. Recording a
refutation here would be the harness-as-suspect failure the project keeps paying for — a faulty TEST,
not a wrong idea. The lane stays open and stays out of every training table until it has a ruler that
can see it.

One thing about it needs no gold at all. **PF OA takes 267 of the 876 flags (30%) while being roughly
8% of positives**, and PF OA was the second-worst column in the gold reading. `negation_scope.py`
already documents LOCATION as its unremovable residue; PF OA may be where that residue concentrates.
Asked as Q-AS.

## Where the bigger lane is, and what blocks it

`2026-09-01_q14_label_cleaning_go_no_go.md` decided **GO** on both remaining lanes and neither has been
executed. The larger one is the rubric-bearing re-read: `rubric_gap` finds **502 backed contradiction
cells** in the 1,768 English reports, scaling to roughly **1,250 cells** corpus-wide, across Medial
Meniscus, Lateral Meniscus, ACL and Baker's. Backed means the radiologist annotated those cells
negative — the rubric and the gold agree against our label. The 1,098 refused cells (MCL, Effusion, the
OA columns) are not targets and rewarding agreement with them would be perverse.

**It was recorded as blocked and it was not.** The lane was designed as an LLM re-read with the rubric
in the system prompt, and the reader's API key does not exist on this machine — so it was filed as
blocked on the key. But `rubric_gap` decides the subthreshold class **lexically, per cell, from the
published criteria, with no ground truth and no model in the decision** — its own docstring says so:
"THE RULES SUPPLY THE ANSWER, SO THIS NEEDS NO GROUND TRUTH." The block was a fact about how I had
imagined executing it, not about the problem. This is the same failure the project keeps paying for in
the other direction: a harness limit read as a limit on the idea.

It is now built: `core.data.labels.table_correction`, which turns that decision into a table under the
rules the annotations BACK, with the backing gate evaluated at runtime rather than written down. 502
cells enter — Medial/Lateral Meniscus 165, ACL 98, Baker's 239; MCL, Effusion and the three OA
compartments are refused. The module is proved by an equality oracle: `--rule fracture_contusion` over
`blend_hybrid.csv` reproduces the hand-written `blend_hybrid_fx.csv` to zero difference on all twelve
target columns.

**The gold reading, and its denominator first.** Only **5 of the 502 corrected cells** fall inside the
57 annotated studies, so gold is again judging 1% of what the correction touches — the same shape as
the negation pass, and it is why neither number below is a verdict.

| column | `blend_hybrid_fx` | `blend_hybrid_fxrub` |
|---|---|---|
| Medial Meniscus | 0.9573 | **0.9597** |
| Lateral Meniscus | 0.8776 | **0.8801** |
| Baker's | 0.9447 | **0.9466** |
| ACL | 0.9828 | 0.9828 |

Nothing regresses. Over all 696 gold cells — a wider ruler than the five — agreement goes 79.9 → 80.5%
at the 0.50 cut and precision 66.3 → 67.2% at unchanged recall. That is +0.0006 on a mean of twelve:
directional, an eighth of what `fracture_contusion` was worth, and far below the bar.

**No arm is queued for it, deliberately.** `stack_increments_fx` already prices the label lane on a gold
gain ten times larger. If that reads NULL, this is priced out with it and the card is not spent finding
out twice. Bead `knee-uwjf`.

**Where it could actually become large.** `rubric_gap` is English-only by design, so 502 is a FLOOR:
59.9% of the corpus is not English and carries the same wording. If the subthreshold vocabulary can be
written for es/tr/hr/el/de/bg/nl the corrected population is roughly **1250 cells for zero GPU time**.
Asked as Q-AU, with the per-language sourcing spelled out — that, not another English pass, is the
version of this lane with a size worth arming.

The `subthreshold` bucket — 46 of 108 false positives, the largest, and a coin flip at 0.49 — is NOT
free to bank and this file will not propose it. `severity_boundary` measured the size qualifier against
the gold classes for Effusion and found it neither separating nor monotone (small 50% positive,
moderate-or-larger 67%, unqualified 86%), and `rubric_gap` REFUSED the subthreshold reading for MCL and
the whole size family while backing it for meniscus, ACL and Baker's. It is already gated, per finding,
by measurements this repo made.

## Rarity, and where our measurement noise actually lives

Prompted by the question of whether rare pathologies carry the score. On `calib_dense_hold_s2026`'s
909-study holdout: **corr(prevalence, AUC) = +0.353**, rarer half 0.8482 against commoner half 0.8786,
**−0.0304**. Directionally true and modest.

The sharper half is the noise. Hanley-McNeil standard error per column: **Fracture 0.0316**, Synovitis
0.0217, MCL 0.0215, against Medial Meniscus 0.0104. The score is a flat mean of twelve, so

- **Fracture alone is 27% of the variance of every holdout number this project quotes**, and
- **Fracture + MCL + Synovitis together are 53%.**

Summing those variances reconstructs the known 0.0053 Hanley-McNeil floor from first principles, so this
is not a new floor — it is an **attribution** of the existing one. The consequence is a fact about our
instrument, not our model: every member ranking separated by less than the floor is being decided
disproportionately by the three columns with the fewest positives.

## What is on the card

`stack_increments_fx` — byte-identical to `stack_increments_base` except `--labels`, so the arm already
running is a free matched control. It is armed as an INSTRUMENT, not an improvement: P(clears the 1.5%
bar) ≈ 0.10. **A NULL prices the whole lane**, including the blocked rubric re-read, which would
otherwise be planned for on the strength of a 57-study gold number that has never been shown to transfer
to training. Bead `knee-hbml`.
