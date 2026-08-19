# What the labels actually contain, and why the local ruler compresses

2026-08-13. Three measurements against the 57 fully-annotated studies and the raw report text, none of
which needed the GPU. They agree with each other, and together they re-price two P0s and explain a
number the project had observed without a mechanism.

## 1. The graded targets are miscalibrated, and four of five sit outside their own interval

The reader emits five values and we train on them as probabilities. Measured over 696 fully-annotated
cells — how often the RADIOLOGIST agrees the finding is present:

| assigned | measured | 95% CI | n | assigned inside? |
|---:|---:|---|---:|---|
| 0.08 | 0.041 | [0.013, 0.068] | 196 | **no** |
| 0.28 | 0.136 | [0.085, 0.186] | 177 | **no** |
| 0.68 | 0.378 | [0.282, 0.474] | 98 | **no** |
| 0.82 | 0.688 | [0.611, 0.764] | 141 | **no** |
| 0.94 | 0.958 | [0.912, 1.004] | 72 | yes |

Every one of the four is too high, worst in the middle: we train the model to predict **0.68 for cells
that are positive 38% of the time**, and 0.28 for unmentioned findings that are positive 14% of the
time. `targets.py` reads the value straight from the label table's column, so correcting it is a
load-time remap — no reader, no new labels, no compute.

The arm scored **0.8495 against the control's 0.8457**, +0.0038, inside the 0.0053 floor. Its
per-label prediction — that the gain concentrates where miscalibration is largest — **failed in
direction**: PF OA (+0.0198, predicted to move least) was the largest of the twelve, MCL moved +0.0045
and Effusion went the wrong way. But not one delta clears its label's own interval, so the pattern is
**unread rather than refuted**, and that is what forced the paired floor into existence.

## 2. The gradation is real signal, confirmed two independent ways

The three YES levels could have been LLM certainty rather than severity. They are not:

- **Against the annotations**: agreement runs 0.378 / 0.688 / 0.958, monotone across a span of 0.58 —
  enormous beside anything this project has measured. And `__conf` is *constant per verdict* (NO 0.85,
  UNK 0.05, YES 0.95), so confidence is demonstrably not what separates them.
- **Against the report text**: splitting report-YES cells by strong severity language (severe, grade 3,
  full-thickness) versus weak (mild, grade 1, partial-thickness), the assigned level runs **0.833 vs
  0.759** — and **twelve of twelve labels** point the same way.

Two methods, two different ground truths, same conclusion. The per-label distributions agree on their
face too: MCL 495/219/53 and Effusion 1599/910/157 skew low, while **PF OA 543/658/790 is the only
label skewed high** — and OA is the one finding routinely graded mild/moderate/severe.

## 3. Why the local ruler compresses, which the project had observed but not explained

`knee-7d1` recorded that local gains roughly double on the leaderboard (0.0107 → 0.020) and correctly
refused to call two points a law. The mechanism:

```
report says YES on 311 cells; the radiologist agrees on 203 = 65.3%
report says NO  on 196 cells; the radiologist calls  4.1% positive
report says UNK on 177 cells; the radiologist calls 13.6% positive
```

**A third of the positives we score against are not positives.** A model that learns the true finding
is penalised on every cell where the label is wrong, roughly in proportion to how much better than the
labels it is — so two good models are separated LESS on the noisy ruler than on the clean one. Local
attenuation and leaderboard amplification are the same fact, and the observed shape is what label
noise predicts rather than a coincidence of two rows.

It does not licence rescaling the ledger by a factor: the attenuation depends on where a model sits
relative to the noise, so it is not one constant. What it supports is the operating rule already in
that bead — the local number picks WHICH recipe to submit, only the leaderboard says whether it is
better — and it explains why free sub-bar deltas are worth keeping.

## What it prices

**A better reader is worth more than a better model**, if a third of our positive labels are wrong —
and the vocabulary is there to build one: 74.1% of English-looking reports carry an explicit severity
marker (grade N 17.8%, mild 19.8%, moderate 9.2%, severe 4.6%, full-thickness 5.7%).

But the two branches of `knee-mxj` turn out to be **coupled, not independent**. English-looking reports
are only 39.6% of the corpus, so a monolingual reader upgrade reaches about **29%** of it; the rest
needs the multilingual encoder that branch (a) is about. And the reader already captures perhaps a
quarter to a third of what an ordinal one would extract — explicit severity moves the level 0.074 on a
range of 0.26 — so branch (b) buys the remainder, not the whole thing.

### And a regex-grade reader is already priced in

Extracting per-FINDING severity properly — sentences naming that finding's anatomy, severity read from
those only — reproduces the ladder marginally (weak 52.8%, none 65.2%, strong 89.5% agreement) and adds
**nothing within a level**:

| level | weak | none | strong |
|---:|---:|---:|---:|
| 0.68 | 40.7% (27) | 37.1% (70) | — |
| 0.82 | 87.5% (8) | 68.2% (132) | — |
| 0.94 | — | 94.4% (54) | 100% (17) |

Inconsistent in sign, small in every cell. The marginal monotonicity is the LEVEL showing through, not
new information — so the reader already captures what a regex can see, and the "quarter to a third"
estimate above is an upper bound on what this kind of extraction could add.

**That raises the bar for branch (b) rather than lowering it.** A better reader has to extract what a
regex cannot: negation and hedging, finding-specific grading the words do not carry ("grade 2 signal
not extending to the articular surface"), and the 60% of reports that are not English. "Use a stronger
model to read severity" is no longer a sufficient statement of that plan. Coverage limit: only 44.1% of
report-YES cells had their finding's own sentence located, so this holds on the half we can read.

### The reader's actual defect is negation, and that is measurable

If a regex adds nothing on severity, what would a stronger reader fix? Splitting located report-YES
cells by whether the finding's own sentence carries a negation — *no, not, without, intact, normal,
unremarkable, ruled out*:

| finding sentence | n | radiologist agrees |
|---|---:|---:|
| no negation | 111 | **73.9%** |
| contains negation | 26 | **42.3%** |

Difference 0.316, SE 0.105, **z = 3.00**, 95% CI [0.109, 0.523] — excludes zero. When the reader calls a
finding present and the sentence naming it carries a negation, it is **wrong more often than right**,
and 19% of located report-YES cells are in that state.

This is the opposite case to severity, and that contrast is the point. Severity is already captured, so
a stronger model adds little. Negation is a *known* failure mode of clinical text extraction, is
measurably present here, and is exactly what a language model handles and a pattern-matcher does not.
It converts "use a better reader" into a specific defect with a measured cost — and into a cheap
action, since only the negated cells need re-reading.

The detector is crude: "ACL intact, medial meniscus torn" is one sentence carrying both, so some
flagged cells are false alarms — which means the true effect on genuinely-negated cells is **larger**
than 42.3%, not smaller.

**But size it before building on it.** Corpus-wide, 1,057 of 16,129 report-YES cells are negated —
17.8% of the located subset but only **6.6% of all positives**, because only 36.9% of cells can have
their finding's sentence located at all. Down-weighting them is free and sound and moves the effective
label noise by a couple of points; it is not a lever. The value is in the **63% we cannot currently
inspect**, which is the population a stronger multilingual reader reaches. The negation gap's job is to
demonstrate that the reader has extractable errors — not to be the fix.

One limit worth stating: the non-English figure (16.6% carrying severity) is a **lower bound, not a
measurement**. English patterns against non-English text find little by construction, and those reports
very likely grade their findings in their own vocabularies. Reading a tool's blind spot as a property
of the data is a mistake this project has already made once.

## The lower bound was measured, and it was the tool

2026-08-13, same day, no GPU: `python -m kneemri.labels.report_language`. The caution above was right and the
correction is larger than the caution implied.

The corpus is not English with a tail. **English is 40.1% of it**, and the rest is a set of complete
radiology reports in six other languages:

| | n | share | graded, own words | graded, English patterns |
|---|---:|---:|---:|---:|
| en | 1,769 | 40.1% | 64.7% | 64.7% |
| es | 727 | 16.5% | **55.2%** | 1.5% |
| tr | 546 | 12.4% | **80.8%** | 51.5% |
| *unclassified* | 412 | 9.3% | — | 0.0% |
| el | 321 | 7.3% | **75.1%** | 24.6% |
| de | 259 | 5.9% | **83.8%** | 2.7% |
| bg | 220 | 5.0% | **61.4%** | 30.5% |
| nl | 153 | 3.5% | **80.4%** | 2.6% |

Across the 2,226 classified non-English reports: **70.0% grade their findings in their own vocabulary,
against 20.2% visible to the English patterns** — and 20.2% is within reach of the 16.6% quoted above.
The old figure was measuring the ruler.

**Denominators, since this section exists to correct one.** 61.3% of **all 4,407** reports carry an
explicit severity marker when each is read in its own language, against 36.1% under English patterns
alone. The 61.3% is itself a lower bound: the 9.3% unclassified are counted as ungraded, and Dutch and
Greek were only added to the detector after their reports turned up as "unknown".

### What it re-prices

`knee-mxj`'s two branches were called coupled, on the reasoning that a monolingual reader upgrade
reaches "about 29%" of the corpus. That 29% was English share times English-visible severity. The
severity information is actually present in **61.3%** of the corpus — more than double the assumed
reach — and it is present in ordinary graded words (`parcial`, `geringgradig`, `hafif`, `grade III`)
rather than anything exotic.

This does **not** say a better reader wins. The English-only finding still stands that per-finding regex
extraction adds nothing a level does not already carry, so *presence of a marker* is not *severity
attached to a finding*. What it removes is the reason to believe the information is unavailable outside
English: it is there, in most of the corpus, and only the tool could not see it.

It also makes the negation defect worth re-measuring. That was sized on located English sentences —
6.6% of all positives — and located-ness was itself an English-regex property.

## And the reader is not measurably worse in any of those languages

`python -m kneemri.labels.reader_agreement`, same day, no GPU. Branch (a) of `knee-mxj` assumes the reader
struggles outside English. Measured against the 58 annotated studies — the only ground truth that is not
the reader's own output — it does not:

| | cells | radiologist agrees |
|---|---:|---:|
| nl | 13 | 92.3% |
| el | 21 | 71.4% |
| en | 135 | **66.7%** |
| bg | 15 | 66.7% |
| tr | 36 | 63.9% |
| de | 13 | 61.5% |
| es | 47 | 57.4% |

**English 66.7% against non-English 64.2%: +0.025, 95% CI [−0.076, +0.126].** The interval is
bootstrapped over STUDIES rather than cells, because twelve cells sharing one report and one radiologist
are not twelve independent draws — a cell-wise interval would be about half this width and would be
wrong.

**State what this can and cannot establish.** It rules out an English advantage larger than about 12.6
points. It cannot rule out a small one, and with 26 English against 30 non-English studies nothing
finer is available. What it does refute is the *large* multilingual deficit that branch (a) was
premised on: whatever is wrong with our labels is close to language-independent.

### The two findings converge, and they sharpen the P0

- the severity vocabulary is present in **61.3%** of the corpus, not the 29% assumed
- the reader reads those languages about as well as it reads English

So "reach the other 60% with a multilingual reader" is not the lever — that ground is already covered.
What remains is what the English-only analysis already isolated: **ordinal extraction within a report the
reader can already read**, where a regex adds nothing and the measured defect is negation. The plan that
survives is narrower and better aimed than the one that went in.

`reader_agreement` is also the acceptance test for any replacement reader: it has to agree more, and a
reader built for a language has to close that language's gap.

## MCL is the one label whose weakness is real, and Fracture is the one that taught us to doubt it

2026-08-13, no GPU: `python -m kneemri.evaluation.label_stability`, reading the per-label AUCs every run has been
writing since the first epoch and which nothing had ever read ACROSS runs.

The project's plan was once "track per-label AUC and treat the worst label as the work item". It was
refuted, correctly, by our own folds: **Fracture** reads 0.806, 0.903 and 0.864 on three holdouts under
an identical recipe, because an AUC from 62 positives has a standard error near 0.04. The refutation was
then applied to per-label work in general. That over-generalised.

**MCL is the worst label in all fifteen runs that have ever scored** — across DINOv2 and DINOv3, LoRA
and full fine-tuning, whole-stack and windowed, and on three different holdouts (fold 0 mean 0.761, fold
1 0.802, fold 2 0.760). No sampling error survives a change of holdout that reliably.

| | fold 0 (7 recipes) | fold 1 | fold 2 |
|---|---|---|---|
| MCL | 0.761, spread 0.037 | 0.802 | 0.760 |
| Fracture | 0.806, spread 0.025 | **0.903** | **0.864** |

**Read the two correlations correctly.** Recipes sharing one fold also share that fold's sampling error,
so their per-label profiles agree almost by construction: median rank correlation **+0.92** within fold 0
and only **+0.51** across holdouts. The within-fold number is a floor, not evidence — which is exactly
the mistake the original per-label plan made, and the reason `label_stability` reports them apart.

### What it is worth, stated honestly

MCL sits about 0.09 under the twelve-label mean. Closing that gap entirely would be **+0.0075** on the
metric — below the 1.5% bar, so directional, not a plan on its own. What changes is that per-label work
is no longer refuted wholesale: there is one label whose deficit is reproducible across encoders and
holdouts, and it is the same label every time.

### A mechanism worth testing before any arm is queued

Two facts already in this file point the same way. MCL's severity levels skew **low** — 495/219/53
across the three YES levels, the second-most-skewed of the twelve — so MCL is dominated by MILD injury,
which is where both the image evidence is faintest and the reader's calibration was worst (0.28 assigned
against 0.136 measured, 0.68 against 0.378).

And the obvious rival explanation is already dead: MCL is read coronally, and the nine-slot arm that
added 1,139 coronal series bought **+0.0002**. So it is not that the model cannot see a coronal — it is
that mild collateral injury is faint in the image and unreliable in the label.

### And MCL is where the ruler is least trustworthy, which makes it the worst thing to optimise locally

Same day, `python -m kneemri.labels.reader_agreement`. Agreement on the reader's POSITIVE calls, per finding:

| | calls | radiologist agrees | mean AUC (3 folds) |
|---|---:|---:|---:|
| Contusion | 36 | **44.4%** | 0.856 |
| **MCL** | 18 | **50.0%** | **0.782** |
| Lateral OA | 16 | 56.2% | 0.822 |
| … | | | |
| Medial Meniscus | 32 | 78.1% | 0.894 |
| Synovitis | 16 | 81.2% | 0.858 |

MCL is the **second-noisiest label** as well as the worst-scoring one. Across the twelve, agreement and
AUC correlate **+0.51** (Spearman) — but twelve points put roughly ±0.6 on that, so it is suggestive and
nothing more, and **Contusion refutes any simple version of it**: worst agreement of all, above-median
AUC. Label noise is a contributing factor, not the explanation.

**What this changes about the MCL finding.** MCL's AUC is agreement with a label that is right about
half the time. So an arm that genuinely improves mild collateral detection may not move the number, and
something that moves the number need not have improved the finding. MCL is simultaneously the one
reproducible deficit and the one place our ruler is least able to tell us whether we fixed it.

**And it cannot be settled locally.** The clean test — MCL scored against the radiologist rather than
the reader — has **9 MCL positives among 58 annotated studies**. An AUC from nine positives has a
standard error near 0.15, which is twenty times the effect being chased. Only the leaderboard can
arbitrate this one, which is `knee-7d1`'s standing conclusion arriving from a new direction.

So: keep MCL as the named weakness, do NOT build a loss or a sampler around it — that is the
loss-engineering the working rules forbid, and here the measurement cannot even confirm it worked.

### The reader grades severity in every language, and only partly in all of them

2026-08-14, CPU only. Splitting each language's reports by whether they lean on STRONG severity words
(severe, grade 3, full-thickness and their equivalents) or WEAK ones (mild, minimal, grade 1, partial),
and reading the level the reader assigned to report-YES cells:

| lang | YES cells | level given strong | given weak | gap |
|---|---:|---:|---:|---:|
| en | 2,609 | 0.841 | 0.762 | **+0.078** |
| es | 989 | 0.843 | 0.779 | +0.064 |
| el | 568 | 0.844 | 0.788 | +0.055 |
| de | 538 | 0.806 | 0.759 | +0.047 |
| bg | 341 | 0.802 | 0.771 | +0.031 |

Every gap is positive and they are the same order of magnitude, so the reader is reading severity in
each language rather than only in English. That is the third independent measurement pointing the same
way, after the language census (severity vocabulary present in 61.3% of the corpus) and the agreement
gap (English 66.7% against non-English 64.2%, interval spanning zero).

**What it settles.** There is no language-specific severity blind spot to fix, so `knee-mxj` branch (a)
— "the rest needs a multilingual encoder" — has now failed three different ways of looking for it.

**What survives, and it is the same everywhere.** The gap is 0.03–0.08 on a level range of 0.26, so the
reader captures perhaps a quarter to a third of what the words carry — in EVERY language equally. That
is branch (b), ordinal extraction quality, and the uniformity is the useful part: an improvement there
applies to the whole corpus rather than to a slice of it.
