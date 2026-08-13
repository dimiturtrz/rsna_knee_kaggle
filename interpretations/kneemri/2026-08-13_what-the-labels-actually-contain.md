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
