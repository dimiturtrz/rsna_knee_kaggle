# What actually moved the number — and what conspicuously did not

*Fold 0, every arm matched on seed, data and fold; scored on the pilkwang label table so that the
training-label arms earn no self-agreement credit. Baseline is the recipe that produced our 0.868
leaderboard submission.*

## The table

| change | holdout | delta | verdict |
|---|---|---|---|
| baseline (constant LR, no augmentation) | 0.8092 | — | what we submitted at 0.868 |
| + OneCycle schedule | 0.8153 | **+0.0061** | shipped |
| + augmentation actually applied | 0.8202 | **+0.0110** | shipped — it was never called |
| + intensity augmentation (gamma/blur/contrast/noise) | 0.8221 | +0.0019 | kept, not defaulted |
| better label table (v4_blend, +0.028 vs gold) | 0.8194 | −0.0008 | **neutral** |
| 32 inference slices instead of 12 | 0.8137 | **−0.018** | refuted |
| overlap TTA (inference only) | +0.0035 | +0.0035 | shipped |
| + weight floor at 0.25 | 0.8221 | +0.0019 | kept |
| + loss normalised by count (their `.mean()`) | 0.8199 | −0.0022 | refuted, on top of the floor |

With arm E the reference recipe's differences from ours are now all measured. Normalising the loss by
label count instead of by weight sum is the last of them and it does not transfer: a study with two
confident findings and ten unmentioned ones contributes the same total gradient as one with twelve
confident findings, which is exactly the reweighting the confidence floor already does more carefully.

Five folds on the shipped recipe: **0.8269** against **0.8162**, +0.0107, every fold improved.

## A correction, and then a correction to the correction

I first read the seed runs as showing a **±0.0121 run-to-run spread** (0.8202 against 0.8323) and used
it to downgrade every small row above to "not detectable at n=1".

**That reading was wrong, because of a bug in the trainer.** The fold split was derived from the
TRAINING seed: `ReportFolds(folds, arguments.seed)`. So seed 7 did not train the same model on the same
split — it trained on a *different split*, whose fold-0 holdout shares only **192 of 909** studies with
seed 2026's. The two numbers were never measurements of the same thing, and the "seed variance" was
mostly split difficulty.

Worse, the member-count experiment built on top of it cross-scored those models on seed 2026's
holdout, of which **717 studies were seed 7's training data**. That is why it reported an implausible
0.8807 for a member whose own run logged 0.8323 — leakage, not ensembling.

**What is actually true, after fixing the split to be a property of the corpus (`--split-seed`):**

- The ablation arms all ran at seed 2026 and therefore shared one split. Those comparisons stand as
  matched pairs, and the small effects (+0.0019, −0.0008) are real *as single-run measurements*.
- The run-to-run noise floor is **now measured, and it is small**. Two seeds under one fixed split and
  one identical recipe: **0.8233 (seed 7) against 0.8221 (seed 13), a spread of 0.0012**. That is an
  order of magnitude below the ±0.0121 I had wrongly inferred from the split bug, and it means the
  ledger's +0.0061 and +0.0110 rows are far outside noise while the +0.0019 rows sit at roughly 1.6x
  it — suggestive, not established, and not worth defending.
- The five-fold result stands: +0.0107 with five of five folds improving, all on one split family.
- The marginal-member curve is **unmeasured**; the numbers it produced are void.

The lesson is not the one I drew an hour ago. It is narrower and sharper: **a split is a property of
the corpus, and anything that makes it move with a run's configuration silently destroys
comparability** — and then quietly rewards you with better-looking numbers, which is the dangerous
direction for a bug to fail in.

## The two findings that were bugs, not ideas

**Augmentation was written, unit-tested, and never called.** `fit_epoch` did not invoke it. Both
leaderboard submissions and all five original folds trained unregularised. The evidence was in a curve
nobody had looked at: a 40-epoch run peaked near epoch 16 and then declined monotonically to 0.8043,
which is what unregularised training looks like, and explains how the public baseline trains 60 epochs
where we degraded after 16.

Both quality gates passed it. `vulture` saw the method as *used* — because its own test referenced it.
The mirror gate saw it as *tested* — because it had a `test_augment`. **A method can be fully covered
and completely disconnected: every gate checked the callee and none checked the caller.** The fix was a
test on `fit_epoch` asserting it routes pixels through augmentation.

**The data path read four times what it used.** `images[batch][:, :, s:s+3]` fancy-indexed first,
materialising all twelve cached slices (97.5 MB per batch) before discarding three quarters. One
indexing operation instead: 35 ms → 3.3 ms, and the epoch fell from ~40 s to ~30 s.

## Four public ideas that needed adapting, not copying

Each came from a real, measured result elsewhere. None transferred unchanged, and the *reason* was
different every time.

1. **Flips** (cardiac-seg augments with both axes). A cardiac label is not side-defined; five of our
   twelve targets are. A horizontal flip would undo the laterality normalisation they depend on.
2. **Translation** (cardiac-seg measured it NEUTRAL and disabled it). Their encoder is a
   fully-convolutional U-Net and therefore translation-equivariant; ours is a ViT with position
   embeddings and is not. The measurement cannot be inherited — only the method for making it.
3. **Label fusion** (the 0.903 system's headline mechanism). Fusing our four public readers scores
   0.8966, *below* the best single reader at 0.8982. Their fusion presumably worked because their
   private Qwen reader was independent of pilkwang's; ours are all derived from the same reports, and
   the best table is already a blend.
4. **32 inference slices** (the 0.903 recipe trains at 12, infers at 32). Worth −0.018 here. Our
   encoder eats three consecutive slices *stacked as channels*, so raising the slice count over a fixed
   band shrinks each group's physical thickness and shows the model an input distribution it never saw.
   Their pooling is max over independently-encoded slices, which has no such coupling. The idea is
   conditional, not dead: it needs the group's physical extent held fixed, or per-slice encoding.

## Two hypotheses of mine that the data refuted

**Laterality was not the cause of the medial/lateral asymmetry.** Both medial/lateral pairs favour
medial (−0.043 meniscus, −0.066 OA), which is exactly what a broken mirror would produce. But the
asymmetry is *identical* across tag-resolved and geometry-resolved studies (the two rules agree 98.6%
of the time on 2,135 studies), and it is **smaller on the knees we actually mirror** (+0.015 against
+0.071). A broken mirror does the opposite. The cause is most likely prevalence: medial pathology is
commoner and more often named in a report.

**Label quality did not become model quality.** The tables differ by +0.028 in agreement with the 58
gold studies and the models differ by −0.0008. Both readers work from the *same reports*, so their
disagreement concentrates on studies whose reports are genuinely ambiguous — where neither label is
reliable and the image cannot settle it either. Unbiased confirmation needs a cross-fitted evaluation
on the gold studies, which costs five folds per arm.

## Two methodology errors worth more than the gains

**Different rulers.** The first ablation queue would have scored the v4_blend arm against v4_blend
targets and the baseline against pilkwang targets. A holdout AUC rises when labels agree with
*themselves* more, and that would have read as a better model. Forty-five minutes of running GPU were
killed to add `--eval-labels` and restart. The ruler chosen is deliberately the one that *disfavours*
the interesting arm.

**A refactor moved a boundary the deployment path depended on.** Extracting `Submission` was right on
every local measure — LCOM4, testability, 95% duplication between two kernels — and it broke the next
kernel run with `ModuleNotFoundError`, because the kit carrying our source was not rebuilt.
`verify_kit.py` catches this in a second and was skipped, because the change felt like it was "only in
the kernels". The kernels *import* the kit; a change to either is a change to both.


## The ruler went missing, and that is worse than any row in the table

Every number above was scored against a report-derived label table — the one with a confidence per
(study, target), which is what we actually train on, since `train.csv` annotates only 58 of 4,407
studies. **That table was not on disk.** A drive-wide search for any csv or parquet written since
Aug 5 under `D:/data` returned nothing; there was no copy in the repo and none in the Kaggle kit.

It was recoverable — it is a public Kaggle dataset, `pilkwang/rsna-knee-llm-labels` — and it now has a
documented fetch step in `comp/README.md`. But the consequence for this document stands: **a new arm
cannot be compared against a row above** unless the same table was in use, which cannot now be
verified. New comparisons are being run as matched pairs instead, both arms fresh, both on the
recovered `report_labels_v2.csv`.

The lesson generalises past this repo. The gates here check code, and they check it well; nothing
checked that an *input* was reproducible. An external input with no fetch step is not a dependency, it
is a memory — and it fails silently in the direction of "the numbers I already have look fine".

## Half the corpus is not in English, and it scores 0.016 worse

**2,303 of 4,407 reports (52%) contain no common English word.** Spanish, Dutch, French, Greek and
Cyrillic all appear. This was not on anyone's list here; it turned up while reading a public notebook
that ships a rule-based *multilingual* parser, complete with Turkish dotted-i folding and the
Croatian d-with-stroke — a detail that only makes sense if its author had hit the same thing.

The LLM label table does cover them, so this is not missing supervision. It is *weaker* supervision:

| | studies | mean confidence | UNK verdicts |
|---|---|---|---|
| English report | 2,104 | 0.717 | 20.2% |
| non-English report | 2,303 | 0.611 | 33.8% |

And it shows in the score. One member, fold-0 holdout, overlap TTA, split by report language:

| | studies | holdout mean AUC |
|---|---|---|
| English | 420 | **0.8446** |
| non-English | 489 | **0.8286** |
| all | 909 | 0.8352 |

**A 0.016 gap across 54% of the holdout, against a measured noise floor of 0.0012.** That is larger
than every lever in the table above except the augmentation bug.

Two things must stay attached to that number. First, it is scored against the *same* weaker labels, so
it conflates "the model is worse on these studies" with "the ruler is noisier on these studies". Both
readings point at the same remedy, but they imply very different ceilings, and only gold labels can
separate them — of which there are 58. Second, **language cannot be an inference-time input**:
`test.csv` has no `Report` column. Nothing here can become a language-conditioned model. The only
lever available is better *supervision* on the non-English half, which is precisely what a rule-based
parser in the report's own language would provide.


## A matched trio on the recovered ruler

All three arms below share a split (2026, fold 0), a recipe (one-cycle, geometric augmentation, weight
floor, 8 epochs) and the recovered `report_labels_v2` ruler. They are compared to each other and to
nothing above, for the reason given in the previous section.

| arm | holdout | vs baseline |
|---|---|---|
| channel-stacked groups (baseline) | 0.8081 | — |
| per-slice max pooling | 0.8099 | +0.0018 |
| anatomical prior on slot attention | 0.8005 | **−0.0076** |

Note the baseline lands at 0.8081 where the ledger recorded 0.8221 for the same nominal recipe. That
difference *is* the missing ruler, and it is why nothing here is compared upward.

### The noise floor is ~0.006, and it demotes most of this document

Three seeds, one recipe, one split, the recovered ruler, scored identically with overlap TTA:

| seed | holdout |
|---|---|
| 2026 | 0.8171 |
| 7 | 0.8209 |
| 13 | 0.8233 |

**A spread of 0.0062** — five times the 0.0012 I measured on the lost table at 30 epochs, and enough to
change several verdicts in this file. What survives it and what does not:

- **+0.0110 (augmentation applied)** and **+0.0107 (five folds)** — still outside it. These stand.
- **+0.0061 (OneCycle)** — comparable to the floor. Weaker than stated.
- **+0.0019 (intensity aug), +0.0019 (floor), −0.0008 (labels), +0.0018 (per-slice max)** — all noise.
  Not "small effects", *no measured effect*.
- **−0.0076 (anatomical prior)** — about one seed spread. **That regression is not established either,**
  and I called it "six times the noise floor" an hour ago on a floor that was wrong.

The general lesson costs more than any row: **a noise floor is a property of the ruler, the epoch
budget and the split, not of the method.** Quoting one measured elsewhere is how a single run gets
promoted to a finding — which is exactly what I did with it, in both directions, within one session.

The practical consequence is a change of strategy, not just of wording. At a floor of 0.006, every knob
this project has left is unmeasurable at n=1. Either a change is big enough to clear 0.006 on one fold,
or it needs multiple seeds per arm — and at ~4 minutes a run, that is affordable but it has to be
budgeted deliberately rather than discovered afterwards.

### Members two and three buy almost nothing

Every subset of the three seed members above, rank-averaged:

| members | mean | range | gain |
|---|---|---|---|
| 1 | 0.8204 | [0.8171, 0.8233] | — |
| 2 | 0.8223 | [0.8211, 0.8237] | +0.0018 |
| 3 | 0.8229 | — | +0.0007 |

**The best single member (0.8233) beats the three-member ensemble (0.8229).** Ensembling here buys
about 0.002 for three times the inference cost — less than the seed spread it is averaging over.

One thing this does *not* measure: these three differ only by initialisation, trained on the same fold
of the same data. Our submitted ensemble is five **fold** members, which also see different training
data and should decorrelate more. So this is a lower bound on ensembling value, not a verdict on the
submission — but it is the relevant number for the Efficiency Prize, where a one- or two-member entry
at nearly full accuracy is worth ranks 29–50 while the same accuracy run expensively is worth 638.

### Per-slice max pooling: refuted at the size that mattered

`knee-wif` predicted 0.02–0.03 and wrote its own kill criterion: *"if it lands in the noise, it is a
refutation, not a tuning opportunity."* It landed at **+0.0018**, about 1.5x the 0.0012 noise floor, for
three times the encoder cost. Refuted as a headline change.

**And the mechanism I proposed for it was wrong.** The argument was that channel-stacking ties a
group's physical thickness to the slice count, which is why reading 32 slices cost −0.018; per-slice
pooling breaks that coupling and should therefore unlock the wider read. It does not:

| | 12 slices, overlap | 32 slices, disjoint | 32 slices, overlap |
|---|---|---|---|
| channel-stacked | 0.8171 | 0.8085 | 0.8080 |
| per-slice max | 0.8177 | 0.8078 | 0.8070 |

The 32-slice penalty is **the same size under both** (−0.009 and −0.010). Removing the coupling did not
remove the penalty, so the coupling was not the cause. Whatever makes a 32-slice read worse here is
something else — a candidate is that a fixed band sampled 32 ways gives each window less physical
extent and averages many near-duplicate views — and that remains untested. The published recipe's
32-slice inference stays refuted for our pipeline, now for an unknown reason rather than a wrong one.

### The prior's regression was one row of the table, and the row was mine

Scoring per target rather than in aggregate: **the loss is Baker's cyst at −0.0589**. The other eleven
findings average −0.0009 and six of them improve.

| worst | delta | | best | delta |
|---|---|---|---|---|
| Baker's | −0.0589 | | MCL | +0.0077 |
| Effusion | −0.0108 | | Fracture | +0.0047 |
| Synovitis | −0.0059 | | Medial OA | +0.0040 |

It also refutes the mechanism I expected. The hypothesis was that forcing anatomy hurts where the
anatomically-correct slot is often absent — the correlation between per-target change and that slot's
presence is **−0.091**, nothing at all, and the two findings with the rarest correct slot (`COR_T1`,
present in 62% of studies) are among those that improved.

The cause is simply that the row was wrong. A Baker's cyst is identified by its *neck*, which runs
between the semimembranosus tendon and the medial head of gastrocnemius — an axial relationship — and I
had axial as secondary behind sagittal. The baseline, which routes almost everything axially, scores
0.8511 on this finding; the prior pulled it toward sagittal and lost 0.06. Axial is now co-primary and
the arm is rerunning.

So the prior is not refuted. What was refuted is one line of anatomy I wrote, which the per-target view
found in a minute and the aggregate number would have hidden as "the idea does not work".

### The anatomical prior: it worked, and the score fell

Giving each diagnosis's slot attention the anatomy it was being asked to learn cost **−0.0076**, six
times the noise floor. The prior is not failing to take — it takes exactly as designed. Correlation
between the learned attention and the anatomy table, averaged over findings:

| | correlation with anatomy |
|---|---|
| baseline | +0.127 |
| with the prior | +0.809 |

What the baseline actually learned is the interesting part. **It routes nearly every finding to one
slot — `AX_FLUID_FS`** — including MCL, both menisci and both tibiofemoral OA targets, none of which a
radiologist reads axially. With the prior, routing becomes radiologically sensible (MCL to coronal
fluid, OA to coronal T1) and the model gets worse.

That is evidence against the premise in `SlotHead`'s own docstring. The head is not localising evidence
to the sequence that shows it; it is selecting a slot that makes a good *study-level* embedding, and
one consistently-present fat-saturated axial series apparently serves that better than the anatomically
correct one. Which is coherent with the supervision: the label is attached to the study and never says
where the finding was seen, so nothing in the loss rewards localisation — and slot routing is itself
recovered from headers rather than given, so the anatomically-correct slot is also the noisier target.

Kept opt-in and documented rather than deleted. What is refuted is the prior at strength 2.0 on one
fold, not the idea that anatomy is real; what is *learned* is that this head does not use its slots the
way we said it did, which is worth more than the arm.

## The language gap is a site gap wearing a language coat

The 0.016 English/non-English gap survives the obvious deflation. Restricting the score to labels the
reader was confident about (conf >= 0.7, 71% of pairs) lifts both halves and leaves the gap intact:

| | all labels | conf >= 0.7 |
|---|---|---|
| English | 0.8242 | 0.8431 |
| non-English | 0.8123 | 0.8314 |
| gap | 0.0119 | 0.0117 |

So it is not simply that the ruler is noisier where it is unsure. But **scanner model predicts report
language almost deterministically**: `MAGNETOM Avanto fit` is 0% English over 262 studies, `SonataVision`
0.9%, `Optima MR450w` and `SIGNA EXCITE` 100%. Reports in Spanish come from Spanish hospitals, with
their own scanners and protocols, so "non-English" and "from a particular site" are very nearly the same
column.

Scoring the same predictions by scanner instead:

| group | n | holdout | English share |
|---|---|---|---|
| Siemens Healthineers | 223 | 0.8020 | 20% |
| SIEMENS | 168 | 0.8157 | 38% |
| GE Medical Systems | 155 | 0.8181 | 68% |
| Philips Medical Systems | 146 | **0.8256** | 50% |
| Philips | 111 | 0.8106 | 46% |
| 1.5 T | 525 | 0.8159 | 39% |
| 3.0 T | 322 | 0.8156 | 48% |

The manufacturer spread (0.024) is larger than the language gap and does not order by English share —
the best group is 50% English. Field strength does nothing at all, which rules out a simple
resolution or SNR story.

**These subgroups are small**, 111–223 studies against 420/489 for the language split, so the
manufacturer ranking is not itself established; the honest claim is that variation of at least this
size exists between sites and that language is confounded with it. That is enough to change what to
build: a nine-language label parser is aimed at the language reading, and the language reading is the
one the evidence does not isolate. It should not be started on this evidence.
