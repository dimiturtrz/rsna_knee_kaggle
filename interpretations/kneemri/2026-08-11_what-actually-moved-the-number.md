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
0.8511 on this finding; the prior pulled it toward sagittal and lost 0.06. Axial is now co-primary, and the rerun confirms it: **0.8073 against the baseline's 0.8081**, where the
first version scored 0.8005. Correcting one row recovered +0.0068 of a −0.0076 regression, leaving the
prior indistinguishable from no prior at a noise floor of 0.006.

So the final verdict on the anatomical prior is *neutral*, not harmful — and it took a per-target view
to get there. The aggregate number said "the idea does not work" twice: once when it was really one
wrong row, and once more when the row was fixed and the remainder was noise. It stays opt-in and off by
default, on its merits: it costs a parameter and buys nothing measurable.

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


## The encoder is the first lever in this session to clear the noise floor

DINOv2-**base** against DINOv2-**small**, matched on split, ruler, recipe, epochs and batch:

| encoder | parameters | holdout | epoch |
|---|---|---|---|
| small | 22M | 0.8081 | 55 s at batch 12 |
| base | 86M | **0.8166** | 56 s at batch 12 |

**+0.0085**, against a measured floor of 0.0062. It is the only change this session that clears it, and
it is still one seed — a second is queued, because the whole point of measuring the floor was to stop
reading single draws as findings.

Two things worth keeping from how it was run.

**It nearly ran confounded.** The first attempt used `--batch 6` out of caution about a larger model,
while every arm it would have been compared against used batch 12 — encoder size and batch size varying
together, which is the same class of error as the different-rulers one further up this file. It was
caught seven minutes in and rerun. Batch 12 also turned out to use 13.7 GB of 32 and run **four times
faster** per epoch, so the caution cost validity and wall-clock at once.

**Base looks undertrained at eight epochs.** Its curve was still climbing at the end — 0.8097, 0.8123,
0.8166, 0.8162 over the last four epochs, where small had flattened. A 20-epoch base arm is queued on
that basis: the mechanism (four times the parameters need more steps to fit) is stated in advance rather
than fitted to whatever the run returns.


## The whole ablation table was measured on an undertrained model

Raising the epoch budget from 8 to 24, small encoder, everything else matched:

| epochs | holdout |
|---|---|
| 8 | 0.8081 |
| 24 | **0.8205** (still climbing at the end) |

**+0.0124** — twice the noise floor, and larger than the encoder change. A 24-epoch *small* model beats
an 8-epoch *base* one (0.8166), which is not what "bigger encoder" intuition predicts and is exactly
what "both models are undertrained" predicts.

This is the most useful thing measured today, and not because of the number.

**Every arm in the table at the top of this file was run at 8 epochs.** OneCycle, intensity
augmentation, the weight floor, the label table, per-slice max, the anatomical prior — each was
compared against a baseline that had not finished learning. A knob that matters only once a model is
near convergence would look like noise under that protocol, and eleven of them did. So "no measured
effect at 8 epochs" is a much weaker statement than it sounded when written, and the honest reading of
that table is now: *these were measured in the wrong regime*, not *these do not work*.

It also explains the shape of the results without needing a story about our head or our data being
special. Nothing knob-sized cleared the floor; both things that did were capacity or training time.
That is the signature of a model that is under-fitted, not of one that is out of ideas.

The two consequences for what happens next:

- The five members currently on the leaderboard were trained for 8 epochs. Retraining the same recipe
  for ~24 is worth about +0.012 before anything else changes.
- Any knob worth re-testing should be re-tested at the longer budget, and re-tested *cheaply* — but the
  queue for that is long and the capacity levers are not exhausted yet, so it waits.


## Corpus facts, finally measured on our own copy

`knee-25n` asked these of *our* data rather than of a public notebook. One header sweep over all 24,371
series answers the three that were still open.

**The 130 mm crop is a crop, not a pad.** Field of view (Rows x PixelSpacing) has a median of 160 mm and
a 1st percentile of 130 mm: only **0.43%** of series are narrower than the crop, and 5.1% are narrower
than 140 mm. The borrowed constant survives our own measurement.

**Pixel spacing is finer than the crop needs, mostly.** Median 0.3125 mm, 95th percentile 0.5625 mm.
Resampling 130 mm onto 336 px gives 0.387 mm/px, so for the median series we are slightly downsampling
and for the **9.6%** of series coarser than 0.5 mm we are interpolating past their native resolution —
which is the group where a 1 mm meniscal tear was never resolvable in the first place.

**Half the corpus has no laterality tag, and which half is decided by the vendor.**

| vendor | studies | share | Laterality tag present |
|---|---|---|---|
| Siemens | 1,951 | 44.3% | 84.3% |
| Philips | 1,301 | 29.5% | 38.3% |
| GE | 868 | 19.7% | **0.0%** |
| Toshiba | 181 | 4.1% | **0.0%** |
| Canon | 45 | 1.0% | **0.0%** |

The tag is absent on 50.7% of series and on 50.0% of studies entirely. It is not missing at random: it
is missing for *every* GE, Toshiba and Canon study and for most Philips ones.

### That confounds one of my earlier refutations

I refuted the laterality-bug hypothesis partly on this: *"the asymmetry is identical across
tag-resolved and geometry-resolved studies"*. Those two groups are now revealed to be, near enough,
**Siemens-and-some-Philips versus GE-Toshiba-and-most-Philips** — a vendor comparison wearing a
rule-comparison's clothes. Any difference between them would have been a vendor difference; any
absence of difference could equally be two errors cancelling across two populations.

The other leg of that refutation is unaffected and remains the stronger one: where both rules can be
evaluated, they agree on 98.6% of 2,135 studies, and the asymmetry is *smaller* on the knees we
actually mirror, which is the opposite of what a broken mirror produces.

But it leaves a real gap, and it is a big one: **the geometry rule is validated only where the tag also
exists — that is, on Siemens and Philips — and it is the sole rule for the 24.8% of studies from
vendors that never write the tag.** Nothing in this project had ever checked it there. A left/right
error on a fifth of the corpus would suppress exactly the medial/lateral targets, and it would be
invisible to every check run so far.

### Checked, and the rule holds

No eyeballing needed. After normalisation every knee should sit in one canonical orientation, so the
*mean* coronal image of the tag-resolved vendor and of a geometry-only vendor should resemble each
other and not each other's mirror. A knee is asymmetric enough for that to bite: the medial compartment
carries the larger, more concave tibial plateau, and the lateral side has the fibular head.

| vendor | tag present | corr vs Siemens mean | corr vs its MIRROR |
|---|---|---|---|
| Siemens (reference) | 84.3% | 1.000 | 0.780 |
| Philips | 38.3% | 0.904 | 0.709 |
| **GE** | **0.0%** | **0.925** | 0.730 |
| **Toshiba** | **0.0%** | **0.913** | 0.721 |

The 0.780 in the first row is the test's floor — Siemens against its own mirror — so a genuinely
mirrored population would score at or below that. Every vendor scores 0.90+ upright and ~0.72 flipped.
**The geometry rule agrees with the tag-resolved orientation on exactly the vendors that never write a
tag.**

What this does *not* rule out: it is a population-mean test, so it detects a systematic flip and not a
sporadic one. A rule that mirrored a random 5-10% of studies would leave these numbers much as they
are. It closes the question of whether a quarter of the corpus is upside down; it does not certify the
rule study-by-study.


## The ladder, finished

Every rung matched on split, ruler, recipe and batch, fold 0, against a 0.0062 noise floor:

| encoder | epochs | holdout | vs the 8-epoch small baseline |
|---|---|---|---|
| small | 8 | 0.8081 | — |
| base | 8 | 0.8166 | +0.0085 |
| small | 24 | 0.8205 | +0.0124 |
| **base** | **20** | **0.8255** | **+0.0174** |

The two capacity levers **add** rather than substitute: +0.0085 and +0.0124 separately, +0.0174
together. Nothing knob-sized in this whole document reached a third of that.

That is the submission recipe — base, 20 epochs, batch 12 — and a five-fold retrain is queued. The
members currently on the leaderboard are small-at-8-epochs, so this is the first change since the
augmentation bug that should move the public number rather than the holdout one.

## Label content is not our lever, and the reader's own doubts are why

The cheapest way to find out whether a 2000-line multilingual report parser is worth porting is not to
port it. It is to ask whether label CONTENT can move our score at all — and the current reader hands us
a free instrument for that, because it marks its own uncertain calls.

`ConfidentLabels.struck()` clears the value of every `UNK`-verdict target, which drops it to weight zero
in `Targets.load`. That strikes **27% of all (study, finding) pairs** — 84% of Synovitis, 56% of
Fracture, 46% of Baker's, 6% of Medial Meniscus. If those 27% are noise, removing them should help or do
nothing. If they carry signal, removing them should hurt.

Matched pairs, small@24, one-cycle, weight floor, fold 0, **scored against the full table in both arms**
so the ruler never moves:

| seed | full labels | confident-only | Δ |
|---|---|---|---|
| 2026 | 0.8205 | 0.8132 | **−0.0073** |
| 7 | 0.8248 | 0.8192 | **−0.0056** |

Mean **−0.0065**, and — the part that matters — *the same sign in both seeds*. The unpaired noise floor
is 0.0062, which a single arm here would not clear. But these are paired: same split, same schedule,
same evaluation table, one variable. Two paired arms agreeing in sign and magnitude is a far stronger
read than one number sitting 0.0065 from a floor.

**Verdict on the report reader: deprioritised, and now for a measured reason rather than a hunch.** The
labels this reader is least sure of are still worth more than nothing. That is what a soft, weighted
target is FOR — an `UNK` does not enter training as a coin flip, it enters at low weight, and the model
extracts the residual signal that survives. Deleting those rows throws away real supervision to remove
imagined noise.

### What this does NOT rule out

It kills one specific idea — *filtering* the current labels by the current reader's confidence. It does
not show that a BETTER reader is worthless. The 0.899 notebook's parser does not strike uncertain rows;
it resolves more of them correctly, which is the opposite operation. This test says our supervision is
not noise-limited in the way filtering would fix; it cannot say what a reader with Turkish negation
scoping would add. Re-test that separately, and do not treat this as having closed it.

Recorded so a later session does not re-run the same arms: `knee-5od` closed, `knee-crl` stays P2.

## EMA: real, positive, and too small to be a lever

Every strong public notebook here keeps an exponential moving average of the weights. Ours existed as a
module, was unit-tested, and had never been constructed by a training run — the augmentation bug's exact
shape, and invisible to both the dead-code gate (its own test referenced it) and the mirror gate (it had
a test).

Wired and measured, matched, small@24, fold 0:

| seed | control | EMA | Δ |
|---|---|---|---|
| 2026 | 0.8205 | 0.8217 | +0.0012 |
| 7 | 0.8248 | 0.8255 | +0.0007 |

**+0.0010 on average, the same sign twice, and far inside the 0.0062 noise floor.** By this project's
own stated rule that is not a result, and it goes in the ledger next to per-slice max pooling (+0.0018)
and the weight floor (+0.0019) rather than next to capacity.

### A claim I made mid-run and then had to withdraw

Watching the seed-7 EMA arm pass 0.8220 at epoch 10 while the control had been at 0.8148 at *its* epoch
10, I said EMA reached the control's final score in half the epochs. That was wrong, and it was wrong in
the way single-point comparisons usually are. Time to within 0.001 of each arm's own best:

| arm | best | at epoch | within 0.001 at |
|---|---|---|---|
| control, seed 7 | 0.8248 | 13 | 13 |
| EMA, seed 7 | 0.8255 | 15 | 12 |
| EMA, seed 2026 | 0.8217 | 16 | 12 |

The control peaks at epoch 13 too. One epoch of difference is not a convergence story; comparing two
noisy curves at one arbitrary epoch produced a speedup that a proper time-to-target measurement does not
support.

### Kept anyway, and for a reason the fold-0 number cannot show

The final fit uses `--fold -1` and has no holdout, so it keeps its **last** epoch by construction. Its
answer therefore depends entirely on where training happened to stop — which, given a curve that wanders
0.004 between epochs 13 and 24, is a real coin-flip we currently take for free. An average removes it.

That is an argument about the *variance of the shipped artefact*, not about the mean of a fold-0 score,
and no arm measured here can confirm or refute it. Stated as a design choice rather than a measured
gain, which is what it is.

## The bar moved, and it demotes almost everything above

The owner's threshold for calling a result significant is **1–2% absolute, ~1.5% working**. This
document was written against a *measured noise floor* of 0.0062, and that was the wrong quantity to
judge against. A noise floor says what is **detectable**; it says nothing about what is **worth
building on**. Clearing it means "probably not zero", which is a much weaker claim than this project
needs.

Under the 1.5% bar, re-scored:

| change | Δ | old verdict | verdict now |
|---|---|---|---|
| DINOv2-base | +0.0085 | cleared the floor | directional |
| 8 → 24 epochs | +0.0068 | cleared the floor | directional |
| ranking term | +0.0067, +0.0036 | cleared the floor | directional |
| EMA | +0.0010 | inside the floor | noise |
| encoder depth 6 → 12 | +0.0008 | — | noise |
| per-slice max pooling | +0.0018 | inside the floor | noise |
| weight floor | +0.0019 | inside the floor | noise |

**Nothing measured in this project has ever cleared 1.5% as a single change.** That is the honest
statement, and it is more useful than the previous framing because it says where the effort should go:
not into another knob worth +0.005.

### What survives is the CUMULATIVE number

Increments below the bar still compose, and the composition is above it:

| recipe | score |
|---|---|
| small, 8 epochs (the original baseline) | 0.8081 |
| small, 24 epochs | 0.8205 |
| base, 20 epochs | 0.8255 |
| **base + ranking, 24 epochs** | **0.8283** |

**+0.0202 from the original baseline** — 2%, above the bar. So the rule going forward is to track the
running total against the *original* baseline and to stop reporting each +0.005 step as a finding.

### The consequence for how experiments get chosen

A change now needs a *mechanism that could plausibly deliver 1.5%*, not merely a hypothesis that it
helps. Knob-sized ideas are kept only when they are free and physically sound — ranking costs nothing
at inference and is the quantity the metric literally scores, so it stays in the recipe — but they no
longer justify a confirmation seed of their own, and they never gate a plan.

## Every number in this document is measured with the wrong ruler

`train.csv` carries the twelve annotations for **exactly 58 studies, all or nothing** — 696 of 52,884
(study, label) cells, 1.32%. There are no partial annotations to scavenge: a study has all twelve or
none.

So every score here — 0.8081, 0.8205, 0.8283 — is agreement with **LLM-read report labels**, and the
competition scores agreement with **radiologist annotations**. Those are different targets, and this
project selects its epochs, its members and its recipe on the proxy.

It cannot be fixed locally, and it is worth being precise about why:

- The 58 annotated studies are **trained on**, at `ANNOTATED_WEIGHT = 3.0`. Scoring them is scoring
  training data.
- Held-out, fold 0 gets about a fifth of them — twelve studies, carrying between 2 and 7 positives per
  label. That is the regime `knee-6xm` already computed a Hanley-McNeil standard error for: ±0.17 at
  95%, far wider than any difference this project separates.

**The only true ruler available is the leaderboard.** That is not a reason to distrust the local
numbers — it is a reason to know what they are: a proxy that ranks recipes, validated against the real
metric only when we submit.

### The proxy appears to COMPRESS, by roughly a factor of two

Two rows of the submission ledger have both numbers:

| local | LB | |
|---|---|---|
| 0.8162 | 0.868 | 5 members, augmentation silently disabled |
| 0.8269 | 0.888 | augmentation actually applied + OneCycle + overlap TTA |

**+0.0107 local became +0.020 on the leaderboard.** If that ratio holds, our local deltas understate
the thing being scored by about 2x, and the 1.5% bar applied to the LB corresponds to roughly +0.008
locally.

Stated with its limits, because it is two points and a confounded pair: three changes landed together,
the member count is the same but the recipe is not, and no error bar exists on either number. It is
enough to justify *submitting to find out*, and not enough to re-score the ledger by.

### What follows for how this project should spend its submissions

Five per day, kernel pushes unlimited. The local ruler is free but proxied; the LB is scarce but true.
So the local number decides WHICH recipe to submit, and only the LB decides whether a recipe is better.
A change that looks like +0.008 locally is exactly the case that cannot be settled without spending one.

## The capacity era is over: we are supervision-limited now

Capacity was the only axis that ever paid here, and it has stopped paying. Three independent readings,
all fold 0, all 24 epochs unless noted:

| recipe | best | at epoch |
|---|---|---|
| small | 0.8205 | 15 |
| small + ranking | 0.8272 | — |
| base (20 epochs) | 0.8255 | — |
| base + ranking | 0.8283 | — |
| **large, every block, + ranking** | **0.8304** | **10** |
| large, 6 of 24 blocks, no ranking | 0.8156 | — |

**1. The ladder saturates.** small → base was +0.0085. base → large, both with ranking and both with
half the encoder trainable, is **+0.0021** — inside noise, for 3.5x the parameters and 185s an epoch
against 50s.

**2. Large overfits our supervision.** It peaks at **epoch 10** and gives back 0.009 by epoch 23, on a
one-cycle schedule that has not even finished decaying. The small and base arms peak at 13-15 and hold.
A model that reaches its best a third of the way through and then declines has more capacity than the
labels can constrain.

**3. The labels are 98.7% read out of text.** Only 58 of 4,407 studies carry an image-read annotation.

Together these say the binding constraint has moved. Capacity paid early because the model was
UNDER-FITTED — that was the correct reading of the ladder at the time, and the 8-epoch ablation table
was invalid for exactly that reason. It is not the reading now.

### What that predicts, so it can be wrong out loud

- The encoder-rate bracket should be **neutral to negative**: letting the encoder move faster is more
  capacity to fit noise, not more signal to fit. If a higher rate helps materially, this whole reading
  is wrong and the encoder was simply frozen too hard.
- **Slice coverage should be one of the few things that still pays**, because it adds EVIDENCE rather
  than parameters — a third of the band we chose to read is currently discarded.
- **The full-data fit should pay**, for the same reason: 4,407 studies against 3,527 is more
  supervision, not more model.
- A larger encoder should be run with a SHORTER schedule if it is run at all — 24 epochs is past its
  peak.

### What it does NOT license

It does not resurrect the label-filtering idea, which was refuted twice over on matched seeds. Being
supervision-limited means more or better-READ supervision, not less of it.

## Seed 7 is worth +0.0038 on this fold, which invalidates a "best" of mine

Arm J — base, every block trainable, ranking, **seed 7** — scored 0.8313, the highest number in this
document. It is not an improvement, and the arm was mis-designed: it changed the unfreeze depth AND the
seed against its comparator (`rankbase24_s2026`, 0.8283, depth 6, seed 2026).

The seed is not free on this split. Every matched pair we have:

| recipe | seed 2026 | seed 7 | Δ |
|---|---|---|---|
| small, 24 epochs | 0.8205 | 0.8248 | +0.0043 |
| confident-only labels | 0.8132 | 0.8192 | +0.0060 |
| EMA | 0.8217 | 0.8255 | +0.0038 |
| small + ranking | 0.8272 | 0.8284 | +0.0012 |

**Mean +0.0038, positive four times out of four.** Seed 7 draws a better model on fold 0 — or, more
precisely, the pair (initialisation, data order) that seed 7 produces suits this holdout better.

Subtracting it puts arm J at ≈0.8275 against 0.8283: **training every block of base is neutral to
slightly negative**, which is what the depth bracket already said (+0.0008 at depth 12 on small).

### The methodological consequence, which outlives this arm

A seed is not a nuisance parameter here, it is worth as much as most of the ideas in this document.
Three rules follow:

1. **Never compare across seeds.** A cross-seed comparison carries a systematic ±0.004 that is larger
   than every effect this project has measured except capacity.
2. **A "new best" from a different seed is not a result** — this one survived about four minutes.
3. The measured seed spread (0.0062 over three seeds, and +0.0038 between these two) is a floor for
   *detectability*, and the reason the owner's 1.5% decision bar is the right one to plan against.

I should have caught this when queueing the arm: it was written as "the stacked recipe, seed 7" without
a seed-7 comparator existing for it.

## Slice coverage: my prediction was wrong, and the reason was already written down

I predicted more slices would pay, on the supervision-limited reading: 89% of series hold more than 12
slices inside the 0.20-0.80 band we read, so a third of the evidence we had already decided was worth
reading was being discarded.

Matched, base + ranking, seed 2026, fold 0, only the slice count moved:

| slices | best |
|---|---|
| 12 | 0.8283 |
| **20** | **0.8267** |

**−0.0016.** Inside noise, and in the wrong direction. The prediction is refuted.

### Why — and it is not "more data does not help"

`model.py` already carries the mechanism, in a comment written after a different failure:

> Channel stacking couples the group's PHYSICAL THICKNESS to the slice count, which is measurably why
> raising inference slices from 12 to 32 cost −0.018 here.

Under `slice_pooling="channels"` a group is three CONSECUTIVE cached slices stacked as the encoder's
three channels. The cached slices span a fixed anatomical band, so at 20 slices they sit closer
together and a three-slice group covers **40% less tissue** than at 12. I did not add evidence per
view. I made every view thinner, and then added more of them.

So the arm did not test slice coverage. It tested slice coverage MINUS group thickness, and those two
moved in opposite directions — which is exactly the confound that cost 0.018 the first time this
project met it, in a comment I had read this morning and did not connect.

### The deconfounded test

Group size is fixed at three by the RGB trick, so thickness cannot be restored by widening the group.
`slice_pooling="max"` is the escape the codebase already has: each slice is encoded ALONE and the
slot's embedding is the per-dimension maximum, so a group's physical extent stops depending on how
finely the band was sampled. It is also what the 0.903 reference does, and per-slice max pooling
measured +0.0018 at 12 slices — inside noise, and tested at the one slice count where the confound it
fixes does not bite.

Queued on small (3x the encoder passes, so base was too expensive for a signal test): max pooling at 12
slices to isolate the pooling change, then max pooling at 20 to move the slice count with thickness
held constant. Control: small + ranking at 12 slices, 0.8272.

**If 20+max still does not beat 12+max, slice coverage is genuinely not a lever here** and the
supervision-limited reading loses its evidence branch, keeping only the label branch.
