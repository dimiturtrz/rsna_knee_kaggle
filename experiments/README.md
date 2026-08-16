# experiments — the arms, in git, runnable by shell

Every number this project reports came out of one of these. They lived in `/tmp` until 2026-08-11,
which meant the definition of each measurement was unversioned, unreviewable, and one reboot from
gone — the same failure the no-notebooks rule exists to prevent. An arm is code. It belongs here.

```bash
bash experiments/arms/compose_base.sh            # one arm, foreground
nohup bash experiments/arms/whole_stack.sh &     # one arm, detached
```

Each arm sources `lib.sh`, which resolves the data root through `core.paths` (never a hardcoded
`D:/data`), fails on a missing input **before** spending GPU, and tees full output to
`<processed>/experiments/<arm>.full.log`. Logs are data and stay out of the repo; arms are code and
stay in it.

An arm file carries its hypothesis, its control number, and its result in the header. Reading the
directory is meant to be the same as reading the ledger.

## The checks, and when to run them

Each exists because the failure it catches happened, silently, and cost either a queue or a slot. Only
the last wants the card, and for two minutes.

```bash
uv run python -m kneemri.kaggle.kit_check      # BEFORE shipping a kit (refuses stale or dirty)
uv run python -m kneemri.kaggle.kit_check --mounted   # BEFORE a submission — asks KAGGLE what it will run
# then push comp/kernels/kneemri-loadcheck and read it: loads, forwards and chunks every
# member on a real T4, spends no submission, and answers what only the runtime knows
uv run python -m kneemri.experiments.arm_check      # BEFORE arming a queue (--require-caches: refuse an undecoded reading)
uv run python -m kneemri.experiments.ledger_check   # AFTER runs finish
uv run python comp/witness_check.py --root <members> --device cpu   # BEFORE a submission
```

**`kit_check` — before shipping a kit, and `--mounted` before a submission.** Two slots have been spent
on code that was already fixed in the repo: one on an OOM whose fix landed three hours after the kit was
built, one on a bf16 member served in fp16 because the kit predated `Member.precision`. Bare, it compares
the kit on this disk to HEAD, and only in the packages a kernel runs — a docs commit that failed the check
would train the reader to ignore it. `--mounted` asks the question a slot actually cares about, because
uploading is a manual `kaggle datasets version` step: it downloads the published `kit.json` alone (46
bytes, listed as its own file, so no weights move) and judges it by the same rule. On 2026-08-15 its first
run refused the live dataset — mounted `4faac3d`, five shipped-package commits behind.

**`--studies N` — the one check that DOES want the card, and wants it for two minutes.** `arm_check`
proves an arm's flags parse; it cannot prove the recipe they select survives contact with the corpus,
the encoder and the loss. `--studies 60` caps the corpus positions a run touches, keeping the fold's
train:holdout ratio so both halves of the loop execute, and reports a score that means nothing by
construction. Measured 2026-08-16: 48 train / 12 holdout, one epoch in **2 s**, checkpoint written and
reloaded — against the forty minutes a real fit holds the GPU to reach the same first assertion.

```bash
uv run python -m kneemri.training.train --labels "$LABELS" --studies 60 --epochs 1 $RECIPE --out /tmp/smoke.pt
```

It earned itself on first use, on the arm that could least afford it. `final_fit.sh` — the deliverable,
`--fold -1`, no score to notice a problem by — asked for batch 12 of whole-stack 392 px inside
`--vram-gb 22` **and** `--compile-encoder`, while the member it ships (`res392_ws_s2026`, holdout 0.8523)
records `vram_gb 28.0` and `compile_encoder False`. The smoke died in the attention backward with
`Tried to allocate 1.03 GiB ... 22.00 GiB allowed`, two minutes in. Batch size is what sets that peak, not
corpus size, so the forty-minute run would have died at the same step — and `arm_check` cannot see it,
because the flags were valid. The cap is now 28 and the same smoke passes in 41 s.

That one arm's rescue is now every arm's, under `arm_check --smoke <dir>`: each accepted invocation is
run for one epoch at `--smoke-studies` (60), its checkpoint deleted as soon as it has proved the write
path, and a `BROKEN` line printed the moment a recipe dies. It wants the card and the better part of an
hour, so it is not the pre-queue check — it is what to run before a queue whose arms have been edited.

```bash
uv run python -m kneemri.experiments.arm_check --smoke /tmp/smoke_out
```

Its own first run reported all thirty-four arms broken on the same missing `labels.csv`: `stub_lib.sh`
fabricates a `/tmp/arm_check` root, which proves flags parse and is what CI gets, but is not a file a
trainer can read. The stub now takes the resolved processed root from the checker, which already holds
`core.paths`, so no arm pays a subprocess to learn where the data is. A check whose own harness lies
reports thirty-four defects and contains one.

Measured 2026-08-16 once the harness was honest: **34 arms, 0 refused, 0 uncached, 0 broken** — every
committed recipe survives a real epoch on the card. That is the pipeline's readiness answered by
execution rather than by inspection.

**`arm_check` — before arming a queue.** Sources every arm against `stub_lib.sh`, whose `run_train`
prints its flags instead of training, and hands each line to the real `TrainingOptions.parse`. Catches
a flag typo in seconds rather than after the card has been held. It is sourced rather than parsed
because bash is what runs it — an arm once died on a quoting error and no regex finds that. It then asks
the accepted flags a second question: is the cache that reading names decoded? `--image-size`,
`--slices`, `--crop-mm`, `--scheme` and `--tie` pick a cache file, and `train.main` takes the card BEFORE
it maps that file, so an undecoded reading costs a slot exactly as a typo does. Reported by default and
refused under `--require-caches`, which is what to run when arming a queue that builds no cache itself.

**`ledger_check` — after runs finish.** Reports any run whose metrics file exists and whose best score
is quoted nowhere above. Five arms once sat in this table as "queued" for two days after they had run,
and the next planning pass read them as unstarted work. It found four more on its first execution,
*after* the table had just been updated by hand.

**`witness_check` — before a submission.** Re-derives what training predicted for eight named studies
through the INFERENCE path and refuses the member if the numbers move. It is the only check that spans
both paths, and it found three defects on first use — `torch.compile` renaming every state-dict key,
bf16 members served in fp16, and its own tolerance being tighter than bf16's epsilon. A member trained
before the witness has nothing to re-derive, so read the summary rather than the exit code: a run that
verifies NOTHING says so out loud and still exits 0, because every packaged member predates the witness
(knee-stp) and the fix is a card, not a flag. `--require-witness` makes the absent claim fail, for a
caller that can demand the strong one.

What none of them check is whether a SENTENCE is true. A number can be present and its interpretation
stale; that still needs a reader.

## Two rules encoded in `lib.sh` rather than remembered

- **`wait_for` keys on a completion marker.** A gate that watches an output path fires when the
  producer *starts*. That once queued an arm against a 59.7 GB cache still mostly zeros; it would not
  have crashed, it would have produced a plausible wrong number.
- **`run_train` tees before it greps.** Arms used to pipe straight into `grep`, so any traceback whose
  wording missed the filter vanished and the run looked merely empty.

## Ledger — fold 0, seed 2026, 24 epochs, one-cycle, weight floor, ranking on

Control for the small arms is `rank24_s2026` = **0.8272**; for base arms, `rankbase24_s2026` = **0.8283**.
The noise floor on this ruler is **0.0062**, and this project's bar for a *result* is ~1.5% absolute —
so everything below is **directional**, and only the cumulative total clears the bar.

| arm | change | score | Δ | verdict |
|---|---|---|---|---|
| `rate_isolate` | encoder rate 8e-6 → 4e-5, depth 6 | 0.8323 | +0.0051 | kept, single-variable |
| — | same rate at depth 12 | 0.8316 | +0.0044 | depth is not a lever |
| `slice_count_maxpool` Q | max slice pooling, 12 slices | 0.8310 | +0.0038 | kept |
| `slice_count_maxpool` R | max pooling, 20 slices | 0.8309 | −0.0001 vs Q | **refuted** |
| `severity_flat` | three YES levels collapsed to one | 0.8228 | −0.0044 | gradation is live signal |
| `report_supervision` | report-embedding term @0.3 | 0.8219 | −0.0053 | **refuted** |
| `report_supervision` | report-embedding term @1.0 | 0.8209 | −0.0063 | refuted, monotone |
| `compose_base` | rate + pooling on base | **lost at epoch 7/24** | — | not rerun |
| **`whole_stack`** | **all 12 slices per step, 12 epochs** | **0.8457** | **+0.0147** | **largest change measured** |
| `wholestack_folds` 1 | same recipe, fold 1 | **0.8505** | — | ensemble member |
| **`resolution`** | **336 → 392 px, 448 cache — AND encoder rate 8e-6 → 4e-5** | **0.8523** | **+0.0066** | **CONFOUNDED, see below** |
| `slice_attention` | attention over slices, not max | 0.8473 | +0.0016 | kept as a member |
| `wholestack_folds` 2 | same recipe, fold 2 | **0.8518** | — | best single member; epoch-5 partial |
| `slices16_ws` | 16 cached slices, whole-stack | 0.8452 | −0.0005 | **refuted**, 33% slower for nothing |
| `extended_slots` | nine slots, +1,139 coronal series | 0.8459 | +0.0002 | **refuted** on the mean |
| `intensity_augment` | gamma/blur/contrast/noise at 392 | 0.8459 | −0.0064 vs res392 | **refuted**, behind at every epoch |
| `dinov3` | DINOv3-S/16 at 336, then 224, then S+ | 0.8080 | −0.0377 | **refuted**; deficit is focal findings |
| `calibrated_targets` | each level set to its MEASURED agreement rate | 0.8495 | +0.0038 | directional, inside the floor |
| `lora` v3small 224 | rank-8 adapters, base frozen | **0.8217** | **+0.0160 vs v3 fine-tuned** | **recipe was damaging v3** |
| `schedule_length` | 8 epochs instead of 12 | 0.8429 | −0.0028 | compressing the cycle costs, two thirds the wall-clock |
| `dinov3` 224 | v3-S at its own native read | 0.8057 | −0.0400 | read size is not the axis: 0.0023 from 441 patches to 196 |
| `dinov3` S+ 224 | wider small, 28.7M | 0.8012 | −0.0445 | capacity scaled the WRONG way inside the family |
| `lora` v2 336 | rank-8 adapters on DINOv2 | 0.8127 @ e3 | behind at every matched epoch | **killed at epoch 3** — the arm's own predicted refutation for v2 |
| `crop80_tissue` | 80 mm about the tissue centre, 0.238 mm/px | 0.8345 | −0.0112 vs `whole_stack`, but see below | mean is the WRONG read — gate is per-label |

### `crop80_tissue` — the control is not schedule-matched, and the mean was never the gate

Two things have to be said before that −0.0112 is quoted anywhere. **First, the control is 12 epochs and this
arm is 8** (`epochs_done` on both checkpoints). `schedule_length` above prices exactly that axis at
−0.0028, so the crop itself costs about **−0.0084** on the mean, not −0.0112. The arm should have been run
at 12 to match; it was not, and the adjustment is arithmetic over another arm rather than a measurement of
this one. **Second, `crop80_tissue.sh` forbids reading the mean at all** — its prediction is two-sided (small
structures up, diffuse findings down), so the mean averages the two halves of the prediction against each
other and a null there is what you would expect whether the mechanism works or not. The decision belongs to
the per-label paired read, which is what gates the multi-scale build (knee-1nn).

## The eleven runs nothing had written down

`python -m kneemri.experiments.ledger_check` used to scan `*.metrics.jsonl` and report "15 finished runs, 0
unrecorded". There are **69 checkpoints** on disk and 21 metrics files, so that pass was computed over
15 of 69 — the wrong denominator, inside the check whose whole job is catching exactly that. It now
counts a run finished on its CHECKPOINT (each carries its own `holdout`), and reads the ledger *and*
`interpretations/` as the record, because a run written up in an interpretation is written down.

That takes it to 65 finished runs and 11 whose number appeared in no document at all, recovered here
2026-08-15 with no GPU:

| run | holdout | what it was, from its own checkpoint |
|---|---|---|
| `nf7_fold0` | 0.8121 | noise-floor seed 7, 8 epochs — see the ruler note below |
| `nf13_fold0` | 0.8158 | noise-floor seed 13, 8 epochs |
| `member_fold1` | 0.8305 | five-fold ensemble member |
| `member_fold3` | 0.8325 | five-fold ensemble member |
| `member_fold4` | 0.8306 | five-fold ensemble member |
| `deep24_s2026` | 0.8213 | unfreeze 12, **ranking off** |
| `shallow24_s2026` | 0.8143 | unfreeze 3, **ranking off** |
| `lr20x_s2026` | 0.7524 | backbone rate 1.6e-4 at depth 12 — the 20× arm, collapsed |
| `base_s7_fold0` | 0.8180 | base encoder, 8 epochs, seed 7 |
| `aug_fold0` | 0.8201 | options block EMPTY — predates option stamping, so name and score is all there is |
| `labaug_fold0` | 0.8203 | options block EMPTY, same |

Two readings that matter more than the rows:

**The noise floor's own inputs were unrecorded.** `nf` is noisefloor: `nf2026`/`nf7`/`nf13` are the three
seeds behind the **0.0062** spread every verdict in this ledger is read against. The interpretation
records them at their **overlap-TTA** scores (0.8171 / 0.8209 / 0.8233); the checkpoints hold the raw
ones (0.8081 / 0.8121 / 0.8158), whose spread is **0.0077**. Same three runs, two rulers, and only one
of the pairs was ever written down. The 0.0062 stands — it is the TTA-scored spread and verdicts are
read on TTA-scored numbers — but the check cannot tell "never recorded" from "recorded under the other
ruler", and that limitation is why these eleven need a reader rather than an auto-appended row.

**The five-fold members average to the number already quoted.** 0.8202 / 0.8305 / 0.8205 / 0.8325 /
0.8306 has mean **0.8269**, which is the "five folds on the shipped recipe: 0.8269" figure — so the
ensemble result was recorded while three of the five runs under it were not. Their options blocks are
empty, so nothing on disk says what recipe produced them beyond the ensemble's own description.

## Our best single member changed TWO things, and the ledger credited one

`res392_ws_s2026` is the best member this project has produced (0.8523) and its row has read
"336 → 392 px, +0.0066, clears the floor" since 2026-08-12. The checkpoint's own recorded options say
otherwise:

| member | backbone lr | cache | read |
|---|---|---|---|
| `wholestack_s2026` (the control) | **8e-6** | 336 | 336 |
| `res392_ws_s2026` | **4e-5** | 448 | 392 |
| `intensity392_ws_s2026` | 4e-5 | 448 | 392 |

**Five times the encoder rate, on top of the resolution change.** And the rate was independently
measured at **+0.0044** in the windowed era (`knee-3om`), which is most of the +0.0066 being attributed
here to resolution. If that transfers, resolution alone is ~+0.002 — comfortably inside the 0.0053
floor and not a finding at all.

It may not transfer: +0.0044 was measured under windowed reading, before whole-stack existed, so the
rate's value under the current recipe is genuinely unknown. What is *not* uncertain is that this row
cannot be read as a resolution result.

**The clean test is one arm**: 392 px at 8e-6, matched to the control's rate. Either resolution
survives on its own or our best member is mostly a rate change we already knew about.

This was found by hand, three days after the row was written. It is now mechanical: every run records
its recipe beside its scores, and `python -m kneemri.evaluation.run_comparison --control <ctl> --arms <...>`
prints a `vs control:` line naming every option the two disagree on — for this arm,
`image_size 336→448, backbone_lr 8e-06→4e-05`. Runs older than the recorder were backfilled from their
own checkpoints (`python -m kneemri.runs.metrics_recipe`); `lora_r8_small_336_s2026` and `wholestack_fold2`
have no checkpoint left and can never be matched controls again.

## A peak is not a rate: OneCycle delivers half of the constant it replaced

Measured at our exact settings — peak 8e-6, `pct_start` 0.15, 12 epochs of 291 steps — by summing the
learning rate over every step:

| regime | total LR over the run | mean rate |
|---|---|---|
| constant 8e-6 | 0.02794 | 8.0e-6 |
| **OneCycle at peak 8e-6** | **0.01405** | **4.0e-6** |

**OneCycle delivers 50.3% of the movement the constant did.** The backbone starts at 3.2e-7
(`div_factor` 25, a default nobody chose) and ends at 3.2e-11 (`final_div_factor` 1e4), so the number
in the arm header is the top of a curve the run barely visits.

Every arm quoted by its peak carries the same halving: the "5x rate" arm at 4e-5 delivers a mean 2e-5,
which is **2.5x** the old constant rather than 5x. This is a framing correction and not a defect —
OneCycle's value is the warmup and the anneal, and the schedule measurably works — but a peak and a
constant are not the same quantity and were being read as one.

## The running total, which is the number the bar applies to

Individual arms keep landing inside the noise floor, and that is the wrong frame to judge them by: the
rule is to track the cumulative total against the ORIGINAL baseline, because increments below the bar
still compose.

| | holdout | step | total vs baseline |
|---|---|---|---|
| `rank24_s2026` — the control every early arm was read against | 0.8272 | — | — |
| whole-stack reading | 0.8457 | +0.0185 | **+0.0185** |
| + 392 px AND encoder rate 4e-5 | **0.8523** | +0.0066 | **+0.0251** (confounded — two changes, see above) |

Both peak at epoch 6. 392 px is the last rung that fits: patch-14 needs multiples of 14, so the ladder
is 336 (576 tokens) → 392 (784) → 420 (900) → 448 (1024), and 392 already holds 28.4 GB of a 32 GB card
while 448 OOMed outright. Buying the next rung would mean a smaller batch, which changes the ranking
term's within-batch pair count and stops it being one variable.

**+0.0251 clears the 1.5% bar**, against a floor of 0.0053. No single arm did — whole-stack came
closest at +0.0185 on its own control. Two things this does NOT license: reading the total as evidence
for any individual knob inside it, and expecting it on the leaderboard, where our holdout gains have so
far transferred at roughly half (`knee-lv2`).

## Whole-stack and overlap TTA are the same mechanism

Each member scored twice: once by training at its best epoch, once by the corrected curve with overlap
TTA at inference.

| member | trained holdout | with overlap TTA | gain |
|---|---|---|---|
| `lr5x_d6` — windowed | 0.8323 | **0.8475** | +0.0152 |
| `max12` — windowed | 0.8310 | 0.8451 | +0.0141 |
| `large_deep` — windowed | 0.8304 | 0.8450 | +0.0146 |
| `wsfold0_e6` — whole-stack | 0.8457 | 0.8457 | **0.0000** |
| `slices16` — whole-stack | 0.8452 | 0.8452 | **0.0000** |

Averaging ten overlapping three-slice windows means the model **does** see every slice — aggregated in
PREDICTION space rather than in FEATURE space. So the windowed lift (+0.0146) and whole-stack's
training-time advantage (+0.0147) are the same thing bought twice, and they do not compose.

**This partly re-prices whole-stack.** Its +0.0147 was measured against a control scored WITHOUT the
TTA that control gets at inference. Where both are given their own best reading, `lr5x_d6` (0.8475)
edges `wsfold0_e6` (0.8457). What this re-pricing does NOT touch is `res392` (0.8523), which is scored
with no TTA available to it at all — but that member is not a resolution result either: it changed the
encoder rate 5× as well, and the section above prices resolution alone at about +0.002 if the rate's
independently measured +0.0044 transfers. So the highest number in the ledger stands as a number and
stands CONFOUNDED as a cause. See `knee-eeo` and `knee-a8x`.

## What is ours, and what is grafted

**0.894 is not our number.** Twenty of its twenty-three models are `pilkwang/rsna-knee-weights` — CC0
weights someone else trained, grafted on 2026-08-10 and run through our data layer. Ours is three
models. Legal (rule 2.6.b permits external models) and fine as a competition entry; it is NOT evidence
that our pipeline is good.

| what | LB | whose |
|---|---|---|
| published 20, alone | 0.888 | theirs |
| our 5, old recipe | 0.888 | **ours** |
| our 3, whole-stack | 0.882 | **ours** |
| the 23 fused | **0.894** | mostly theirs |

**Our own best is 0.888.** Quote that one. The fused number's marginal gain from our three members is
+0.006 over their twenty.

What IS ours and worth keeping is the FINDING: members that fail differently help, members that fail
the same way do not. Our third whole-stack member bought 0.000; a different recipe bought +0.006. That
is reproducible without anyone else's weights, and the all-ours ensemble is the test of it.

## Ensemble SIZE is worth more than anything we can do to a single model

| members | holdout per member | public LB |
|---|---|---|
| our strongest model ever | 1 | 0.8518 | 0.868 |
| whole-stack | 2 | 0.8481 | 0.882 |
| whole-stack | 3 | 0.8493 | 0.882 |
| ours, old channel-stacked windowed recipe | 5 | 0.8269 | 0.888 |
| the published CC0 members | 20 | 0.8398 | 0.888 |
| **ours 3 + the published 20** | **23** | mixed | **0.894** |
| 2 — whole-stack | 0.8481 | **0.882** |
| 5 — old, channel-stacked, windowed | 0.8269 | **0.888** |

Six entries built to isolate one variable. The best single model this project has produced loses to
five much weaker ones by **0.020** — more than whole-stack (+0.0147) and more than every knob combined.

**And DIVERSITY beats more of the same.** A third member of our own recipe bought nothing (0.882 →
0.882), while adding our three to the twenty published ones — a different recipe, not a different draw
of ours — bought **+0.006 over either component alone**. Members that fail the same way do not help
each other. That is the mechanism worth building on, and it is neither folds nor seeds.

This does NOT defend folds or seeds, which remain banned and rightly so: the noise floor is the
holdout's own sampling error (`knee-2rx`), so re-running a recipe cannot buy a better measurement. It
says that AVERAGING MANY PREDICTORS is worth ~0.02, whatever produced them. So the members have to
differ by MECHANISM rather than by RNG — and every arm here is fitted on the same fold-0 split, which
makes every arm a member as well as a verdict. See `knee-9pl`.

## The leaderboard disagreed with the holdout, and that is the finding of the night

| entry | members | holdout mean | public LB |
|---|---|---|---|
| ours v2 | 5 (channel-stacked, windowed) | 0.8269 | **0.888** |
| whole-stack | 2 (fold 0 + fold 1) | 0.8481 | **0.882** |

Two members that are **+0.021 better each** scored **0.006 lower**. The prediction written before that
submission — "my honest expectation is an improvement on 0.888" — is refuted.

It is NOT evidence against whole-stack, which is +0.0147 on a matched holdout comparison measured on
two folds. It is evidence that member QUALITY alone does not carry this leaderboard, and that our
holdout gains are not transferring one-for-one. 0.006 is also small enough to be LB noise on a public
subset, so the honest reading needs the rest of the curve — 1, 3 and 23 members are in flight on
nearly the same members. See `knee-lv2`.

## Whole-stack is the result of the day

Reading every cached slice per step instead of one random three-slice window, against arm Q (identical
recipe, windowed, best of 24 epochs) = 0.8310:

| | holdout | note |
|---|---|---|
| whole-stack, 24 epochs | 0.8413 | peaked epoch 6, then declined |
| **whole-stack, 12 epochs** | **0.8457** | peaked epoch 6 — *higher* AND half the wall-clock |
| whole-stack, fold 1 | **0.8505** | second fold, same recipe |
| the five members behind our 0.888 LB | 0.8269 | mean of `member_fold0..4` |

**The shorter budget scored higher, not merely cheaper.** One-cycle is defined over `total_steps`, so a
24-epoch budget stretches the anneal across epochs the model does not need and leaves the rate far from
annealed where the model actually converges. An oversized epoch budget detunes the schedule; it is not
just wasted GPU. Both folds peaked at epoch 6, so the arms now run **8**.

It is also a replication rather than an invention: MRNet combines per-slice features by an elementwise
max over ALL slices of a series. Our three-slice window was the deviation.

## The biggest lever is not architecture

The metric is the **unweighted** mean of twelve AUCs, and our labels are far from level — MCL 0.755 and
Fracture 0.784 against Baker's 0.896. Lifting those two to 0.873, a level five other labels already
reach, is worth **+0.0173** on the mean: larger than whole-stack, and above the 1.5% bar.

Rarity and label noise were both tested and refuted as explanations: Synovitis has *fewer* positives
than MCL (580 vs 767) and the corpus's *lowest* reader confidence (0.189), and scores 0.873. So MCL is
hard to **see** — a thin ligament at the medial margin — which is what the resolution arm now targets.

## Three arms were lost on 2026-08-11, and only one cause is known

Recorded here because the ledger must not read as if the empty cells were merely pending.

- **`compose_base`** died at epoch 7 of 24 **and printed its own completion marker**, so the queue
  advanced on a corpse. Cause unrecoverable: it ran from a pre-`experiments/` script that piped into
  `grep` with no `tee`, and the traceback went with it.
- **The 448 px cache build** was killed at 2,200 of 4,407 with a log that simply stops — no traceback,
  and it was a plain redirect, so the OS killed it. RAM pressure from a 63.7 GB memmap beside a 27 GB
  trainer is suspected and **unproven**; there is no OOM event in the system log.
- **`whole_stack`** died with `unexpected EOF while looking for matching quote`. A sweep editing every
  arm's trailing marker rewrote the file **while bash was executing it** — bash reads a script
  incrementally by byte offset, so it resumed at a stale offset and died.

The plumbing changes that came out of it (`PIPESTATUS`, success-only markers that also require a
`best holdout` line, verdicts teed into the arm's own log, `run_snapshot`, `wait_for_cache` keyed on
the mask) are in `lib.sh`, each documented at the point it prevents. See `knee-7gr`.

`compose_small` was **dropped from the queue unrun**: an hour of GPU to confirm that two sub-bar
levers add cannot change any decision.

## Why the arms above mostly could not have mattered

Every row that moved less than 0.006 is a knob *inside* one design, and that design uses about a tenth
of the data it is handed — 18 of ~180 acquired slices per step, 12 cached of a median 30 per series,
17.4% of series never routed to a slot at all, 336 px for findings a few voxels wide. We are at 0.888
on the leaderboard against a 0.942 frontier, and a knob inside a family returns ~0.005 by
construction. `whole_stack` and `resolution` are the first two arms that leave it. See `knee-jk3`.

## Results that cost no GPU, and belong here anyway

This ledger is a table of ARMS, so a day spent on measurement rather than training leaves it empty and
reads as a day with no results. 2026-08-14 was such a day, and its findings change what the arms above
mean:

- **The ruler was five times too small.** The twenty published members carry their own out-of-fold
  predictions, four seeds over five folds, whose id-sets partition all 4,407 studies exactly. Every
  study therefore has a held-out prediction, on a ruler with a sampling floor near **0.0024** against
  our one fold's **0.0053** — on a CPU, with no new labels.
- **The label profile is not the one the plan was built on.** Fracture, which this ledger names a target
  at 0.784, is our BEST finding at 0.8923 over the corpus. The weakest is Lateral Meniscus at 0.7981,
  never named. The `+0.0173 from MCL and Fracture` target above is dead, exactly as `label_power`
  predicted from the standard error of 62 positives.
- **A run kept eight of the 881 studies it predicted, and threw the rest away.** `Fitting.ranked`
  already scores the whole holdout to rank each epoch, and that array is what every per-label number in
  this ledger was read off; eight studies of it became the witness and 873 were dropped. They are now
  written beside the checkpoint (42 KB, no extra compute), so `run_comparison` marks each arm against
  **this pair's own paired interval** instead of the unpaired bound — and says in the arm's block which
  of the two it used. Arms that predate the record still print against the wide bound and are UNREAD,
  not refuted; back-filling them is an inference pass (`knee-j81m`), not a training slot.
- **Six findings are held back by their LABELS and six by the MODEL**, and the split is by structure
  size. Measured against radiologist annotations, held out.
- **Caching 12 slices of a median 36 shows no measurable cost** — one of `knee-jk3`'s four named causes,
  tested for the first time and not supported.
- **Their router keeps the LAST of equally dense candidates; ours keeps the first.** The one difference
  found in eleven tested axes, replicated on 4 of 4 published members across 3 seeds. Correcting it makes
  the routing signature vanish entirely (`corr(residual, series_count)` +0.647 → −0.070). It is worth
  about a third of the residual, not all of it.
- **Ten other axes say our decode already IS theirs** — percentiles, DICOM windowing, crop rounding,
  band sampling, resize filter, slice order and window policy — several of them assumptions recorded in
  `comp/SUBMISSIONS.md` that had never been tested.

Interpretation of the whole ledger, including what each refutation rules out, lives in
`interpretations/kneemri/` — for the above, `2026-08-14_the-ruler-was-five-times-too-small.md`,
`2026-08-14_labels-or-pixels.md` and `2026-08-14_their-router-not-ours.md`. The build log is git history.
