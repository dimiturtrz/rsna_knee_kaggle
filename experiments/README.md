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

## What "seat value" means in a row below — and what it does NOT mean

Eighteen rows quote one. **Every one of them is the price of an ADDITIONAL member**, computed by
`MemberCurve.marginal`: the shipped roster plus this candidate, five members against four. That is the
right number for *does this member bring anything the roster lacks* and the wrong one for *should this
member ship*, because `comp/champion.json`'s FOUR MEMBERS principle says the best subset falls
monotonically past five. A positive add-price is a diversity signal, not a seat.

`MemberCurve.swap` (2026-08-20) prices the move the pin actually permits: hold the count at four,
report the best seat a candidate could take and whose it was. **The two orderings differ**, so the
distinction is not pedantry — over nine candidates against `calibspan_v1`:

| candidate | add (5th member) | best swap (held at 4) | displaces |
|---|---|---|---|
| `attend_dense` | +0.0010 (1st) | **+0.0010 (1st)** | `lr4e-5_ws` |
| `focal_dense` | +0.0007 (2nd) | +0.0003 (4th) | `lr4e-5_ws` |
| `band_span` | +0.0006 (3rd) | +0.0001 (6th) | `lr4e-5_ws` |
| `band_wide` | +0.0006 (4th) | +0.0004 (3rd) | `lr4e-5_ws` |
| `depth_regions_narrow` | +0.0005 (5th) | +0.0007 (2nd) | `lr4e-5_ws` |
| `depth_regions` | +0.0004 (6th) | +0.0001 (7th) | `lr4e-5_ws` |
| `band_density` | +0.0001 (7th) | +0.0002 (5th) | `lr4e-5_ws` |
| `res392_ws` | +0.0000 (8th) | −0.0002 (9th) | `lr4e-5_ws` |
| `lr1e-4_ws` | +0.0000 (9th) | −0.0001 (8th) | `lr4e-5_ws` |

The rows below are not restated: their numbers are correct as add-prices and re-running nine months of
them would replace a labelled number with an unlabelled one. Read them as diversity signals.

**The result in that table is the last column.** Every candidate's best seat is the SAME one, so
`lr4e-5_ws` is the roster's weakest member by marginal contribution and no other seat is contestable by
anything in the pool. The whole spread is 0.0012, a fifth of the 0.0062 floor — nine candidates the
ruler cannot tell apart, queued for one seat.

### That table is against a SUPERSEDED roster, and re-pricing it is the finding — 2026-08-20

`greedy4_v1` scored 0.930 and took the pin, so the roster the table above prices against no longer
ships. `lr4e-5_ws`, the seat every one of nine candidates was contesting, is not in the champion any
more. Re-run against the new pin (`python -m kneemri.kaggle.seat_value --candidates <processed>/*_s2026.pt`,
which reads the roster from `champion.json` rather than a typed list):

| candidate | best swap against `greedy4_v1` | displaces |
|---|---|---|
| `attend_dense` | **+0.0002** | `depth_regions_narrow` |
| `crop80_span` | +0.0001 | `band_span` |
| `band_wide` | −0.0001 | `band_span` |
| `focal_dense` | −0.0002 | `band_dense` |
| `calibrated_targets` | −0.0003 | `band_span` |
| `band_density` | −0.0003 | `band_span` |
| everything below | −0.0007 or worse | mostly `band_span` |

**THE MOUNTED POOL IS EXHAUSTED.** The best available move fell from +0.0010 to **+0.0002** — a
thirtieth of the noise floor — and only two of the whole pool are positive at all. Against the old
roster nine candidates were queued for one contestable seat; against this one there is effectively no
contestable seat. `band_span` is now the most-displaced incumbent, so it is the champion's weakest link
by marginal contribution, and nothing in the pool can beat it.

**CONSEQUENCE, and it is a spending rule as much as a research one.** No submission on the current pool
is worth a slot: the offline maximum has already been fired and it is the champion. The next board move
needs a member that does not exist yet, which is exactly what `focal_dense`'s row predicted when it
closed the decode/pooling family. It also re-reads the seat numbers quoted in the rows below one more
time: they are add-prices against a roster that has now been replaced twice, so read them as diversity
signal and re-run the tool before any of them is used as a decision.

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

**`# bead: knee-xxxx` — an arm may not be ARMED against an issue that never priced it.** The rule the
project keeps breaking is its own: price an experiment as wall-clock ÷ expected gain *before* running it.
The 140 mm cache build ran on a stated cost with no stated odds, which is the argument the rule exists to
refuse. So an arm's shell carries one header line naming its bead, `arm_check` reads that bead's text out
of `bd`, and it demands **both** marks — a `PRICE:` (what the card costs) and a `P(clears ...)` (the odds
it clears the 1.5% bar). A cost alone is not a price. knee-yuqf is the format both were read off.

It **refuses only under `--only`**, and the asymmetry is the same one `unstated` draws: seventy arms
predate the rule and nearly all of them already ran, so refusing the full sweep would fail every
invocation of this tool until they are back-filled — which is how a pre-flight check earns a
`--no-verify`. The rule exists to stop momentum at the moment of arming, and the run that arms an arm is
the run that names it. The full sweep prints one summary count instead.

**`--studies N` — the one check that DOES want the card, and wants it for two minutes.** `arm_check`
proves an arm's flags parse; it cannot prove the recipe they select survives contact with the corpus,
the encoder and the loss. `--studies 60` caps the corpus positions a run touches, keeping the fold's
train:holdout ratio so both halves of the loop execute, and reports a score that means nothing by
construction. Measured 2026-08-16: 48 train / 12 holdout, one epoch in **2 s**, checkpoint written and
reloaded — against the forty minutes a real fit holds the GPU to reach the same first assertion.

```bash
uv run python -m kneemri.training.train --labels "$LABELS" --studies 60 --epochs 1 $RECIPE --out /tmp/smoke.pt
```

**`--cache-studies N` — the smoke for a recipe whose cache does not exist yet.** `--studies` subsamples a
cache that is already on disk, so an arm on a reading nothing has decoded could not be smoked at all: the
only way to reach the first assertion was a two-hour full build. `--cache-studies N` decodes the first N
studies of the split, and it is in the cache NAME (`_first{N}`), so a partial decode can never be written
over the full arrays every member on disk was fitted on — they differ only in one axis length, with no
shape change to notice. Both CLIs take it, from `CacheFlags`, so the build and the run cannot disagree;
the run stamps it, so a member fitted on a truncated cache re-reads its own 60 studies rather than the
full 4407.

```bash
uv run python -m core.data.corpus.build_cache --cache-studies 60 --band-low 0.1 --band-high 0.9 --workers 20
uv run python -m kneemri.training.train --labels "$LABELS" --cache-studies 60 --studies 60 --epochs 1 \
  --band-low 0.1 --band-high 0.9 --out /tmp/smoke.pt
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

Re-measured 2026-08-18, after the landed-check rework and three new arms: **37 arms, 0 refused, 0
uncached, 0 broken**, in 14 minutes rather than the hour this section warns about — the per-arm cost is
startup (weights, cache map, compile), not the 2 s epoch, so it is cheap enough to run after any change
to `kneemri/training/`, not only before a queue.

**`--switches` — the axis the arm sweep cannot see, because it only runs the paths the arms take.**
Diffing `TrainingOptions` against every `--flag` under `experiments/**.sh` on 2026-08-18 found five real
switches that were landed-checked, stamped and never once RUN: `--pool cls_mean_focal`,
`--anatomical-prior`, `--normalise-by-count`, `--tie` and `--head-lr`. A break in one of those surfaces
at a queue slot, after the card is held. `kneemri/experiments/switch_coverage.py` now holds one default
recipe per uncovered switch and a written reason for every option that is not a method at all (`--seed`,
`--folds`, `--paths`, …); a unit test asserts the partition is TOTAL, so a new knob has to be armed,
covered or written off before it can merge. `--tie last` earns the integration: the router is a cache
axis, so the truncated decode `arm_check` already does is the only way to smoke it.

```bash
uv run python -m kneemri.experiments.arm_check --switches --smoke /tmp/smoke_out
```

Measured 2026-08-18: all five reach a written checkpoint at 60 studies / 1 epoch.

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

## Three rules encoded in `lib.sh` rather than remembered

- **`retired "<why>"` refuses a refuted arm.** An arm its gate declined stays committed — it is the
  record of what was tried and why it lost — but it must not be runnable, and `multiscale.sh` was
  runnable for a build priced at +0.0016. The declaration exits 1 under `lib.sh` (a queue that reaches
  a retired arm should say so) and exits 0 under `stub_lib.sh`, so `arm_check` captures no invocation
  and stops reporting a cache nobody should build. A check that reports work you have decided against
  is a check people learn to skim.
- **`wait_for` keys on a completion marker.** A gate that watches an output path fires when the
  producer *starts*. That once queued an arm against a 59.7 GB cache still mostly zeros; it would not
  have crashed, it would have produced a plausible wrong number.
- **`run_train` tees before it greps.** Arms used to pipe straight into `grep`, so any traceback whose
  wording missed the filter vanished and the run looked merely empty.
- **A NEW ARM SHOULD NOT NAME `--vram-gb` AT ALL** (2026-08-19, knee-ylp0). The reserved peak is a
  function of the slice count and nothing else — 12 slices reserve 16.24 GB, 16 reserve 21.57, 20
  reserve 26.92, and `band_density`, `band_full` and `depth_regions_span` land on their slice count's
  value to the centibyte despite differing in band, pooling and region count. `DeviceBudget` now
  derives the cap from `--slices` (1.335 GB a slice, plus a slice of slack), so a hand-typed number is
  a THIRD statement of a fact the recipe already makes — and the third one goes stale silently:
  thirteen arms overstate by 1 to 12 GB, `depth_regions_narrow` because it was `depth_regions` with
  the slices dropped and the 20-slice cap copied along. Slack is not free once the other project holds
  the card: `await_headroom` blocks until `free >= vram_gb`, so an overstated ask makes a run poll for
  memory it will never allocate, burn the wait bound, and start into the PCIe crawl. Name a number
  only to *contradict* the derivation, and say in the arm why.
  **NO ARM NAMES ONE NOW** (2026-08-19): `arm_check` grew an OVERSTATED report, it named 37 of 50 —
  not thirteen — and the count is the finding. The 20 / 26 / 28 the arms had converged on were picked
  as fractions of the CARD, months before a cap was derived from the RECIPE, so 26 on a 12-slice arm
  is 8.4 GB over need for the same reason 28 is 10.4. There was no honest threshold between "copied"
  and "rounded", which is what the first version of that report tried to legislate; the flag was
  simply obsolete everywhere, and it is now stripped from all 36 arms that carried it. The three
  headers that argued for a specific number keep the MEASUREMENT and drop the prescription.

## Ledger — fold 0, seed 2026, 24 epochs, one-cycle, weight floor, ranking on

Control for the small arms is `rank24_s2026` = **0.8272**; for base arms, `rankbase24_s2026` = **0.8283**.
The noise floor on this ruler is **0.0062**, and this project's bar for a *result* is ~1.5% absolute —
so everything below is **directional**, and only the cumulative total clears the bar.

**The head-side rows have a second home, `head_axes.jsonl`**, which pairs each head-only axis with its
control so the frozen-encoder proxy (`python -m kneemri.training.head_trial --correlate`) can rerun it in
seconds and be checked against what the pixel arm measured. It scores only pairs this ruler RESOLVES —
today exactly **one** (the anatomical prior, −0.0076), because the other two head-side pairs are inside the
floor and a third is inexpressible over cached features. So the proxy is not yet shown to rank anything,
and adding a pair there means running the pixel arm, not editing the file.

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
| **`resolution`** | **336 → 392 px, 448 cache — AND encoder rate 8e-6 → 4e-5** | **0.8523** | **+0.0066** | **the RATE, not the pixels — see 2026-08-18 below** |
| `resolution` 392 @ 8e-6 | 392 px at the control's own encoder rate | 0.8459 | +0.0002 | resolution alone is nothing (knee-a8x) |
| **`rate_wholestack`** | **encoder rate 8e-6 → 4e-5, held at 336 px** | **0.8525** | **+0.0068** | **clears the floor — and matches res392 with no 448 cache** |
| `rate_1e-4` | encoder rate 1e-4, same recipe | 0.8490 | +0.0033 | inside the floor of 4e-5 — the plateau spans 4e-5..1e-4 |
| `rate_2e-4` | encoder rate 2e-4, same recipe | 0.7391 | −0.1066 | past the edge: loss RISES epoch 0→1 and plateaus at 1.07 |
| `slice_attention` | attention over slices, not max | 0.8473 | +0.0016 | **MISLABELLED — IT MEASURED THE MEAN (knee-7gyw, found 2026-08-21).** `StudyModel.parameter_groups` built its head group as `self.head.parameters()`, and the aggregator is a SIBLING of the head, so the gate's tensors were in no param group and the optimiser never touched them. `SliceAttention`'s score layer is zero-initialised on purpose so that an untrained gate is exactly the arithmetic mean over slices — so this row is **mean pooling vs max pooling**, not attention vs max. The checkpoint and the 0.8473 are real; only the name is wrong. **RESOLVED 2026-08-21 by `attend_true`, the row below**, which ran the same recipe from the fixed snapshot and is the number this row was supposed to be |
| `attend_true` | attention over slices with a gate the optimiser actually reaches — `slice_attention`'s recipe re-run after knee-7gyw | 0.8476 | +0.0019 vs max, **+0.0003 vs the mean** | **THE POOLING OPERATOR IS NOT A LEVER, and this is the first honest reading of that axis.** Plumbing verified BEFORE the verdict, which is the whole point of the re-run: `slices.gate.3.weight` reads absmax **8.596939e-02** in the epoch-5 checkpoint against exactly **0.000e+00** in `attend_dense`, so the gate trained. Converged and overfit — peak 0.8476 at epoch 5, then 0.8441 / 0.8451 / 0.8438 / 0.8428 / 0.8434 / 0.8432 while the loss falls 0.8031 → 0.6405. **Three operators on one recipe now span 0.0019**: elementwise max 0.8457, unweighted mean 0.8473, trained 64-wide gate 0.8476 — under a third of the 0.0062 floor, and the ordering is monotone but every gap is noise. Seat value **+0.0004** into the pinned five, 0 of 1 clearing +0.0015. **WHAT IT DOES NOT RULE OUT:** all three operators are permutation-blind, so slice ORDER is untested (`DepthAttention`, knee-tqn) — but the prior for it drops, because this row read against `knee-ezz` (20 cached slices vs 12 under max: **−0.0001**) and `extended_slots` (nine slots, 83,297 unseen slices: **+0.0002**) says the slice-AGGREGATION STAGE is saturated from both ends at once: neither more inputs to it nor a better rule for combining them moves the number. The limit is upstream, in what one slice's feature vector contains |
| `depth_true` | `DepthAttention` — the ORDER-AWARE aggregator, slices mixed with their through-plane neighbours before the gate, re-run from the fixed snapshot after knee-7gyw | 0.8461 | **−0.0015 vs `attend_true`**, −0.0012 vs the mean, +0.0004 vs max | **SLICE ORDER IS NOT A LEVER EITHER, and this closes knee-tqn — the clause `attend_true` left open.** Peak 0.8461 at epoch 6, then 0.8431 / 0.8440 / 0.8439 / 0.8436 while the loss falls to 0.6387, the same converge-and-overfit shape as the row above. **Four operators on one recipe now span 0.0019** — max 0.8457, order-aware gate 0.8461, mean 0.8473, permutation-blind gate 0.8476 — and the order-aware one lands SECOND-WORST, so the extra structure is not merely flat but slightly costly. Seat value **−0.0002**, negative, which the board charges at about 3× → roughly −0.0005 (knee-c0u9): not a free look, so it does not mount. **The frozen-feature proxy called both signs in advance** — trained gate +0.0027, through-plane neighbour mixing −0.0007 — and the pixel model reproduced them, which is the first time that proxy has been checked against a full arm on this axis and makes it usable to gate future pooling work at ~2 min instead of ~45. **WHAT IT DOES NOT RULE OUT:** through-plane mixing at the FEATURE level is what died; a 3D encoder that sees the volume before any per-slice feature exists is a different mechanism and untouched. **What it confirms:** `attend_true`'s reading that the limit is upstream, in what one slice's feature vector contains — the aggregation stage is now saturated from three directions (more slices −0.0001, more slots +0.0002, every combining rule within 0.0019) |
| `encoder_adapt_thawed` | the SSL-adapted encoder with **every block trainable** (`--unfreeze-last 12`) — the "unfreeze deeper" removal `encoder_adapt` named | 0.7162 | **−0.1416 vs `band_dense`**, +0.0260 vs the frozen case | **the third collapse, and it inverts the hypothesis it was built to test** — see below. Found unrecorded by `ledger_check`. No seat |
| `slotmean` | pooling across SLOTS by mean instead of attention (`--slot-pooling mean`) | 0.8406 | −0.0051 vs `wholestack_s2026` | **found unledgered 2026-08-19** — a full 12-epoch arm on an axis this table records nowhere else, sitting on disk since it ran. Directional against attention, not a refutation, and the reasons are on the comparison rather than in it: the reading is on the **UNPAIRED** bound because the control kept no per-study predictions, and `run_comparison` reports the control as predating the `slot_pooling` option entirely — so "the control ran attend" is read off the era's default, not off its record. No label clears its interval; the largest rows are Lateral Meniscus −0.0187 and Baker's −0.0162, both regional and focal moving together (focal −0.0051, regional −0.0052), which is a uniformly slightly worse model rather than a redistribution. Attention over slots stays the default it already was; what this row buys is that the axis is now MEASURED once instead of assumed |
| `wholestack_folds` 2 | same recipe, fold 2 | **0.8518** | — | best single member; epoch-5 partial |
| `slices16_ws` | 16 cached slices, whole-stack | 0.8452 | −0.0005 | **refuted**, 33% slower for nothing. TWO runs share this name: the one quoted here (epochs 0–11, peak 0.8452 at epoch 11, and the checkpoint on disk is its epoch 11 at 0.84518) and a later re-run that was KILLED at epoch 8 having peaked at **0.8411**. Both sets of epochs are in one metrics file because runs used to append; every reader that took the last group quoted the dead run's 0.8411 for a member that scores 0.8452 (knee-1m2g). `RunMetrics.begin` now rotates a previous run's epochs aside, so no file written after 2026-08-19 can hold two runs |
| **`band_wide`** | slice band 0.20–0.80 → **0.10–0.90**, 16 slices at the control's mm-per-slice | **0.8553** | **+0.0028 vs `rate_wholestack`** | best single member yet; directional on the mean, and the per-label pattern is the finding |
| **`band_dense`** | the wide band at restored central density: 0.10–0.90, **20 slices** | **0.8578** | **+0.0053 vs `rate_wholestack`**, +0.0025 vs `band_wide` | best single member; **PF OA and Synovitis resolve** — first per-label rows in this family whose intervals clear zero. Dilution confirmed and symmetric: the centre comes back, the collaterals `band_wide` bought largely go. Pairs with `band_wide` at 0.8633 |
| **`depth_regions`** | region pooling (`--slice-regions 4`) on the wide dense band | 0.8549 | −0.0029 vs `band_dense` | mean is a non-result under the 0.0062 floor; the per-label read is the finding. Two labels clear their interval and disagree: **PF OA −0.0193\***, which is `band_wide`'s −0.0194 to four decimals (the patellofemoral loss is the BAND, not the pooling), and **Lateral Meniscus +0.0165\*** against `band_wide`'s +0.0007 (the only thing attributable to the pooling). Its own written prediction — centre holds, periphery recovers — is wrong on both clauses. [interpretation](../interpretations/kneemri/2026-08-19_depth_regions.md); `depth_regions_narrow` is the single-variable follow-up |
| **`depth_regions_narrow`** | the same region pooling on the band every OTHER member reads (0.20-0.80, 12 slices) | 0.8554 | +0.0029 vs `lr4e-5_ws` | the transfer question, ANSWERED IN TWO PARTS. The PF OA loss was the BAND: on a narrow band it does not merely vanish but reverses, **+0.0067**, confirming `depth_regions`' decomposition. The Lateral Meniscus gain does NOT transfer at strength: **+0.0056 [-0.0112, +0.0206]** here against +0.0165\* there, and read directly against `depth_regions` it is **-0.0112**. Both arms are flat on the mean, which is this arm's own pre-registered refutation condition for the pooled-budget argument. Seat value **+0.0006** against the shipped four. See below |
| **`band_density`** | the fourth corner of the span × density square: 20 slices at the CONTROL's 0.20–0.80 span, so density moves alone | 0.8540 | **+0.0014 vs `lr4e-5_ws`** | ~~**the additive decomposition of `band_dense` is refuted.**~~ **CORRECTED 2026-08-19 by `band_span`, the cell that did not exist when this was written:** on the MEAN the square is additive to the fourth decimal (+0.0039 span, +0.0014 density, sum +0.0053 against `band_dense`'s observed +0.0053). Non-additivity was inferred from the per-label signature and asserted of the mean, which was a claim three cells could not support. What survives, and is the better finding, is below: the signature is not additive because the two levers move DIFFERENT LABEL POPULATIONS. Density alone buys +0.0014 — a quarter of the floor — and the per-label signature is not a weaker version of `band_dense`'s, it is a different one. `band_dense` resolved **PF OA +0.0192\*** and **Synovitis +0.0218\***; holding the span, those two rows *fall* (PF OA −0.0076, Synovitis −0.0140) and the two rows that clear their intervals are **Lateral Meniscus +0.0191\*** and **ACL +0.0179\***. So the central/OA gain is the SPAN or the span×density interaction, never the extra slices, and the arm's own pre-registered refutation condition — a collateral gain — is what landed. Consistent with the fixed representational budget: slices added inside an unchanged span redistribute toward focal ligament and meniscus rows at the centre's expense. Seat value **+0.0002** against the shipped four, the smallest any same-split candidate has posted. `band_span` (0.10–0.90 at 12 slices) is the matching span-alone corner |
| **`band_span`** | the last corner of the span × density square: 0.10–0.90 at the CONTROL's 12 slices, so span moves alone | **0.8565** | **+0.0039 vs `lr4e-5_ws`** | **the square closes, and it is ADDITIVE ON THE MEAN** — span +0.0039 at 12 slices and +0.0038 at 20, density +0.0014 at 0.20–0.80 and +0.0013 at 0.10–0.90, interaction ≈0.0001. Each main effect reproduces at both levels of the other, which is what a set of sub-floor cells can say and a single sub-floor cell cannot; it retires the fixed-budget model that motivated the square and corrects `band_density`'s row above. **The pre-registered clause is REFUTED BY INVERSION, not by margin**: the mean cleared its 0.8560 threshold, and then every predicted riser fell (MCL −0.0039, Lateral Meniscus −0.0054, Lateral OA −0.0038) while the predicted fallers rose (PF OA +0.0028, Synovitis +0.0024). The replacement mechanism comes from the tool's own structure-scale line and is the better finding: **widening the span buys REGIONAL findings +0.0079 against FOCAL +0.0011** — ACL +0.0148, Contusion +0.0108, Baker's +0.0105, Effusion +0.0079 — while adding slices INSIDE a span does the opposite (focal +0.0036, regional −0.0019, starring Synovitis +0.0194\* and PF OA +0.0164\*). The two levers are orthogonal in their label populations as well as in their means, which is why the SIGNATURE is not additive though the mean is. Directional only: no label clears its paired interval (half-widths reach ±0.0411) and +0.0039 is under the 0.0062 floor. Seat value **+0.0014** against the shipped four — the largest any same-split candidate has posted, 7× `band_density`'s +0.0002 — and it reads a pixel group no seated member reads, so it adds a decode group, the axis measured to transfer at 2.4× (knee-7d1). `band_full` (0.00–1.00 at 12 slices) tests whether the coverage mechanism keeps paying |
| **`band_full`** | the third point on the span axis: 0.00–1.00 at 12 slices, the whole stack, so span moves alone against `band_span` | 0.8427 | **−0.0138 vs `band_span`** | **the span axis has an INTERIOR OPTIMUM, and this arm is the point that proves it.** Three points now: 0.20–0.80 = 0.8525, 0.10–0.90 = **0.8565**, 0.00–1.00 = 0.8427. Coverage does not keep paying — the outer tenths cost 2.2× the 0.0062 floor, which is a real negative rather than scatter. Flags verified in the recorded recipe (`band_low 0.0`, `band_high 1.0`, matched to `band_span` on every other field). **The pre-registered refutation condition FIRED on both of its halves**: the mean fell below 0.8540 (0.8427) AND the regional group went negative against `band_span` (regional −0.0185 vs focal −0.0104), with Baker's −0.0247\*, ACL −0.0236\*, Contusion −0.0216\* each clearing their own paired interval. So the outer tenths are ACTIVE DISTRACTION, not weak signal, and the band is a FILTER rather than a budget. **But the clause's prescribed next move — a band NARROWER than 0.20–0.80 — does not follow, and I am not queueing one.** That clause was written when the axis had two points; the third makes the ordering 0.10–0.90 > 0.20–0.80 > 0.00–1.00, an interior peak, and narrowing past 0.20–0.80 walks AWAY from it. A fired clause is evidence, not an instruction — the filter reading is what survives, the direction it was written to imply is refuted by the same three points. Seat value **−0.0005** against the shipped four (roster 0.8648 → 0.8643), so no seat either. The axis is closed: 0.10–0.90 is the operating point every later band arm should read from |
| **`depth_regions_span`** | `band_span`'s 0.10–0.90 span stacked ON TOP of `depth_regions_narrow`'s four-region depth pooling, so the two "where along the stack" levers run together | 0.8505 | **−0.0049 vs `depth_regions_narrow`** | **span and depth-region pooling are SUBSTITUTES, NOT COMPLEMENTS.** Span bought +0.0039 at `slice_regions=1`; on top of `slice_regions=4` the same widening costs −0.0049. Flag verified in the recorded recipe: `band [0.1, 0.9]` against the control's `[0.2, 0.8]`, with every other config key identical (slice_regions 4, slices 12, crop_mm 130, centre image, backbone_lr 4e-05, slice_pooling max, unfreeze_last 6, whole_stack true, bf16, 336 px) — and `band_span` differs from this arm only by `slice_regions 1`, so the square is closed on both diagonals. **The pre-registered refutation condition FIRES on both halves**: mean 0.8505 ≤ 0.8554, AND the regional group is negative on 3 of 4 (group −0.0081 vs focal −0.0026) — ACL −0.0123, Contusion −0.0291, Effusion −0.0081 against only Baker's +0.0029. The predicted signature is absent, not merely small. Converged at epoch 9 (0.8355 → 0.8486 → 0.8502 → **0.8505** → 0.8496 → 0.8491), 16.24 GB peak every epoch — a fourth independent confirmation of the 1.335 GB/slice + 0.22 rule. Honest size: −0.0049 sits INSIDE the 0.0062 floor, so this is "no gain with the predicted mechanism absent", not a large loss, and no label clears its paired interval (half-widths to ±0.0624). Seat value **+0.0009** against the shipped four (roster 0.8648 → 0.8657) — positive but a seventh of `band_span`'s own +0.0014, which is what a member that duplicates a seated decode axis rather than adding one looks like. Same shape as `band_full`'s filter finding: both levers answer WHERE ALONG THE STACK signal lives, so the second one applied adds distraction rather than coverage. **THE ACCIDENTAL REPLICATE LANDED, and it says which numbers in this whole table are comparable (knee-9g15).** An identical recipe at the identical seed re-entered `run_train` under the same `--out`, so run 1 is preserved as `depth_regions_span_s2026_run1.*` and both recorded recipes match on every key including `seed 2026` and `split_seed 2026`. The curves are NOT the same: run 2 read 0.7455 / 0.7696 / 0.8011 / 0.8226 / 0.8355 / 0.8455 / 0.8474 / 0.8479 / 0.8487 against run 1's 0.7457 / 0.7708 / 0.8106 / 0.8246 / 0.8355 / 0.8375 / 0.8486 / 0.8486 / 0.8502. Matched-epoch deltas reach **−0.0095 at epoch 2 and +0.0080 at epoch 5**, while the best-so-far through epoch 8 differs by only **0.0015**. Run 2 was truncated at 9 of 12 epochs by its own wall-clock `timeout`, not by a fault, so the 12-epoch bests are unpaired. Two consequences, and the first is the load-bearing one: **a single epoch's AUC is not a comparable number** — bf16 + `compile` + non-deterministic kernels move one epoch by up to 1.5× the 0.0062 sampling floor, so any read that sets one arm's epoch against another arm's epoch is reading noise, and only best-over-schedule may be quoted. **The best IS stable**: 0.0015 is a quarter of the sampling floor, so the floor stays 0.0062 and gains no material nondeterminism term. This arm's −0.0049 therefore stands as written |
| **`crop80_span`** | the 80 mm tissue-centred crop read at the settled 0.10–0.90 span, so the in-plane and through-plane coverage levers run together | 0.8466 | **−0.0099 vs `band_span`** | **THE SPAN'S REGIONAL GAIN DOES NOT REPRODUCE AT 80 mm, and the pre-registered refutation FIRES.** Flag verified in the recorded recipe: `run_comparison` names `crop_mm 130.0→80.0, centre image→tissue` and NOTHING else, so this is the single-variable pair the arm was written to be — same 12 slices, same 0.10–0.90 band, same 4e-5, same 12 epochs, 336 px. The clause required the regional group (ACL, Contusion, Baker's, Effusion) to reproduce at 80 mm **and** a positive seat value; refutation was that group coming out flat or negative. **It comes out negative on all four**, and three of them clear this pair's own paired half-widths (±0.0314): ACL −0.0263\*, Contusion −0.0198\*, Baker's −0.0164\*, Effusion −0.0079. By structure scale, regional −0.0104 against focal −0.0095 — the predicted signature is absent, not merely small. The mean landing at 0.8466 was itself pre-registered (predicted 0.8420–0.8500) so it is not the news; the group is. **THE MECHANISM READS AS ANTAGONISM BETWEEN THE TWO COVERAGE AXES.** The widened band buys regional findings by covering more anatomy ALONG the stack; an 80 mm crop removes anatomy ACROSS the plane, and the two biggest losses are exactly the structures at the field's margins — MCL −0.0265\* and ACL −0.0263\*, the collaterals and cruciate the band widening had bought. So in-plane extent and through-plane span are not independent coverage knobs: cropping in gives back what widening out paid for. That is the same substitution shape as `depth_regions_span`, on a different pair of axes, and it is the third time this family has produced it. Converged with the best at epoch 6 (0.8466) and a monotone decline after — 0.8436 / 0.8412 / 0.8397 / 0.8401 — so the best is not a truncation artefact. 16.24 GB peak every epoch, a **fifth** independent confirmation of the 1.335 GB/slice + 0.22 rule, and the first on an arm carrying no `--vram-gb` at all. Seat value **+0.0010** against the shipped four (`calibspan_v1`, roster 0.8670 → 0.8680; it read +0.0015 against `band_v1`'s 0.8648 before slot 4 replaced the champion, and the seat shrank when the roster got better) — positive, but it is an ADD as a fifth member, and `comp/champion.json`'s own FOUR MEMBERS principle records the best subset falling monotonically past five. A fifth seat is not a seat; a swap would have to beat a seated member, and this one loses to its own control by −0.0099. Honest size: +0.0015 is a quarter of the 0.0062 floor, directional only. **CONSEQUENCE:** the crop/centre axis knee-nas5 wanted for roster diversity does not pay at the settled operating point, and the decode-geometry family (span, density, crop, depth regions) is now closed on all four of its axes at about +0.005 total — a twelfth of the 0.054 frontier gap (knee-jk3) |
| **`calib_span`** | the label calibration measured on the recipe we ship: `band_span`'s 0.10–0.90 span at 12 slices with `--calibrated-targets` added and nothing else | 0.8545 | **−0.0020 vs `band_span`** | **THE CLAUSE DID NOT MISS, IT INVERTED — and that is the finding.** Flag verified in the recorded recipe: `run_comparison` names `calibrated_targets False→True` and nothing else, the single-variable pair the arm was written to be. The prediction, inherited unchanged from `calibrated_targets` so it could not be re-fitted, was that the gain CONCENTRATES where miscalibration is largest — MCL (495/219/53) and Effusion (1599/910/157), the two labels most skewed toward the 0.68 level — while PF OA (543/658/790), skewed high and nearly calibrated already, barely moves. **Every clause came back backwards**: Effusion **−0.0212\***, MCL +0.0029 (flat), and PF OA **+0.0170\*** is the largest positive row in the arm. The label predicted to have nothing to gain gained most; the two predicted to gain most are flat and significantly negative. The written refutation condition was a mean at or below control WITH MCL and Effusion flat; the mean fell (−0.0020) and MCL is flat, so it fires — but in a stronger form than it was written for, because Effusion is not flat, it is the worst row on the board. **MECHANISM: the remap is not adding truth at this operating point, it is moving mass off the low-grade positives it was built to help.** By structure scale, focal +0.0037 against regional −0.0099 — the exact mirror of what the span bought (regional +0.0079, focal +0.0011), so the calibration spends its gain on the centre and pays for it with the ends the widened band had just purchased. That is the SUBSTITUTION shape a fourth time (`depth_regions_span`, `crop80_span`, `band_full`, now this), and the first time it has appeared between a LABEL lever and a DECODE one rather than between two decode levers. **THE STALE-OPERATING-POINT TRAP IS THE TRANSFERABLE LESSON.** `calibrated_targets` scored +0.0038 against `wsfold0_e6` on 2026-08-16 at the default rate and the 0.20–0.80 band, and it is a member of the 0.928 champion package on the strength of that number. Re-measured on the recipe the project actually ships it is NEGATIVE. A lever's sign is a property of the operating point it was measured at, not of the lever, and every directional result in this ledger taken before the rate and span settled is now suspect in the same way. Converged with the best at epoch 9 (0.8470 / 0.8542 / 0.8527 / 0.8540 / **0.8545** / 0.8537 / 0.8532), so the best is not a truncation artefact; 16.24 GB peak every epoch, a **seventh** confirmation of the 1.335 GB/slice + 0.22 rule and the second on an arm naming no `--vram-gb`. Honest size: −0.0020 is under a third of the 0.0062 floor, so the MEAN is a non-result and the per-label inversion is what carries the row — three rows do clear this pair's own paired half-widths (±0.0331): Effusion −0.0212\*, PF OA +0.0170\*, Baker's −0.0167\*. Seat value **+0.0022** against the shipped four (roster 0.8648 → 0.8670), the largest any same-split candidate has posted — but it is an ADD as a fifth member, and `comp/champion.json`'s FOUR MEMBERS principle records the best subset falling monotonically past five, so a fifth seat is not a seat and a swap is what would have to be shown. **CONSEQUENCE:** `calib_dense` (knee-ep74), written to be the span × density × calibration corner, is dead by its own pre-registered gate and was not launched — a calibration that is negative on the span corner cannot be positive on the denser one for any reason this evidence supports |
| **`attend_dense`** | `band_dense`'s recipe with `--slice-pooling attend` in place of `max` — a learned 64-wide gate over the pooled slices instead of the elementwise maximum | 0.8569 | **−0.0009 vs `band_dense`** | **NEITHER CLAUSE FIRED, AND THE THIRD OUTCOME IS THE FINDING: THE TRADE STOPS AND NOTHING IS RECOVERED.** Flag verified in the CHECKPOINT config, not just the recipe: `slice_pooling max→attend` with `slices 20`, `band [0.1, 0.9]`, `pool cls_mean` and `slot_pooling attend` identical across the pair, so this is the single-variable pair the arm was written to be. The prediction was that attention BEATS `band_dense` **and** both structure groups come back non-negative; the refutation condition was another trade — one group up, one down, whatever the mean does. By structure scale the arm posts **focal −0.0006 against regional −0.0016**: both negative, neither clearing anything, no trade in either direction. So the arm landed in the case it did not enumerate, and that case is more informative than either. **MECHANISM: the elementwise max is not the binding constraint.** Three arms (`band_span`, `band_density`, `calib_span`) produced the one-group-gains-one-pays shape, and the max was the operator that most plausibly caused it — it keeps whichever slice shouted loudest per feature and discards which slice that was, so two structures competing for a feature cannot both be represented. Replacing it with a learned gate makes the trade disappear **and buys nothing**: −0.0009 is a seventh of the 0.0062 floor, and the two group deltas are smaller still. A fixed budget whose allocator is swapped for a strictly more expressive one and does not move is a budget bounded somewhere else. That **exonerates the pooling operator without retiring the re-allocation reading** — the substitution shape stands as a description of the decode family, but its cause is not the max, so the remaining suspects are capacity and supervision (different beads), exactly the branch the arm named as its consequence. The +0.0016 the same lever bought at 12 slices in a narrow band does not scale with how much is pooled, which was the arm's stated reason for re-measuring it; that reason is now spent. Converged with the best at epoch 8 (0.8530 / 0.8519 / **0.8569** / 0.8549 / 0.8542 / 0.8544), so the best is not a truncation artefact. 26.92 GB reserved / 24.93 GB allocated on every one of twelve epochs, byte-identical to `band_dense`'s own run — the gate's parameters and activations are invisible against 20 slices, so the operator swap is free in memory as well as in score. Seat value **+0.0010** against the shipped four (`calibspan_v1`, roster 0.8670 → 0.8680; it read +0.0015 against `band_v1`'s 0.8648 before slot 4 replaced the champion, and the seat shrank when the roster got better), a quarter of the floor and directional only — and it is an ADD as a fifth member, which `comp/champion.json`'s FOUR MEMBERS principle records as not a seat. **CONSEQUENCE:** the pooling-operator cell of the decode family is closed; `focal_dense` is the queued follow-up on the capacity branch rather than another allocator. **CORRECTION 2026-08-21 (knee-7gyw): THE LEARNED GATE NEVER LEARNED.** `parameter_groups` built its head group as `self.head.parameters()` and the aggregator is a SIBLING of the head, so the gate's six tensors were in no param group; a checkpoint dumped after a full epoch has `slices.pool.gate.3.weight` at exactly `0.000e+00`. The gate is zero-initialised so that an untrained gate is the arithmetic MEAN over slices, so this arm compared **mean vs max**, not attention vs max. The −0.0009 stands as a number and the checkpoint is a valid member; what does NOT stand is the mechanism sentence — "a strictly more expressive allocator does not move the score" was never tested, because the allocator that ran was a fixed mean. **The pooling-operator cell REOPENS**, and with it the part of the fixed-budget refutation that rested on this arm; `focal_dense`'s width axis is unaffected and still reads negative |
| **`focal_dense`** | `band_dense`'s recipe with `--pool cls_mean_focal` in place of `cls_mean` — a third part concatenated to the pooled vector (a focal tail over the token grid) rather than a different rule for combining the existing parts | 0.8534 | **−0.0044 vs `band_dense`** | **NEITHER CLAUSE FIRED LITERALLY, AND THE ARM'S OWN CONSEQUENCE LANDS ANYWAY — HARDER THAN IT ASKED FOR.** Flag verified in the CHECKPOINT config, not just the recorded recipe: `pool: cls_mean_focal` with `slices 20`, `slice_pooling max`, `whole_stack true`, `backbone_lr 4e-05` identical to the control, so this is the single-variable pair the arm was written to be. By structure scale at each arm's own best epoch (focal e9, control e8): **focal +0.0001, regional −0.0114.** The prediction required the focal group to gain ABSOLUTELY; +0.0001 is a sixtieth of the noise floor, which is flat, not a gain. The refutation condition named another trade — focal up, regional down — and this is not that either: nothing was traded because nothing was bought. Widening the pooled vector cost the regional group 0.0114 and returned the focal group nothing. **MECHANISM: the fixed-budget reading is now refuted on BOTH of its axes, which is exactly the consequence this arm pre-registered.** *(CORRECTION 2026-08-21, knee-7gyw: only the WIDTH axis — this arm — survives. The other half rested on `attend_dense`, whose gate was outside every optimiser group and therefore ran as a fixed mean, so the allocator axis was never measured and reopens.)* `attend_dense` swapped the pooling OPERATOR and the trade stopped while nothing was recovered; this arm widens the pooled vector's WIDTH and the focal group still does not gain. A budget that does not grow when its allocator is made strictly more expressive AND does not grow when the vector is made wider is not a budget being mis-allocated — re-allocation is retired as an explanation of the `band_span` / `band_density` / `calib_span` substitution shape, and the bottleneck moves to supervision or to the encoder, which are different beads and different arms. **THE ONE THING THAT WENT AS PREDICTED, and it is worth keeping.** The named riser was Lateral Meniscus, the largest model-side gap in the corpus (−0.1932, knee-dtk) and a 1–3 mm finding: it rose 0.846 → **0.856**. The named second, Fracture, went the other way (0.809 → 0.796). And the focal group re-allocated INTERNALLY as well — ACL +0.006, Lateral OA +0.005, Lateral Meniscus +0.010 against MCL −0.008, Medial Meniscus −0.006, Medial OA −0.003, PF OA −0.003 — summing to nothing. So the focal tail does reach the fine-scale finding it was argued to reach; it pays for it inside its own group and then again, four times over, in the diffuse one. That is a real observation about WHERE the tail spends attention, and it does not license the tail as a lever. Converged with the best at epoch 9 (0.8405 / 0.8473 / 0.8532 / 0.8504 / 0.8517 / **0.8534** / 0.8521 / 0.8521), flat over the last three, so the best is not a truncation artefact. 27.51 GB reserved / 24.93 GB allocated, unchanged from `band_dense` — the tail is a head-side concatenation and the encoder cost does not move, as the arm predicted. Seat value **+0.0007** against the shipped four (`calibspan_v1`, roster 0.8670 → 0.8677; it read +0.0015 against `band_v1`'s 0.8648 before slot 4 replaced the champion), below `attend_dense`'s +0.0010 under the same roster and a quarter of the floor: an ADD as a fifth member, which `comp/champion.json`'s FOUR MEMBERS principle records as not a seat, and which tonight's board read (five entries, no relation between offline edge and board score inside 0.0062) says is not worth a submission slot either. **CONSEQUENCE:** the decode/pooling family is closed on all its axes; `band_dense_long` is already on the card asking the REGIMEN question, and after it the evidence asks for a member that does not exist yet rather than another knob on this one |
| **`encoder_layer 9`** | the head reads ViT block 9 of 12 instead of the last one | 0.8230 | -0.0227 vs `wsfold0_e6` | 3.7x the floor, converged (flat over its last three epochs), flag verified in the recipe. Seat value **-0.0034**, the largest negative any same-split candidate has posted, so diversity does not rescue it. The linear-probe convention turns out to be right for this head. See below |
| **`focal_pool`** | pooled vector gains a third part: the mean of the top eighth of each channel over the patch grid | 0.8432 | -0.0025 vs `whole_stack` | Flat (inside the floor). Seat value -0.0012. Pre-registered clause half-holds: focal group -0.0004 vs diffuse -0.0065, the predicted direction, but carried by the diffuse group losing rather than the focal group gaining. Fracture +0.020, directional only (one crossing in twelve). See below |
| **`unfreeze_4`** | four of the encoder's twelve blocks adapt instead of six — the +25% throughput arm, run at length | 0.8320 | −0.0137 vs `wsfold0_e6` | **refuted**, 2.2x the floor, flag verified (`unfreeze_last 6→4`). Peaked at epoch 6 and flat for five epochs after, so it is converged, not short. Eleven of twelve labels negative, six clearing their interval; by structure scale **focal −0.0170 vs regional −0.0091**, which is the arm's own stated risk — the fine findings are the ones that needed the deeper blocks. Seat value **−0.0017**, so diversity does not rescue it either. The default stays six; the speed was real and is not worth 0.0137. See below |
| `extended_slots` | nine slots, +1,139 coronal series | 0.8459 | +0.0002 | **refuted** on the mean |
| `intensity_augment` | gamma/blur/contrast/noise at 392 | 0.8459 | −0.0064 vs res392 | **refuted**, behind at every epoch |
| **`aug_long`** | the same intensity block on the 24-epoch schedule `band_dense_long` proved long enough to overfit — the setting the old row could not see | 0.8511 | **−0.0076 vs `band_dense_long`** | **refuted, both clauses literally** (peak epoch 9 ≤ 10, and 1.2x the floor below control). Flag verified in the recorded recipe: `intensity_aug False→True` with all 46 other keys identical. **And the refutation is the informative kind — the regulariser demonstrably FIRED and the overfitting did not care:** train loss higher at every epoch past 4 (0.463 vs 0.454 at the end), so the fit was made harder, while the peak-to-finish decline is **unchanged at −0.0169 against the control's −0.0182**. The peak did move late (6 → 9), which is the only part of the prediction that held, and it bought nothing. The predicted regional concentration is refuted too: focal −0.0084 vs regional −0.0067, both down, the diffuse group slightly LESS hurt. Seat value **+0.0007** as a fifth and **+0.0000** as a swap — no seat. See below |
| **`aprior_dense`** | `--anatomical-prior`: `SlotHead.prior` initialised from the 12x6 plane-by-finding table in `core/data/corpus/anatomy.py` at PRIOR_STRENGTH 2.0 instead of zeros — the mapping the head otherwise has to discover from study-level labels alone | 0.8557 | **−0.0021 vs `band_dense`** | **THE REFUTATION CLAUSE LANDED LITERALLY, AND IT IS THE ONE THAT NARROWS THE MOST.** Flag verified in the CHECKPOINT config: `anatomical_prior false→true`, with `slices 20`, `slice_pooling max`, `whole_stack true`, `pool cls_mean`, `backbone_lr 4e-05` identical to the control (the only other diff, `slice_regions` absent vs `1`, is the recorded default and not a variable). Both arms peak at **epoch 8** on the same twelve-epoch cycle, so the comparison is at matched schedule position and the −0.0021 is not a truncation artefact. The PREDICTION needed the peak to clear 0.8578 by more than the 0.0062 floor; it came in below the control. The THIRD OUTCOME needed the per-label pattern to separate along the table's own peaked/flat split, and it does not usefully: **peaked rows (MCL, both OA, PF OA, Synovitis) −0.0008 against flat rows −0.0033**, a 0.0025 separation which is under half the floor, and inside it the largest single mover is Synovitis at **−0.010** — an axial-weighted, PEAKED row moving the wrong way. **MECHANISM, and the training loss is what says it: the warm start washed out rather than failed to land.** Train loss tracks the control to the fourth decimal at every epoch (1.2117/1.2162 at e0, 0.6508/0.6529 at e8 — the arm slightly LOWER), so 72 numbers of initialisation on an `nn.Parameter` the data may overrule are gone by the time the fit matters. Cost was as predicted: parameters, shapes, VRAM (26.92 GB reserved / 24.93 GB allocated) and 368 s an epoch all identical to `band_dense`. Seat value **+0.0009** as a fifth member and **+0.0004** as the best swap (displacing `band_dense`), 0 of 1 clearing +0.0015 — no seat on either ruler. **CONSEQUENCE, written into the arm before it ran and now owed:** the head had ALREADY recovered the plane-by-finding mapping from study-level supervision alone, so "supervision never says where" is wrong about THIS head, and the localisation argument has to buy a real localisation signal (knee-mxj) rather than a warm start. Fourth consecutive knob on the decode/head family returning nothing (−0.0009, −0.0044, +0.0014, −0.0021); the second of the five never-run switches to be spent. See below |
| **`half_data`** | `band_dense` fitted on HALF the training studies (`--train-fraction 0.5`), everything else identical — an instrument, not a candidate | 0.8344 | **−0.0234 vs `band_dense`** | **data volume is the binding constraint, and it is the only axis that has cleared the bar.** 3.8× the 0.0062 floor, and ~0.019 of it survives the registered schedule confound charged at its maximum (`schedule_length` prices step compression at 0.0028). Peak moved 8 → 6 as memorisation predicts; the peak-to-finish declines (−0.0024 vs −0.0006) are both sub-floor and unreadable. After its peak the curve PLATEAUS six epochs inside 0.0021 while the training loss keeps falling 0.8015 → 0.5973 — fitting continued, generalisation stopped. Slope: 1,763 → 3,526 studies buys ~0.020, so the bar needs ~1.7× the corpus. External data is PERMITTED (§2.6.a/b); `knee-ombx` promotes from backlog to the decision that gates the schedule. MRNet alone prices at ~+0.0095 and does NOT clear the bar. No seat — it is a handicapped control and was never a member. See below
| `adapt_null` | the flag with NOTHING adapted: stock DINOv2-small written through `pretrain.py`'s own `save` at `--lr 0`, loaded via `--encoder-weights` | 0.8240 @ e3 | **+0.0008 vs control at matched epoch 3** | **the load path is clean.** Plumbing removed as a suspect for the whole SSL family; the four collapses are evidence about the adaptation, as the ledger already said. Drift on the null reads 0.0000 on the stem and all twelve blocks — the drift instrument's first negative control. 4 epochs, instrument only, no seat (−0.0029). The magnitude anomaly that bought the slot was already resolved at `encoder_adapt_gentle`; see below
| `mae_matched` | `band_dense`'s recipe with `--encoder-weights` pointing at **`pretrain_small_matched_e8`** — DINOv2-small given 8 epochs of masked reconstruction over 402,600 knee-MRI slices, the only encoder on the board that has seen nothing but this domain | 0.6735 @ e3 | **−0.1505 vs `adapt_null` at matched epoch 3** | **the FIFTH collapse, and it closes the SSL-as-init family across two objectives and two import depths.** Every suspect worked before the verdict and every one clears: plumbing (`adapt_null` is the same `--encoder-weights` path and reads 0.8240), data (same cache key `336px_20sl_0.1-0.9band`, stamped in the checkpoint), variant (`PretrainedEncoder.into` RAISES on a mismatch and did not fire — checkpoint `small` equals the `TrainingOptions` default, and `band_dense` names no `--variant` either, so the byte-identity claim holds), test (matched epoch, both controls already on the ledger, neither re-run). **The fix I reached for was already run**: `pretrained_encoder.py:47-58` documents that pretraining rewrites the TOP block (block 11 moved 55% / 21% of its norm where supervision never moves it past 0.8%) and argues the axis only concerned blocks 0-5, which is exactly `--encoder-adapt-depth 6` — and that is `encoder_adapt_lower`, 0.7309, its diagnosis already recorded as refuted. The family now reads **0.6735 / 0.6875 / 0.6902 / 0.7162 / 0.7309**, every one 0.09-0.15 below a control that simply starts from stock. Nearest-neighbour 0.4333 against a seated spread of 0.88-0.89 — not diversity, a trunk predicting something nearly uncorrelated with everything, which is what a broken representation looks like. Seat −0.0041, 0 of 1 clearing +0.0015, no seat. **The pretrain was already spent, so running it was right on sunk cost — but it should have been priced as an INSTRUMENT retiring the axis, not as an improvement, because `knee-jk3`'s board re-read already listed SSL-as-init CLOSED and I flagged that prior before the arm started.** Does NOT rule out volume-native pretraining, which never passes through this repo's adapt-and-import path at all (`knee-rpkw`). See below
| `quarter_data` | `band_dense` on a QUARTER of the training studies — the third point, so the data slope is fitted rather than assumed | 0.7907 @ e9 | −0.0671 vs `band_dense`, **+0.0437 over `half_data`** | **the slope is CONCAVE and log-linear is refuted.** 25→50 buys +0.0437 against 50→100's +0.0234, 1.87× steeper; charging the schedule confound per point (0.0048 and 0.0097, downward) makes it 2.09×, so flattening is not the schedule. Each doubling buys ~0.48 of the last, which prices the NEXT doubling — the whole external-data programme — at **~+0.009** and ALL remaining doublings at ~+0.017. Takes down `half_data`'s '1.7× the corpus' arithmetic and reprices `knee-ombx` (MRNet ~+0.004) and `knee-5895` (final_fit ~+0.0029). **No lever on this board is now priced above 1.5%.** Instrument, no seat (−0.0035). See below
| `dinov3` | DINOv3-S/16 at 336, then 224, then S+ | 0.8080 | −0.0377 | **refuted**; deficit is focal findings |
| `calibrated_targets` | each level set to its MEASURED agreement rate | 0.8495 | +0.0038 | directional, inside the floor |
| `lora` v3small 224 | rank-8 adapters, base frozen | **0.8217** | **+0.0160 vs v3 fine-tuned** | **recipe was damaging v3** |
| `schedule_length` | 8 epochs instead of 12 | 0.8429 | −0.0028 | compressing the cycle costs, two thirds the wall-clock |
| `dinov3` 224 | v3-S at its own native read | 0.8057 | −0.0400 | read size is not the axis: 0.0023 from 441 patches to 196 |
| `dinov3` S+ 224 | wider small, 28.7M | 0.8012 | −0.0445 | capacity scaled the WRONG way inside the family |
| `lora` v2 336 | rank-8 adapters on DINOv2 | 0.8127 @ e3 | behind at every matched epoch | **killed at epoch 3** — the arm's own predicted refutation for v2 |
| `crop80_tissue` | 80 mm about the tissue centre, 0.238 mm/px | 0.8345 | −0.0112 vs `whole_stack`, but see below | mean is the WRONG read — gate is per-label |
| `pair_memory` | cross-step memory of 8 pairs in the ranking loss | 0.8155 @ e3 | −0.0302 vs `wsfold0_e6` | **refuted on the mean**; 0.8353 was a first run whose checkpoint was overwritten, see below |
| `dinov3_base_lora` | DINOv3-**base**/16 + rank-8 adapters, 224 | **0.8268** | +0.0051 vs `lora` v3small | best v3 member we have, and still **not a member** — see the oracle below |
| `slot_mean` | uniform mean over present slots instead of learned slot attention | 0.8406 | −0.0051 vs `whole_stack` | attention is worth ONE floor-width — a knob, not a lever (knee-7gv) |
| `per_label_loss` | loss normalised per column, so each of the twelve weighs one twelfth | 0.8414 | −0.0043 vs `wsfold0_e6` | **the arm's own null, and its falsifiable clause resolves with the WRONG SIGN** — see below |
| `weight_average` | EMA of the weights, scored and saved in place of the live ones | 0.8460 | +0.0003 vs `wsfold0_e6` | flat where a holdout ranks epochs, exactly as written; the last-epoch case is +0.0013, directional — see below |
| `centre_tissue` | the control's 130 mm extent held, the crop CENTRED ON THE TISSUE instead of the image | 0.8409 | −0.0048 vs `wsfold0_e6` | **null, as pre-registered** — inside the 0.0062 floor, flag verified (`crop_mm=130.0, centre=tissue` against a control that predates both). Seat value **−0.0016**. Two labels clear their paired interval and both are NEGATIVE (Contusion −0.0128\*, Medial Meniscus −0.0096\*); nothing positive resolves. This is the arm that licenses reading `crop80_tissue` as a RESOLUTION result rather than a centring one. See below |

### Moving the crop's CENTRE, at the control's extent, does nothing — which is the point (2026-08-19)

`centre_tissue` holds the 130 mm extent every canonical member reads and moves only where that extent is
centred: from the image centre to the tissue centre `SlotReader` computes. The recorded recipe carries
`crop_mm=130.0, centre=tissue` against a control that predates both keys, so the flag landed.

**0.8409 against `wsfold0_e6`'s 0.8457, −0.0048** — inside the 0.0062 floor, and the arm pre-registered
exactly this: *"a whole-corpus gain here would be surprising, and the honest expectation is roughly
nothing"*. Seat value against the shipped four is **−0.0016**, so diversity does not rescue it either.

The per-label read does not rescue it in the other direction. Marked against this pair's own paired floor
(half-widths up to ±0.0290, INDICATIVE — the control's scores are reconstructed from its kept predictions),
two labels clear their interval and **both are negative**: Contusion −0.0128\* and Medial Meniscus −0.0096\*.
Nothing positive resolves; by structure scale, focal −0.0051 against regional −0.0044, i.e. flat.

**What this arm buys is the READING OF ANOTHER ARM.** `crop80_tissue` (0.8345) changed two things at once —
80 mm instead of 130 mm, and tissue-centred instead of image-centred — and its per-label table was credited
to the tighter extent. That credit was a subtraction with an unmeasured term. This is the term: at the
control's extent, centring alone is a null. So `crop80_tissue`'s pattern is attributable to the RESOLUTION
change, and the centring is neither the gain nor the loss.

One caveat stated rather than buried: epoch 4 took 1378 s against a median of 258 s, because a job from
another project landed on the same card. The arm's own STALLED guard caught it and the epoch still scored
(0.8374, on-trend between 0.8294 and 0.8390), so the curve is not distorted — but the wall-clock in the
ledger is not this recipe's cost.

### The shipped recipe is defended by two labels, and the mean ruler charges for it (2026-08-20)

A free re-read — no card time — of every vision/decode arm with a surviving checkpoint, re-marked
against the recipe we actually ship (`band_dense`, 0.8578) instead of against the control each was
written for. Eleven arms × twelve labels. **PF OA is negative in 11 of 11 arms** (seven clearing their
own pair's paired half-width) and Synovitis in 10 of 11, across levers with nothing in common — band
span, band density, slice pooling, region pooling, crop geometry, augmentation schedule, epochs. The
mean is negative in 11 of 11 as well, and in the flattest arms it is *made of those two rows*
(`band_span`: focal −0.0036, regional +0.0019, MEAN −0.0013). So `band_dense` sits at a local optimum
those two labels defend, and a flat mean on this control is not "the lever did nothing", it is "the
lever paid the local-optimum tax and broke even elsewhere". Consequence for reading future arms:
`band_dense` is not a neutral control for anything that moves the band, and an arm whose mechanism is
not about the central compartment should be marked on the other ten rows before its mean is believed.

**The positive side is chance and is reported as such.** Three cells clear their half-width upward
(`band_density` Lateral Meniscus +0.0187\*, `depth_regions` Lateral Meniscus +0.0165\*, `aug_long` ACL
+0.0187\*); 132 cells at a two-sided ~5% rule predicts ≈3.3 per side. That is exactly what chance
predicts, no label repeats with consistency, and none of it is a finding. Three arms
(`resolution_420`, `extended_slots`, `multiscale`) could not be re-read at all — their checkpoints are
gone from the data root. And the obvious follow-on, a per-label roster, is **prior art in this repo**:
`kneemri/evaluation/label_combine.py` already combines members per label with a split-half estimator
built specifically around the selection optimism such a table invites.
[interpretation](../interpretations/kneemri/2026-08-20_per_label_reread.md)

### Per-label loss weighting is refuted a second time, now from the training side (2026-08-19)

`per_label_loss` normalised the loss per column so each finding weighs one twelfth, against a data
mismatch it stated in numbers: Effusion is positive in 2,666 studies and Fracture in 314, so pooled
cross-entropy spends about ten times more gradient on one than on the other while the score counts them
the same. It scored **0.8414 against the control's 0.8457, −0.0043**.

The mean is inside the 0.0062 floor and on its own would be a shrug. **The pre-registered falsifiable
clause is what settles it**: the arm wrote that a gain must concentrate in the rare labels — Fracture
(314 positives), Synovitis (580), MCL (767) — and that a uniform shift would mean something else moved
it. Read against `weight_average`, which is the same recipe at the same epoch count and the closest
per-label control we have printed (0.8460 vs the control's 0.8457, i.e. matched on the mean):

| | Fracture | MCL | Synovitis | Lat Men | ACL | Med Men | Effusion |
|---|---|---|---|---|---|---|---|
| positives | 314 | 767 | 580 | — | — | 2,347 | 2,666 |
| `weight_average` | 0.785 | 0.752 | 0.880 | 0.816 | 0.876 | 0.880 | 0.846 |
| `per_label_loss` | 0.763 | 0.750 | 0.878 | 0.809 | 0.869 | 0.874 | 0.849 |
| Δ | **−0.022** | −0.002 | −0.002 | −0.007 | −0.007 | −0.006 | +0.003 |

It is not a uniform shift: eleven labels sit within −0.007 and one is three times further out. It
concentrates exactly where the arm said it would — **and it is negative there.** The rarest label takes
the largest loss, and it is the label the reweighting up-weighted most.

That is a mechanism, not a coin flip. Up-weighting a 314-positive column multiplies the variance of its
gradient without adding any information to it: the same handful of studies simply speak louder, so the
encoder fits their noise. AUC being rank-based and invariant to the positive rate — which the arm named
first as the reason this might do nothing — means there was no compensating gain to set against that.

**Caveat on the denominator, which is the weakest part of this read:** 314 positives is the least stable
AUC in the table, so −0.022 on Fracture carries a wider interval than any other cell and this is
directional evidence, not a measured effect size. What is *not* soft is the sign and the ordering.

This is now the second independent refutation of treating labels separately, and the two come from
opposite ends of the pipeline: [the oracle line](#the-ceiling-of-combining-what-we-have-is-the-plain-average-2026-08-17)
killed per-label weighting at the *ensembling* stage (−0.004 to −0.005 against plain averaging), and this
kills it at the *objective*. Prevalence in this dataset is not a mismatch to be corrected; it is the
budget that buys the common labels their scores. **Per-label treatment is closed on both sides.**

### An EMA buys nothing where a holdout picks the epoch, and ~+0.0013 where it does not (2026-08-19)

`weight_average` scored **0.8460 against 0.8457** — flat, and its written prediction is why that is the
expected result rather than a disappointment: keeping the BEST epoch by holdout score is *itself* an
average over the trajectory, so an EMA was predicted to help LEAST here and MOST in the final fit, where
no holdout ranks epochs and the last one is kept by construction.

So the mean is the wrong number to read this arm on. Its own stated targets were the best-to-last gap and
the last-epoch value:

| | best | last (e11) | gap |
|---|---|---|---|
| `wsfold0_e6` control | 0.8457 | 0.8394 | 0.0063 |
| `weight_average` | 0.8460 | **0.8407** | **0.0053** |

The gap shrinks by 0.0010 and the last epoch gains **+0.0013** — both far under the 0.0062 floor, so
neither is a result. The verdict is the significance bar applied literally: **directional, mechanism-sound
and free** (one weight copy, no extra step time, measured at no throughput cost), so it is kept for the
final fit where there is no holdout to select on, and it is not a reason to spend a slot, rebuild a
member, or plan anything around. It does NOT rescue the banned axis it was scouting for: an EMA damping
the last few hundred steps recovers about a fifth of the 0.0062 three-seed spread, which is a knob-sized
piece of a floor-sized problem.

### The ceiling of combining what we have is the plain average (2026-08-17)

`member_curve` over seven members reports `oracle 0.8590 vs plain average 0.8588 (headroom +0.0002)`.
The oracle picks, per finding, whichever member scores that finding best ON THIS HOLDOUT — a choice made
with the answers, so it bounds *every* per-label weighting scheme that could be written over these
members. It sits two orders below the 0.0062 floor above the flat rank-average, so **per-label member
weighting is worth nothing here**, and no more members of this kind will change that: the members do not
disagree per label in a way any rule could exploit. The same run prices the DINOv3 diversity bet —
`best 4` (0.8612) excludes both v3 members, `best 6` with them scores 0.8603, `best 7` 0.8588. Full
reading: `interpretations/kneemri/2026-08-17_the-oracle-ceiling-is-the-average.md`.

### `crop80_tissue` — the control is not schedule-matched, and the mean was never the gate

Two things have to be said before that −0.0112 is quoted anywhere. **First, the control is 12 epochs and this
arm is 8** (`epochs_done` on both checkpoints). `schedule_length` above prices exactly that axis at
−0.0028, so the crop itself costs about **−0.0084** on the mean, not −0.0112. The arm should have been run
at 12 to match; it was not, and the adjustment is arithmetic over another arm rather than a measurement of
this one. **Second, `crop80_tissue.sh` forbids reading the mean at all** — its prediction is two-sided (small
structures up, diffuse findings down), so the mean averages the two halves of the prediction against each
other and a null there is what you would expect whether the mechanism works or not. The decision belongs to
the per-label paired read, which is what gates the multi-scale build (knee-1nn).

**THE GATE RETURNED, 2026-08-16: multi-scale does NOT earn its build.** The per-label read at matched
epoch 7 landed the diffuse half (Baker's −0.028, Synovitis −0.019, Effusion −0.015) and missed the half
the build depended on — Medial Meniscus −0.012, ACL and MCL at the floor, only Fracture up (+0.019, and
the worst-agreement label). Multi-scale adds tight slots to the wide ones, so its value is what those
slots buy: ≈ **+0.0016 on the mean** for a 71.6 GB cache and a training slot. `multiscale.sh` is now
`retired` in `experiments/lib.sh` terms — it refuses to run and the checker no longer reports its cache.
Resolution at CONSTANT field of view was untouched here — but it was tested directly the next day and
came back flat at both rates (see "The confound came apart", 2026-08-18); what
failed is paying for sampling with context. See
`interpretations/kneemri/2026-08-16_crop80_resolution_gate.md`.

**AND IT DOES NOT PAY AS A DIVERSITY MEMBER EITHER, 2026-08-17.** The remaining argument for it was that
a member which fails differently can add more than it scores — it is the only member we own that reads a
different pixel group, and the board prices decode diversity at 2.4x its offline size. Measured on cached
probabilities, no slot: the six all-ours members score 0.8598 and the same six plus crop80 score
**0.8595**. Joint selection never wants it — best-4 of the six is 0.8612, best-6 *including* crop80 is
0.8607, best-7 is 0.8595, so every subset containing it sits below the best subset of the same size
without it. Two independent routes, the same verdict. The caveat is stated rather than buried: this
member ran 8 epochs against the pool's 12, so it prices an undertrained member of the crop family, not
the crop axis — `centre_tissue` has still never run, and the slice band has never been varied at all.

### `pair_memory` — refuted on the mean, and its own gate cannot be run

Recipe-matched to `wsfold0_e6` on every axis the metrics line records except `pair_memory: 8` (336 px,
12 slices, whole-stack, one-cycle, weight floor, backbone 8e-6, seed 2026, fold 0).

**The score is 0.8155, not the 0.8353 this ledger carried until 2026-08-16.** The file holds TWO runs:
a twelve-epoch one peaking at .8353 @ e6 (e0 .7326 · e3 .8160 · e8 .8347 · e11 .8338), then a
four-epoch one peaking at .8155 @ e3. A re-run to the same `--out` appends its rows and REPLACES the
checkpoint, so the weights on disk are the second run's and the number quoted was the first run's.
The oracle that caught it is the kept holdout record: it is `trained`, so exact by construction, and
it scores 0.8155 at epoch 3. Fixed in `RunMetrics.rows`, which now answers for one run.

Against the control's **0.8457** that is **−0.0302**, six times the noise floor. The verdict was
already "does not pay on the mean" at −0.0104; the artefact that survived is worse still.

**Its own prediction is refuted too, on its own instrument.** The arm predicted a gain concentrated on
the RARE labels — Fracture (314 positives) and Synovitis (580) — which an unweighted 12-label mean
averages away exactly as it did for `crop80_tissue`. The per-label read against `wholestack_s2026` at
matched best epochs:

| | delta |
|---|---|
| Lateral Meniscus | **−0.0426*** |
| Medial Meniscus | **−0.0304*** |
| Synovitis | −0.0144 |
| Fracture | +0.0044 (interval ±0.061 — unreadable) |
| everything else | −0.010 … −0.001 |

`*` = clears that label's own 95% sampling interval. The predicted half is inside the noise; the
damage is real and it lands on the two menisci, the labels with nothing to do with the mechanism.
Cross-step pair memory does not trade the mean for the rare labels — it just costs. **Refuted.**

The read was nearly not run at all, on a reading error worth recording: a scan of the metrics files for
a recipe looked at the FIRST line of each and reported every older run as unrecorded, which read as
"there is no matched control on disk". `kneemri.runs.metrics_recipe` backfills the recipe from the
checkpoint's `options` block and APPENDS it, so the line it was looking for was at the end of every
file it declared empty — and the tool had already been run. A missing-data conclusion drawn from a
scan that only looked at one line of the file. The one confound the read does report is
`compile_encoder False→True`. The cached `pair_memory_s2026.holdout.npz` was written off at the time as
unusable, "a second, truncated run (epochs 0–3) was appended to the same file, so the npz is epoch 3" —
which had it backwards. Epoch 3 of the second run is exactly what the checkpoint beside it holds; the
npz was the only thing on disk telling the truth, and reading it is what corrected the score.

One curiosity worth not losing: epoch 11 records **27,266 s** against ~234 s for every other epoch. That
is the machine, not the model — but it means wall-clock from this file is unusable.

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
survives on its own or our best member is mostly a rate change we already knew about. Run it as
`bash experiments/arms/resolution.sh 392 448 8e-6`; it lands as `resolution_392_lr8e-6` /
`res392_lr8e-6_ws_s2026.pt`, because a non-default rate now tags the name the way a non-default batch
does — without that, the confound-breaking run wrote the name of the very member it is the control for.

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

Both peak at epoch 6. Patch-14 needs multiples of 14, so the ladder is 336 (576 tokens) → 392 (784) →
420 (900) → 448 (1024). 392 was the last rung that fit while it held 28.4 GB of a 32 GB card and 448
OOMed outright, and buying the next one then meant a smaller batch — which changes the ranking term's
within-batch pair count and stops it being one variable.

**That wall moved on 2026-08-17 and the ledger said so a day late** (knee-d2fc). Activation
checkpointing took 392/batch-12 from 28.4 GB to 20.5, and the freed headroom buys the next rung at the
SAME batch. Measured, one epoch on 400 studies, whole-stack + compile + checkpointing + bf16, `--vram-gb
28` on a card another tenant already held ~2.8 GB of:

| train size | batch | peak | throughput | epoch |
|---|---|---|---|---|
| 392 (shipping) | 12 | 20.46 GB | 733 img/s | 55 s |
| **420** | **12** | **23.42 GB** | 618 img/s | 89 s |
| 448 | 9 | 20.00 GB | 560 img/s | 88 s |
| 448 | 12 | OOM in six seconds | — | — |

420 px is therefore a ONE-VARIABLE arm at 1.6× the wall-clock, and it needs no decode: the cache is
already 448 px, so `--train-size 420` reads what is on disk. 448 buys the rung after it and does move
the batch, so it is only worth reading if 420 pays.

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

### The confound came apart, and resolution is not the cause (2026-08-18)

`resolution.sh 392 448 8e-6` — 392 px at the CONTROL's encoder rate, the one arm knee-a8x asked for —
scored **0.8459**, against `wsfold0_e6`'s 0.8457. **Resolution alone is +0.0002.** Not inside the floor:
an order below it. The estimate above (+0.002, if the rate's +0.0044 transfers) was generous by tenfold.

**The axis is flat at the OTHER rate too (2026-08-19).** The pair above is 336→392 at 8e-6. The same
step at 4e-5 is already on disk and had never been read as a pair: `lr4e-5_ws_s2026` (which is what
`rate_wholestack` is STORED as — the arm name is not the member name, and looking for the arm name
costs two failed lookups) against `res392_ws_s2026` reads **MEAN −0.0002**, with the recipe diff the
tool prints confirming the only change is `image_size 336→448, train_size 0→392`. So resolution at
constant field of view is +0.0002 at one rate and −0.0002 at the other: flat twice, independently,
against a 0.0062 floor. A 420 px rung is a knob inside a family measured flat — it does not get a
queue slot, and `knee-d2fc`'s rungs are now refused on evidence rather than left needing a reason.

So the whole +0.0066 of `res392` belongs to the encoder rate 8e-6 → 4e-5, and the rate is now the
largest unpriced effect in the ledger — bigger, on this evidence, than anything else we have moved
except whole-stack itself. `rate_wholestack` (336 px, 4e-5, everything else the control) is the arm
that prices it under the current recipe; `rate_isolate`'s +0.0051 was measured in the windowed era.

**Two claims elsewhere have to be read down.** `knee-jk3` lists "336 px for findings a few voxels
across" among four causes that PAID — that one did not. And the 448 px cache's rungs (`knee-d2fc`) were
bought as resolution rungs; they now need their own reason, because the axis they climb is flat.

**And the rate confirmed it the same day.** `rate_wholestack` — 336 px, 4e-5, everything else the
control — scored **0.8525, +0.0068**, over the floor and level with `res392`'s 0.8523. So the best
single member this project has produced is reproducible at 336 px on the cache we already had: the
64 GB of 448 px arrays bought the rate change, which was free. What this does NOT settle is whether the
rate composes with 392 px or replaces it — `res392` at 4e-5 IS that combination and scores the same,
which reads as one effect, not two.

**The sweep then found both ends of the useful range in one sitting.** 1e-4 scored **0.8490** and 2e-4
**0.7391**, against 8e-6's 0.8457 and 4e-5's 0.8525. So the axis is a broad plateau with a cliff at its
top: 4e-5 and 1e-4 differ by 0.0035, *inside* the 0.0053 floor, so this evidence cannot rank them — the
honest reading is that anything from 4e-5 to 1e-4 buys the gain and the optimum is somewhere in there.
2e-4 is past the edge, and it failed in exactly the shape the arm predicted before it ran: the loss went
UP from epoch 0 to epoch 1 (1.2296 → 1.2739), then flattened at 1.07 where every healthy arm reaches
0.70. That is a pretrained encoder being overwritten, not a rate that merely underperforms — which is
why it reads as "too far", not as "the rate stopped mattering".

**The axis is spent for now.** A rung between 4e-5 and 1e-4 would be measuring inside the floor, which
buys a number and no decision. The sweep's real leftover is `lr1e-4_ws_s2026.pt` — 0.8490, a member
trained a full 2.5× off the best rate, which is diversity that cost nothing extra to produce.

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

> **SUPERSEDED 2026-08-16 — quote 0.919 now, and it is entirely ours.** `kneemri-efficient` v4 (two
> post-LoRA whole-stack members, 2 passes/study, no CC0 weights) scored **0.919**, and `kneemri-trained`
> v8 (five all-ours members) scored **0.911**. Both beat the 24-member mixed fusion's 0.898 — by +0.021
> and +0.013 — so the grafted members are now a COST, not a contribution, and the fusion track no longer
> carries the score. The table and the diversity paragraph below are kept as the record of what was true
> at the earlier member quality; the reading of them changed when the pool did, not the mechanism.
> See `interpretations/kneemri/2026-08-16_the_grafted_members_became_a_cost.md`.

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

## The six throughput arms had no row here at all (2026-08-17)

Six one-epoch runs priced compiling, activation checkpointing, bf16 and a narrower unfreeze, and none of
their numbers were written down anywhere until today. They are prices, not results — every arm lands
between 0.7539 and 0.7612, inside the floor of every other — but they set what an hour of GPU buys:

- **`torch.compile` is +47%** (815 → 1195 img/s) for 0.5 GB, holdout unmoved. A twelve-epoch whole-stack
  arm costs ~55 min compiled against ~73 eager. This is the evidence for compiling being the default.
- **Activation checkpointing costs 29% and buys 3.6x memory** (785 vs 1016 img/s at batch 6; 7.38 vs
  26.24 GB). At the shipping batch of 12 the off-side would want ~52 GB on a 32 GB card, so the tax is
  what buys the batch. The docstring's "about 30% from the literature" turns out to be right for us.
- **bf16 is +7%**, not the reason it is on: the scaler falling away, not faster matmul.
- **`--unfreeze-last 4` is +25% and −0.0065 at one epoch** — inside the floor, unreadable as a score, and
  the only negative direction here. A question for a real arm, not an adopted default.

Read the pairs as the RECIPES record them, not as `throughput.sh` now reads: the script was edited after
these ran, and five of six recipes carry `compile_encoder False`, so every row except the compile pair is
a comparison among the eager arms. Full write-up:
`interpretations/kneemri/2026-08-17_what_the_throughput_arms_priced.md`.

## Almost every per-label delta that resolves is a LOSS (2026-08-17)

Seven arms re-read against `wsfold0_e6` (0.8457) with each pair's own bootstrap interval, now that the
back-fill has given them per-study predictions. Two results, and neither is the one the re-read was
run to find:

- **The Fracture rows above do not survive.** extended_ws (+0.0200), slices16 (+0.0120) and schedule_8
  (+0.0158) are UNREAD against their own paired floors (0.0288 / 0.0190 / 0.0254) — a floor two to three
  times TIGHTER than the unpaired bound that first flagged them. Only intensity392's +0.0372 clears its
  0.0311. `res392` cannot be asked at all: it kept no per-study predictions, so its +0.0330 Fracture and
  +0.0273 MCL still print against the wide bound.
- **Across seven arms, exactly two positive deltas clear their pair's floor** (extended_ws PF OA +0.0156,
  intensity392 Fracture +0.0372); every other resolvable per-label move is negative. These arms are not
  lifting some findings and holding the others — they are trading, and the trades that resolve go the
  wrong way. `pair_memory` is the extreme: eleven of twelve down, nine starred, MEAN −0.0302.

All seven are marked INDICATIVE — the control's scores were reconstructed from its kept predictions.
`interpretations/kneemri/2026-08-17_which-per-label-findings-survive-the-paired-floor.md`.

Interpretation of the whole ledger, including what each refutation rules out, lives in
`interpretations/kneemri/` — for the above, `2026-08-14_the-ruler-was-five-times-too-small.md`,
`2026-08-14_labels-or-pixels.md` and `2026-08-14_their-router-not-ours.md`. The build log is git history.

## Region pooling: the patellofemoral loss was the band, and the meniscus gain does not transfer

`depth_regions` starred on two labels out of twelve and they said opposite things — PF OA −0.0193 and
Lateral Meniscus +0.0165 — so it was one arm carrying two candidate mechanisms and a multiple-comparison
risk. `depth_regions_narrow` holds the reading at the pool's own band (0.20–0.80, 12 slices, 336 px,
backbone 4e-5) and varies nothing but `--slice-regions 4`, against `lr4e-5_ws_s2026 = 0.8525`. The flag
is verified in the recorded recipe rather than assumed: `slice_regions: 4`, `band_low: 0.2`,
`band_high: 0.8`, `slices: 12`.

**The PF OA clause is confirmed, and more strongly than predicted.** The wide-band arms both lost the
patellofemoral centre by the same amount to four decimals; on a band that never adds the outer tenth
there is nothing to pay, and PF OA reads **+0.0067**. The loss belongs to the BAND, not to the pooling.
That was the useful half of the decomposition and it survives.

**The Lateral Meniscus clause does not.** It reproduces in SIGN and not in size: +0.0056
[−0.0112, +0.0206] against +0.0165* on the wide band, and read directly against `depth_regions` the
narrow arm is **−0.0112** on that label. A single star out of twelve labels that returns at a third of
its size with an interval spanning zero is what a multiple-comparison artefact looks like, and this arm
named that risk as the second reason to run it.

**On the mean, both arms are flat** — `depth_regions` −0.0029 against `band_dense`, this one +0.0029
against `lr4e-5_ws`, both inside the 0.0062 floor. The arm pre-registered "flat in BOTH is the
pooled-budget argument refuted", so that clause resolves against the argument by its own writing.

What it is NOT: a verdict on region pooling as an idea. What is refuted is that one maximum per depth
region buys a general gain over one maximum across the band, on this band and this depth. What remains
open is the same treatment on a band wide enough that the regions have different anatomy in them, which
is the case `depth_regions` was and where the mean also did not move. The asset is not dead; this use of
it is.

**Seat value +0.0006** against the shipped four, below `depth_regions`' +0.0010 and below every member
already mounted. It does not get a slot.

## Reading block 9 instead of block 12 costs 0.023, and it is not rescued by diversity

`encoder_layer 9` is the only arm this project has run on an axis nobody had varied for either encoder:
which block of the ViT the head reads. The argument for it was not weak — the final block of a
self-supervised ViT is the most specialised toward its pretraining objective, which is exactly why both
DINOv2 and DINOv3 ship `get_intermediate_layers` for dense work, and our findings are 1–3 mm read off
pooled PATCH tokens rather than an image-level embedding. It costs nothing extra:
`output_hidden_states` returns every block from the same forward pass.

It loses, and not by a margin that needs interpreting: **0.8230 against the control's 0.8457, −0.0227**,
which is 3.7x the 0.0062 floor. The flag is verified in the recorded recipe (`encoder_layer: 9`), and
the curve was flat over its last three epochs (0.8228 / 0.8230 / 0.8229), so it converged there rather
than being cut short. Seat value against the shipped four is **−0.0034** — the largest negative any
same-split candidate has posted — so the usual rescue for a weak-but-different member does not apply
either. It is worse AND it is not usefully different.

WHAT IS ACTUALLY REFUTED: that block 9 of 12 carries our findings better than block 12. Not that no
intermediate layer does. But the gradient argues against spending a second slot: the endpoint we
already use is the best number we have, and a drop of 0.023 nine blocks back means an interior peak at
block 10 or 11 would have to be a sharp non-monotonicity rather than a gentle one. The convention the
linear-probe literature uses turns out to be the right one for this head, which is worth knowing
precisely because it was assumed rather than measured.

## Focal pooling: the focal/diffuse split appears in the predicted direction, and buys nothing on the mean

`focal_pool` concatenates a third part to the pooled vector — the mean of the top eighth of each channel
over the 24x24 patch grid — so a 1-3 mm finding that lands in a handful of tokens is not divided by two
orders of magnitude before the linear head sees it. Per channel rather than per patch, because selecting
whole patches would commit every channel to one location and the twelve findings do not share one. The
axis was unvaried for us and for every published member; `StudyModel._pool` had implemented it since the
head was written and nothing had ever run it.

**0.8432 against the control's 0.8457, -0.0025** — inside the 0.0062 floor, so flat. Flag verified in the
recorded recipe (`pool: cls_mean_focal`, `whole_stack: True`, `encoder_layer: -1`, fold 0 / split 2026).
Seat value against the shipped four is **-0.0012**, small enough to sit inside the region where this
ruler is known to be wrong by 2.4x (knee-7d1), so it is not evidence of anything on its own.

THE CLAUSE IS THE INTERESTING PART, and it did not fail the way a dead mechanism fails. The arm
pre-registered that a real effect concentrates in the small focal findings (meniscus, ACL, MCL,
Fracture) while the diffuse ones (Effusion, the three OA columns) hold or dip, and that a UNIFORM lift
would mean the extra width helped rather than the tail. Read against `whole_stack` at the same epoch on
the same split, the focal group means **-0.0004** and the diffuse group **-0.0065** — a 0.006 separation
in exactly the predicted direction. The mechanism's signature is there. What is absent is the gain: the
focal group held rather than rose, so the separation is the diffuse findings losing, not the focal ones
winning, and the mean nets out flat.

Fracture, the smallest and most focal of the twelve, is the one label that moves: **+0.020**. That is the
mechanism's best case and it is also one crossing out of twelve labels, which is not surprising by
chance — the same multiple-comparison trap `depth_regions_narrow` walked into with Lateral Meniscus and
then failed to reproduce. DIRECTIONAL ONLY. Not a finding, and not worth a confirmation slot on its own.

WHAT THIS DOES NOT KILL: focal pooling as an asset. What is refuted is this use — a third part bolted
onto the pooled vector of a whole-stack member, where it widens a linear head's input by 50% on the same
number of studies and the tail statistic is noisier than the mean beside it. The arm named that risk
first, and it is the reading the numbers support: the tail carries signal about focal findings (the
direction is right) and the head cannot afford to estimate it here.

## Unfreezing four blocks instead of six: the speed is real, and it costs the fine findings

`throughput_unfreeze4` measured +25% img/s at four blocks against six (1019 vs 815, eager both ways,
13.11 GB peak against 14.56) and left the score unsettled: its epoch-0 holdout was 0.7539 against
0.7604, a 0.0065 gap sitting exactly ON the floor and read from the single weakest measurement in the
ledger. 25% compounds across every arm that follows, which is worth more than most single-arm results,
so knee-jvwu asked for one twelve-epoch arm to settle it. This is that arm.

**0.8320 against the control's 0.8457, −0.0137** — 2.2x the floor. Flag verified in the recorded recipe,
by the comparison itself: `unfreeze_last 6→4`, nothing else moved. It is CONVERGED, not short: it peaked
at epoch 6 and spent its last five epochs between 0.8270 and 0.8294 while the loss kept falling from
0.7552 to 0.6302, which is the shape of a model that has run out of representation to adapt rather than
one that needed more schedule.

WHERE THE LOSS SITS IS THE ARM'S OWN HYPOTHESIS. The file named two possibilities: a shallower adaptation
could catch up, because our ledger says capacity is not the constraint (base over small was +0.0011), or
it could fall behind, because the frozen blocks are the GENERIC ones and knee MRI is a long way from
ImageNet. By structure scale the answer is the second: **focal −0.0170 against regional −0.0091**. The
three biggest losses are Baker's −0.0360\*, Lateral Meniscus −0.0345\* and Medial Meniscus −0.0271\* —
the fine structures — and eleven of twelve labels are negative. The intervals are INDICATIVE rather than
paired-tight (the control's per-label scores are reconstructed from its kept predictions, half-widths up
to ±0.0280), so the individual stars are soft; the mean and the consistent sign are the strong part.

Seat value against the shipped four is **−0.0017**, so the second ruler agrees with the first — this is
not a weak member that earns a seat on diversity. **The default stays six blocks.**

WHAT THIS DOES NOT KILL: cheaper adaptation as a lever. What is refuted is spending the saving HERE, on
the generic blocks a knee needs. The measurement says the specialisation reaches deeper than four blocks,
which is a statement about where the encoder is wrong for this domain — the same direction knee-c5n
(unlabelled adaptation of the encoder) attacks from the other side, and a cheaper adaptation that keeps
depth (LoRA on all twelve rather than full fine-tuning on four) is untested and is not covered by this
result.

### `band_dense_long` — the schedule question is refuted and the checkpoint is still the best seat move on the card — 2026-08-20

`band_dense` is a member of the shipped roster and it was fitted on the twelve-epoch OneCycle every arm
in this ledger uses. The arm asked one thing: does the same recipe at **24 epochs** score higher, i.e. is
the schedule a bound on the band members? Flag verified in the recorded recipe — `epochs 24`, everything
else identical to `band_dense` (`slices 20`, band `0.1–0.9`, `backbone_lr 4e-05`, `slice_pooling max`,
`whole_stack True`, `intensity_aug False`).

**Pre-registered clause:** peak at epoch 16–20 confirms the long regimen; a peak still near 8 is scatter
on the same trajectory. **It peaked at epoch 6 (0.8587) and never beat it in eighteen more epochs**, with
a near-tie at epoch 9 (0.8581) and then a monotone slide to 0.8405 at epoch 23 while the training loss
halved from 0.85 to 0.4536. That is the SCATTER limb, and stronger than the clause asked for: not a peak
that failed to move late, but a decline of **0.0182 from peak to finish, 2.9x the 0.0062 floor**, which is
memorisation rather than a schedule that wanted more room. **The long regimen is refuted for this recipe.**

Against its matched control `band_dense` (0.8578) the peak is **+0.0009**, a seventh of the floor —
directional at best, and the honest reading is that eighteen extra epochs bought a different draw of the
same number.

AND THE SEAT RULER DISAGREES WITH ALL OF THAT. Priced against `comp/champion.json`'s `greedy4_v1` roster:

| ruler | reading |
|---|---|
| matched control | +0.0009 over `band_dense` — inside the floor |
| its own curve | peak at 6 of 24, −0.0182 to the end — question refuted |
| seat value | **+0.0012**, displacing `band_span`; best-four goes 0.8679 → 0.8691 |
| as a fifth member | +0.0012 (0.8692), which the FOUR MEMBERS principle forbids |

**+0.0012 is six times the best move the mounted pool had an hour ago** (+0.0002, `attend_dense`), and it
comes from an arm whose own question failed. This is the `calib_span` principle again — a member's control
and its seat are different quantities — but sharper: there the member merely lost its control, here the
arm's PREMISE was refuted and the artefact it dropped out is still the largest available improvement to
the champion. A run that answers "no" still produces a checkpoint, and the checkpoint is not bound by the
answer.

It also names `band_span` as the displaced incumbent for the third candidate running, which is now a
finding about the roster rather than about any one candidate: `band_span` is the champion's weakest seat
and everything positive in the pool is competing for it.

**+0.0012 is still a fifth of the noise floor** and a fiftieth of knee-jk3's 0.054 frontier gap. It is a
seat move worth holding for the next entry, not a result.

WHAT THIS DOES NOT KILL: the long schedule anywhere else. What is refuted is 24 epochs on THIS recipe,
which is exactly the setting `aug_long` was written to change — a regulariser that has no room to pay in
twelve epochs, tested on the schedule this arm just proved is long enough to overfit without one. The
decline measured here (0.0182) is nearly three times the whole gap `intensity_augment` originally lost by,
which is the quantitative form of the argument that the old twelve-epoch refutation could not see this
axis. `aug_long` launched on this arm's completion marker.

### `aug_long` — the regulariser fired, the overfitting did not care, and that is worth more than the number — 2026-08-20

The arm asked whether the intensity block changes the SHAPE of the long curve, on the schedule
`band_dense_long` had just proved long enough to overfit. Single variable, verified in the recorded
recipe by diffing the two headers key by key: `intensity_aug False→True`, and all forty-six other keys
identical (`epochs 24`, `slices 20`, band `0.1–0.9`, `backbone_lr 4e-05`, `head_lr 0.001`,
`slice_pooling max`, `whole_stack True`, `batch 12`, same cache).

**Both halves of the refutation clause fire literally.** It required the peak epoch still at or below 10
AND the peak mean more than 0.0062 below control. **Peak epoch 9, peak mean 0.8511 against 0.8587 —
−0.0076, 1.2x the floor.** Paired against its control, MEAN −0.0077 with three labels past their own
95% interval and all three DOWN (MCL −0.0270, Baker's −0.0219, Lateral OA −0.0202); nothing rose past
its interval.

**THE PART THAT IS EVIDENCE RATHER THAN A VERDICT.** Before reading this as "augmentation doesn't work",
check the flag actually did something — and it did, mechanically and visibly. `aug_long`'s training loss
is higher than the control's at every epoch from 4 onward and finishes at 0.4633 against 0.4536: the
perturbation made the fit measurably harder, which is exactly what a regulariser is supposed to do. What
it did NOT do is change the decline:

| | control `band_dense_long` | `aug_long` |
|---|---|---|
| peak | **0.8587** at epoch 6 | 0.8511 at epoch 9 |
| epoch 23 | 0.8405 | 0.8342 |
| peak → finish | **−0.0182** | **−0.0169** |
| final train loss | 0.4536 | 0.4633 |

The two curves are near-PARALLEL from epoch 5 on, `aug_long` sitting 0.006–0.008 under the control the
whole way. So the arm's premise — that the tail decline is memorisation, and that perturbing the input is
the cheapest thing here that makes memorising harder — is half confirmed and half refuted in the most
useful possible way: **the input perturbation landed, and the decline did not move.** Whatever the tail
decline is, it is not the kind of overfitting an input-space regulariser prevents.

The one part of the PREDICTION that held is the peak epoch, 6 → 9, three epochs later. It bought nothing.
The predicted concentration is refuted outright: gains were to land in the regional group, and by
structure scale it went focal −0.0084, regional −0.0067 — both down, the diffuse group very slightly less
hurt, which is not the shape argued for.

**Seat value: +0.0007 as a fifth member, +0.0000 as the best swap.** Below the 0.0015 the board ruler has
ever been shown to resolve, so unlike `band_dense_long` this checkpoint does not rescue itself on the
second ruler either. Both rulers agree, which they have not always done here.

**CONSEQUENCE, and it points somewhere specific.** The arm's own third outcome (peak moves late AND curve
flattens AND mean does not rise) did not occur — the curve did not flatten — so this is not that branch.
But the branch it IS on carries the same instruction more strongly: the third outcome would have said
"the regulariser works and supervision is the bound"; what actually happened is that the regulariser
worked *on the training loss* and the holdout decline was untouched, which rules out input perturbation
as the lever without needing the flattening to argue it. The next arm on this axis should not be another
input perturbation. `intensity_augment` is now refuted at twelve epochs AND at twenty-four, at two
operating points, on the mean and on the curve shape — which retires augmentation on this corpus rather
than on any one schedule, and is the strongest form the original row could have been given.

WHAT THIS DOES NOT KILL: augmentation of a KIND this repository does not have. Everything measured here
is intensity-space (gamma, blur, contrast, noise) — a photometric perturbation on a modality whose
findings are geometric. Nothing has tested a geometric or slice-space perturbation, and the parallel
curves are evidence about the surface that exists, not about the idea of regularisation.

### `aprior_dense` — the head already knew where to look, so a warm start had nothing to teach it — 2026-08-20

`SlotHead` gives every diagnosis its own softmax over the six acquisition slots and initialises
`self.prior` at zeros, so which sequence shows which finding is something 4,407 study-level label
vectors — carrying no localisation at all — have to teach it. `--anatomical-prior` initialises that same
12x6 parameter from `core/data/corpus/anatomy.py` instead, generated from two facts per finding (which
planes show the structure, whether it needs fluid weighting) at PRIOR_STRENGTH 2.0. It stays an
`nn.Parameter`, so the data can overrule it; every entry is non-negative, so it is a preference over
slots and never a mask.

**The flag landed and the arm is single-variable.** Verified in the checkpoint config rather than the
recorded recipe: `anatomical_prior false→true`, with `slices 20`, `slice_pooling max`, `whole_stack
true`, `pool cls_mean` and `backbone_lr 4e-05` identical to `band_dense`. The one other config
difference is `slice_regions` absent from the control's record against `1` in this one — that is the
`StudyDesign` default arriving in the manifest, not a behaviour change.

**Both arms peak at epoch 8**, on the same twelve-epoch one-cycle, which makes this a matched read
rather than a comparison of two different points on two different schedules:

| epoch | 5 | 6 | 7 | **8** | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|
| `band_dense` | 0.8540 | 0.8550 | 0.8554 | **0.8578** | 0.8572 | 0.8556 | 0.8554 |
| `aprior_dense` | 0.8490 | 0.8541 | 0.8532 | **0.8557** | 0.8526 | 0.8529 | 0.8525 |

**Which clause fired.** The prediction required the peak to clear 0.8578 by more than the 0.0062 floor;
it came in 0.0021 below the control. That is the REFUTATION clause, literally. The third outcome — mean
flat but the per-label pattern moving along the table's peaked/flat split — is the one worth checking
carefully, because it would have said the prior does what it claims while the ceiling sits elsewhere.
At matched epoch 8:

| row | control | arm | delta | prior row |
|---|---|---|---|---|
| MCL | 0.793 | 0.796 | +0.003 | PEAKED (coronal alone) |
| Medial OA | 0.880 | 0.879 | −0.001 | PEAKED |
| Lateral OA | 0.816 | 0.820 | +0.004 | PEAKED |
| PF OA | 0.858 | 0.858 | +0.000 | PEAKED |
| Synovitis | 0.876 | 0.866 | **−0.010** | PEAKED |
| ACL | 0.874 | 0.877 | +0.003 | flat |
| Medial Meniscus | 0.905 | 0.901 | −0.004 | flat |
| Lateral Meniscus | 0.846 | 0.851 | +0.005 | flat |
| Effusion | 0.860 | 0.853 | −0.007 | flat |
| Baker's | 0.907 | 0.908 | +0.001 | flat |
| Contusion | 0.871 | 0.863 | −0.008 | flat |
| Fracture | 0.809 | 0.796 | −0.013 | flat |

**peaked −0.0008, flat −0.0033.** The split is in the predicted direction and it is 0.0025 wide, under
half the holdout floor, so it does not carry the third outcome; and the single largest mover in the
whole table is Synovitis at −0.010, a PEAKED, axial-weighted row going the wrong way. Reading a 0.0025
group separation as the mechanism working would be exactly the number-watching the earlier rows in this
ledger were corrected for.

**MECHANISM: the warm start washed out, and it did not merely fail to land.** Those are different
failures and the training loss separates them. If the prior had never taken effect, the two runs would
be identical; if it had fought the data, the arm's loss would sit above the control's. Neither happened —
the losses track to the fourth decimal at every epoch (1.2117/1.2162 at e0, 0.7518/0.7524 at e6,
0.6508/0.6529 at e8, with the ARM slightly lower), so the 72 numbers were absorbed and then overwritten
by gradients that had already been finding the same structure. `landed.py` asserts the parameter was
constructed and `switch_coverage` smokes it to a written checkpoint, so the plumbing was never the
question here; the science was, and it answered.

**Cost was exactly as the arm predicted, which is worth recording because it is the one clause that
held.** 26.92 GB reserved / 24.93 GB allocated and 368 s an epoch, byte-identical to `band_dense`. A
warm start on an existing parameter is free, and it bought nothing.

**Seat value: +0.0009 as a fifth member, +0.0004 as the best swap** (it would displace `band_dense`,
the member it is a variant of, which is the shape a low-diversity candidate makes). 0 of 1 clears the
+0.0015 the board ruler has ever been shown to resolve. No seat on either ruler.

**CONSEQUENCE, and the arm pre-registered it so it is owed rather than invented.** The refutation branch
says: the head had already recovered the plane-by-finding mapping from study-level supervision alone.
"Supervision never says where" is wrong about THIS head. That retires the warm-start form of the
localisation argument and hands the axis to a real localisation signal — knee-mxj — which is a different
bead and a different kind of input, not another initialisation of an existing parameter.

It is also the fourth consecutive knob on the decode/head family to return nothing: `attend_dense`
−0.0009 (operator), `focal_dense` −0.0044 (pooled width), `band_density` +0.0014 (slice count at fixed
span), `aprior_dense` −0.0021 (head initialisation). Four independent levers on one subsystem, none
clearing a seventh of the floor, is evidence about the subsystem rather than about any of the four.

### `severity_weight` — the knob fired, and moved the graded findings the WRONG way — 2026-08-20

Scaling each label's loss by the severity of the finding, so an obvious-severe case counts for more than
a borderline one. The argument was that a binary target throws away the grade the report actually
carries, and that the graded findings — the osteoarthritis rows, the menisci — are where a recovered
grade should pay.

**The flag landed and the arm is single-variable.** `severity_weight = True` in the checkpoint's own
options block, against a control that predates the flag; every other recorded option matches
`band_dense`, and the one config difference is `slice_regions` absent from the control's record against
`1` here — the `StudyDesign` default arriving in the manifest, not a behaviour change.

**Both arms peak at epoch 8**, same twelve-epoch one-cycle, so this is a matched read:

| epoch | 5 | 6 | 7 | **8** | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|
| `band_dense` | 0.8540 | 0.8550 | 0.8554 | **0.8578** | 0.8572 | 0.8556 | 0.8554 |
| `severity_weight` | 0.8452 | 0.8512 | 0.8503 | **0.8530** | 0.8529 | 0.8530 | 0.8528 |

Mean −0.0048 against a 0.0062 floor: inside noise, wrong sign.

**Which clause fired — and it is the refutation, on the ruler that was pre-registered rather than on the
mean.** The arm predicted a gain CONCENTRATED in the graded rows: Medial and Lateral Meniscus, Medial
and Lateral OA, PF OA. A uniform lift was pre-declared a refutation, and so was a lift concentrated in
MCL. At matched epoch 8:

| row | delta | predicted? |
|---|---|---|
| ACL | **+0.0153** | no |
| Lateral Meniscus | +0.0064 | yes |
| Medial Meniscus | −0.0010 | yes |
| Medial OA | −0.0033 | yes |
| PF OA | −0.0058 | yes |
| Lateral OA | −0.0072 | yes |
| MCL | −0.0035 | no |
| Baker's | −0.0072 | no |
| Fracture | −0.0109 | no |
| Synovitis | −0.0116 | no |
| Effusion | **−0.0144** | no |
| Contusion | **−0.0148** | no |

Four of the five predicted rows are NEGATIVE. The pattern is not uniform, so the mechanism did fire and
did change the fit — this is not a no-op wearing an arm's name. It changed it against the prediction.

**By structure scale the reversal is cleaner than any single row: focal +0.0001, regional −0.0118.** The
graded, regional findings — exactly the ones a severity scale was supposed to serve — are the ones that
lost, and the diffuse fluid rows (Effusion, Contusion, Synovitis) lost most. The honest reading is that
up-weighting severe cases does not teach the grade; it spends the batch on the cases the model already
gets right and starves the borderline ones, and the borderline cases are where a large-fluid AUC is
actually decided. Paired-floor half-widths reach ±0.0238, so no single row above is a result on its own;
the SIGN AGREEMENT across nine of twelve, and the focal/regional split, is what carries the read.

**Seat value: none, on either ruler.** Into the roster of four, 0.8676 vs 0.8679 (−0.0003); the best seat
it could take from any current member, 0.8669 (−0.0010). 0 of 1 clears the +0.0015 the board ruler has
ever been shown to resolve.

**What this does and does not retire.** It kills THIS use of severity scaling — a per-example multiplier
on the existing binary loss. It does not touch the underlying observation that the reports carry a grade
the targets discard; an ordinal target or a graded head consumes that information without reweighting
the batch, and neither has been tried. Recorded so a later ordinal arm is re-tested rather than assumed
dead by this row.

**That hatch was shut the next day, and the family closes with it (2026-08-21).** `grade_ladder` IS the
ordinal target this paragraph names as untried. It was run, and on the OA triple — the only labels whose
target shape it changes — it moves three for three negative, mean −0.0013. So the grade has now been
spent three independent ways against this same control: the target VALUE (`calib_span`, −0.0020), the
loss WEIGHT (this arm, −0.0048), and the target SHAPE (`grade_ladder`, −0.0013). Three parameterisations,
three negatives, none clearing the floor in the right direction. What survives untouched is the
observation itself — the reports carry a grade the binary targets discard. What the three arms jointly
refute is that the grade is recoverable **by a supervision edit** at this corpus size. A graded HEAD (an
output structure rather than a target rewrite) is a different mechanism and remains untried.

**The axis stays open by the owner's call (2026-08-20): the grade is real information and the loss still
throws it away.** What this row retires is the multiplier, not the signal. The successor is an ORDINAL
target — the three YES levels as ordered classes with a cumulative-threshold head — which consumes the
grade without touching any example's weight in the batch, so it cannot reproduce the starvation
mechanism above. Filed rather than queued ahead of the encoder-adaptation arm, which holds the card.

### encoder_adapt — on-domain SSL init, 0.6902 vs 0.8578 control (2026-08-20)

**The flag landed.** The checkpoint's `options` block records
`encoder_weights=.../adapt_small_e6.pt`, and `PretrainedEncoder.into` raises on a variant mismatch or an
unnamed key, so the load could not have been a silent no-op wearing the arm's name.

**The number is not a knob-sized loss, it is a collapse.** Best holdout 0.6902 at epoch 11 against
`band_dense` 0.8578 — **−0.1676**, twenty-seven times the 0.0062 noise floor, and the arm was still
climbing when it stopped (0.6191 → 0.6902 over twelve epochs) rather than converging to a worse optimum.
Seat value is moot at that distance: into the roster of four 0.8616 (−0.0063), best seat swap 0.8578
(−0.0101).

**The pre-registered clause reads UNIFORM, in the losing direction.** By structure scale the drop is
focal −0.1680 / regional −0.1669 — identical to three decimal places, with every one of the twelve
labels down between −0.11 and −0.24. The clause said a uniform lift would credit a better
initialisation rather than the spectral term; a uniform loss says the same thing in reverse. **This arm
did not test the spectral hypothesis.** It tested an initialisation, and the initialisation was broken.

**The mechanism, measured rather than argued.** `kneemri.evaluation.encoder_drift` now reads an SSL
adaptation as well as a fitted member (they key the encoder differently, which is why this check was
missing while the arm ran). Against stock DINOv2-small, six epochs of masked reconstruction moved:

```
stem 0.0794 | b0 0.0712  b1 0.0601  b2 0.0792  b3 0.0340  b4 0.0832  b5 0.0620
             b6 0.0476  b7 0.0432  b8 0.0351  b9 0.0441  b10 0.1812  b11 0.5510
```

Supervision, across every member ever shipped here, moves its most-adapted block by **2.1%** of that
block's own norm and blocks 0–5 by **exactly 0.0000**. Self-supervision moved blocks 0–5 by **3.4–8.3%**
— three to four times the largest movement supervision has ever produced anywhere — and block 11 by
**55%**. Six epochs is nowhere near enough to rebuild a representation (published MAE recipes run
hundreds), so what those numbers describe is a partially destroyed encoder, not an adapted one.

**And the arm's own precondition is what makes the damage permanent.** `--unfreeze-depth 6` freezes
blocks 0–5. The argument for this axis was that the fine-tune has no gradient to overwrite what SSL
writes there; the same fact means it has no gradient to REPAIR it either. The precondition cuts both
ways and only one direction was written down beforehand.

**What is refuted and what is not.** Refuted: this pretraining RECIPE — 6 epochs at lr 1e-4 over the
whole encoder. Not refuted, and not tested: the spectral loss, on-domain adaptation as an axis, and the
frozen-half argument. The successor holds the objective, the data and the schedule fixed and lowers only
the rate to 1e-5, with a **pre-registered gate that is now measurable: early-block drift must land in
supervision's own range (≤ ~0.02) before the checkpoint is allowed to spend a training slot.** A drift
profile is a one-minute CPU read; the arm it guards is seventy-five minutes of card.

**Queued as the successor (2026-08-20): `encoder_adapt_gentle`, same objective at lr 1e-5.** It cleared
the pre-registered drift gate before it was allowed a slot — stem 0.0174, blocks 0–5 between 0.0046 and
0.0179, 0.821 of all drift in the top four blocks — while the spectral term still fell 1.4482 → 0.6196,
so the adaptation happened and spent itself where a representation can afford to move. The clause is
carried over unchanged, because the hypothesis was never tested; only the harness was broken. A third
outcome is now reachable that the first arm could not produce: a result INDISTINGUISHABLE from control,
which would bracket the axis (1e-4 destroys, 1e-5 does nothing, the answer if any lies between) rather
than kill it.

### encoder_adapt_thawed — the third collapse, and it says the adapted encoder is WORSE than what it started from (2026-08-21)

**The arm.** Best holdout **0.7162** at epoch 10, twelve epochs, against `band_dense` 0.8578 (**−0.1416**).
Recipe: the adapted encoder (`--encoder-weights $ADAPTED --encoder-adapt-depth 6`) with
**`--unfreeze-last 12`** — every block trainable — at `backbone_lr 4e-5`, batch 2 × accumulate 6, 20 slices,
band 0.1–0.9, 336 px. Roughly 800 s an epoch against the control's ~230, so ~2.7 h for the slot.

**Found by `ledger_check`, not by a watcher.** It sat finished and unrecorded for the length of a session:
`epochs_done 11` in the checkpoint while the tail of its own log still read mid-epoch. That is the exact
failure the check exists for, and the second time it has paid.

**The mechanism, and it is the informative part.** This is the *unfreeze-deeper* removal the
`encoder_adapt` interpretation named as one of two ways the axis could still be wrong. It fired, and it
moved the number the predicted direction: **0.6902 (freeze) → 0.6875 (gentle lr) → 0.7162 (all twelve
blocks trainable)**, +0.0260 over the frozen case. But the reading that buys is not "unfreezing helps" —
it is that **the score rises monotonically with how much of the SSL representation the fine-tune is
allowed to ERASE**, and it is still 0.1416 short of a control that starts from stock DINOv2. The limit is
therefore not that the fine-tune lacks gradient to overwrite the adaptation (the hypothesis this arm was
built to test). It is that **the adapted representation is worse than the one it replaced**, and the best
thing training can do with it is destroy it.

**What it does NOT rule out.** The other named removal is untouched: the MAE converged on the DEFAULT
12-slice cache while this arm reads a 20-slice 0.10–0.90 one, so the adaptation was fit on a different
slice distribution than it is asked to serve. That is a data-plumbing suspect, not an idea suspect, and it
has to be cleared before the SSL axis can be called refuted rather than three-times-failed. It also does
not rule out a longer MAE — published recipes run far past six epochs, and `encoder_drift` measured the
representation only 55% rebuilt.

**Seat: none.** 0.1416 below the control is not a diversity argument, it is a broken member.

### encoder_adapt_gentle — 0.6875, and the pair is what carries the finding (2026-08-20)

**The arm on its own.** Best holdout 0.6875 at epoch 10 against `band_dense` 0.8578 (−0.1703). Into the
roster of four 0.8616 (−0.0063). No seat.

**THE PAIR IS THE RESULT, and it overturns yesterday's reading of the first arm.** lr 1e-4 gave 0.6902,
lr 1e-5 gave 0.6875 — 0.0027 apart, well inside the 0.0062 floor — from two adaptations whose blocks 0–5
differ FOURFOLD in drift (0.034–0.083 against 0.0046–0.0179). If the quality of the initialisation were
the variable, those two numbers could not agree. **So the early-block damage I named as the mechanism
was not the mechanism.** It was real, and it was not what cost the 0.17.

**What the two arms actually share.** Block 11 moved 0.5510 and 0.2144 of its own norm; supervision has
never moved that block more than 0.0080 (`band_dense`: stem and blocks 0–5 exactly 0.0000, top block
0.0080, largest anywhere 0.0225). The head reads block 11 and the final norm. Both arms therefore handed
the classifier a representation reshaped for reconstruction, and twelve epochs at 4e-5 cannot walk back
21–55%.

**Not a plumbing failure — checked before it was believed.** The fitted arm's own drift profile shows
blocks 6–8 moving 0.0205–0.0239 against control's 0.0167–0.0225, so the backbone received gradient and
trained normally. The collapse is what the encoder was, not whether it was fitted.

### encoder_adapt_thawed — KILLED at epoch 4 of 12, on the ratio rather than on the number (2026-08-21)

**Not a result. A schedule decision, recorded so the arm is re-runnable rather than assumed dead.**

The arm was `encoder_adapt_lower`'s recipe with the imported blocks THAWED — `--unfreeze-last 12
--backbone-lr 4e-5` over an encoder initialised from `adapt_small_e6_lr1e5.pt` at depth 6, testing the
"do not freeze the adapted blocks" half of what `encoder_adapt_lower` said it did not rule out.

**Five epochs, five behind its own matched control**, at 780–863 s an epoch against the control's ~365 s:

| epoch | thawed | loss | `encoder_adapt_lower` |
|---|---|---|---|
| 0 | 0.6651 | 1.3041 | |
| 1 | 0.6533 | 1.3136 | |
| 2 | 0.6456 | 1.2963 | |
| 3 | 0.6739 | 1.2908 | 0.6906 (loss 1.2193) |
| 4 | 0.6600 | 1.2739 | |

The loss reads the same story the whole encoder-adaptation family reads: floored near the control's
epoch-ZERO 1.2162 and still above where `band_dense` starts.

**WHY IT WAS KILLED RATHER THAN FINISHED.** Seven epochs remained, ~1.6 h of card, and the best case was
a number near `encoder_adapt_lower`'s 0.7309 — i.e. a NULL on an axis already 0.13 below the shipped
`band_dense` 0.8578. Under the standing rule (price wall-clock over expected gain BEFORE running, and
justify a predicted-null run as an instrument or not at all) that was the worst ratio on the board, so
the hour went back to the queue. The kill is the honest application of the rule to our OWN work: this
arm was ours, argued, and running.

**WHAT IT DOES AND DOES NOT SAY.** It does not refute thawing — 5 of 12 epochs under a one-cycle
schedule is before the decay tail where these arms do most of their recovering, so no verdict is
written here. What it does say is that thawing does not RESCUE the axis EARLY: the flat-rate 12-block
arm is behind the frozen-6 control at every matched epoch, not ahead of it, so there is no early signal
that unfreezing is the missing piece.

**A CONFOUND THE SUCCESSOR MUST REMOVE, and it is the reason to re-run rather than to conclude.** This
arm gave all twelve blocks ONE flat 4e-5, which is peak rate on blocks 0–5 — exactly the SSL-imported
blocks the whole axis is about. `--layer-decay` (built 2026-08-21) is what removes it. A re-run at
`--layer-decay 0.7` is a different experiment, not a repeat, and it is the one the axis is owed.

### encoder_adapt_lower — 0.7309, and the diagnosis it was built on is refuted (2026-08-20)

**The flag landed.** `encoder initialised from adapt_small_e6_lr1e5.pt (113 tensors, 110 left stock)` —
the stem and blocks 0–5 imported, blocks 6–11 and the final norm stock. Checked before the number was
read, because a depth that silently loaded everything would have reproduced `encoder_adapt_gentle` under
a new name.

**The arm on its own.** Best holdout 0.7309 at epoch 11 against `band_dense` 0.8578 (−0.1269). Seat value
into the roster of four: 0.8591 (−0.0089), 0 of 1 clearing +0.0015. No seat.

**PRE-REGISTERED OUTCOME 3, and it costs me the mechanism I wrote yesterday.** The arm was built on
"the rewritten TOP is the whole collapse" — block 11 moved 0.5510 and 0.2144 where supervision never
moves it past 0.0080, so leaving the top stock should recover roughly control. It imported ONLY the
bottom and still lost 0.1269. **So the damage is not in the top blocks. It is in the frozen bottom.**
The recovery is real but small: +0.0407 over `encoder_adapt` (0.6902) and +0.0434 over
`encoder_adapt_gentle` (0.6875). Dropping the reconstruction-shaped top bought about a quarter of what
the axis lost, and three quarters of it came in through the six blocks the argument said were the safe
part.

**The instrument that says why is the TRAINING LOSS, not the AUC.** Control `band_dense` runs
1.2162 → 0.5780 over twelve epochs. All three encoder arms floor near the control's epoch-ZERO loss:
1.3452 → 1.1978, 1.3393 → 1.1960, and this one 1.2874 → 1.1073 — still falling at epoch 11, still above
where the control started. **These arms are not fitting worse, they are barely fitting at all.** The
head sits on frozen blocks 0–5 that were rewritten for spectral reconstruction, and the six blocks
supervision CAN write cannot re-encode what the frozen half no longer represents. AUC alone said
"worse"; the loss curve says "not training", which is a different failure and a different fix.

**What this does NOT rule out.** The adaptation was six epochs and neither loss term had converged, and
it was run on the DEFAULT 12-slice cache while every arm importing it trains on 20 slices at 0.10–0.90 —
so the encoder is off-domain for the pixels it was imported into, which the stamp added on 2026-08-20
now records. Both are cheap to remove and both point the same way: converge the MAE first, on the arm's
own cache, and either do not freeze the adapted blocks or unfreeze deeper. The axis is not closed; the
`--encoder-adapt-depth` mechanism is proven correct and the initialisation it was fed is not yet a fair
test of it.

**Prior version of this section, kept because it was wrong in a way worth seeing:**

**Successor queued: `encoder_adapt_lower`, `--encoder-adapt-depth 6`.** New option, tested, that takes
the patch embedding and blocks 0–5 from the adaptation and leaves blocks 6–11 and the final norm stock
(the norm stays with the blocks it is calibrated to). This is the axis's original argument stated
exactly — supervision cannot write into the frozen six, so it cannot repair them either, and the top
blocks it CAN rewrite were never the part the argument was about. Its three outcomes are pre-registered
in the arm: a focal lift supports the spectral mechanism, a recovery to roughly control confirms this
diagnosis and closes the axis honestly, and a third collapse near 0.687 refutes the diagnosis and sends
the question back to the harness.

### `base_dense` — a bigger encoder loses on its own ruler and comes back level on the other one — 2026-08-20

**The arm.** `--variant base --batch 6`, everything else the settled band recipe: fold 0, one-cycle,
`--weight-floor --rank-weight 1.0`, 12 epochs, 20 slices, band 0.10–0.90, whole-stack, 336 px,
`--slice-pooling max`, `--backbone-lr 4e-5`, `--recompute-activations`. Verified from the checkpoint's own
`options`, not from the invocation: `variant: base`, `batch: 6`, `slices: 20`, `band_low/high: 0.1/0.9`,
`backbone_lr: 4e-05`, `whole_stack: True`, `slice_pooling: max`, `one_cycle: True`, `weight_floor: True`,
`rank_weight: 1.0`. The batch is halved because 336 px at 20 slices on a 768-wide encoder does not fit
otherwise — peak 27.85 GB reserved, 25.38 GB allocated, on a 32 GB card.

**The curve, against its matched control `band_dense` (DINOv2-small, batch 12).**

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_dense | .7625 | .7759 | .8144 | .8232 | .8355 | .8371 | .8486 | .8459 | **.8512** | .8505 | .8490 | .8491 |

Control `band_dense` peaks at **0.8578** (epoch 8) and finishes 0.8554. Both peak at the same epoch, so
the schedule is not the difference. **-0.0066 at the bests, just outside the 0.0062 three-seed floor**,
at **921 s an epoch against ~370** — 2.5× the card time for a member that reads worse alone.

**Read as a member, it is the opposite.** Rank-averaged over the 909-study fold-0 holdout, added as a
fifth to the `longseat` four (the current board maximum, 0.930) it reads **0.8690 against their 0.8691** —
one ten-thousandth, a sixtieth of the floor — and **+0.0011 against the pinned champion `greedy4_v1`**
(0.8679). Its best swap is -0.0006. **This is the first member this project has mounted whose add is not
negative**: `calib_span` went in at -0.0020, `crop80_tissue` at -0.0005, `dinov3_base_lora` at -0.0013.

**The pre-registration, judged.** The arm asked three things. (1) *Does a bigger encoder beat small on the
settled recipe?* **No** — -0.0066, outside the floor, and the answer is unambiguous at 2.5× the price.
(2) *Is it worth a seat anyway?* **Yes, on the seat ruler** — a checkpoint 0.0066 behind the control on
its own that comes back level once mixed is the signature of an encoder whose errors are not a
re-parameterisation of the other four's, which is precisely what the pool has never had. (3) *Is the
halved batch the alibi?* **Not yet answerable, and it is being answered** — `batch_teardown` is on the
card now running `band_dense`'s exact recipe at `--batch 6` and at `--batch 6 --accumulate 2`. If
`small_batch6` lands materially below the 0.8578 control, base's -0.0066 is partly a batch effect and the
encoder verdict must be re-read at effective 12.

**What it does NOT say.** It does not say DINOv2-base is worse than DINOv2-small. It says base at batch 6
on a recipe tuned for small at batch 12 is worse, which is a statement about an operating point, and the
one confound in it is on the card. `band_dense_long` is the precedent for what happens next: an arm whose
premise was refuted produced the checkpoint that carried the 0.930.

**Shipped.** Mounted in `comp/members.roster` and named by `comp/kernels/kneemri-basedense`, a one-member
delta off the known 0.930 so its board reading attributes to this member alone.

**BOARD RESULT: 0.931 — a new champion, and the third question above is now answered.** The entry took the
board up a tick (55654130). So the member that lost its own matched control by 0.0066, *outside* the noise
floor, is the member that produced this project's best entry. Question (3) resolved by `small_batch6`
below: the batch is not the alibi, and this stays an encoder reading. Full ledger entry in
`comp/SUBMISSIONS.md`.

### `batch_teardown` arm 1 — `small_batch6`: the batch is not a lever, so `base_dense` has no alibi — 2026-08-20

**The question, pre-registered in the arm script before the run.** `base_dense` ran at `--batch 6` because
336 px x 20 slices on a 768-wide encoder asks 56.3 GB against a 32 GB card, so its -0.0066 carried a second
variable it did not choose. This arm prices the batch **alone**, on the encoder where 12 *is* affordable:
`band_dense`'s recipe byte-identical except `--batch 6`.

**Flag landed.** `experiments/arms/batch_teardown.sh` passes `--batch 6` and nothing else off the control;
`$RECIPE` and `$THROUGHPUT` unchanged from `band_dense`.

```
e0 .7514(419s)  e1 .7684  e2 .8084  e3 .8191  e4 .8345  e5 .8396
e6 .8550  e7 .8553  e8 .8598(peak)  e9 .8575  e10 .8576  e11 .8577    (~366 s/epoch)
control band_dense: e8 0.8578 (PEAK), e9 .8572, e10 .8556, e11 .8554
peak memory 13.56 GB reserved, 12.61 GB allocated (vs the control's ~27 GB at batch 12)
```

**Read against its matched control.** Peak **0.8598** vs **0.8578**, **+0.0020** — a third of the 0.0062
floor, and on the *positive* side. Both peak at epoch 8, so the schedule is not what separates them either.
**The batch is not a lever at this operating point.** The first pre-registered branch fires: *"the batch is
NOT a lever here and `base_dense`'s halved batch is not an alibi for whatever it read."*

**What that settles, immediately.** `base_dense`'s -0.0066 is an **encoder** reading, not a batch artefact,
and the 0.931 it went on to score is not one either. The confound named in the champion entry's own
pre-registration is closed by measurement rather than by argument — which is the point of naming it there.

**Seat value: none sought.** This is a dinov2-small member of a pool already seven small members deep, so a
tie on the mean is a correlated member, not a vote the ensemble lacks. Not sent to `member_curve`, not
mounted. The arm's product is the correction, not the checkpoint.

**The bonus fact, and it is not free.** Batch 6 costs **half the memory** (13.56 GB vs ~27 GB) and **369 s
an epoch vs the control's ~361** — under 3% slower — for a delta inside the floor. Under the standing rule
that an offline delta inside the floor is a real board cost of roughly 3x its size, +0.0020 offline is not
a licence to switch the default; it is a statement that batch 6 is *available* when a member needs the VRAM,
which is exactly the case `base_dense` was in.

### `cnx_dense` — the first non-transformer trunk, and it closes encoder diversity as an axis — 2026-08-21

**The flag landed.** `run_comparison` against `base_dense` names `variant base→cnxbase`, with
`train_size 0→352` and `batch 6→2` — the two deviations the arm stated before it ran, both forced by a
stride-32 trunk without gradient checkpointing, and the third (`accumulate 3`) invisible here only
because the control predates the option. Slices, band, rate, pooling, whole-stack and schedule are the
control's.

**One number in that line is `base_dense`'s and not the recipe's, and the distinction cost a smoke the
next day.** `base_dense` itself ran at batch 6, so `batch 6→2` with `accumulate 3` holds THAT control's
effective batch. The DEFAULT is 12 (`options.py:42`), which is what every arm naming no `--batch` runs
at — `band_dense` and `encoder_adapt_lower` among them. So an arm matched against one of those has to
multiply back to 12, not 6, or the batch stops being held and starts being a second variable.

**The arm on its own.** Best holdout **0.7927 at epoch 11** against `base_dense` 0.8512 (**−0.0585**) and
`band_dense` 0.8578. Seat value into the pinned five: 0.8655 as a sixth (−0.0035), and 0.8648 as a swap
for `base_dense` (−0.0042), 0 of 1 clearing +0.0015. At the measured 3× board cost that swap reads as
roughly −0.0127.

**TWO PRE-REGISTERED CLAUSES FIRE, and they are the two that close things.** REFUTATION asked for seat
value at or below zero, like `v3base_dense`; it came in at −0.0042, six times more negative than
`v3base_dense`'s −0.0005. THE EXPENSIVE NEGATIVE asked for solo below 0.80; 0.7927 is below 0.80. The
arm's own text makes the second clause send it to 448 px before the trunk is judged, and **that retry
should not be run** — see the next paragraph, which is the part of this row that generalises.

**THE 448 RETRY IS DEAD ON THREE INDEPENDENT READINGS, and none of them is the mean.** (1) `run_comparison`'s
structure-scale line reads **focal −0.0585 against regional −0.0585**, equal to four decimals: a
uniformly worse model, not a redistribution, and it prints its own verdict — "nothing gained at either
scale, so resolution was not the bottleneck". A coarse-grid hypothesis predicts the focal findings lose
and the regional ones hold; they lose identically. (2) The resolution axis is already priced on this
data: `resolution` at 392 px held at the control's own encoder rate scored **0.8459, +0.0002**
(knee-a8x) — the +0.0066 that row used to claim was the RATE, not the pixels. (3) At MATCHED TRAIN LOSS
rather than matched epoch, which is the comparison that removes "it just needed more steps":
`base_dense` holds 0.8371 at loss 0.8456, this arm holds 0.7927 at loss 0.8332. **A 0.044 generalisation
gap at equal fit** — it is not fitting less, it generalises far worse from the same fit. A 14×14 grid
buying that back is not an arguable mechanism.

**The per-label read does not rescue it and is not close.** Eleven of twelve labels are negative and
**ten clear this pair's own paired half-widths** (up to ±0.0508): Baker's −0.1475\*, Lateral Meniscus
−0.0927\*, Medial Meniscus −0.0863\*, Contusion −0.0804\*, MCL −0.0779\*, PF OA −0.0532\*, ACL −0.0437\*,
Lateral OA −0.0370\*, Effusion −0.0232\*, Medial OA −0.0189\*. Synovitis is the sole positive at +0.0070
and does not clear. This is the widest significant sweep any arm in this ledger has produced.

**The trajectory is a generalisation gap, and the reading was revised twice while it ran — recorded here
because both revisions predate the number.** Holdout 0.7050 / 0.7350 / 0.7495 / 0.7659 / 0.7681 / 0.7760
/ 0.7858 / 0.7908 / 0.7921 / 0.7909 / 0.7927 / 0.7927 against loss 1.2612 → 0.8332. At epoch 6 it was
climbing +0.0098 where the control had flattened to +0.0026, which read as UNDERFIT and made the
schedule (knee-9otf) the likelier suspect than geometry. By epoch 8 that was wrong, and by epoch 11 the
holdout is flat for three epochs while the loss falls another 0.05. **Loss still dropping and holdout
already flat is a generalisation gap; more epochs or a larger rate buy more of the same fit.** knee-9otf
was retitled off the refuted underfit claim on the strength of this.

**CONSEQUENCE, which is what this arm actually bought.** Its own REFUTATION clause states it: with seat
value at or below zero, "the fifth seat was bought by DINOv2-base's capacity and not by diversity of any
kind, *encoder diversity* stops being an axis this project has evidence for, and the next member has to
come from the DATA side rather than the encoder side." Three encoders now say it — `v3base_dense`
−0.0005 within the transformer family, `cnx_dense` −0.0042 outside it entirely, against `base_dense`'s
one paying seat. A different inductive bias is not a different error structure. **The axis closes.**

### `v3base_dense` — "DINOv3 is weak" was an ADAPTATION finding wearing an encoder's name — 2026-08-21

`base_dense`'s invocation with `--variant v3base` substituted for `--variant base`. Rate, slices, band,
pooling, whole-stack, epochs, schedule and batch all held, so `base_dense` is the matched control and the
only variable is the encoder family.

```
e0 .7563(720s)  e1 .7887  e2 .8167  e3 .8307  e4 .8367  e5 .8435
e6 .8455(peak)  e7 .8428  e8 .8451  e9 .8439  e10 .8432  e11 .8429    (~680 s/epoch)
control base_dense: e8 0.8512 (PEAK)   |   LoRA sibling dinov3_base_lora: 0.8268
```

**Against its matched control: 0.8455 vs 0.8512, −0.0057 — INSIDE the 0.0062 floor.** DINOv2-base and
DINOv3-base are the same size at the same recipe and land level. The family, by itself, is worth nothing
here either way.

**Against its LoRA sibling: +0.0187, and that is the finding.** `dinov3_base_lora` read 0.8268 — the same
encoder, the same 336 px, under LoRA r8 over a frozen trunk at 5e-4. Full adaptation at 4e-5 on the settled
band recipe lifts it to 0.8455. The arm's PREDICTION branch fires almost exactly as written (it predicted
"between 0.84 and 0.86" and "~0.02 separating adapted from adapted-frozen"), and the REFUTATION branch —
solo at or below 0.8268 — does not.

**So the pool's two DINOv3 members have been MIS-ATTRIBUTED since 2026-08-16.** Both are the roster's
weakest (0.8215, 0.8268, taken 16th and 17th by greedy over eighteen), and the ledger has been reading that
as evidence that DINOv3 does not transfer to this task. It is not. It is evidence that **LoRA over a frozen
trunk costs ~0.019 here**, measured on the one encoder where both regimes now exist. Every sentence in this
repo of the form "DINOv3 is weak on this task" needs its scope narrowed to the adaptation regime it was
measured under.

**It also re-reads `base_dense`.** That arm lost its own control by 0.0066 and then produced the 0.931
champion, and the reading was "a different encoder FAMILY brings different errors". This arm holds the
recipe and swaps only the family, and gets a **tie**. Two DINOv2-base and DINOv3-base checkpoints at the
same size and recipe are not different in the way that mattered — which means the seat `base_dense` won may
have been bought by its 768 WIDTH against a pool of 384-wide members, not by its pretraining corpus.

**SEAT VALUE IS THE UNFINISHED HALF AND IT IS THE HALF THAT DECIDES.** `base_dense` is the standing proof
that solo score does not predict seat value — it lost its control outside the floor and took the board. This
member has not been through `member_readings` / `record_mix` against `basedense_v1` (0.8690, five members),
so nothing here says whether it earns a seat. **Do not retire it on the −0.0057.** The arm's third branch
(solo clears 0.84, seat value at or below zero) is live and unmeasured, and it is the branch that would say
encoder diversity paid ONCE and not TWICE — which would make the next arm a genuinely different inductive
bias (a CNN trunk), not a fourth ViT.

**SEAT VALUE, measured 2026-08-21 — flat, and the third branch fires on the letter of its wording.**
`record_mix` over the 909-study shared holdout, roster held at the champion five:

```
roster of 5:                        0.8690
  +v3base_dense (sixth seat):       0.8683   (-0.0007)
  v3base_dense FOR base_dense:      0.8684   (-0.0005)
0 of 1 clear +0.0015
```

Both readings are negative and both are an order of magnitude inside the ~0.001 the holdout resolves at,
so what this measures is a TIE, not a loss. The arm's `THE INTERESTING NEGATIVE` branch — "solo clears
0.84 and seat value is at or below zero" — fires as written, and its consequence stands: **a second ViT
of a different family does not buy a second diversity seat.** `base_dense` paid once. The next member has
to be a different INDUCTIVE BIAS (a CNN trunk), not a fourth ViT.

But read the sign, not the magnitude, and note what this arm cannot settle. `base_dense` also read as an
offline tie (−0.0001) against the incumbent four and then paid a board tick — the holdout's one recorded
miss in the UP direction, and it happened on exactly this shape: offline-flat, structurally different.
This is the same shape again. **The −0.0007 is not evidence that a v3base seat would cost a tick; it is
evidence that this ruler cannot tell.** Only a slot can, and a slot is worth more spent on a member the
ruler can actually see. So: the member stays on disk and mounted-eligible, the axis is closed as a
*source of new seats*, and the finding that carries forward is the adaptation re-attribution above, not
the seat number.

**Cost note.** 680 s/epoch against `base_dense`'s ~920, at 25.9 GB — the patch-16 token discount (441 vs 576
tokens at 336 px) is real and shows up in wall-clock as predicted. It ran only after `--vram-gb 27` was
named explicitly; the derived cap refused it at 21.68 GB with 9.17 GB of the card free (see the arm header
and the filed intercept bug).

### `batch_teardown` arm 2 — `small_accum`: three points inside one floor, and the axis closes — 2026-08-20

`--batch 6 --accumulate 2`, everything else held. The arm's own clause was already weakened before it ran:
`small_batch6` did **not** land below the control, so the branch asking whether accumulation *buys the batch
back* had nothing left to buy. What remained was the ranking-pair question — batch 6 forms 15 pairs a step
against batch 12's 66, and `--accumulate 2` does not restore them, by construction.

```
e0 .7412  e1 .7794  e2 .8161  e3 .8389  e4 .8418  e5 .8506
e6 .8538  e7 .8553  e8 .8579(peak)  e9 .8562  e10 .8553  e11 .8551    (~365 s/epoch)
```

**Three arms, one floor, one conclusion.** Control `band_dense` (batch 12) **0.8578** · `small_batch6`
(batch 6) **0.8598** · `small_accum` (batch 6 + accum 2) **0.8579**. All three peak at **epoch 8**, and the
whole spread is **0.0020** against a **0.0062** floor. **The batch is not a lever, accumulation is not a
lever, and the ranking-pair count is not a lever** at this operating point. The axis closes.

**The pair question is answered in the negative, and that is a real finding.** `small_accum` restores the
optimiser step of batch 12 while leaving the pairs at 15 a step. If pairs were what a larger batch bought,
`small_accum` should have tracked `small_batch6` and both should have sat below the control. Instead all
three are level, so at `--rank-weight 1.0` on this corpus the pair count between 15 and 66 does not matter.
**`--pair-memory` is not urgent**, and the knee-55k8 accumulation equality now has a scale check behind its
unit test rather than only a six-study CPU proof.

**Seat value: none sought, for either arm.** Both are dinov2-small members of a pool already seven small
members deep. Neither was sent to `member_curve`; neither is mounted. A `record_mix` swap of `small_accum`
for `base_dense` in the champion five reads **0.8694 (+0.0004)**, 0 of 1 clearing +0.0015 — a correlated
member, exactly as the arm predicted.

**What the pair of arms bought.** Not a member and not a tick: a **correction**, and it landed exactly where
it was needed. `base_dense` shipped with a named confound, `basedense_v1` scored 0.931 with that confound
still open, and these two closed it inside three hours — the encoder reading stands, the champion is not a
batch artefact. Also a standing operating fact: **batch 6 is free here**, at half the memory (13.56 GB vs
~27) and under 3% wall-clock, which is what makes a 768-wide encoder affordable at 336 px in the first place.

### band140 — the FOV axis closes, and MCL is why (2026-08-21)

**The mean said nothing, nine epochs running.** Matched-epoch holdout against `band_dense`: +0.0058,
−0.0016, −0.0011, +0.0015, +0.0069, −0.0081, +0.0006, −0.0068, −0.0037 (e0–e8). Scatter around zero with
no drift, against a 0.0062 floor, at matched loss (e8 0.6585 vs 0.6529). Best 0.8556 at e6 vs 0.8578.

**The pre-registered clause was PER-LABEL on the MCL / ACL / Baker's periphery group, and it reads
negative on the label the hypothesis was actually about.**

| label | e6 | e7 | e8 |
|---|---|---|---|
| MCL | −0.016 | −0.021 | −0.017 |
| ACL | −0.000 | −0.013 | +0.007 |
| Baker's | −0.003 | −0.002 | +0.002 |

MCL is negative at every matched epoch, by the largest margins in the twelve-label table, all clearing
the floor. ACL and Baker's scatter around zero. A group that was supposed to move together did not.

**THE MECHANISM, AND IT IS A CONFOUND THIS ARM COULD NOT SEPARATE.** 140 mm of span across the SAME
pixel grid is coarser mm/px than 120 mm. Sampling density is what thin structures spend first, and MCL
is the thinnest label on the sheet — while the thick ones held flat (Medial Meniscus +0.004, Baker's
+0.002, Medial OA −0.006). So the arm did not measure "does peripheral anatomy carry signal"; it
measured that against a resolution loss on the same knob. The reading that survives is: whatever the
extra 20 mm contains, it does not pay for the density it costs.

**THE knee-n3yx GATE DOES NOT FIRE.** Its PASS branch required periphery POSITIVE with the mean flat
before a 140 mm @ 362 px re-run was allowed a slot. Periphery is negative on its own deciding label, so
that arm is NOT run, and ~75 min of card stays in the queue.

**What this does NOT rule out.** It does not say peripheral anatomy is signal-free — the confound is
sitting on exactly that question. Separating them needs 140 mm at a pixel budget that holds mm/px fixed,
which is a bigger grid, not a re-crop. Resolution is already priced at +0.0002 for 392 px, so that arm
would be buying back a loss rather than reaching for a gain: it prices as an INSTRUMENT (it would close
the FOV axis cleanly) and not as a lever, and it is not queued.

### `grade_ladder` — ordinal supervision lands, and buys nothing the roster can see (2026-08-21)

**Arm** `experiments/arms/grade_ladder.sh`, 12 epochs at 366 s, ~1265 img/s, 291 steps an epoch, peak
26.92 GB reserved. Control `band_dense` @ e8 = 0.8578, the matched recipe with binary heads.

| epoch | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|
| holdout mean AUC | 0.8528 | **0.8565** | 0.8555 | 0.8563 | 0.8552 | 0.8556 | 0.8551 |

**Result.** Best 0.8565 against a 0.8578 control — **−0.0013 solo**, on the wrong side and a fifth of the
0.0062 noise floor. The seat test is the one that decides: over the five held members the best seat
`grade_ladder` could take is **+0.0001**, inside the +0.0015 the ruler can resolve. It is not a member.

**Which link broke.** Not plumbing — `GRADE_LADDER` is one of the switches `Landed` compares before the
card is spent, and this run passed that check, so the ordinal targets reached the objective. Not data:
the same cache and the same fold-0 split as the control, target coverage 1.0 on all twelve labels. Not
the test: the seat test is the ruler five members were selected under. So the reading is about the IDEA,
in this USE: an ordinal ladder over grades adds no signal that twelve binary heads did not already carry
at this corpus size. The labels are already a severity encoding; laddering them re-expresses the same
supervision rather than adding any.

**What this does NOT rule out.** The severity AXIS is not this arm. `grade_ladder` changes how a grade is
SUPERVISED; it does not change how the loss WEIGHTS a severe case against a mild one, which is the axis
the owner kept alive by explicit call and which `severity_weight` carries. A ladder may also be doing
work the mean AUC averages away — this arm was read on the twelve-label mean only, and a per-label read
of the graded labels (the OA triple, which is where a ladder should bite) is still owed and cheap: the
checkpoint is written at `grade_ladder_s2026.pt`, so it costs an evaluation and no training.

**The per-label read, run the same day: the mean was not hiding a win (knee-t607).** It cost neither a
model load nor a GPU-second — both `.holdout.npz` were already on disk, so `run_comparison` read saved
predictions:

    Synovitis -0.0231*  ACL +0.0212*  Lateral OA -0.0115  Lateral Meniscus +0.0093  Medial OA -0.0092
    Fracture +0.0089  PF OA -0.0065  Medial Meniscus -0.0036  Contusion +0.0029  Effusion -0.0024
    Baker's -0.0019  MEAN -0.0013  MCL +0.0003
    half-widths up to +/-0.0367 on this pair's own paired floor
    by structure scale: focal +0.0000  regional -0.0031

The OA triple is the whole test, and all three of it move NEGATIVE: Medial −0.0092, Lateral −0.0115,
PF −0.0065. Every delta sits inside its own ±0.0367 half-width, so the honest claim is not that the
ladder HURT those labels — it is that the ladder produced **no positive signal on the only labels whose
target shape it changed**. A mechanism that works points the right way on three labels at once;
three-for-three wrong-way is what an inert switch looks like. The two starred labels are not graded
labels and disagree in sign with each other, which is what a paired floor of that width predicts.

This closes the target-SHAPE axis on its own pre-registered clause rather than opening a finer one. What
survives is the loss-WEIGHT axis (`severity_weight`, knee-t1ho): reweighting which STUDIES the loss
listens to is a different mechanism from reshaping what the target IS.

### `band_dense_llrd` — the arm changed two things, and the training loss is what says so (2026-08-21)

**Arm** `experiments/arms/band_dense_llrd.sh` (knee-a2q7), 12 epochs at ~580 s, 782 img/s, 1746 steps an
epoch at `--batch 2 --accumulate 6`, peak 6.14 GB reserved. Control `band_dense` = **0.8578** at epoch 8.

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| holdout AUC | 0.7230 | 0.7615 | 0.7899 | 0.8104 | 0.8226 | 0.8353 | 0.8435 | 0.8407 | 0.8481 | 0.8478 | **0.8483** | 0.8481 |
| train loss | 1.2527 | 1.1244 | 1.0480 | 0.9889 | 0.9232 | 0.8767 | 0.8419 | 0.8038 | 0.7699 | 0.7308 | 0.7146 | 0.7013 |

Best **0.8483**, i.e. **−0.0095** against the control — past the 0.0062 floor and the wrong way. The arm's
own READ-IT-ON ladder says a BELOW result means "the early blocks do not want to move at any rate, which
closes the depth axis for good". **That reading is refused here, and the arm's own second instrument is
why.**

**The pre-registration named the training loss as the second instrument, and it fired against the arm.**
The docstring says this arm's loss "should track `band_dense`'s 1.2162 -> 0.5780 curve". It does not:
1.2527 -> 0.7146 at e10, still descending, against the control's 0.5898 at the same epoch. The two start
level and diverge monotonically. A loss that floors ~0.13 high while still falling is the signature of
too little rate, not of wrecked early blocks — the arm is UNDERFIT relative to its control, and the
holdout gap is what underfitting looks like on the ruler.

**The cause is a second variable the arm did not price.** `StudyModel.parameter_groups`
(`kneemri/models/model.py:580`) anchors the ladder at the TOP opened block —
`backbone_lr * layer_decay ** (len(opened) - 1 - depth)`. Under `--unfreeze-last 12` that makes blocks
6-11 — **the exact six the control trains at a flat 4e-5** — run at 4e-5, 2.8e-5, 2.0e-5, 1.37e-5,
9.6e-6, 6.7e-6: a geometric mean of `0.7^2.5` = **0.41 of the control's rate**. So the arm did not add six
whisper blocks to the control. It added six whisper blocks AND cut the control's own six by 2.4x. The
docstring saw the re-grading ("blocks 6-11 are re-graded by depth instead of sharing 4e-5") and then read
the result as though only the depth had moved.

**Link by link, before the verdict.** PLUMBING green, and mechanically so: `encoder_rate_decay`
(`trainer.py:166`) is a `Landed`-compared field read off the optimiser's top two backbone groups rather
than off the flag that asked, so the run could not have passed without the decay reaching the optimiser —
the measured ratio is 0.7. DATA green: identical cache, band, slices, schedule, top-block rate. TEST
green: same holdout, same ruler, delta well past the floor. The BATCH substitution is green and was
priced in advance rather than discovered here (`small_accum` 0.8579 against the control's 0.8578, three
points inside a 0.0020 spread); it also explains the 580 s epoch against the control's 361 s, since batch
2 at accumulate 6 runs 6x the loader and optimiser iterations for the same effective batch. **TRAINING is
the red link**, and it is red for a reason no gate could catch: both variables are legitimate values of
flags that landed correctly.

**What this does NOT rule out — which is most of what the arm was built to ask.** Layer-wise decay as a
mechanism is untested by this run, and so is depth. `Adaptation`'s docstring should keep calling depth
and rate ENTANGLED: this arm did not disentangle them, it entangled them a second way. The one-variable
version anchors the ladder at block 6 so that blocks 6-11 keep the control's flat 4e-5 and only 0-5
decay — control plus six whisper blocks, nothing else moved.

**That clean arm is priced and DECLINED.** It needs an anchor argument in `parameter_groups` (small), a
smoke, and 12 x 580 s = 1.93 h of card, so ~2.2 h all in. The flat 12-block bracket measured depth at
+0.0008; the hypothesis is that this figure is correction-minus-damage and the correction alone is
bigger, but a generous ceiling is ~+0.005 — still under the project's 1.5% bar — at maybe one-in-five
odds of clearing even the 0.0062 floor. 2.2 h for a one-in-five shot at a sub-bar number loses to
`through_dense`'s 40 min, which is why `through_dense` takes the card instead. Re-testable, not dead.

**One contamination to flag rather than hide.** The e4 (672 s) and e5 (599 s) epoch times are inflated by
my own CPU inference job contending for the machine — I launched a `member_curve` seat read believing it
would read saved predictions and it began a full CPU inference pass instead. AUC is unaffected, since
contention changes the wall-clock and not the arithmetic, but those two epoch times are not the arm's.

**The seat test agrees with the mean, so there is no roster rescue.** Against the shipped roster of five
(0.8690): adding it reads 0.8682 (**−0.0008**), and the best seat it could take from any incumbent is
0.8683 (**−0.0007**, displacing `base_dense`). Zero of one clears the +0.0015 this ruler can resolve, and
a negative is charged at roughly 3x on the board (~−0.0020, knee-c0u9). Not a member, not a free look.

**The wall-clock note, because it was not the queue.** The arm was suspended mid-epoch-11 when the
machine entered sleep at 16:45 UTC and did not finish until 20:27:57 UTC on resume; the watcher fired 56 s
later and `through_dense` started immediately. So ~3.7 h of card went to MACHINE SLEEP, not to a stalled
gate — `queue_check` read `live` and `0/1 started` throughout and was right both times. The trainer logs
`wakelock: held`, so a wakelock is taken and it did not prevent this; that gap is filed separately. Worth
writing down because the failure looks exactly like a wedged watcher from the log side, and the second
time I mistake one for the other it will cost more than the reading did.

### `through_dense` — stem inflation at a depth window of 3, and the checkpoint outlived the answer (2026-08-21)

**Arm** `experiments/arms/through_dense.sh` (knee-40wh), 12 epochs at ~379 s, ~1215 img/s, 291 steps an
epoch. Matched to `band_dense` (0.8578) on every flag except `--through-plane`, which fills the encoder's
three input channels with slices *i−1, i, i+1* clamped at the slot's depth ends and gathered strictly
within a slot, instead of three copies of *i*. Same slice axis, same tokens, same parameters, same FLOPs —
free by construction, differing from the control on `slices−2` of `slices` slices.

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| holdout AUC | 0.7403 | 0.7812 | 0.8141 | 0.8312 | 0.8457 | 0.8513 | **0.8558** | — | 0.8558 | 0.8534 | 0.8518 | 0.8526 |
| train loss | 1.2205 | 1.0830 | 0.9899 | 0.9141 | 0.8475 | 0.8014 | 0.7488 | — | 0.6481 | 0.6082 | 0.5883 | 0.5733 |

**The answer is no.** Best 0.8558 against the control's 0.8578: **−0.0020**, a third of the 0.0062
three-seed floor. The pre-registered BELOW-or-FLAT branch fired. It peaked at epoch 6 and slid while the
training loss fell by a further 0.18 — the same late-overfit shape `band_dense_long` showed on the same
recipe.

**The second instrument is the one that reads.** Matched-epoch training loss tracked the control within
0.009 at every epoch (1.2205/1.0830/0.9899 against 1.2162/1.0743/0.9851). The input change was almost
invisible to the optimisation: the model fits the training set at the same rate with neighbours in the
channels as with three copies of one slice. That is a sharper statement than the flat AUC — it is not
that the depth information hurt, it is that the trunk had no way to use it.

**What it retires, stated narrowly, and it is narrower than the arm was filed as.** The arm's own
mid-run correction establishes that `Conv2d(3, D, k=14, s=14)` over channels (*i−1, i, i+1*) is
mathematically IDENTICAL to `Conv3d(1, D, k=(3,14,14), s=(1,14,14))` over that 3-slice volume — same
parameters, same output, exactly. So this **is** stem inflation at a depth window of 3, not a cheap proxy
for it, and the flat reading genuinely discharges the exclusion clause `depth_true` wrote into its own
close, *at that window*.

What it does **not** retire, and this is the honest residue:

- a depth window **larger than 3**, which needs a real `Conv3d` because the channel trick is capped at 3
  by the pretrained stem's channel count;
- a stem that emits **several depth positions** instead of collapsing the window to one linear projection
  per output dimension — the thing no reshaping of a 2D stem can do at any window size.

Both are what a volume-pretrained trunk buys (knee-jucq), and the pre-registration named the mechanism
before the number: the stem's three channel filters are distinct learned kernels, so it can express a
depth derivative rather than only a smear, but there is one output token per patch and those weights were
pretrained to mean **colour**. It has no capacity to carry "the slice at *i*" and "how it changes with
depth" as separate quantities. A flat result is what that predicts, and it points harder at a trunk
pretrained on volumes rather than away from one.

**The checkpoint outlived the answer, and by the largest margin yet.** Priced against the 0.931 champion
five, `through_dense` is **+0.0008** as an add and **+0.0010** taking `base_dense`'s seat — the best of all
**twenty-four** candidates scored on 2026-08-21, mounted and unmounted, against a distribution whose next
value is +0.0005 and whose mounted best is +0.0002. Five times the best move the rest of the mounted pool
can make. It is still under the +0.0015 `BOARD_RESOLVABLE` threshold and is not claimed as an edge.

The reason to believe the seat number over the arm number is mechanical rather than statistical: every
other member in the pool reads the **same pixels**, and their disagreement is recipe, band, rate,
resolution or encoder — all downstream of an identical input tensor. This is the only member that changes
what the encoder is shown at the first op. knee-nas5 measured that axis as under-priced by this holdout by
2.4× (0.0059 offline became 0.014 on the board), and that argument has never before applied at the input
rather than at the decode.

That makes three arms in a row whose premise was refuted and whose checkpoint the roster wanted anyway —
`calib_span` (−0.0020 on its own arm, carried the 0.929), `band_dense_long` (premise refuted outright,
carried the 0.930), and now this. A run that answers "no" still produces a checkpoint, and the checkpoint
is not bound by the answer.

### `half_data` — data volume is the binding constraint, and it is the only axis that has cleared the bar — 2026-08-22

> **PARTLY SUPERSEDED THE SAME DAY BY `quarter_data`, below.** The −0.0234 measurement stands. The
> LOG-LINEAR EXTRAPOLATION built on it does not: a third point at 25% shows the curve is concave, so the
> slope quoted here is the one BELOW us and every price computed forward from it (a doubling at ~+0.020,
> the "~1.7× the corpus" bar, MRNet at +0.0095, `final_fit` at +0.0064) is an overestimate. Corrected
> prices are in `quarter_data`. Read this section for the measurement, that one for the arithmetic.

`half_data` is `band_dense` with `--train-fraction 0.5` and nothing else moved: same twelve-epoch
OneCycle, `slices 20`, band `0.1–0.9`, `backbone_lr 4e-05`, `slice_pooling max`, `--whole-stack`. It was
armed as an INSTRUMENT — it cannot raise the score, it can only say whether the corpus is the wall — and
its three branches were written down before the number existed.

| | control `band_dense` | `half_data` |
|---|---|---|
| best holdout mean AUC | **0.8578** | **0.8344** |
| best epoch (of 12) | 8 | **6** (tied at 10) |
| finish | 0.8554 | 0.8338 |
| peak → finish | −0.0024 | −0.0006 |
| epoch cost | ~368 s | ~204 s |

Curve: 0.7103 / 0.7436 / 0.7774 / 0.7961 / 0.8145 / 0.8174 / **0.8344** / 0.8323 / 0.8333 / 0.8324 /
0.8344 / 0.8338. (Epoch 5's 682 s is a stall — another process took the card — and epoch 4's 237 s its
leading edge. The stall is a TIMING contaminant only; the AUCs on both epochs are unaffected, and the
six epochs after it ran at 203–204 s.)

**THE BRANCH THAT LANDED IS THE THIRD ONE, BELOW BY 0.02+: DATA IS THE BINDING CONSTRAINT.** The raw
delta is **−0.0234**, 3.8× the 0.0062 holdout floor and the largest single-variable delta any arm on
this project has produced against `band_dense`.

**THE SCHEDULE CONFOUND WAS REGISTERED BEFORE THE NUMBER AND IT DOES NOT EXPLAIN IT.** Half the studies
is half the optimiser STEPS, and `--one-cycle` is defined over `total_steps`, so the arm compresses the
cycle as well as the corpus. `schedule_length` prices that effect directly: 8 epochs instead of 12 —
0.67× the steps at the same cycle shape — cost **0.0028**. Scaling that per doubling of steps, 0.5×
costs somewhere in **0.0028–0.0048**, and the bias is DOWNWARD. Net of the widest reading of the
confound the data term is **~0.019**, still 3× the floor and still above the 1.5% bar. This was the
pre-registered interpretability threshold ("only a drop of order 0.02+ is interpretable as data") and it
is met with the confound charged at its maximum.

**THE SECOND READING — MEMORISATION — LANDS ON ITS ONE READABLE CLAUSE.** The registration said that if
the peak-to-finish decline is memorisation of a finite corpus, then half the corpus should be memorised
SOONER: peak epoch EARLIER, decline LARGER. The peak moved **8 → 6**, which is the memorisation
direction. The decline sizes are **−0.0024 and −0.0006**, both a fraction of the floor, so that clause
carries no signal in either direction and is reported as unreadable rather than as a refutation. What
the half-data curve actually does after its peak is not decline but **plateau**: 0.8344 / 0.8323 /
0.8333 / 0.8324 / 0.8344 / 0.8338, six epochs flat inside 0.0021 while the training loss keeps falling
0.8015 → 0.5973. Fitting continued; generalisation stopped. That is the shape of a corpus exhausted.

**MECHANISM: THE MODEL IS DATA-BOUND, NOT CAPACITY-BOUND, NOT DECODE-BOUND, NOT SUPERVISION-BOUND.**
This is the reading that matters, and it is earned by the other arms rather than by this one. Every
knob family has now been measured flat inside the 0.0062 floor — decode and pooling (`attend_dense`
−0.0009, `focal_dense` −0.0044, `depth_regions` −0.0029, `aprior_dense` −0.0021), span and density
(+0.0014 to +0.0053), resolution (±0.0002 twice), schedule (+0.0009), regularisation (`aug_long`
−0.0076), ordinal supervision (`grade_ladder`), layerwise rates, batch and accumulation. Capacity went
the WRONG way at the one rung above (`base_dense` −0.0066 at 2.5× the card time). Against that
background a single lever that moves the number by 0.019–0.023 is not one finding among many; it is the
only axis on the board whose mechanism has ever delivered above the bar.

**THE SLOPE, AND THE HONEST DENOMINATOR.** The corpus is 4,407 studies; the training side is ~3,526 and
this arm fitted ~1,763. So the doubling measured is **1,763 → 3,526 buys ~0.020**. Read log-linearly —
and one point is not a curve, which the arm said in advance — the NEXT doubling is 3,526 → 7,052 and
would need **~3,500 external studies** to clear the same 0.020. Clearing the 1.5% bar needs ~0.75 of a
doubling, about **1.7× the corpus**.

**CONSEQUENCE, and it is the one the arm pre-registered.** External data and models are PERMITTED
(§2.6.a/b, no host prohibition — see `comp/README.md`), so the axis is legal as well as priced.
`knee-ombx` — does the winner licence inherit MRNet's non-commercial clause — stops being a backlog item
and becomes the decision that gates the schedule, exactly as the arm said it would on this branch.
**And the first thing to say about MRNet is that it does not, on its own, clear the bar:** 1,370 knee
MRIs is 0.39× the training corpus, ~0.47 of a doubling from 3,526 → 4,896, which the slope prices at
**~+0.0095** — directional, under the bar, and that is the OPTIMISTIC read because MRNet's labels are a
different and coarser table (abnormal / ACL / meniscus), so it is an auxiliary or pretraining corpus
rather than 1,370 more supervised studies. The corpora that could carry a doubling are OAI and the
combination of several; pricing them is the next question, not this arm's.

**WHAT IT DOES NOT RULE OUT.** It says nothing about WHICH data — a doubling of studies, more slices of
the same studies, or a better-labelled subset are three different mechanisms and this arm separates
none of them. It also does not license the linear extrapolation above beyond its use here as a price:
one point separates "flat" from "steep" and cannot locate the knee. A 25% point costs another ~18
minutes of card and would price the slope properly; that is now worth running, where before the number
existed it was not.

**THE ONE FREE MOVE THIS BUYS IMMEDIATELY, AND IT IS ALREADY BUILT AND ALREADY BROKEN.** The shipped
members are all fitted on the ~80% training split with ~20% held out. A final fit on 100% is +25% data,
~0.32 of a doubling, priced by this slope at **~+0.0064** — under the bar, unverifiable offline by
construction (there is no holdout left to read it on), and free at ship time. That path is not new:
`final_fit.sh` runs `--fold -1` and holds nothing out, so it collects this already. What is new is the
price, and the price is what makes **`knee-5895`** expensive rather than tidy — that bead records that
`final_fit.sh` names neither `--slices` nor the band and therefore takes the parser defaults (12 slices
at 0.20–0.80) instead of the settled recipe (20 at 0.10–0.90). So the one move this arm licenses is
currently spent fitting the wrong recipe, on the one run that has no holdout to notice.

### `adapt_null` — the load path is clean, and the anomaly that bought the slot was already answered in this file — 2026-08-22

`adapt_null` is `band_dense`'s recipe with `--encoder-weights` pointing at **stock DINOv2-small written
through `pretrain.py`'s own `save`** at `--epochs 1 --max-steps 1 --lr 0`. It holds the flag and removes
the adaptation, which nothing in the four collapsed SSL arms had ever separated.

**THE NULL IS CONSTRUCTED AND CHECKED, not assumed.** AdamW's update and its decoupled weight decay are
both scaled by the rate, so a zero-rate step is the identity. `kneemri.evaluation.encoder_drift` then
reads the written checkpoint at **0.0000 on the stem and all twelve blocks** — which confirms the null
AND gives the drift instrument its first negative control. Every gate on this axis has been steered by
that reading and none of it had ever been checked against a known zero. That part of the arm keeps its
value whatever the score did.

**THE SCORE: THE LOAD PATH CARRIES NO COST.** Four epochs against the control at matched schedule
position:

| epoch | `adapt_null` | loss | control `band_dense` | collapsed arms |
|---|---|---|---|---|
| 0 | 0.7451 | 1.2042 | | ~0.63 |
| 1 | 0.7880 | 1.0441 | | |
| 2 | 0.8180 | 0.9452 | | |
| 3 | **0.8240** | 0.8631 | **0.8232** | ~0.63 |

**+0.0008 at epoch 3**, an eighth of the 0.0062 floor, against a separation of ~0.19 the arm was written
to detect. The training loss says the same thing from the other side: 1.2042 → 0.8631 is the control's
own trajectory, where all three encoder arms floored near the control's epoch-ZERO loss (1.1978, 1.1960,
1.1073). **So `--encoder-weights` and the preprocessing on either side of it cost nothing.** Plumbing is
removed as a suspect for the whole family, and the four collapses are evidence about the adaptation, as
every write-up here already said.

**AND THAT IS THE PART WORTH RECORDING AGAINST MYSELF.** The arm was armed on a "magnitude anomaly": two
adaptations whose blocks 0–5 differ FOURFOLD in drift (0.034–0.083 against 0.0046–0.0179) scoring 0.0027
apart, which is dose-response behaviour a representational cost should not show. That observation is
correct and it is **already in this file**, made better, at `encoder_adapt_gentle` — where it is resolved
rather than merely noticed: blocks 0–5 are the wrong blocks to have measured, block 11 moved 0.5510 and
0.2144 of its own norm where supervision never moves it past 0.0080, and both arms therefore handed the
classifier the same reconstruction-shaped top. `encoder_adapt_lower` then refutes even that and lands on
the reading the family now carries — the damage is in the FROZEN BOTTOM, and the head cannot re-encode
through six blocks rewritten for spectral reconstruction. I read lines 1320–1370 and 1455–1500 of this
file when arming the arm and not 1401–1440, which is the one section that had already done the work.
**`CLAUDE.md` says to read the ledger before filing a finding; the cost of not finishing that read is
~25 minutes of card, and this is the second time it has been paid.**

**WHAT THE ARM STILL BOUGHT, honestly priced.** A suspect converted from "the ledger's argument is
probably right" into a measurement, a negative control for the instrument the axis is judged by, and a
`--encoder-weights` path that can now be used without an asterisk. That is worth something and it is not
worth 25 minutes; the same conclusion was available for the cost of forty more lines of reading.

**NO SEAT, as declared.** Best seat into the five held members −0.0029, which the board charges at ~3×.
It was never a candidate.

**WHAT IS STILL OPEN ON `knee-c5n`, unchanged by this arm and now the only live suspects.** The MAE ran
six epochs with both loss terms still falling steeply (pixel 0.9830 → 0.8519, spectral 0.9381 → 0.3489),
and it was fitted on the **DEFAULT 12-slice 0.20–0.80 cache** while every arm importing it trains on 20
slices at 0.10–0.90 — so the encoder is off-domain for the pixels it was imported into. Both are named in
`encoder_adapt_lower`'s section, both are cheap to remove, and neither is touched by a null control.

**AND ONE THING `half_data` NOW SAYS ABOUT THIS AXIS, which changes its price rather than its status.**
The bottleneck measured today is data VOLUME — ~0.020 per doubling of training studies. An SSL MAE over
the same 4,407 studies adds no volume; it offers a better INITIALISATION, which is a different mechanism
against a different failure. So the data finding does not price SSL up, and the two axes have to be
argued separately rather than folded into one "representation" story.

### `quarter_data` — the slope is CONCAVE, and it takes `half_data`'s headline down with it — 2026-08-22

`quarter_data` is `band_dense` at `--train-fraction 0.25`, everything else identical. It exists because
`half_data` priced three downstream decisions off a SINGLE interval, and one interval cannot tell a
straight line from a curve.

Curve: 0.7106 / 0.7062 / 0.7344 / 0.7540 / 0.7629 / 0.7623 / 0.7763 / 0.7861 / 0.7899 / **0.7907** /
0.7898 / 0.7897, at 119 s an epoch. Best 0.7907 at epoch 9, then flat inside 0.0010 for three epochs
while train loss keeps falling 0.7493 → 0.7113 — the same plateau signature `half_data` showed.

**THE THREE POINTS, and the first branch is the one that landed.**

| training studies | share | best holdout | interval gained |
|---|---|---|---|
| ~882 | 25% | 0.7907 | |
| ~1,763 | 50% | 0.8344 | **+0.0437** |
| ~3,526 | 100% | 0.8578 | **+0.0234** |

**25→50 is 1.87× steeper than 50→100 on the raw numbers.** The pre-registered reading for that is
explicit: *the curve is flattening, we sit past the steep part, and every extrapolation above is an
overestimate.*

**THE CONFOUND CORRECTION STRENGTHENS IT RATHER THAN EXPLAINING IT.** Fewer studies is fewer optimiser
steps against a OneCycle spanning `total_steps`. `schedule_length` prices 0.67× the steps at 0.0028,
which is 0.0048 per doubling of step-reduction, so 50% carries ~0.0048 and 25% ~0.0097, both downward.
Adding those back:

| share | corrected | interval |
|---|---|---|
| 25% | 0.8004 | |
| 50% | 0.8392 | **+0.0388** |
| 100% | 0.8578 | **+0.0186** |

Ratio 2.09×. Flattening survives the correction and gets slightly sharper, so it is not the schedule.
The correction is if anything UNDER-priced — the 25% arm ends at train loss 0.7113 against the control's
0.5780 on the same twelve epochs, so it is underfitting, not overfitting, and any residual under-pricing
pushes the curve flatter still. The direction of every uncertainty here is the same.

**WHAT THAT PRICES, and it is the whole point of the arm.** Each doubling buys about 0.48 of the last
one. The next doubling — 3,526 → 7,052 studies, which is the entire external-data programme — is
therefore **~+0.009**, and the geometric sum of ALL remaining doublings is **~+0.017**. So infinite data
barely reaches the 1.5% bar, and the first (and only affordable) doubling of it does not.

**CORRECTIONS OWED TO `half_data`, written the same day it was committed.** That section reads "data is
the binding constraint, and it is the only axis that has cleared the bar", with a slope of ~0.020 per
doubling and a bar needing ~1.7× the corpus. **The first clause stands and the arithmetic does not.**
Halving the corpus really does cost ~0.019–0.023 — that measurement is untouched — but it measures the
slope BELOW us, on the steep part, and using it to price moves ABOVE us was the log-linear assumption
this arm was built to test and has now refuted. Restated:

* the corpus is genuinely a binding constraint at 3,526 studies, and this is still the largest
  single-variable effect the project has measured;
* **going FORWARD from where we sit, data does not clear the bar.** ~+0.009 for a doubling.
* `knee-ombx`: MRNet at 0.39× the corpus is ~0.47 of a doubling, now **~+0.004** rather than +0.0095 —
  a third estimate, the lowest of the three, and the one built on the most points.
* `knee-5895`: `final_fit`'s 80→100% step is 0.32 of a doubling, now **~+0.0029** rather than +0.0064.
  Still worth fixing — it is the deliverable running the wrong recipe — but it is a knob-sized gain and
  must be reported as directional.

**MECHANISM: THERE IS NOW NO LEVER ON THIS BOARD PRICED ABOVE 1.5%.** Every knob family measures flat
inside the 0.0062 floor; capacity goes the wrong way at the rung above; the SSL axis is repriced as an
initialisation argument with two unremoved suspects and no volume claim; and data — the one axis that
looked like an exception this morning — is concave and prices its next affordable step at 0.009. That is
a statement about the corpus and the label table, not a statement about any one method, and it is what
`reader_agreement` predicted from the other end: the board's labels come from the same reports ours do.

**WHAT THIS DOES NOT RULE OUT, and it is where the remaining hours should go.** Three points fix a
curvature but not a functional form; the 0.48-per-doubling decay is a fitted choice, not a measurement,
and the ~0.017 asymptote inherits that. It says nothing about data QUALITY or LABEL quality — a corpus
twice as large is a different thing from a corpus half as noisy, and only the first was measured. And
`CLAUDE.md` is explicit that increments below the bar still COMPOSE against the original baseline: with
no single lever above 1.5% left, a chain of directional gains that sums past it is the only route the
evidence supports, and it is a legitimate one.

**NO SEAT, as declared.** Best seat into the five held members −0.0035, charged ~3× by the board. It was
never a candidate.


### `residual_coupling` / `label_probe` — the label-correlation head is REFUTED for zero card (2026-08-22)

**COST: no GPU at all.** knee-w8h6 proposes CoPAS component (C), a learned interaction over the label
axis, priced at ~1 h of build plus a fold-0 arm. The build was gated on a CPU measurement instead, and the
measurement kills it. The rule that made this cheap is the distinction between the two questions: "do the
labels correlate" (they do, hugely) and "does the correlation survive out of sample" (it does not).

**WHY A HEAD COULD ONLY MINE THE RESIDUAL.** Our twelve sigmoids already read ONE pooled study feature, so
every correlation the IMAGE predicts is already available to all twelve of them. Whatever a label-
interaction block adds has to come from what the image did not explain.

**RESIDUAL COUPLING IS NOT ZERO.** On `band_dense_s2026`, over DECIDED pairs only (7712 of 10908 — the
UNK cells sit at the 0.28 default and a residual there is distance from a prior, not from a fact):

| pair | truth r | residual r | studies |
| --- | --- | --- | --- |
| Medial OA / Lateral OA | 0.6071 | 0.3068 | 563 |
| Contusion / Fracture | 0.4657 | 0.2742 | 358 |
| Lateral OA / PF OA | 0.5663 | — | 570 |
| Medial OA / PF OA | 0.5598 | — | 617 |

**AND IT IS NOT USABLE.** The held-out probe IS the proposed head with its nonlinearity removed: fit a
ridge from all twelve logits onto one label's truth on half the holdout, score it on the other half
against that label's own raw logit, on the SAME rows. Every one of 15 cells loses AUC:

| member | ridge 0.01 | 1.0 | 10 | 100 | 1000 |
| --- | --- | --- | --- | --- | --- |
| band_dense | −0.0128 | −0.0125 | −0.0129 | −0.0145 | −0.0285 |
| base_dense | −0.0235 | −0.0227 | −0.0195 | −0.0165 | −0.0291 |
| calib_span | −0.0149 | −0.0143 | −0.0133 | −0.0128 | −0.0257 |

The penalty spans 10^5 and the sign never turns, so the null is not the regulariser's. Three different
members, so it is not one run's.

**THE HARNESS WAS SUSPECTED FIRST, AND CLEARED BY A NULL CONTROL.** `HeldOutProbe.alone` restricts the
same machinery to the label's own logit; a ridge on one monotone feature is monotone, so it must reproduce
raw AUC exactly. `null_control_max_abs` = **0.0** on all three members. The −0.0125 is mechanism, not
arithmetic — the same question `adapt_null` had to answer before four collapses could be read as evidence
about an idea rather than about a load path.

**THE MECHANISM, NAMED.** What survives in the residual is SHARED STUDY-LEVEL ERROR: when a study is hard,
all twelve logits are wrong in the same direction, so label A's logit points the same wrong way as B's and
cannot correct it. Correlated errors are not recoverable errors. That is why a matrix with r = 0.31 in it
still yields a probe that loses.

**THE PRE-REGISTERED LEAK FIRED EXACTLY AS WRITTEN.** knee-w8h6 said in advance that "a gain that lives
entirely in the RARE labels is the prior leaking, not correlation being used". The only two labels that
improved are the rarest — Synovitis is 65 scored studies of 909.

**WHAT THIS DOES NOT RULE OUT.** It kills the LINEAR label-interaction refinement on OUR twelve-head,
one-pooled-feature architecture. It does not kill label structure as a supervision signal (a joint loss
shapes the shared feature during fitting, which is a different intervention from refining frozen logits),
and it does not kill CoPAS component (A). It RAISES (A)'s value: if the label half is worth ~0, the whole
of CoPAS's expected gain has to come from (A), which is what an instrument is for. knee-pq8u repriced.

**NO SEAT, no submission, no card spent.**

### vjepa_probe — volume pretraining is a TIE on frozen features, and the tubelet path was unarmable (2026-08-22)

| arm | trunk | read | holdout mean AUC |
| --- | --- | --- | --- |
| control | DINOv2-**large**, 1024 wide, 24 blocks, 304M | 252 px, per-slice, mean pool | **0.7669** |
| `vjepa_probe` | **V-JEPA 2** ViT-L, 1024 wide, 24 blocks, 326M, tubelet 2 | 256 px, whole 12-slice stack as frames | **0.7620** |

**−0.0049, inside the holdout's 0.0062 floor.** The arm pre-registered three branches and this is the
middle one, verbatim: *"WITHIN the floor — the trunk is not BETTER, which does not decide the seat
question. A seat is bought by DISAGREEMENT and the holdout is structurally blind to it. Do not read a
tie as a kill."* So the reading is a tie, not a refutation, and it licenses fitting ONE member priced by
`seat_value` against the champion five.

**The capacity confound was removed by construction and that is what makes the tie readable.** The
obvious probe would have set a 1024-wide 24-block trunk against the 384-wide 12-block DINOv2-small cache
already on disk, where a win reads "large beats small" — a lever `basedense` already refuted — and a
loss reads as nothing. DINOv2-large was encoded here specifically so the surviving difference is
PRETRAINING DOMAIN rather than capacity. 252 against 256 because patch 14 cannot tile 256 and patch 16
was fitted at it; forcing either onto the other's grid means interpolated position embeddings, so each
reads the nearest size its own grid tiles.

**WHAT THE TIE DOES NOT RULE OUT, and this is the load-bearing half.** V-JEPA's objective is prediction
in LATENT space, not linear separability, so a flat frozen probe is a known property of that objective
rather than evidence about a fine-tuned member. The frozen reading is the cheap path; a tie blocks the
cheap path only.

**IT ALSO DOES NOT RULE OUT THE READING, because the tubelet trunk has never actually had a window wider
than 3.** Audited the same day: `Trainer.whole_stack` is the only path to a window wider than `GROUP` = 3,
and its guard demanded per-slice pooling — which a volumetric read is not, by definition. So the whole
depth axis this trunk exists for was unreachable from any flag combination, and every number above was
taken with the trunk's advantage partly spent. Two defects fixed the same day, both card-free:

* `knee-2nxh` — the VRAM cap ignored the depth-axis stride. It OVERSTATED a tubelet trunk 2x (`units` is
  per-slice; a tubelet trunk costs grid²/tubelet per slice). At the only read that exists — `VolumeRead
  .check_tokens` asserts the native 256 crop — 20 slices, batch 6, 6 blocks open derives **38.3 GB
  uncorrected against a 31.8 GB card, 21.7 GB corrected**. The encoder was never too big; the cap was
  double-counting depth.
* `Model.check_whole_stack` — the whole-stack guard moved off the caller and onto the model, which is the
  only thing that knows what it can be handed. It now admits a tubelet trunk at any window its tubelet
  divides, and refuses channel stacking, which genuinely has three channels and no more. This is the
  same bug class the guard's own docstring already recorded once: it tested `!= "max"` and silently
  rejected attention pooling the model supported.

So the honest state of this axis is **not "volume pretraining is flat"** — it is *"volume pretraining is
flat on a frozen probe, taken through a reading that could not use the trunk's depth axis, and the
harness holes that made that so are now closed."* The re-test is a fitted member at `--whole-stack
--slices 20 --train-size 256`, which is armable for the first time as of this entry.

### vjepa_warm — the re-test lands in the TIE band, and the finding is the DISTANCE, not the score (2026-08-22)

The arm the entry above licensed: V-JEPA 2 ViT-L fitted as a member at `--whole-stack --slices 20
--train-size 256 --batch 6 --backbone-lr 4e-5`, six epochs, warm-started from `vjepa_member_e5.encoder.pt`
so the schedule is not spent re-learning what the cold run already learned.

| ruler | `vjepa_warm` | reference |
| --- | --- | --- |
| solo holdout, best over schedule | **0.8337** (epoch 3) | roster members 0.8525-0.8578; cold `vjepa_member` 0.8313 |
| seat value, add and best swap | **+0.0002** | 0 of 1 clears +0.0015 |
| ordinal concordance, mean over pairs | 0.743 | `lr4e-5_ws` 0.748, `aug_long` 0.738 |
| residual distance, nearest neighbour | **0.8210** (mean 0.8100) | roster's MOST DISTANT seated member is 0.8801 |

**Solo lands in the pre-registered TIE band.** The arm registered ≥0.845 as licensing the
volume-pretraining axis, 0.83-0.845 as the tie that retires it, <0.82 as "suspect the harness". 0.8337 is
the middle branch, so the axis is retired for this use.

**The warm start bought SPEED, not CEILING.** +0.0024 over the cold run's 0.8313 is a third of the 0.0062
floor — not a number. What did move is where the schedule starts: epoch 0 scored 0.8247, above the cold
run's epoch-4 endpoint of 0.8177. The bridge carried perfectly and then had nothing to carry the trunk
past. That distinguishes the two claims cleanly: the warm-start asset works, the trunk's ceiling on this
task is ~0.834.

**THE ACTUAL FINDING IS THE DISTANCE.** Nearest seated neighbour 0.8210 against a seated spread whose
most distant member sits at 0.8801 — this is the first genuinely different prediction family this project
has produced, by 0.059, and it is not the broken-representation kind (`mae_matched`'s 0.4333 above is
what uncorrelated-because-broken looks like; 0.82 is a working model that disagrees). It bought **+0.0002**.

`knee-053z` asked for DISTANT-AND-STRONG: nearest under 0.88 and solo over 0.85. This clears the distance
clause by a mile and misses the solo clause by 0.017. **Distant-and-weak buys nothing.** Distance is
necessary and demonstrably not sufficient — the ruler's verdict was correct, what failed was the
INFERENCE from distance to seat.

**NAMED HARNESS SUSPECT: the regimen confound.** Six epochs against the roster's twelve. Argued against
by the arm's own overfit signature rather than dismissed: train loss fell monotonically 0.9198 → 0.6189
across all six epochs while holdout stopped improving at epoch 3 and drifted 0.8337 → 0.8311 → 0.8306. A
schedule that is still generalising when it stops looks different from that. Not conclusive, but the
cheap read points away from "it needed more epochs".

**WHAT THIS DOES NOT RULE OUT.** (a) A twelve-epoch schedule closing the solo gap while keeping the
distance — that is the one arm that would overturn the seat verdict, and it costs ~3.2 h of card.
(b) The warm-start bridge asset itself: three witnesses, 587 tensors, 0 left stock, and it opened above
the cold endpoint. It is re-testable against any trunk. (c) The distance ruler, whose reading was right.
(d) V-JEPA at a larger read — every number on this axis was taken at 256 px with 20 slices.

### `mae_matched` — the fifth collapse, and the fix I reached for was already in this file (2026-08-22)

`mae_matched` is `band_dense`'s recipe with one flag added: `--encoder-weights` pointing at
`pretrain_small_matched_e8`, DINOv2-small given eight epochs of masked reconstruction over 4,407
studies x 20 slices = 402,600 knee-MRI slices. Every other encoder on this board was pretrained on
natural images. This one has seen nothing but the domain.

It scored **0.6735 at epoch 3**, against `adapt_null` 0.8240 and `band_dense` 0.8232 at the same epoch.

**The arm's own header said a null would not land here, and it did.** It reads: "Collapsed arms sit at
0.63-0.73, so the floor that says 'the weights loaded and broke the trunk' is far below the band this
lands in — a null here will not be that." 0.6735 is inside that band. So the pre-registration's
`< 0.824` clause applies, and which link broke comes before any verdict.

| suspect | verdict | what says so |
|---|---|---|
| plumbing | CLEARED | `adapt_null` is the same `--encoder-weights` path with a null adaptation and reads 0.8240 |
| data | CLEARED | checkpoint stamps cache `336px_20sl_0.1-0.9band`, the control's own key |
| training (variant) | CLEARED | `PretrainedEncoder.into` RAISES on a variant mismatch and did not fire; checkpoint `small` equals the `TrainingOptions` default, and `band_dense` names no `--variant` either, so byte-identity against the control holds |
| test | CLEARED | matched epoch, both controls already on the ledger, neither re-run |

**The remedy was already run, and it is documented eleven lines above the code that loads the weights.**
`pretrained_encoder.py:47-58` records that a reconstruction objective rewrites the TOP block — block 11
moved 55% and 21% of its own norm across two measured adaptations, where supervision has never moved it
more than 0.8% — and that the head reads that block. It then argues the axis only ever concerned blocks
0-5, which supervision provably cannot write (drift exactly 0.0000) and therefore cannot repair. That
argument stated exactly is `--encoder-adapt-depth 6`, and that arm exists: `encoder_adapt_lower`,
**0.7309**, with its diagnosis already recorded here as refuted.

So the family now reads across TWO pretraining objectives and TWO import depths:

| arm | objective | import | score |
|---|---|---|---|
| `mae_matched` | spectrally-shaped MAE | all blocks | 0.6735 |
| `encoder_adapt_gentle` (1e-5) | SSL-adaptive | all blocks | 0.6875 |
| `encoder_adapt_gentle` (1e-4) | SSL-adaptive | all blocks | 0.6902 |
| `encoder_adapt_thawed` | SSL-adaptive | bottom six, thawed | 0.7162 |
| `encoder_adapt_lower` | SSL-adaptive | bottom six, frozen | 0.7309 |

Five arms, 0.09 to 0.15 below a control that simply starts from stock DINOv2. Importing less of the
adapted trunk helps by about 0.05 and never closes the gap, so the damage is not confined to the top
block the mechanism names — it is the representation, not the layer.

The diversity numbers say the same thing from a second direction. `mae_matched`'s nearest seated
neighbour is **0.4333** against a seated spread of 0.8801-0.8911. That is not a distant member, it is a
model whose predictions are nearly uncorrelated with everything on the roster, which is what a broken
representation looks like rather than a complementary one. Seat value **-0.0041**, 0 of 1 clearing
+0.0015. No seat.

**MECHANISM.** An encoder adapted by any reconstruction-shaped objective in this repo hands the
classifier a representation shaped for reconstruction, and the classifier does worse with it than with
ImageNet features it has to adapt itself.

**HONEST PRICING NOTE.** `knee-jk3`'s 2026-08-21 board re-read already listed SSL-as-init CLOSED, and
that prior was flagged before this arm started. Running it was still the right call on sunk cost — the
65-minute pretrain was already spent and the card was free — but it should have been queued as an
INSTRUMENT retiring the axis, not written up as an improvement with a `>= 0.838` upside clause.

**WHAT THIS DOES NOT RULE OUT, kept re-testable.** Volume-native pretraining is a different asset: the
trunk is pretrained by someone else on video volumes and imported whole, so it never passes through this
repo's adapt-and-import path at all, and none of the five collapses above is evidence about it
(`knee-rpkw`). A pixel-weighted or much longer MAE is also untouched, but five collapses across two
objectives and two depths makes it a poor bet, and it should not take a card ahead of anything on
`knee-jk3`'s surviving list.
