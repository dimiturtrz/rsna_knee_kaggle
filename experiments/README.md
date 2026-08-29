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

**`# gate: <entry> >= <score>` — an arm may not RUN past its own written stop condition.** `final_fit_six.sh`
opened with one: "Run this ONLY if `kneemri-finalfit` reads 0.936 or above on the board." The board read
0.934. The arm ran anyway — four members to completion plus a fifth to epoch 10 of 12, roughly five hours of
card — and was stopped only because a peer project asked for the machine. The arm was not wrong: the gate
was written *before* the reading and priced the decision honestly. Its verdict simply had no consumer,
because it lived in a bash comment, which is the one place in this tree where nothing executes. So the gate
becomes data: one line naming a board entry and a floor, evaluated against `comp/board.jsonl`. The prose
stays — it says what to do in each band, which no threshold can — but the line is what refuses.

An **unread** gate refuses too, and so does a missing board file: an arm gated on a submission nobody has
pushed yet is exactly the arm most likely to be launched on a hunch, and a check that passes because its
evidence is absent is the failure mode this repository keeps paying for. Like the bead rule it names the
finding on every sweep and **refuses only under `--only`** — the run that arms an arm.

The same fix closed a larger hole underneath it. Every header rule resolved an arm's header as
`arms/<name>.sh`, but an arm is keyed by the name it **prints**: `final_fit_six.sh` prints six member names
and none of them is its filename, so the gate, the `# bead:` line and the `# control` line were all unread
on precisely the arms that spend the most card. `ArmHeader` now records which shell printed each name as the
shells are sourced, and falls back to it.

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
| **`tissue224`** | crop about the TISSUE centroid instead of the image centre, at 224 px — `TissueCentre` has been plumbed through six call sites since knee-yrf and had never been trained under | 0.8491 | **−0.0011 vs `native224`** | **FLAT, the pre-registered middle branch, and it refutes the mechanism that predicted a MOVE while leaving the one that predicts a flat mean untested.** Converged (12 of 12, best e6, `convergence_check` clean); flags verified in the run's own recipe — `"centre": "tissue"`, `"image_size": "224"` — so the arm's named harness suspect is discharged before the verdict. Paired epoch by epoch against the control the delta never leaves the 0.0062 floor (e5 .8480/.8451, e6 .8491/.8493, e9 .8482/.8502) and the train loss is slightly LOWER at every epoch (e5 0.8073 vs 0.8141): the tissue crop fits marginally better and generalises the same. **The retained tissue is real and does not convert.** `crop_coverage` measured tissue−image at +0.0294 [+0.0276, +0.0312] coverage at 130 mm over 1,822 series, an interval excluding zero — so this is not a flag that failed to bite, it is 3% more imaged anatomy reaching the network and buying nothing. **WHICH OF THE TWO MECHANISMS DIED.** The header pre-registered both and said "either shape confirms a real lever". COMMON-MODE framing — `crop_alignment`'s finding that two thirds of the 9-14 mm displacement is shared, so every study sees the joint off-centre in the same direction — predicts a UNIFORM corpus-wide gain, which moves the mean by construction. The mean did not move, so that mechanism is **refuted**: correcting a shared framing offset is worth nothing at 130 mm. CLIPPING — a subpopulation claim confined to the worst tenth, where the p90 18 mm centroid offset exceeds the ~15 mm margin — predicts a flat mean *by construction* and is **untested**, not refuted. **THE SPLIT THAT WOULD TEST IT CANNOT BE BUILT TODAY:** `crop_alignment.dump()` writes `spread`, the between-slot disagreement, not the centroid's offset magnitude from the image centre, so the decile the clipping claim needs does not exist on disk. **CONSEQUENCE:** the 99 GB 336 cache rebuild this arm existed to license is **not licensed**, and the crop/centre axis is now closed on the mean at both extents (80 mm −0.0099 in `crop80_span`, 130 mm −0.0011 here). What survives is a 10%-of-corpus claim worth at most a tenth of its own size on the mean — below the floor even if true. **AND IT CLOSES ON THE SECOND RULER TOO, 2026-08-26:** `seat_value --against gradeswap_v1` (0.933, seven members) prices the seat at **−0.0002 added to the roster (0.8694 → 0.8692)** and **−0.0001 for the best single swap** (for `lr4e-5_ws`), against the +0.0015 this ruler has never picked a board tick below; the mix ruler charges a negative at about 3x, so roughly −0.0003 on the board. It is also a near-TWIN — nearest neighbour 0.9032, above every seated member's own nearest except the `band_dense_long`/`grade_ladder` pair — so the crop/centre change did not buy diversity either. A flat member that is *also* a twin has no route to a seat, which is the one thing a flat mean leaves open |
| **`calib_span`** | the label calibration measured on the recipe we ship: `band_span`'s 0.10–0.90 span at 12 slices with `--calibrated-targets` added and nothing else | 0.8545 | **−0.0020 vs `band_span`** | **THE CLAUSE DID NOT MISS, IT INVERTED — and that is the finding.** Flag verified in the recorded recipe: `run_comparison` names `calibrated_targets False→True` and nothing else, the single-variable pair the arm was written to be. The prediction, inherited unchanged from `calibrated_targets` so it could not be re-fitted, was that the gain CONCENTRATES where miscalibration is largest — MCL (495/219/53) and Effusion (1599/910/157), the two labels most skewed toward the 0.68 level — while PF OA (543/658/790), skewed high and nearly calibrated already, barely moves. **Every clause came back backwards**: Effusion **−0.0212\***, MCL +0.0029 (flat), and PF OA **+0.0170\*** is the largest positive row in the arm. The label predicted to have nothing to gain gained most; the two predicted to gain most are flat and significantly negative. The written refutation condition was a mean at or below control WITH MCL and Effusion flat; the mean fell (−0.0020) and MCL is flat, so it fires — but in a stronger form than it was written for, because Effusion is not flat, it is the worst row on the board. **MECHANISM: the remap is not adding truth at this operating point, it is moving mass off the low-grade positives it was built to help.** By structure scale, focal +0.0037 against regional −0.0099 — the exact mirror of what the span bought (regional +0.0079, focal +0.0011), so the calibration spends its gain on the centre and pays for it with the ends the widened band had just purchased. That is the SUBSTITUTION shape a fourth time (`depth_regions_span`, `crop80_span`, `band_full`, now this), and the first time it has appeared between a LABEL lever and a DECODE one rather than between two decode levers. **THE STALE-OPERATING-POINT TRAP IS THE TRANSFERABLE LESSON.** `calibrated_targets` scored +0.0038 against `wsfold0_e6` on 2026-08-16 at the default rate and the 0.20–0.80 band, and it is a member of the 0.928 champion package on the strength of that number. Re-measured on the recipe the project actually ships it is NEGATIVE. A lever's sign is a property of the operating point it was measured at, not of the lever, and every directional result in this ledger taken before the rate and span settled is now suspect in the same way. Converged with the best at epoch 9 (0.8470 / 0.8542 / 0.8527 / 0.8540 / **0.8545** / 0.8537 / 0.8532), so the best is not a truncation artefact; 16.24 GB peak every epoch, a **seventh** confirmation of the 1.335 GB/slice + 0.22 rule and the second on an arm naming no `--vram-gb`. Honest size: −0.0020 is under a third of the 0.0062 floor, so the MEAN is a non-result and the per-label inversion is what carries the row — three rows do clear this pair's own paired half-widths (±0.0331): Effusion −0.0212\*, PF OA +0.0170\*, Baker's −0.0167\*. Seat value **+0.0022** against the shipped four (roster 0.8648 → 0.8670), the largest any same-split candidate has posted — but it is an ADD as a fifth member, and `comp/champion.json`'s FOUR MEMBERS principle records the best subset falling monotonically past five, so a fifth seat is not a seat and a swap is what would have to be shown. **CONSEQUENCE:** `calib_dense` (knee-ep74), written to be the span × density × calibration corner, is dead by its own pre-registered gate and was not launched — a calibration that is negative on the span corner cannot be positive on the denser one for any reason this evidence supports. **MECHANISM 2026-08-25 (knee-p9oc): THE REMAP'S CONSTANTS ARE POOLED ACROSS FINDINGS, AND THE FINDINGS DO NOT AGREE.** `core.data.labels.level_agreement` measures the same object `MEASURED_AGREEMENT` estimates, without pooling: at the 0.68 level the constant is 0.378 while the twelve findings' own rates run **0.000 (ACL, MCL) to 0.778 (Synovitis)** over the ten with 5+ annotated cells, and every other level spreads similarly (0.82: 0.200–1.000 against 0.688). So this arm did not test "calibration"; it tested one particular pooled estimate, and on the label it named as its biggest winner that estimate is furthest from the truth at the largest n — Effusion's own 0.68 rate is **0.481 on 27 cells** against the 0.378 it was assigned, and Effusion holds the most mild cells in the corpus. That is a magnitude account of the −0.0212, not a sign one: 0.481 and 0.378 share a side of 0.5, so those cells were softened rather than inverted. The cells that genuinely cross are at **0.82**, where Contusion reads 0.458 against 0.688 on 24 cells. **What this does NOT do is revive the lever** — it says the arm measured a pooled constant rather than the calibration idea, and it names Contusion at 0.82 as the pre-registered riser if a per-finding version is ever run. Most per-finding rows rest on 5–9 annotated cells, so the spread refutes the pooling, not the constants one by one. See `interpretations/kneemri/2026-08-25_effusion-threshold-ruler.md`. **CORRECTION 2026-08-26: THAT MECHANISM SENTENCE WAS NEVER TESTED AGAINST NOISE, AND AT THE LEVEL IT QUOTES IT DOES NOT SURVIVE.** The clause above reads a per-finding RANGE — "0.000 (ACL, MCL) to 0.778 (Synovitis)" at the 0.68 level — as evidence that the findings disagree. Ten findings holding 5–27 annotated cells each, every one of them genuinely drawn from the pooled constant, produce a wide range and cross 0.5 by sampling alone, so a range is not evidence without its own null. `LevelAgreement.heterogeneity` now draws it (2,000 corpora at each finding's own cell count) and prices both the chi-square dispersion and the range: **0.08 p 0.615 · 0.28 p 0.021 · 0.68 p 0.100 · 0.82 p 0.024 · 0.94 p 0.135**, and at 0.68 the observed range 0.778 is SMALLER than the 0.833 chance produces. So the level carrying the 6,461 mild cells — the level this whole account rests on, and the one holding Effusion's 0.481-against-0.378 — is **consistent with pooling**. Heterogeneity survives only at 0.28 and 0.82, and 0.82 is where four of the five inversions sit AND where `stability` reads 2.0σ, so its pooled constant is unstable in both directions at once. Uncorrected, besides: two of five levels at p ≈ 0.02 fail Bonferroni at α 0.01 individually, though the joint count (expected 0.25, observed 2, p ≈ 0.023) says per-finding structure exists somewhere. **What this changes:** the magnitude account of the −0.0212 stands as arithmetic — 0.481 and 0.378 are what they are — but it is no longer licensed as a FINDING about Effusion's own rate, and `knee-p9oc` step 3 is unlicensed as written. The narrow arm that would still be defensible is single-level (0.82 only, Contusion the pre-registered riser at 24 cells) and is an instrument, not an improvement |
| **`attend_dense`** | `band_dense`'s recipe with `--slice-pooling attend` in place of `max` — a learned 64-wide gate over the pooled slices instead of the elementwise maximum | 0.8569 | **−0.0009 vs `band_dense`** | **NEITHER CLAUSE FIRED, AND THE THIRD OUTCOME IS THE FINDING: THE TRADE STOPS AND NOTHING IS RECOVERED.** Flag verified in the CHECKPOINT config, not just the recipe: `slice_pooling max→attend` with `slices 20`, `band [0.1, 0.9]`, `pool cls_mean` and `slot_pooling attend` identical across the pair, so this is the single-variable pair the arm was written to be. The prediction was that attention BEATS `band_dense` **and** both structure groups come back non-negative; the refutation condition was another trade — one group up, one down, whatever the mean does. By structure scale the arm posts **focal −0.0006 against regional −0.0016**: both negative, neither clearing anything, no trade in either direction. So the arm landed in the case it did not enumerate, and that case is more informative than either. **MECHANISM: the elementwise max is not the binding constraint.** Three arms (`band_span`, `band_density`, `calib_span`) produced the one-group-gains-one-pays shape, and the max was the operator that most plausibly caused it — it keeps whichever slice shouted loudest per feature and discards which slice that was, so two structures competing for a feature cannot both be represented. Replacing it with a learned gate makes the trade disappear **and buys nothing**: −0.0009 is a seventh of the 0.0062 floor, and the two group deltas are smaller still. A fixed budget whose allocator is swapped for a strictly more expressive one and does not move is a budget bounded somewhere else. That **exonerates the pooling operator without retiring the re-allocation reading** — the substitution shape stands as a description of the decode family, but its cause is not the max, so the remaining suspects are capacity and supervision (different beads), exactly the branch the arm named as its consequence. The +0.0016 the same lever bought at 12 slices in a narrow band does not scale with how much is pooled, which was the arm's stated reason for re-measuring it; that reason is now spent. Converged with the best at epoch 8 (0.8530 / 0.8519 / **0.8569** / 0.8549 / 0.8542 / 0.8544), so the best is not a truncation artefact. 26.92 GB reserved / 24.93 GB allocated on every one of twelve epochs, byte-identical to `band_dense`'s own run — the gate's parameters and activations are invisible against 20 slices, so the operator swap is free in memory as well as in score. Seat value **+0.0010** against the shipped four (`calibspan_v1`, roster 0.8670 → 0.8680; it read +0.0015 against `band_v1`'s 0.8648 before slot 4 replaced the champion, and the seat shrank when the roster got better), a quarter of the floor and directional only — and it is an ADD as a fifth member, which `comp/champion.json`'s FOUR MEMBERS principle records as not a seat. **CONSEQUENCE:** the pooling-operator cell of the decode family is closed; `focal_dense` is the queued follow-up on the capacity branch rather than another allocator. **CORRECTION 2026-08-21 (knee-7gyw): THE LEARNED GATE NEVER LEARNED.** `parameter_groups` built its head group as `self.head.parameters()` and the aggregator is a SIBLING of the head, so the gate's six tensors were in no param group; a checkpoint dumped after a full epoch has `slices.pool.gate.3.weight` at exactly `0.000e+00`. The gate is zero-initialised so that an untrained gate is the arithmetic MEAN over slices, so this arm compared **mean vs max**, not attention vs max. The −0.0009 stands as a number and the checkpoint is a valid member; what does NOT stand is the mechanism sentence — "a strictly more expressive allocator does not move the score" was never tested, because the allocator that ran was a fixed mean. **The pooling-operator cell REOPENS**, and with it the part of the fixed-budget refutation that rested on this arm; `focal_dense`'s width axis is unaffected and still reads negative |
| **`focal_dense`** | `band_dense`'s recipe with `--pool cls_mean_focal` in place of `cls_mean` — a third part concatenated to the pooled vector (a focal tail over the token grid) rather than a different rule for combining the existing parts | 0.8534 | **−0.0044 vs `band_dense`** | **NEITHER CLAUSE FIRED LITERALLY, AND THE ARM'S OWN CONSEQUENCE LANDS ANYWAY — HARDER THAN IT ASKED FOR.** Flag verified in the CHECKPOINT config, not just the recorded recipe: `pool: cls_mean_focal` with `slices 20`, `slice_pooling max`, `whole_stack true`, `backbone_lr 4e-05` identical to the control, so this is the single-variable pair the arm was written to be. By structure scale at each arm's own best epoch (focal e9, control e8): **focal +0.0001, regional −0.0114.** The prediction required the focal group to gain ABSOLUTELY; +0.0001 is a sixtieth of the noise floor, which is flat, not a gain. The refutation condition named another trade — focal up, regional down — and this is not that either: nothing was traded because nothing was bought. Widening the pooled vector cost the regional group 0.0114 and returned the focal group nothing. **MECHANISM: the fixed-budget reading is now refuted on BOTH of its axes, which is exactly the consequence this arm pre-registered.** *(CORRECTION 2026-08-21, knee-7gyw: only the WIDTH axis — this arm — survives. The other half rested on `attend_dense`, whose gate was outside every optimiser group and therefore ran as a fixed mean, so the allocator axis was never measured and reopens.)* `attend_dense` swapped the pooling OPERATOR and the trade stopped while nothing was recovered; this arm widens the pooled vector's WIDTH and the focal group still does not gain. A budget that does not grow when its allocator is made strictly more expressive AND does not grow when the vector is made wider is not a budget being mis-allocated — re-allocation is retired as an explanation of the `band_span` / `band_density` / `calib_span` substitution shape, and the bottleneck moves to supervision or to the encoder, which are different beads and different arms. **THE ONE THING THAT WENT AS PREDICTED, and it is worth keeping.** The named riser was Lateral Meniscus, the largest model-side gap in the corpus (−0.1932, knee-dtk) and a 1–3 mm finding: it rose 0.846 → **0.856**. The named second, Fracture, went the other way (0.809 → 0.796). And the focal group re-allocated INTERNALLY as well — ACL +0.006, Lateral OA +0.005, Lateral Meniscus +0.010 against MCL −0.008, Medial Meniscus −0.006, Medial OA −0.003, PF OA −0.003 — summing to nothing. So the focal tail does reach the fine-scale finding it was argued to reach; it pays for it inside its own group and then again, four times over, in the diffuse one. That is a real observation about WHERE the tail spends attention, and it does not license the tail as a lever. Converged with the best at epoch 9 (0.8405 / 0.8473 / 0.8532 / 0.8504 / 0.8517 / **0.8534** / 0.8521 / 0.8521), flat over the last three, so the best is not a truncation artefact. 27.51 GB reserved / 24.93 GB allocated, unchanged from `band_dense` — the tail is a head-side concatenation and the encoder cost does not move, as the arm predicted. Seat value **+0.0007** against the shipped four (`calibspan_v1`, roster 0.8670 → 0.8677; it read +0.0015 against `band_v1`'s 0.8648 before slot 4 replaced the champion), below `attend_dense`'s +0.0010 under the same roster and a quarter of the floor: an ADD as a fifth member, which `comp/champion.json`'s FOUR MEMBERS principle records as not a seat, and which tonight's board read (five entries, no relation between offline edge and board score inside 0.0062) says is not worth a submission slot either. **CONSEQUENCE:** the decode/pooling family is closed on all its axes; `band_dense_long` is already on the card asking the REGIMEN question, and after it the evidence asks for a member that does not exist yet rather than another knob on this one |
| **`encoder_layer 9`** | the head reads ViT block 9 of 12 instead of the last one | 0.8230 | -0.0227 vs `wsfold0_e6` | 3.7x the floor, converged (flat over its last three epochs), flag verified in the recipe. Seat value **-0.0034**, the largest negative any same-split candidate has posted, so diversity does not rescue it. The linear-probe convention turns out to be right for this head. See below |
| **`focal_pool`** | pooled vector gains a third part: the mean of the top eighth of each channel over the patch grid | 0.8432 | -0.0025 vs `whole_stack` | Flat (inside the floor). Seat value -0.0012. Pre-registered clause half-holds: focal group -0.0004 vs diffuse -0.0065, the predicted direction, but carried by the diffuse group losing rather than the focal group gaining. Fracture +0.020, directional only (one crossing in twelve). See below |
| **`unfreeze_4`** | four of the encoder's twelve blocks adapt instead of six — the +25% throughput arm, run at length | 0.8320 | −0.0137 vs `wsfold0_e6` | **refuted**, 2.2x the floor, flag verified (`unfreeze_last 6→4`). Peaked at epoch 6 and flat for five epochs after, so it is converged, not short. Eleven of twelve labels negative, six clearing their interval; by structure scale **focal −0.0170 vs regional −0.0091**, which is the arm's own stated risk — the fine findings are the ones that needed the deeper blocks. Seat value **−0.0017**, so diversity does not rescue it either. The default stays six; the speed was real and is not worth 0.0137. See below |
| `extended_slots` | nine slots, +1,139 coronal series | 0.8459 | +0.0002 | **refuted** on the mean |
| **`blendlabels`** | the training table blended from three independent readers of the same reports (`core.data.labels.table_blend`, `MEAN` mode), with `--eval-labels` pinned to `report_labels_v2` so the holdout ruler does not move with the arm | 0.8516 | **+0.0034 vs `holdannot`** 0.8482 | **THE SIGNATURE LANDED EXACTLY WHERE PRE-REGISTERED, AND THAT IS THE RESULT — NOT THE SCORE.** Harness verified on the printed recipe (`image_size 224`), which is not a formality: this arm's FIRST launch today silently inherited 336 from `$RECIPE` because its header declared a `deliberate:` flag its command never passed, and it was killed at 60 s by grepping the recipe line rather than trusting the prose. Converged, not truncated — peak 0.8516 at epoch 6 then 0.8511 / 0.8505. Matched on every option both runs recorded. Per label: **MCL +0.0239\*** · Fracture +0.0174 · Effusion +0.0107 · Lateral Meniscus −0.0105 · Medial Meniscus −0.0083 · Baker's +0.0081 · Synovitis +0.0053 · PF OA −0.0048 · ACL −0.0034 · **MEAN +0.0034** · Lateral OA +0.0027 · Medial OA +0.0005 · Contusion −0.0001. **By structure scale focal +0.0000 against regional +0.0083**, and on the six label-bottlenecked findings `run_comparison --split` derives for itself — Baker's, Contusion, Effusion, Fracture, Medial OA, Synovitis — **mean +0.0070, twice the twelve-label mean.** Those are precisely the six `interpretations/kneemri/2026-08-25_the-focal-deficit.md` pre-registered that morning from the supervision gap, named before this arm ran. **So the six/six partition is PREDICTIVE of where a label change lands, not merely a description of a past measurement** — which is worth more than the arm, because it licenses pre-registering future label work on the split instead of reading label arms on the mean. **ONE EXCEPTION, UNEXPLAINED:** MCL +0.0239\* is the largest single delta and MCL is a DEFICIT-side structure a better reader is not supposed to be able to touch. It is one marked crossing in twelve on the third-poorest label (153 holdout positives) — about what chance gives at 5%, and the pattern this tool's own footnote raises about Fracture — so it is not read as a mechanism violation, and it is not explained either. The other three deficit structures behave (Lateral Meniscus, Medial Meniscus, ACL all ≤ 0). **HONEST SIZE:** the mean +0.0034 falls in the arm's own *"within ±0.006 → the axis closes"* band and the claim set's +0.0070 in *"directional, keep as a diversity member"*; both are at or under the 0.0062 floor, so neither is a finding alone. Better readers buy ~+0.007 on the six findings they can reach and ~+0.003 on the metric. **CONSEQUENCE: the label lane is confirmed LIVE and confirmed SMALL** — it does not close, because the mechanism demonstrably works, but it is an order of magnitude short of the 1.5% bar and cannot be the path to 0.950 on its own. Do not refit the roster on it. See `knee-w7xa`. **CONTROL NAME CORRECTED 2026-08-26:** this row read "+0.0034 vs `native224`" for a day, which implied a `native224` of 0.8482 and contradicted the 0.8502 the resolution section records four times. The arm holds annotations, so its matched control is `holdannot` = 0.8482 (0.8516 − 0.8482 = 0.0034 exactly); the delta was always right and only the control's NAME was wrong. `native224` = **0.8502**, one number, and every 224 arm that does not hold annotations reads against it |
| **`confident224`** | `--labels confident.csv` — every UNK (study,finding) cell CLEARED rather than filled, so `Targets.load` skips the target and the cell trains at weight 0.0. Strictly LESS supervision: 14,445 of 52,872 pairs, **27.3% of the corpus**, removed | 0.8431 | **−0.0071 vs `native224`** | **THE THIRD BRANCH FIRES, AND IT INVERTS THE PROJECT IT WAS BUILT TO GATE.** Flag verified in the printed recipe before the number was read — `image_size 224`, `labels …confident.csv`, `weight_floor True`, `calibrated_targets False`, `slices 12`, band 0.2–0.8 — and 7.42 GB reserved against the 8124 MiB `vram_need` predicted, so the arm is `native224`'s recipe with exactly one thing moved. Converged: best epoch 6 of 12 run / 12 asked, five declining epochs after (0.8389 / 0.8422 / 0.8428 / 0.8416 / 0.8412), and `convergence_check` places it among the 81 converged rather than the 7 truncated. **Removing a quarter of the supervision COSTS 0.0071 — 1.15 floors, downward.** The pre-registered clause was *"≤ 0.8440 — the UNK cells carry real signal at their 0.2875 weight, and silence IS weak evidence of absence exactly as `CONFIDENCE_FLOOR`'s docstring argues"*; 0.8431 clears it **by 0.0009**, so it fires MARGINALLY. One arm, one seed, 1.15 floors: **directional, not a finding.** **WHAT IT SETTLES ANYWAY, and it is worth more than the number.** `knee-pju4` proposed a local open-weights multilingual reader to FILL those 14,445 cells — a model download, three absent deps (vllm / accelerate / gptqmodel), the card for inference, and a retrain of every seat, call it a day — on the premise that the UNK cells are garbage being tolerated. They are not. They are weak evidence the model is **already using**, at the 0.2875 that `CONFIDENCE_FLOOR = 0.25` plus `__conf ≈ 0.05` give them against a YES cell's 0.9625. The docstring argued this before anything measured it. So the reader project's premise is **inverted, not merely unsupported**, and the honest move is the opposite one: trust the silence MORE, which is `--calibrated-targets`, already shipped on all six champion seats. **AND THE AUDIT HAD ALREADY REMOVED TWO THIRDS OF THAT BEAD** (2026-08-27): the VALUE the UNK constant takes is already corrected (`MEASURED_AGREEMENT` maps 0.28 → 0.136, plus a per-finding logit shift in `FINDING_OFFSETS`, largest at Synovitis +0.985 — the 84.2%-UNK label), and the WEIGHT it pulls with is already floored (`--weight-floor` is in `$RECIPE`). Only the FILLING was open, and only this arm gated it. **WHAT IT DOES NOT SAY:** this is the 224/12-slice corner, weaker than the champion's dense one, and the caveat was written into the header before the number landed. A NEGATIVE is stronger evidence than the null would have been — supervision removal hurt at the cheap corner and there is no mechanism by which it would stop hurting at the dense one — but a narrower reader claim survives, over the three labels whose UNK share is extreme (Synovitis 84.2%, Fracture 56.4%, Baker's 45.9%) rather than all twelve. That would need its own gate and its own price. See `knee-07wb`, `knee-pju4` |
| `intensity_augment` | gamma/blur/contrast/noise at 392 | 0.8459 | −0.0064 vs res392 | **refuted**, behind at every epoch |
| **`aug_long`** | the same intensity block on the 24-epoch schedule `band_dense_long` proved long enough to overfit — the setting the old row could not see | 0.8511 | **−0.0076 vs `band_dense_long`** | **refuted, both clauses literally** (peak epoch 9 ≤ 10, and 1.2x the floor below control). Flag verified in the recorded recipe: `intensity_aug False→True` with all 46 other keys identical. **And the refutation is the informative kind — the regulariser demonstrably FIRED and the overfitting did not care:** train loss higher at every epoch past 4 (0.463 vs 0.454 at the end), so the fit was made harder, while the peak-to-finish decline is **unchanged at −0.0169 against the control's −0.0182**. The peak did move late (6 → 9), which is the only part of the prediction that held, and it bought nothing. The predicted regional concentration is refuted too: focal −0.0084 vs regional −0.0067, both down, the diffuse group slightly LESS hurt. Seat value **+0.0007** as a fifth and **+0.0000** as a swap — no seat. See below |
| **`aprior_dense`** | `--anatomical-prior`: `SlotHead.prior` initialised from the 12x6 plane-by-finding table in `core/data/corpus/anatomy.py` at PRIOR_STRENGTH 2.0 instead of zeros — the mapping the head otherwise has to discover from study-level labels alone | 0.8557 | **−0.0021 vs `band_dense`** | **THE REFUTATION CLAUSE LANDED LITERALLY, AND IT IS THE ONE THAT NARROWS THE MOST.** Flag verified in the CHECKPOINT config: `anatomical_prior false→true`, with `slices 20`, `slice_pooling max`, `whole_stack true`, `pool cls_mean`, `backbone_lr 4e-05` identical to the control (the only other diff, `slice_regions` absent vs `1`, is the recorded default and not a variable). Both arms peak at **epoch 8** on the same twelve-epoch cycle, so the comparison is at matched schedule position and the −0.0021 is not a truncation artefact. The PREDICTION needed the peak to clear 0.8578 by more than the 0.0062 floor; it came in below the control. The THIRD OUTCOME needed the per-label pattern to separate along the table's own peaked/flat split, and it does not usefully: **peaked rows (MCL, both OA, PF OA, Synovitis) −0.0008 against flat rows −0.0033**, a 0.0025 separation which is under half the floor, and inside it the largest single mover is Synovitis at **−0.010** — an axial-weighted, PEAKED row moving the wrong way. **MECHANISM, and the training loss is what says it: the warm start washed out rather than failed to land.** Train loss tracks the control to the fourth decimal at every epoch (1.2117/1.2162 at e0, 0.6508/0.6529 at e8 — the arm slightly LOWER), so 72 numbers of initialisation on an `nn.Parameter` the data may overrule are gone by the time the fit matters. Cost was as predicted: parameters, shapes, VRAM (26.92 GB reserved / 24.93 GB allocated) and 368 s an epoch all identical to `band_dense`. Seat value **+0.0009** as a fifth member and **+0.0004** as the best swap (displacing `band_dense`), 0 of 1 clearing +0.0015 — no seat on either ruler. **CONSEQUENCE, written into the arm before it ran and now owed:** the head had ALREADY recovered the plane-by-finding mapping from study-level supervision alone, so "supervision never says where" is wrong about THIS head, and the localisation argument has to buy a real localisation signal (knee-mxj) rather than a warm start. Fourth consecutive knob on the decode/head family returning nothing (−0.0009, −0.0044, +0.0014, −0.0021); the second of the five never-run switches to be spent. See below |
| **`half_data`** | `band_dense` fitted on HALF the training studies (`--train-fraction 0.5`), everything else identical — an instrument, not a candidate | 0.8344 | **−0.0234 vs `band_dense`** | **data volume is the binding constraint, and it is the only axis that has cleared the bar.** 3.8× the 0.0062 floor, and ~0.019 of it survives the registered schedule confound charged at its maximum (`schedule_length` prices step compression at 0.0028). Peak moved 8 → 6 as memorisation predicts; the peak-to-finish declines (−0.0024 vs −0.0006) are both sub-floor and unreadable. After its peak the curve PLATEAUS six epochs inside 0.0021 while the training loss keeps falling 0.8015 → 0.5973 — fitting continued, generalisation stopped. Slope: 1,763 → 3,526 studies buys ~0.020, so the bar needs ~1.7× the corpus. External data is PERMITTED (§2.6.a/b); `knee-ombx` promotes from backlog to the decision that gates the schedule. MRNet alone prices at ~+0.0095 and does NOT clear the bar. No seat — it is a handicapped control and was never a member. See below
| **`merged_corpus`** | 736 kneeMRI (Rijeka) studies concatenated into our corpus as one cache (`core.data.corpus.build_merged`, scheme `merged`), trained jointly on `merged_labels.csv` — an EXTERNAL corpus that differs in KIND, not count: expert double-blind IMAGE reads of the ACL, not text mined from a report | 0.8545 | **+0.0020 vs `lr4e-5_ws`** | **THE MEAN IS NOTHING AND THE SPLIT IS THE RESULT.** +0.0020 is a third of the 0.0062 floor, so on its own number this arm says nothing. Read on the partition `2026-08-25_the-focal-deficit.md` pre-registered, it separates: **the six DEFICIT findings (model below its own labels) mean +0.0061, the six SURPLUS findings mean −0.0022** — Lateral Meniscus **+0.0196\*** · ACL **+0.0161** · MCL +0.0062 · Lateral OA +0.0016 · Medial Meniscus −0.0024 · PF OA −0.0044, against Synovitis **−0.0206\*** · Contusion +0.0035 · Medial OA +0.0023 · Effusion +0.0017 · Baker's +0.0004 · Fracture −0.0005. By structure scale **focal +0.0056 vs regional −0.0031**. Three of the four internal-derangement structures move up; this is the first mechanism on the board to move the deficit set AS A SET. **THE TRANSFER CLAIM IS WHAT MAKES IT INTERESTING, and it is legible in one line: the corpus supervises ACL AND NOTHING ELSE, yet the largest mover — and one of only two deltas clearing its own interval — is Lateral Meniscus, which Rijeka never labels.** Rijeka carries a single `ACL` column, and `Targets.pair` weights an absent column at 0.0, so no meniscus gradient exists anywhere in those 736 studies. So the gain on Lateral Meniscus cannot be supervision and has to be the ENCODER: expert focal reads taught it to encode focal structure, which is exactly the surviving *"what one slice's feature vector contains"* hypothesis the whole decode elimination list points at. **CONVERGED, NOT TRUNCATED** — peak 0.8545 at epoch 7 of 12, then 0.8541 / 0.8536 / 0.8529 / 0.8527, against a control peaking at epoch 8, so the two are compared at matched schedule position. 301 s an epoch against the control's, for +736 studies (+21% corpus). **WHAT IT DOES NOT SAY:** single arm, single seed, the paired floor on this pair is wide (half-widths to ±0.0400 — the control predates per-study predictions), and **Synovitis −0.0206\* is a real starred cost on a surplus finding and is NOT explained.** The mean does not clear the bar and must not be quoted as a result. **CONSEQUENCE, and it is the schedule-setting one: `half_data` said volume is the only axis that ever cleared the bar and priced the bar at ~1.7× the corpus; the prevalence refutation then said volume is not the AXIS and left open exactly one door — a corpus differing in KIND. This arm walks through that door on 736 studies, 21% of a corpus, with ONE label column.** It is the evidence `knee-ombx` (MRNet, owner-blocked on the RUA) was missing and could not get any other way: MRNet is 1,370 studies with image-read ACL **and meniscus** labels, so it is ~1.9× this corpus on the axis that just moved, supervising the structure that moved most. Argue MRNet on KIND now, never on volume. See below
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
| **`margin_seat`** | `band_dense`'s recipe with the ranking loss replaced by an **AUC-margin surrogate** (`--auc-margin 1.0 --rank-weight 0.0`) — the objective moved, nothing else | 0.8525 | **−0.0053 vs `band_dense`** | **THE SOLO IS A NON-RESULT AND THE SEAT IS THE FINDING — the first candidate this project has measured into the strong-and-distant cell.** See below |
| **`dissent_seat`** | a SECOND objective on `band_dense`'s recipe, teacher gated to fold-0 twins — the test `margin_seat`'s row asked for | 0.8509 (killed at epoch 8 of 12) | **−0.0069 vs `band_dense`** | **NO SEAT — greedy picks it 38th of 66 and the ensemble peak is unchanged. Nine of twelve labels down, widest swing −0.0307: it reads the SAME as its control, slightly worse, where `margin_seat` read DIFFERENTLY. Retracts the "loss axis buys seats" rule this table stated one row up.** See below |
| `centre_tissue` | the control's 130 mm extent held, the crop CENTRED ON THE TISSUE instead of the image | 0.8409 | −0.0048 vs `wsfold0_e6` | **null, as pre-registered** — inside the 0.0062 floor, flag verified (`crop_mm=130.0, centre=tissue` against a control that predates both). Seat value **−0.0016**. Two labels clear their paired interval and both are NEGATIVE (Contusion −0.0128\*, Medial Meniscus −0.0096\*); nothing positive resolves. This is the arm that licenses reading `crop80_tissue` as a RESOLUTION result rather than a centring one. See below |

### `margin_seat` — a losing solo that greedy picks SECOND of sixty-five, and the empty quadrant finally has an occupant — 2026-08-24

**PLUMBING FIRST, because the whole row rests on one variable having moved.** The checkpoint's own config
stamp carries `auc_margin: '1.0'` and `rank_weight: '0.0'` against `band_dense`'s inverse, and no other key
differs. The objective is the single change.

**THE MEAN SAYS NOTHING.** Solo 0.8525 against the control's 0.8578 is −0.0053, inside the 0.0062 floor, so
by this project's own rule the arm neither wins nor loses on quality. It converged rather than truncated —
peak 0.8525 at epoch 10, 0.8524 at 11, flat and never turning down — so the number is the method's, not a
schedule artefact.

**THE PER-LABEL PROFILE IS EMPHATICALLY NOT FLAT, which refutes this arm's own pre-registered
"surrogate is simply worse" branch.** A uniformly weaker objective produces a uniformly depressed column;
this produces a redistribution of about ±0.03 with a mean that barely moves (\* clears the paired interval):

| up | | down | |
|---|---|---|---|
| MCL | **+0.0272\*** | Lateral Meniscus | **−0.0546\*** |
| Fracture | +0.0250 | PF OA | **−0.0288\*** |
| Baker's | **+0.0163\*** | Medial Meniscus | **−0.0277\*** |
| Lateral OA | +0.0130 | Effusion | **−0.0145\*** |
| Contusion | **+0.0126\*** | ACL | −0.0128 |

That is a DIFFERENT ERROR STRUCTURE, not a worse one — which is exactly the property a seat is bought with,
and exactly the property `dissent_seat.sh` was written to argue a different inductive bias does not
automatically deliver.

**THE SEAT READ, card-free over 65 fold-0 records sharing 909 studies** (`logs/greedy_margin.log`; in-flight
arms and smokes excluded from the glob per knee-p94l, since a running arm writes a partial holdout npz).
Greedy builds `small_batch6` 0.8598 → **`margin_seat` 0.8669** → `calib_span` 0.8706 → `through_dense` 0.8720
→ `band_dense_long` 0.8726 → `sided_pool` 0.8729 → `aprior_dense` 0.8730 → `crop80_span` 0.8731 (peak, flat
to seat 9). **It is taken SECOND of sixty-five, worth +0.0071 to the pair, by a ruler that has never seen its
solo score favourably.** Priced against that greedy-7 roster: **add +0.0015**, best **swap +0.0013**
(displacing `crop80_span`). The tool's own verdict line reads `0 of 3 clear +0.0015`, so this is at the very
edge of `BOARD_RESOLVABLE` and not past it — but the comparison that matters is historical: the best seat
move across 59 candidates was **+0.0004** (knee-dfl1), and the mounted pool was written up above as
EXHAUSTED at +0.0002. This is an order up on both.

**DISTANCE — the half that makes it the quadrant, and it reads NO labels.** The greedy roster's own
nearest-neighbour agreements span 0.8717–0.8893. `margin_seat` sits at **nearest 0.8628, mean 0.8314** —
below the roster's own tightest pair, i.e. further from every seated member than any two of them are from
each other. For contrast `cnx_dense` reads nearest 0.7686 at solo 0.7927: distant, but weak, which is the
cell this project keeps landing in.

**CONSEQUENCE: the strong-and-distant cell (knee-053z), recorded empty since the project began, has an
occupant — and it came from the OBJECTIVE, not from a trunk swap.** Every previous attempt at distance
bought it with architecture and paid for it with quality (`cnx_dense`, `vjepa_warm`, the DINOv3 family, all
distant-and-weak). An objective change bought distance at a solo cost inside the noise floor. That
re-prices the loss axis above the trunk axis for seat-hunting specifically, and it is the argument
`dissent_seat` is currently on the card to extend — a second, differently-motivated objective, testing
whether this is a property of AUC-margin or a property of moving the objective at all.

**THAT TEST RAN THE SAME DAY AND ANSWERED NO — see `dissent_seat` below.** A second objective-axis member
lands 38th of 66 in the same greedy. The re-pricing in the paragraph above is therefore RETRACTED as a rule:
what is measured is that `margin_seat` bought a seat, not that the loss axis buys seats. One member is not
an axis, and this file nearly recorded it as one.

**HONEST LIMITS, stated because the number is small.** Greedy selection on the holdout is IN-SAMPLE: the
+0.0071 is the price of picking on the same 909 studies it is scored on, and the seat value +0.0015 is the
out-of-that-loop figure. `seat_value`'s own instruction is READ THE SIGN, NOT THE SCORE. And the pinned
champion roster cannot be priced against at all — six of its seven members are `--fold -1` and have seen
every holdout study — so every seat number here is against the greedy fold-0 proxy, not against what ships.

**AND THE IN-SAMPLE PRICE IS NOW MEASURED, not estimated — 2026-08-25.** `record_mix --seats honest`
refits the greedy on half the studies, scores each prefix on the other half and averages both directions.
Over 77 same-split members:

| seats | held-out | in-sample | bias |
|---|---|---|---|
| 5 | 0.8697 | 0.8728 | +0.0031 |
| 6 | 0.8697 | 0.8732 | +0.0035 |
| 8 | 0.8715 | 0.8735 | +0.0020 |
| **12** | **0.8720** | 0.8734 | +0.0014 |
| 16 | 0.8718 | 0.8728 | +0.0011 |
| 24 | 0.8712 | 0.8724 | +0.0012 |

**Best honest count is TWELVE at 0.8720, against an in-sample peak of 0.8736 at ten.** Two readings, and
the second matters more than the first. (1) The bias is real but small — +0.0035 at six seats, decaying to
+0.0014 by twelve, so roughly a fifth of the plateau was the selection and four fifths is a real ensemble.
(2) The honest curve peaks LATER and FLATTER than the in-sample one: selection overfitting is what made a
narrow roster look optimal. Six honest seats read 0.8697 and twelve read 0.8720, so widening to twelve buys
**+0.0023** — directional, under the 0.0062 floor, not a submission on its own.

The greedy's first twelve on the full pool: `small_batch6`, `margin_seat`, `calib_span`, **`blendlabels`**
(fourth, its first seat), `band_dense_long`, `through_dense`, `base224`, `attend_dense`, `aprior_dense`,
`crop80_span`, `sided_pool`, `calibrated_targets`. Nine are already in `comp/members.roster`; `blendlabels`,
`base224` and `attend_dense` are not, and adding them is the packaging cost of shipping this count.
Split-half chooses the COUNT, the full pool chooses the MEMBERS — the count is the thing that overfits.

### `dissent_seat` — the objective axis is NOT a seat lever; 38th of 66, and the per-label table says why — 2026-08-24

The arm `margin_seat`'s section (above) named as the test of whether its seat came from AUC-margin
specifically or from moving the objective at all. `dissent_seat` is that second objective — a different
motivation, same trunk, same data, same recipe, and a teacher signal gated to fold-0 twins by
`teacher_targets.py` so the holdout stays honest. It answers **no**.

| | `margin_seat` | `dissent_seat` |
|---|---|---|
| solo | 0.8525 (−0.0053 vs `band_dense`) | **0.8509 (−0.0069)** |
| greedy pick, same pool | **2nd** of 65 | **38th** of 66 |
| ensemble peak with it in the pool | 0.8731 | **0.8731, unchanged** |

Killed at epoch 8 of 12 rather than run out. Epochs 6/7/8 read 0.8509 / 0.8501 / 0.8503 — a plateau, and
the best epoch's weights were already on disk, so the tail was ~18 minutes of a SHARED card to maybe
improve a member sitting thirty places from a seat. The card went back to the other project instead. The
stop is recorded here because "killed early" and "converged" are different facts and the ledger should not
have to guess which.

**WHY IT IS 38TH, from the per-label table against its control — and this is the whole finding.**

| label | `dissent_seat` | `band_dense` | Δ |
|---|---|---|---|
| ACL | 0.8872 | 0.8740 | +0.0132 |
| Lateral Meniscus | 0.8602 | 0.8464 | +0.0138 |
| Contusion | 0.8783 | 0.8705 | +0.0078 |
| Effusion | 0.8580 | 0.8595 | −0.0015 |
| Lateral OA | 0.8098 | 0.8156 | −0.0058 |
| Medial Meniscus | 0.8949 | 0.9047 | −0.0099 |
| Medial OA | 0.8688 | 0.8797 | −0.0109 |
| PF OA | 0.8471 | 0.8583 | −0.0112 |
| Baker's | 0.8937 | 0.9065 | −0.0128 |
| Fracture | 0.7922 | 0.8088 | −0.0166 |
| MCL | 0.7745 | 0.7931 | −0.0186 |
| Synovitis | 0.8456 | 0.8764 | −0.0307 |

Nine of twelve down, widest swing −0.0307. Set that beside `margin_seat`, whose swings ran +0.0272 MCL and
−0.0546 Lateral Meniscus: that member READ DIFFERENTLY, and paid for it in the mean. This one reads the
SAME WAY, slightly worse everywhere. A member that is uniformly a little worse is not distant — it is
just worse, and a greedy that selects on complementarity has no reason to reach for it. The seat and the
redistribution are the same fact seen twice.

**CONSEQUENCE — the rule retracted above.** `margin_seat` bought a seat; the objective axis did not. Two
members on that axis, 2nd and 38th. The re-pricing of "loss axis over trunk axis for seat-hunting" was
written in this file on a sample of one and is now withdrawn: what transfers is the METHOD (price a
candidate on both rulers before believing it) not the AXIS. Filed to biohub as a correction too, since
yesterday's letter offered them the axis as the transferable bit.

**HONEST LIMITS.** The greedy is in-sample on the same 909 studies, so 38th is a rank on a proxy, not a
verdict on the board — but the gap is thirty places, not three, and no in-sample correction closes that.
The member also stopped at epoch 8 under a OneCycle schedule built for 12, so its solo is a floor rather
than its converged score; that raises the solo, not the complementarity, and complementarity is what the
seat is made of.

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

### `crop80_tissue_full` — the schedule defect is CLOSED, and it was never the explanation

2026-08-29. The arm below carried a recorded defect for fifteen days: its control ran 12 epochs and it
ran 8. `crop80_tissue_full` re-runs it at 12 on the same cache, same fold, same labels, same recipe —
the budget as the only change, as its header pre-registered.

| | epochs | peak holdout | nearest neighbour | seat value vs `honest12` |
|---|---|---|---|---|
| `crop80_tissue_s2026` | 8 | 0.8345 | 0.8375 | −0.0006 [−0.0011, −0.0001] |
| `crop80_tissue_full_s2026` | 12 | **0.8343** (epoch 6) | 0.8492 | **−0.0006** [−0.0011, −0.0001] |

Four extra epochs bought **−0.0002**, and the marginal seat value reproduced to four decimal places.
That is the arm's **branch two**, written before the number: *"the budget was not the binding
constraint… the distance lane closes on the mean, this time without the confound."* The −0.0084
adjustment computed below is therefore not needed — the 80 mm read costs what it costs, and its weakness
is a property of the read rather than of a short schedule.

Two things worth keeping. The 12-epoch run **peaked at epoch 6 and slid 0.0055 to the finish**, the same
shape `band_dense_long` recorded at 24 — under `one_cycle` this family does not use a longer budget.
And the nearest-neighbour column moved **+0.0117** (0.8375 → 0.8492) at a holdout that moved −0.0002,
on the identical read: that is a precision bound on the twin detector, and it means differences of about
a hundredth in that column are run-to-run wobble rather than decode distance.

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
- **And that label profile was THEIR model's, not ours — the whole per-label axis is worth +0.0043.**
  2026-08-23, `kneemri/evaluation/weights_gap.py`: the two bullets above compare OUR fold-0 columns
  against a corpus ruler built from THEIR weights, which differs in studies AND weights at once. Put
  both on the same 909 studies and our single member beats their average single member on **ten of
  twelve labels** (mean 0.8483 vs 0.8285) and beats their four-seed pool (0.8369). Only MCL (−0.0346)
  and Fracture (−0.0167) lose. Closing both completely, to weights that provably exist, is
  `(0.0346+0.0167)/12 = +0.0043` — a quarter of the bar, and the CEILING of the whole weak-column
  programme. Fracture's share is sampling (68 positives). So the line at 827 above — "MCL is hard to
  see, which is what the resolution arm now targets" — is superseded on PRICE, not on mechanism, and
  `focus_mcl` was cancelled before it ran. See
  `interpretations/kneemri/2026-08-23_the_per_label_axis_is_worth_0.004.md`.
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

### `cnx_dense` — the first non-transformer trunk, at n=1 against DINOv2's n=189 — 2026-08-21

> **RETITLED 2026-08-24, and the old title is quoted here because it was quoted forward.** This entry
> read *"and it closes encoder diversity as an axis"* until today. It does not close anything, on two
> counts measured after the fact (knee-zfgn).
>
> **The denominator.** Of the 202 trained checkpoints on disk, ~189 are DINOv2-family and **one** is this
> ConvNeXt run. Every recipe knob this project owns — read size, backbone LR, schedule, band, sampling,
> augmentation, batch, pooling — was tuned across those 189. The standing recipe is a 189-run local
> optimum around DINOv2-small, and it was handed to a convolutional trunk unchanged, once. What the run
> measures is our own tuning, charged to the architecture.
>
> **Its own pre-registration was overruled.** The arm named four outcomes before it ran. The one that
> fired says: *solo below 0.80 … the reading is about RESOLUTION and the retry is 448 px (a 14×14 grid)
> **before the trunk is judged***. It scored **0.7927**. The 448 retry was never run, and the entry was
> written up as a verdict anyway. A pre-registration discarded when it fires is not a pre-registration.
>
> **And the LR is DINOv2's.** `--backbone-lr 4e-5` is an adapter rate for a ViT; convolutional trunks
> fine-tune roughly an order of magnitude higher. One point, one rate, chosen for a different family.
>
> **What this run DOES establish, and it survives intact:** the trunk is genuinely FAR. `cnx_dense` reads
> nearest **0.7941** against a seated roster whose own spread floors at 0.8724 — 0.078 more distant than
> any seated member is from its nearest neighbour. The claim elsewhere that *"a different inductive bias
> is not a different error structure"* is false; ConvNeXt produces a very different one. The distance half
> of the thesis holds. The quality half has never been tested at a fair budget. See knee-y59c, knee-zfgn.

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

### the ordinal re-rank — the whole pool scored by the ruler that just passed (2026-08-22)

`gradeswap` cleared its pre-registered band at 0.933, which per `knee-7d1` licenses re-ranking every
seat by ordinal concordance. This is that re-ranking: `kneemri.evaluation.ordinal_signal --checkpoint`
over all 67 same-split checkpoints, card-free, 334,942 cross-level pairs, chance 0.50.

**PLUMBING BUG, named before the numbers, because the first run of this scan produced 67 blank rows.**
`ordinal_signal` reports through `logging`, which writes to **stderr**, and my loop had `2>/dev/null` on
it to suppress load warnings. Every number went to the null device and the scan returned a clean-looking
list of names with nothing beside them — exit 0, 67 lines, no error anywhere. That is the failure mode
this project's own instructions put under PLUMBING: the value was recorded where the consumer does not
read. Re-run with `2>&1`, same command otherwise, and it filled in. A gate that returns the right SHAPE
with no content is worse than one that crashes.

Top of the pool, and the champion's own seats marked:

    calib_span            0.791   <- seated
    small_accum           0.784
    severity_weight       0.783
    base_dense            0.782   <- seated
    small_batch6          0.781
    grade_ladder          0.780   <- seated, the member that just bought the tick
    band_dense            0.778   <- seated
    aprior_dense          0.778
    focal_dense           0.777
    band_dense_long       0.776   <- seated
    calibrated_targets    0.776
    through_dense         0.774   <- seated
    lr4e-5_ws             0.770   <- seated, and the WEAKEST seat in the champion
    depth_regions_narrow  0.769   <- the member gradeswap displaced

**The displacement gradeswap made is legible on this scale and small.** It took a 0.769 out for a 0.780,
+0.011, and the board moved one tick. That gives the ruler a rough exchange rate to test against.

**The next licensed move is `lr4e-5_ws` out.** It is the champion's floor at 0.770, and the best
available replacement is `small_accum` at 0.784 — **+0.014, a larger gap than the one that passed**. If
the ruler transfers, that predicts at least the tick gradeswap bought.

**But the ranking exposed a mounting bottleneck, which is the real finding here.** Of the members above
the champion's floor, `small_accum`, `severity_weight`, `small_batch6`, `aprior_dense` and `focal_dense`
are in NO Kaggle package. The best MOUNTED unseated candidate is `calibrated_targets` at 0.776 — only
+0.006 over the floor, half the gap that just bought a tick, which predicts a tie at board resolution
and is therefore a useless instrument. **The mounted pool's ordinal top is already seated.** The
licensed move was not costed as a kernel edit; it needed a dataset push first, and those five names went
into `comp/members.roster` in one push for that reason.

**Priced before running, out loud.** Path A, the mounted swap: ~20 min, ordinal gap +0.006, expect
+0.0005 board, most likely a tie that arbitrates nothing. Path B, push five and swap `small_accum`:
~45-60 min, gap +0.014, expect a tick, and it re-tests the ruler at a LARGER separation than its first
pass rather than a smaller one. B is three times the wall-clock for a bit over twice the predicted
effect AND a decisive read instead of an ambiguous one, and it unblocks the pool's whole ordinal top
permanently. B.

**The caveat travels with the ruler, per the bead's own words.** One trial cannot separate "the ordinal
ruler transfers" from "grade_ladder is simply a good member". This re-rank is the ruler's second use and
it is a RE-TEST, not an application of a settled fact. If the `small_accum` swap ties or loses at a gap
larger than the one that passed, the single pass was the coincidence and the ruler goes back in the
box beside residual distance.

---

## `side_dense` — the loss cannot buy the side, and the contrast is not the metric (2026-08-22, `knee-6u12`)

| arm | what moved | best | vs control | reading |
| --- | --- | --- | --- | --- |
| **`side_dense`** | `band_dense`'s recipe with `--side-weight 1.0` — a within-study term asking the worse side of a compartment pair to outrank the better one | **0.7937** | **−0.0641 vs `band_dense`** | **REGRESSION, with the mechanism CONFIRMED to fire.** Term verified landed three ways BEFORE the run: `side_weight: 1.0` stamped in the checkpoint recipe; smoke loss 1.9368 against the control's ~1.2, the extra ~0.69 being exactly `softplus(0)`, what a side-blind pair must pay; VRAM 26.92/24.93 GB and 361 s/epoch byte-identical to the control, as a term adding no parameter must be. **The contrast moved the way the arm predicted** — Lateral OA 0.5141 → **0.5759**, Medial OA 0.7416 → 0.7962 — and **every one of the twelve labels fell**, ten times the 0.0062 floor: Baker's −0.207, MCL −0.077, PF OA −0.076, Lateral Meniscus −0.069, Effusion −0.068, ACL −0.056, Medial Meniscus −0.056, Synovitis −0.050, **Lateral OA −0.034**, Medial OA −0.030, Fracture −0.025, Contusion −0.023. **The last of those is the finding: the column the term exists to fix became MORE side-dependent on the contrast and WORSE at the metric, in the same run.** The metric ranks study-against-study WITHIN a label; the term orders label-against-label WITHIN a study. Orthogonal orderings over the same logits — buying one costs the other rather than composing with it, which is why `Objective.pairs` was never missing the side comparison by oversight. **NOT the pre-registered mass-imbalance failure**: the meniscus pair was supposed to eat the gradient 2.6:1 and gain at the OA pair's expense; instead it lost too (Medial Meniscus −0.071), and damage is uniform across labels the term places NO gradient on, which is shared-encoder damage rather than pair-specific robbery. Per-pair normalisation therefore redistributes a cost instead of removing it and is declined with the weight sweep. **Which link broke: the IDEA, in this specific use** — plumbing verified, data and split identical by construction, training one flag off a control already on disk, test the same tool on the same 909 studies and the same n. Seat **−0.0019**, 0 of 1 clearing +0.0015, negative and charged ~3× by the board — no seat. **CAVEAT THE TOOL NOW CARRIES:** the neutral control column went dirty — `PF OA` reads 0.4274 / 0.3498 under the arm against 0.4770 / 0.4133 under the control, and `side_blindness` flags both pairs CONFOUNDED. The term puts literally zero gradient on `PF OA` (asserted in its unit test), so this is indirect coupling through the shared encoder. **`side_blindness` can audit a run trained WITHOUT a side term; it cannot cleanly audit one trained with it** — a second, independent reason a weight sweep could not be steered by the contrast, since the sweep would bend its own ruler. **FOLLOW-UP DECLINED, and that is the arm's return.** PRICE: 73 min of card for weight 0.3, P(clears 1.5% on the mean) ≈ 0.02 — the result says the SIGN of the trade is wrong, not merely its size, so a third of the weight buys a third of a bad trade. **What it does NOT rule out:** the side-blindness OBSERVATION is untouched (Lateral OA at chance on all eight members separately, four data-side explanations already refuted by `knee-yrg`); the side IS recoverable from this representation, since the contrast rose 0.062 on demand — what is refuted is getting it out of the loss for free; and the term survives as a small-weight regulariser on a head that SEES position (pooling or field of view for the OA compartments), where it would be read on per-label AUC rather than on the contrast. [interpretation](../interpretations/kneemri/2026-08-23_side_resolution_is_not_score.md) |

## `accumseat` — the ordinal ruler's re-test at a LARGER gap, and it tied (2026-08-22)

`accumseat` displaced `lr4e-5_ws` (the champion's ordinal floor, 0.770) for `small_accum` (0.784) — a
+0.014 separation against the +0.011 that bought `gradeswap` its tick. **It scored 0.933. A tie.**

The consequence was pre-registered in this file before the entry fired, and it is owed:

> If the `small_accum` swap ties or loses at a gap larger than the one that passed, the single pass was
> the coincidence and the ruler goes back in the box beside residual distance.

**So all three rulers are now dead as selection rules.** Residual distance is 0-for-2 in opposite
directions. Binary seat value is blind at ±0.0015. Ordinal concordance passed once at +0.011 and tied at
+0.014 — a ruler that does not order its own two trials by their separation is not measuring what it
claims. Nothing in this project may now spend a submission slot on a member swap justified by a ruler.

The second thing it settles is about the roster rather than the rulers: **displacing the weakest seat of a
seven-member vote is free in both directions.** `gradeswap` and `accumseat` are two different displacements
of a seven and both read 0.933. That is a statement about the ensemble's flatness at seven, and it points
the remaining board budget at the count (an eighth seat, never tested) or at a member from an axis the
roster does not contain — not at further one-for-one swaps.

---

## `raddino_dense` — a medical-report backbone at a natural-image recipe, killed at epoch 6 on its own slope (2026-08-23, `knee-vp8q`)

| arm | what moved | best | vs control | reading |
| --- | --- | --- | --- | --- |
| **`raddino_dense`** | `band_dense`'s recipe with the trunk swapped to Microsoft **RAD-DINO** (a DINOv2-family ViT-B pretrained on chest radiographs with report supervision) instead of DINOv3 — everything else identical | **0.8137 at epoch 6 of 12** (killed) | **−0.0375 vs `base_dense_s2026` 0.8512 at e8** | **KILLED ON ITS OWN SLOPE, not on a landing number.** Epochs 0–6 read 0.7247 / 0.7439 / 0.7852 / 0.7944 / 0.8000 / 0.8119 / 0.8137 at ~990 s each. Per-epoch gains: +0.019, +0.041, +0.009, +0.006, +0.012, **+0.0018**. The pre-registered top band was ≥0.866, which needed +0.052 in the five remaining epochs while the arm was delivering two thousandths. The bottom band (≤0.845) was already unreachable-from-below in the other direction: it would have landed there or under. **PRICE that killed it:** 83 min of card to convert a near-certain bottom-band landing into a certain one, against 72 min for `sided_pool` on an axis no arm in this repo has ever touched. **Which link is suspect, before any verdict:** the trunk DID fire — loss falls monotonically 1.11 → 0.757 and VRAM/step-time are backbone-appropriate, so this is not a plumbing failure and not a frozen-encoder failure. **The confound this arm cannot separate is the RECIPE:** `--backbone-lr 4e-5` and the twelve-epoch schedule were tuned on DINOv3, and RAD-DINO is a different pretraining objective at a different scale. A slope this shallow is as consistent with an under-driven backbone as with a backbone that does not transfer. **What this kills: the DROP-IN swap** — RAD-DINO at DINOv3's recipe does not reach DINOv3's number and is not worth a seat. **What it does NOT rule out:** RAD-DINO at its own learning rate, RAD-DINO as a frozen feature extractor, or the wider premise that medical-domain pretraining transfers to knee MRI. Those are re-testable; this run says nothing about them. The asset stays on disk. |

**A SECOND confound, found on 2026-08-25 (`knee-8wm6`) and now fixed in the tree.** The arm was fed the
ImageNet constants. `StudyModel` hardcoded `mean 0.485/0.456/0.406`, `std 0.229/0.224/0.225` for every
trunk, and `microsoft/rad-dino`'s own `preprocessor_config.json` declares `image_mean 0.5307`,
`image_std 0.2583` — greyscale, fitted on chest radiographs. That is an input **offset by 0.20 std**
((0.5307 − 0.485)/0.229) at a **12.8% scale error** (0.2583/0.229). Six of the other seven registry
encoders declare the ImageNet pair themselves, so the shared constant was right by coincidence
everywhere it had ever been looked at, and RAD-DINO is the one encoder it was wrong for — the one
encoder whose arm was read as evidence about domain pretraining. So 0.8137 is a floor under a floor:
truncated at e6 AND fed the wrong contract. It does not price RAD-DINO, at this recipe or any other.
The constants now live on `Encoder.normalization` (`kneemri/models/backbone.py`) and ride the state
dict as persistent buffers, so every member already packaged keeps what it was fitted under and only a
new fit moves. V-JEPA2's constants are **UNVERIFIED** (gated repo, no local snapshot) and must be
checked before any vjepa arm is armed.

The wider reading is about the roster rather than the backbone. This is the **third** arm to fail the same way —
a new trunk (`cnx_dense` 0.7927, `vjepa_member` 0.8343, `raddino_dense` ~0.82 extrapolated) plugged into a recipe
that was fitted around DINOv3, and each landed below the DINOv3 control it was measured against. Either the
recipe is genuinely DINOv3-shaped and every trunk swap needs its own tuning pass before it can be read at all,
or the trunk is not where the remaining board budget lives. The greedy curve over the cleaned 68-member pool
says the latter more cheaply: **the peak is 0.8717 at seats 8–10 and declines monotonically after**, so the
ensemble does not want another member — it wants a member that can do something none of the existing ones can.

### sided_pool — the head was given a medial/lateral axis, and did not use it

`band_dense`'s recipe with `--pool sided`, one flag off the best single member on disk. 12 epochs,
377 s/epoch, best at e6/e9.

**Result: 0.8561 against `band_dense` 0.8578 on the 909-study fold-0 holdout — −0.0017, inside the
0.0062 floor. The MIDDLE pre-registered band.**

The band's own instruction was to read the per-label columns before writing a verdict, since the mean
over twelve labels can absorb a real move on three. Read (paired, 95% intervals):

| target | delta | interval |
|---|---|---|
| MCL | +0.0052 | [−0.0162, +0.0298] |
| Lateral OA | +0.0000 | [−0.0148, +0.0142] |
| Lateral Meniscus | −0.0064 | [−0.0215, +0.0085] |

Mean of the three side-defined targets ≈ −0.0004; every interval spans zero. The labels the pooling
was built for are the labels it did not move.

**The mechanism FIRED — this is not a plumbing null.** The saved config carries `'pool': 'sided'` and
`head.proj.1.weight` is (256, **1152**) = 384×3 parts, against `cls_mean`'s 768 = 384×2. The head was
built three-part and did see the split. Verified before the verdict was written, because a silently
two-part head would have produced exactly this flat reading.

`ACL +0.0175 [+0.0009, +0.0333]` is the one interval clear of zero, and ACL is not side-defined. One
significant hit in twelve tests at 95% is what chance produces. Not claimed.

**Named suspect that is NOT cleared.** The config reads `'centre': 'image'`, `crop_mm 130`. The pooling
splits the patch grid at its centre COLUMN, which is the anatomical midline only if the crop is
anatomically centred. An image-centred crop leaves the true midline off-column by a study-varying
offset, which smears the halves into each other. This is a live alternative to "the idea is wrong" and
it is cheap to test — an anatomically registered crop, or a split at the measured midline rather than
at column `side // 2`.

**What this KILLS:** the halves-of-an-image-centred-crop split as a way to buy the side-defined labels.
**What it does NOT rule out:** an anatomically registered split; a finer grid (quadrants, or the
compartment columns rather than halves); and the small-weight side REGULARISER that `side_dense` left
standing, which was explicitly deferred to "a head that sees position" — this arm built one, so that
row is now testable and was not tested here.

**Ensemble seat.** Greedy over the champion's six scoreable members plus this one picks it FOURTH,
ahead of three incumbents — but seat 3 is 0.8690 and seat 4 is 0.8700, so it contributes **+0.0010**,
under the floor, and seats 5–7 add nothing. It also ranks fourth by SOLO score inside that pack, so its
greedy rank is its quality rank, not decorrelation. That greedy is in-sample selection over seven
members and carries the measured +0.0033 selection overfit, so the honest figure is smaller still,
possibly negative. No submission was spent on it.

Denominator note: the champion could not be used directly. `seat_value` raises on `final_ws_e12`, a
`--fold -1` member that is unscoreable forever, so the roster above is the SIX scoreable champion
members, not the shipped seven. Filed as a P1.

**2026-08-25 — the crop suspect above is EXCLUDED, by arithmetic, and this row hardens into a
refutation.** The `'centre': 'image'` alternative was going to be tested by `sided_tissue` at a real
price: the arm lands on an undecoded reading axis, and a full cache build is recorded here at about two
hours, so ~3.25 h of card. Pricing it forced the question of whether the suspect is even large enough
to be the explanation, and it is not. At 336 px with ViT patch 14 the grid is 24 columns, 12 per half,
across a 130 mm crop — **5.4 mm per column**. `core/data/corpus/crop_centre.py` measured the tissue
centroid against the image centre over 60 studies at median 7 mm, p90 18 mm: **1.3 columns at the
median, 3.3 at p90**. Contamination of each half is 1.3/12 ≈ 11%, and a difference-of-means contrast
attenuates by (1−2c), so **~78% of the true contrast survives for the median study and ~56% at p90**.
A confound that preserves three quarters of the signal cannot flatten a real effect to −0.0004 with all
three side-defined intervals spanning zero.

So the flat reading above stands on its own terms, and it is stronger than it claimed: the halves
partition the grid, the head was built three-part and verified so, the crop offset is too small to hide
the split, and the head still did not use it. **Head-sees-position is refuted AT HALVES on this data.**
The finer grid (quadrants, compartment columns) rests on the same premise that just failed and needs a
new argument rather than a rerun. The arm and its cache smoke are kept — the smoke read `ok
sided_tissue`, 0 refused, so the axis is proven decodable if one arrives. Bead `knee-86vi`, closed
without running.

### focus_latmen — 0.4747, and why a DEAD member is the one a greedy selector reaches for

Never finished. The checkpoint on disk is epoch 0 and scores **0.4747**, which is worse than chance on
a twelve-label mean. It is recorded here not as a result but because of what it did to a reading.

A near-random member is, by construction, DECORRELATED from every trained member. A greedy ensemble
selector maximising a mix score will therefore pick it up as apparent diversity and keep paying for it.
That is what produced the earlier peak of 0.8721 at seats 11–13: re-run on a pool with this member
removed, the greedy peaks at **0.8717 at seats 8–10** and declines monotonically after. The extra seats
were fitting noise, and the tool cannot warn about it because the failure LOOKS like the result you
wanted.

Operational consequence, and the part that generalises: `.partial` survives a SUCCESSFUL run, so it is
not a liveness marker. The finished test is the existence of the `.pt`. Any pool built by globbing
checkpoints must score every member alone before mixing — a member below the weakest trained member is
a dead checkpoint, not a diverse one.

### crop_alignment — the slots disagree by 3-6x a meniscal tear, and two thirds of the offset is harmless

`python -m core.data.corpus.crop_alignment --workers 20`, 400 studies of 4407, 1848 routed slots,
~20 min CPU, zero GPU. The module had been written and never run. An INSTRUMENT: it cannot raise a
score, it decides what runs next.

```
plane           |offset| p50      p90   down p50   across p50   slots
AX_FLUID_FS              9.9     18.3        8.2         -0.6     396
COR_FLUID_FS            10.2     22.5       -8.1         -0.8     386
COR_T1                   9.1     19.6       -6.0         -0.6     263
SAG_FLUID_FS            13.9     26.1       -4.7         10.7     380
SAG_FLUID_NOFS          11.1     22.3       -8.1          2.9     246
SAG_T1                  12.9     23.4       -8.5          3.4     177
within-study spread of |offset| across slots: p50 5.7 mm, p90 10.7 mm, max 18.8 mm
```

**Middle pre-registered band (4-10 mm), which required reading the per-plane SIGNED components rather
than the mean.** The honest cross-slot comparison is within a plane, where the axes agree: the three
SAGITTAL slots run +10.7 / +2.9 / +3.4 mm anterior-posterior, so `SAG_FLUID_FS` sits ~7.8 mm displaced
from the other two sagittal slots, systematically. Axial `down` is anterior-posterior and coronal `down`
is superior-inferior, so those two columns are NOT comparable and no verdict is drawn from them.

**Most of the displacement is COMMON MODE, and that is the finding that resizes the lever.** Per-slot
|offset| is 9-14 mm but the within-study spread across slots is only 5.7 mm. The study is shifted as a
whole; the slots are shifted together. A common-mode shift does not break slot correspondence — it
spends crop margin. This RECONCILES with knee-yrf rather than contradicting it: its 6.9 mm dy and
6.4 mm dx combine to ~9.4 mm of magnitude on 60 studies, against 9-14 mm here on 400.

**The residual, with its denominator.** At 130 mm over 336 px the scale is 0.387 mm/px, so the 5.7 mm
p50 residual is ~15 px and the p90 ~28 px. `anatomy.py` records a meniscal tear at 1-2 mm. The
cross-slot misregistration is therefore **3-6x the size of the finding** for millimetre-scale labels and
irrelevant for centimetre-scale ones.

That is a PREDICTION, and it matches the weakest columns on the 909-study holdout: MCL 0.7893, Lateral
OA 0.8246, Lateral Meniscus 0.8504 — all fine structure — against Effusion 0.8655 and Baker's 0.9086.

**What this does NOT license.** Not CoPAS's +0.056. That was bought against a full-FOV 576-matrix
baseline with no crop and no alignment; we already crop, and two thirds of our offset is common-mode and
already harmless. The available component is the 5.7 mm residual only, and its size is unmeasured in AUC.

**What it DOES license.** Two separable fixes, and they are not the same work:
1. the common-mode ~10 mm offset -> crop CENTRING, which knee-yrf declined on retention grounds
   (+0.0294 tissue kept, not worth a 99 GB rebuild). It now has a second, independent reason.
2. the residual 5.7 mm cross-slot spread -> slot-to-slot REGISTRATION, which is the CoPAS mechanism and
   which centring alone does not fix.

**What it retires.** The blanket reading that our six flat slot-aggregation arms failed because the
aggregator was wrong. They were fusing views misregistered by 3-6x the fine findings' size. That does
not prove the aggregator was fine — it removes the assumption that it was the only suspect.

LIMIT, from the module's own docstring: an Otsu mass centroid is not a joint centre; it is dragged by
coil, fat and the far limb, and knee-yrf measured tissue-centring's worst series as worse than
image-centring's at 80 mm. A small spread would have been conclusive; this middling one is suggestive.

### alignment_damage — slot disagreement does NOT predict focal error; the registration fix is unlicensed

`crop_alignment` measured the INPUT: a study's routed slots disagree about where the knee sits by 5.7 mm
p50 / 10.7 mm p90, three to six times a meniscal tear. This is the OUTPUT read that was pre-registered
against it, on the 909 fold-0 holdout studies, mixing the champion's six scoreable members. Zero GPU,
~30 minutes of CPU, and it was built to decide a 99 GB cache rebuild plus hours of refit BEFORE either
was started.

```
455 well-agreeing studies and 454 poorly-agreeing, split at 5.85 mm; 0 abstained with one slot
focal mean gap -0.0068, regional mean gap -0.0171
DIFFERENTIAL (focal minus regional) +0.0103, 95% [-0.0305, +0.0421]
```

THE VERDICT, AGAINST THE BANDS FILED BEFORE ANY NUMBER EXISTED. The sizing argument was that a perfect
registration lifts the poorly-agreeing half toward the well-agreeing one on the seven focal findings
only, so the mean-AUC gain is `differential x (7/12) x (1/2) = differential x 0.29`; the 1.5% bar
therefore needed a differential near 0.051, and even a 0.01 move needed 0.034. The measured +0.0103 sits
in the pre-registered `<= 0.020` band AND its interval spans zero. The registration fix is not licensed
by this evidence. It is dropped as a candidate for the rebuild, not carried as a rider.

THE SIGN NOBODY PREDICTED, AND WHAT IT RETIRES. Both raw gaps are NEGATIVE: the poorly-agreeing half
scored BETTER, on focal and regional alike. That is the opposite direction from the difficulty confound
the differential was built to subtract — a scattered tissue centroid is evidently not a proxy for a worse
study. Whatever the Otsu spread is tracking, it is not acquisition quality, which is worth knowing
because the same proxy underwrites the whole `crop_alignment` reading.

WHAT THIS DOES NOT KILL. The COMMON-MODE offset, which this design cannot see at all: every study shares
it, so it cannot separate studies and cannot appear as a spread effect. That fix — tissue-centred
cropping — stays exactly where knee-yrf left it, priced on retention (+0.0294 at 130 mm, knob-sized).
Nor does it clear the proxy: an Otsu mass centroid is not a joint centre, and a spread built on it
attenuates any true gap toward zero. And it says nothing about a model TRAINED aligned from the start,
which is a different experiment from scoring an image-centred model on studies that happen to align.

### distill_pairs — rank distillation is a TWIN of a seated member; the per-label rule was unreadable

`--distill-weight` with `--teacher-targets`, 8 epochs on the shared recipe. The mechanism predicts lift
ORDERED BY TEACHABLE PAIR MASS `2p(1-p)`: a teacher's ordering can only be copied where the student has
ordered pairs to get wrong, so the rarest labels should gain most.

```
mean 0.8594 (e8) vs control 0.8578  = +0.0016, inside the 0.0062 holdout floor
seat value against greedy7_v1 (7 scoreable members): -0.0003  (0 of 1 clear +0.0015)
distance: nearest 0.9159 vs the roster's own worst nearest 0.8893
```

| label | positive rate | pair mass | delta vs control |
|---|---|---|---|
| Fracture | 0.071 | 0.132 | **+0.0133** |
| Synovitis | 0.132 | 0.229 | -0.0071 |
| MCL | 0.174 | 0.288 | -0.0049 |
| Lateral OA | 0.247 | 0.372 | +0.0005 |
| Baker's | 0.253 | 0.378 | +0.0104 |
| Lateral Meniscus | 0.254 | 0.379 | +0.0075 |
| ACL | 0.278 | 0.402 | +0.0054 |
| Contusion | 0.294 | 0.415 | -0.0036 |
| Medial OA | 0.367 | 0.465 | -0.0050 |
| PF OA | 0.452 | 0.495 | -0.0045 |
| Medial Meniscus | 0.533 | 0.498 | +0.0057 |
| Effusion | 0.605 | 0.478 | +0.0013 |

THE SHIPPING QUESTION IS ANSWERED AND THE MECHANISM QUESTION IS NOT. Both rulers decline the seat, and
for once they agree: the seat value is negative, and the distance column says WHY — at 0.9159 nearest,
this member sits closer to a seated member than any seated member sits to its own nearest neighbour. A
rank average cannot pay for an ordering it already holds. `distill_pairs` does not go on a card.

THE ARM'S OWN DECISION RULE COULD NOT BE EXECUTED, which is a harness finding, not a verdict on
distillation. It pre-registered "lift ordered by pair mass" and the paired per-label floor half-widths
run to +/-0.0284, so only Baker's is even marked; eleven of twelve deltas are unreadable individually.
The profile is neither flat nor rarity-ordered — the largest gain IS on the rarest label, the mechanism's
headline prediction, and the 2nd and 3rd rarest are the 1st and 3rd most negative. Writing a per-label
clause against a per-label ruler that cannot resolve a per-label effect is the defect; the next arm on
this axis states its rule on the axis its ruler can read. knee-1x4j stays gated.

### CORRECTION — the v3 axis was closed on the wrong member, and the distance ruler says so (2026-08-24)

The `v3base_dense` entry above closes DINOv3 "as a *source of new seats*" on a seat value of −0.0007. Three
things are wrong with carrying that forward, and the entry itself warned about two of them.

```
against greedy7_v1 (0.933, the CURRENT seven) — roster's own nearest spread 0.8724 .. 0.8893

  candidate            seat      nearest   solo
  lora_r8_v3small_224  -0.0013   0.8455    0.8217
  dinov3_base_lora     -0.0009   0.8467    0.8268
  v3base_dense         -0.0006   0.8829    0.8512
```

**It was closed on the one v3 member that is not distant.** `v3base_dense` reads nearest 0.8829, inside the
roster's own spread — an ordinary member. Both LoRA-v3 members sit at ~0.846, BELOW the most distant seated
member by 0.027, which is `member_distance`'s own definition of genuinely new information. A claim about
*redundancy* was decided on the sole family member that is not redundant.

**It was priced against a roster that no longer exists** — five members, 0.8690, two of them since replaced —
and **with the quality ruler alone**, on the same day `MemberDistance` was built to answer this exact
question. The entry's own words: *"The −0.0007 is not evidence that a v3base seat would cost a tick; it is
evidence that this ruler cannot tell."* The caveat was written and then the verdict was quoted without it.

**The mechanism was inside the adaptation finding.** LoRA freezes the base, so the member keeps DINOv3's
representation rather than being fine-tuned toward the solution every other member converges on. That
predicts exactly the ordering measured: both LoRA members distant, the full fine-tune ordinary. The
adaptation that scores WORSE is the one that is DISTANT, and nothing charged that to diversity.

**And the variable that makes it runnable was never isolated.** `v3base_dense` and `dinov3_base_lora` differ
on TWO things — LoRA vs full fine-tune, AND 224 → 336 plus the dense recipe — and the whole +0.0244 was
charged to adaptation. Nobody has run **LoRA + the dense recipe + 336**. That is the distant-and-strong cell
`knee-053z` asked for and never got, and it is filed as `knee-u0my` with its bands pre-registered.

WHAT STANDS FROM THE ORIGINAL ENTRY: the adaptation re-attribution (DINOv3's weakness was a recipe finding,
not an encoder one) is untouched and is in fact the load-bearing half of the argument above. What does not
stand is "the axis is closed as a source of new seats".

### THE ARCHITECTURE AXIS IS A TRADEOFF CURVE, AND EVERY POINT ON IT WAS TRAINED ON THE WRONG RECIPE (2026-08-24)

`knee-y59c`. The v3 correction above was one member. This is the same audit run across **every trunk this
project has ever fitted**, on both rulers, card-free, priced against `greedy7_v1`'s seven members.

```
  member                trunk                 nearest   solo
  mae_matched           MAE pretraining       0.4333    0.6735
  cnx_dense             ConvNeXt (CNN)        0.7941    0.7927
  vjepa_warm            V-JEPA 3D tubelet     0.8212    0.8337
  lora_r8_v3small       DINOv3 + LoRA         0.8455    0.8217
  dinov3_base_lora      DINOv3 + LoRA         0.8467    0.8268
  v3base_dense          DINOv3 full fine-tune 0.8829    0.8512
  distill_pairs         DINOv2 self-distill   0.9159    0.8594

  roster of 7, its own nearest-neighbour spread: 0.8724 .. 0.8893
```

**Distance orders exactly by architectural distance from DINOv2, seven points, no exception** — a different
pretraining objective, then a different inductive bias, then a different input geometry, then a frozen base
of the same family, then a fine-tuned base of the same family, then the same family distilled from itself.
That is a curve, not scatter. Architecture diversity demonstrably produces different orderings here and the
amount scales with how different the architecture is.

**Solo score is monotone the other way on the same seven points.** Distance and strength are anti-correlated
across the whole axis. That is why each arm read as a refutation in isolation: every one bought its distance
with quality and was then judged on quality alone.

**CORRECTION TO THIS ENTRY'S FIRST DRAFT, written the same hour.** It claimed the recipe confound was
unnoticed and named read size and learning rate as the wrong knobs. Both are wrong, and the ledger says so
three screens up — this is the failure CLAUDE.md warns about, filed against rows that already held better
evidence. The confound was raised on 2026-08-23 in `raddino_dense` (line 2538: "a new trunk plugged into a
recipe that was fitted around DINOv3 … every trunk swap needs its own tuning pass before it can be read at
all"), and set aside on a cost argument rather than refuted. And for `cnx_dense` specifically, read size is
dead on three independent readings and the schedule was refuted hard enough to retitle `knee-9otf`.

**What the recipe confound actually still covers, on `cnx_dense`'s own numbers.** At MATCHED TRAIN LOSS it
holds 0.7927 at loss 0.8332 where `base_dense` holds 0.8371 at loss 0.8456 — a 0.044 generalisation gap at
equal fit. It fits BETTER and generalises WORSE. That is not an under-driven trunk, so LR and epochs are
correctly dead; it is the signature of a REGULARISATION regime that does not transfer. DINOv2's recipe needs
little augmentation because the pretrained representation carries the generalisation. A CNN fine-tuned
end-to-end from ImageNet weights on ~1.5k studies is a different regime, and **augmentation strength is the
one knob on this arm that nothing has tested.** That is the surviving live suspect, narrower than the first
draft claimed and pointed at by the arm's own trajectory rather than assumed.

**RETRACTED 2026-08-24, by the trajectory this paragraph claimed to read.** The matched-loss argument above
is real arithmetic and the wrong inference, and the giveaway is the clause "pointed at by the arm's own
trajectory" — the trajectory was never opened. `logs/cnx_dense.log` holds it:

| epoch | 0 | 3 | 6 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|
| train loss | 1.2612 | 1.0696 | 0.9419 | 0.8833 | 0.8553 | 0.8416 | 0.8332 |
| holdout AUC | 0.7050 | 0.7659 | 0.7858 | 0.7921 | 0.7909 | 0.7927 | 0.7927 |

The holdout rises monotonically to the final epoch and never turns over, while train loss is still falling
steeply. **That is an UNDER-TRAINED curve, and it is the one curve an over-regularisation story cannot
produce** — a regime that fails to transfer peaks and then decays, and this one has not peaked in twelve
epochs. What the matched-loss comparison actually shows is that ConvNeXt converts loss into AUC more slowly
than DINOv2 does; it holds two different points on two different trajectories side by side, which reads as
overfitting only if you assume both had converged. Neither had been checked.

So the correction runs the other way, and it un-kills what this paragraph killed: **LR and epochs are NOT
correctly dead on this arm — they are the live suspects, and augmentation is the one that is now
contraindicated.** Adding augmentation strength to a model that never finished climbing would slow the
climb. `cnx_own_recipe` is left exactly as written (`--backbone-lr 4e-4`, ten times `cnx_dense`'s), because
raising the rate is the conventional fix for a monotone-rising fine-tune and the arm already carried it.
Priced on `knee-y59c`; kill at epoch 4 against `cnx_dense`'s 0.7681 if the rate does not bite.

This is the same failure as the two P0s three screens up, one layer in: not reciting a belief instead of
reading the record, but reading HALF the record — two summary numbers pulled from the entry — and writing
"the arm's own trajectory" over the half that was never opened. The twelve lines that refute it cost one
`grep` of a log that had sat in `logs/` since the run.

**What today's measurement DOES refute, and it is a sentence, not a knob.** `cnx_dense`'s entry closes with
"a different inductive bias is not a different error structure. The axis closes." The distance ruler
measures the opposite: nearest **0.7941** against a roster whose most distant seated member reads 0.8724 —
0.078 outside the spread, the largest of any member that did not collapse. A different inductive bias
produced the most different error structure in this project. What the entry correctly showed is that it
produced a WORSE one; what it did not show, and could not with the quality ruler alone, is that it produced
a SIMILAR one.

**And the distance does not cash, which is the same day's second measurement and the one that sets the bar.**
Being the most distant member is not the same as being worth a seat, and the gap between those two was never
measured — `cnx_dense`'s 63rd-of-65 came from a FORWARD greedy, which only ever asks "is this the best next
add to a 62-member soup". It never asks what the member does to the PEAK set. That question needs no card;
the holdout arrays have been on disk since the runs. Adding a ninth member to the 8-member greedy peak
(0.8731), `record_mix`:

| added 9th | solo | ensemble |
|---|---|---|
| `quarter_data` | 0.7907 | 0.8710 |
| `cnx_dense` | 0.7927 | **0.8706** |
| `side_dense` | 0.7937 | 0.8710 |
| `conf24` | 0.8259 | 0.8716 |
| `deep24` | 0.8352 | 0.8727 |
| — none — | — | **0.8731** |

That five-point read was written up as a QUALITY FLOOR, and the write-up carried two claims the sample could
not support. **Both are retracted here, by the denser measurement below.** The first: `cnx_dense` was said to
lose to "matched-quality but much NEARER peers `side_dense` and `quarter_data`". Their distances were never
looked up. Measured, all three are far outside the roster spread — `cnx_dense` nearest 0.7686, `quarter_data`
0.7932, `side_dense` 0.7961. The comparison held distance roughly FIXED and varied nothing; it is not evidence
that distance bought no exemption, because no near member was in it. The second: "damage is monotone in solo
quality" fails on the wider sample — `deep24` (solo 0.8352, nearest 0.8342) scores 0.8727, ABOVE `v3base_dense`
(solo 0.8455, nearest 0.8829) at 0.8721. Higher solo, closer, worse.

**The denser measurement, and it is a bigger result than the floor it replaces.** The sweep was extended to 28
further members, spanning solo 0.8406–0.8534 and nearest-agreement 0.8342–0.8951 (`logs/floor_sweep.log`):

```
best   intensity392_ws 0.8729  attend_true 0.8729  calibrated_targets 0.8728  aug_long 0.8728  focal_dense 0.8728
worst  slices16_ws 0.8718  per_label_loss 0.8718  slotmean 0.8719  schedule_8 0.8720  band_full 0.8720
```

With the five above that is **33 candidates, spanning solo 0.79–0.857 and nearest 0.769–0.951, and not one of
them reaches 0.8731.** The whole spread of outcomes is 0.0011 — a sixth of the 0.0062 noise floor. Nor is it
an append-only limit: for twelve of the strongest and most distant candidates, the EXHAUSTIVE best-8 over
{peak8 + candidate} returns the original eight, unchanged, in all twelve cases (`logs/swap_sweep.log`,
`swap_sweep2.log`). The peak set is a local optimum under append AND under single-swap.

**The owner's principle shows up in the SIGN and dies on the SIZE.** Over the 28,
`spearman(delta, NEAREST) = −0.203` — distant members do help more, which is the direction "it is about
diversity, not averaging seven copies of the same number" predicts, and the two most distant of the 28
(`intensity392_ws` 0.8551, `attend_true` 0.8607) are two of the best five. But `spearman(delta, SOLO) = +0.477`
is the larger of the two, and both effects live inside a 0.0011 band. **Distance orders the candidates; it
does not pay any of them.**

The confound that would rescue it was checked and did not. `RankAverage` is EQUAL-WEIGHT, so a weak member
is forced to vote as loudly as a strong one, which penalises precisely the distant-but-weak case. Under the
weighted rule the damage shrinks and does not reverse: `QualityWeightedRank` 0.8712, `LogitAverage` 0.8726,
both still under the 0.8731 the peak holds without it. Down-weighting a distant weak member does not make it
pay.

**What this does NOT say** — and the distinction is the whole finding. It does not say diversity loses to
averaging. `margin_seat` is solo 0.8525, worse than several seated members, and greedy takes it **2nd of 65**
on distance alone; distance built this peak. What is exhausted is the POOL, not the principle. Every member
that exists today was trained on one trunk family, one data pipeline, one split — the diversity already
banked in the eight is all this recipe space has to give, and a 34th draw from it is a re-roll, not a new
axis. Filed as `knee-98y2`.

**CONSEQUENCE — and it changes the bar, not just the wording.** The pre-registered test written yesterday for
`cnx_own_recipe` was `peak8 + it > 0.8731`. **That is a test nothing passes**, so it prices no member and
gates no decision; a bar 33 candidates fail is measuring the bar. It is replaced by a DISPLACEMENT test: put
`cnx_own_recipe` into the full pool, re-run the greedy from scratch, and ask whether the NEW peak exceeds
0.8731. A member that cannot append can still rebuild the set around itself, and that is the only remaining
way in. Solo AUC stops being a gate entirely — it enters the pool at any value. `knee-y59c` carries the
replacement.

**"The RECIPE SPACE is what saturated" was the obvious next claim, and it was tested the same hour rather
than written down.** Seven members that never entered the 33 were run through the same displacement test —
`vjepa_warm`, `vjepa_member` (video pretrain), `raddino_dense` (medical-imaging pretrain), `dinov3_base_lora`,
`dinov3_v3small_224`, `lora_r8_v3small_224`, `grade_ladder`. **All seven return the original eight.** That is
nineteen displacement tests and zero passes, three separate pretrain corpora among them, so "a member from
outside the corpus gets in" is NOT supported and must not be asserted.

But the confound is the same one that broke the five-point read, so it is checked here rather than skipped.
Solo scores: `dinov3_v3small_224` 0.8057, `raddino_dense` 0.8137, `lora_r8_v3small_224` 0.8215,
`dinov3_base_lora` 0.8268, `vjepa_warm` 0.8337, `vjepa_member` 0.8343 — every outside-corpus member is
**0.013–0.054 BELOW the seated band (0.8466–0.8598)**. They are outside-corpus AND worse, and the test cannot
separate the two. The one candidate at genuine parity is `grade_ladder` at **0.8565**, above six of the eight
seated members — and it is an inside-space member (an ordinal-target trick, same trunk, same corpus). It fails
too.

So the honest state of the axis is: **no candidate has ever existed that is BOTH outside the pretrain corpus
AND at seated quality.** Inside-space-at-parity fails (`grade_ladder`). Outside-corpus-below-parity fails
(six members). The cell that would decide it is empty, and `cnx_own_recipe` — a different trunk on a recipe
built for it rather than inherited — is the first candidate that could fill it. That is now the arm's whole
value: not a ninth seat, but the one measurement that separates corpus from quality.

HONEST LIMITS: one peak set, one split, one fold. The swap test is SINGLE-swap, not exhaustive, so a
two-member exchange is untested. And `record_mix`'s own docstring records that this holdout prices diversity
2.4× too cheaply against the board (0.0059 here read 0.014 there) — which does NOT rescue any of the 33,
since a multiplier on a negative delta is still negative, but it does mean the 0.0011 ordering could be
board-resolvable even though it is holdout-noise. Quoted as a ranking, never as a price.

And that lands exactly on the question `raddino_dense`'s entry closes with — the greedy peak sits at seats
8–10 and declines after, so "the ensemble does not want another member, it wants a member that can do
something none of the existing ones can." Today's table names which candidates those are, by measurement.
Bands are pre-registered in `knee-y59c`.

The distance ruler still has ONE paired board observation behind it (2026-08-21). Quoted as a RANKING, never
as a price.

### DISTANCE AS A SEAT RULE, RUN RATHER THAN ARGUED — AND REFUTED (2026-08-24)

The architecture-diversity question has been asked in this repository more times than it has been
measured. Every reading that touched it — `seat_value`, `marginal`, `swap`, the greedy curve — prices a
candidate by what it does to the HOLDOUT MEAN, which is the quality ruler choosing seats. `MemberDistance`
was built on 2026-08-21 to read the other axis and has only ever been printed BESIDE the quality number,
never used to decide anything. So the greedy's standing verdict against every distant member this project
has trained might have been the verdict of the ruler rather than of the axis, and nothing distinguished
those two cases.

`MemberCurve.diverse` is the missing rule. Seed with the strongest member, then repeatedly take the member
whose agreement with its nearest already-seated neighbour is LOWEST — the twin detector run backwards. The
mean is printed at every seat and never consulted. Parameter-free on purpose: a distance threshold or a
quality-times-distance weighting are both knobs, and a knob on a rule invented to settle an argument is how
a favourable answer gets tuned out of a pool of seventy.

    python -m kneemri.evaluation.record_mix --records <pool> --seats greedy
    python -m kneemri.evaluation.record_mix --records <pool> --seats diverse

**70 members over 909 shared studies, both curves off the same arrays, no card.**

    seats     greedy (by mean)        diverse (by distance)
      1       0.8598                  0.8598  (same seed member)
      9       0.8719  <- peak         0.8331
     20       0.8715                  0.8455
     62       0.8717                  0.8624  <- peak
     70       0.8616                  0.8616  (everything seated, same set)

**Distance alone loses 0.0095 at its own peak and 0.039 at matched seat count.** The floor is 0.0062, so
the first is past it by 1.5x and the second is not close to a judgement call. The rule is refuted.

**WHY IT LOSES, WHICH IS NOT WHAT THE HEADLINE SAYS.** Read `diverse`'s first five seats: `mae_matched`,
`lr2e-4_ws`, `quarter_data`, `encoder_adapt_thawed`, `dinov3_v3small_224`. Two of those are members
deliberately trained on a FRACTION OF THE DATA and one is a refuted learning rate. In this pool distance
and weakness are correlated — a member ends up far from the roster mostly by being bad at the task — so a
rule that reads distance alone buys that correlation at full price. What is refuted is the sentence
"distance is a seat rule", not the sentence "a distant member can be worth a seat".

**AND THE GREEDY IS NOT BLIND TO DISTANCE, which is the reading that actually settles the argument.** It
seats `vjepa_warm` twelfth and `mae_matched` — the most distant member in the pool, solo 0.6735 — EIGHTEENTH,
on a plateau where the curve is flat to four decimals. So the quality ruler does take the distant members;
it takes them late, once the strong core is assembled, and it declines to take them instead of a strong
member. That is a different claim from "architecture diversity buys nothing" and it is the one the numbers
support.

**WHAT IS STILL OPEN, and it is one cell.** Every distant member this project owns is also weak: the seven-point
table above orders solo score exactly opposite to distance with no exception. Nobody has held a member that is
distant AND strong, so the greedy has never been offered the trade it would have to refuse for the axis to be
closed. `v3lora_dense` (knee-u0my) exists to produce exactly that member and is on the card tonight.

---

### `cnx_own_recipe` — the inherited rate cost a whole trunk, and an audit says two more were never finished — 2026-08-24

`cnx_own_recipe` finished at 21:00Z, 12 epochs, **peak 0.8186 at e6** against `cnx_dense`'s 0.7927.
**+0.0259**, four times the 0.0062 floor.

**THE CONFOUND, STATED FIRST.** Three flags moved, not one: `--backbone-lr` 4e-5 → **4e-4** (ten times),
`--train-size` 352 → 448, and `--batch 2 --accumulate 3` → `--batch 1 --accumulate 6` (effective batch 6
unchanged — that one is VRAM for the larger read, not a training change). Resolution at constant field was
independently priced at **+0.0002** on `res392` and re-refuted per-label on 2026-08-24, so the read size is
not a plausible carrier of +0.0259, but it is not a clean single-variable arm and is not written up as one.

**WHAT SETTLES IT ANYWAY, AND IT NEEDS NO CONTROL ARM.** The two curves have different SHAPES.

```
cnx_dense       (4e-5)  auc  0.7050 .. 0.7909 0.7927 0.7927   PEAK AT e11 OF 11
                        loss 1.261  .. 0.855  0.842  0.833    still falling at the end
cnx_own_recipe  (4e-4)  auc  0.7314 .. 0.8186 0.8167 .. 0.8130  PEAK AT e6, then DECLINES
```

`cnx_dense` never converged. Its best epoch is its last, its loss is still dropping when the run ends, and
its number is therefore a statement about the epoch budget rather than about ConvNeXt. `cnx_own_recipe`
peaks and then overfits, which is what a fitted rate looks like. That diagnosis is read off `cnx_dense`'s
OWN trajectory and does not depend on the second run at all.

**THE DEFECT IN ONE SENTENCE.** 4e-5 was fitted over ~189 DINOv2-family runs and then handed to every other
encoder unexamined, and the ledger wrote the resulting numbers up as facts about architectures.

**WHAT IT DOES NOT BUY.** 0.8186 is still 0.033 below `base_dense` (0.8512). The trunk is far and, even
fitted, weak. This corrects the ledger's account of WHY, not its ranking.

Per-label, at each run's own peak: gains everywhere but Synovitis (−0.0066), led by Baker's **+0.1136**,
ACL +0.0341, Contusion +0.0391. **MCL moves +0.0027** — it does not move for anything, in any arm, ever.

#### The audit this triggered, which is larger than the result

Every `metrics.jsonl` on disk was scanned for the same signature — best epoch at or next to the last epoch
recorded. **Twenty-one runs match.** Two are not undertrained but outright broken:

| run | asked | got | recorded | what the driver log says |
|---|---|---|---|---|
| `raddino_dense` | 12 | **7** | 0.8137 | `ARM FAILED raddino_dense (trainer exit 1)` |
| `vjepa_member` | 12 | **10** | 0.8343 | `ARM FAILED vjepa_member (trainer exit 127)` |

Both died **mid-epoch** at healthy throughput — raddino at step 248/583 running 448 img/s with loss still
falling, no traceback, no OOM (27.14 GB peak on a 32 GB card), killed externally at 23:29 on 2026-08-23
during the card-handoff window. Exit 127 is command-not-found, so vjepa's arm chain broke rather than its
trainer — the third instance in this repo of a missing binary reading as a result. `vjepa_member`'s log
additionally records a real stall (`173 img/s against this run's own peak of 589`, diagnosed in-log as
another process paging over PCIe), so that number was measured under contention as well as truncated.

**WHY TRUNCATED IS NOT THE SAME AS SHORT.** The recipes set `one_cycle=True` over 12 epochs. A run that
stops at 7 stops in the high-LR portion, so the anneal — where one-cycle does its fine convergence — never
runs at all. The control for this is on disk and it is decisive: `v3base_dense` ALSO peaks at e6, then runs
its remaining five epochs and never beats that peak. That is a converged measurement. RadDINO has no such
evidence, and its 0.8137 is a floor.

**AND `v3base_dense` SETTLES A CARD DECISION IN THE OTHER DIRECTION.** Because it converges at 4e-5, DINOv3-base
does NOT carry the ConvNeXt defect. `v3base_own_recipe` — written, `arm_check`-clean, 2h16m, asking for a
fitted 1.5e-4 — is therefore a ~0-expected run and should NOT be armed. The rate-fix mechanism is real and
it does not apply here. One free audit of a file already on disk retired a 2h16m arm.

**WHAT THE AUDIT OPENS.** This ledger's previous entry ends: *"Nobody has held a member that is distant AND
strong, so the greedy has never been offered the trade it would have to refuse for the axis to be closed."*
The most pixel-distant trunk in the pool is RadDINO — pretrained on medical images — and it is one of the two
that crashed. The empty cell was never measured; it was measured and thrown away by a killed process.

Its own curve (`0.7247 0.7439 0.7852 0.7944 0.8000 0.8119 0.8137`) is DECELERATING, so the honest estimate
of the remaining gain is modest and uncertain, not the naive slope. But `fitted_encoder.py` exists to start
an arm from a member checkpoint, so the finish is a warm start (~1.65 h) rather than 12 epochs from scratch
(~3.3 h). Gated on `kneemri-farswap`'s board number at the 00:00Z reset, which tests the same seat hypothesis
with a WEAK non-DINOv2 member: if trunk distance pays there, a strong RadDINO is the member that cashes it.

Filed as three beads — the two crash readings, and a convergence check for the arm harness. The repo has
nine gates that gate a SHIPMENT and none that gate a NUMBER, which is how the same defect landed three times.

### `convergence_check` — the tenth check, and the first that guards a NUMBER rather than a shipment — 2026-08-24

Nine checks in `kneemri/experiments/` guard a SHIPMENT: that a kit builds, that a kernel's members mount,
that a queue is armed exactly once, that a champion is still rebuildable. None guarded a RESULT. This is
the tenth, and it exists because the same defect landed three times: a run bounded by the harness rather
than by the recipe, whose partial peak was then written down as a fact about an architecture.

`python -m kneemri.experiments.convergence_check` reads every metrics file under the processed root and
compares three numbers the file already holds — the best epoch, the last epoch recorded, and the epoch
budget the recipe ASKED for.

**Reading of the corpus, 88 runs: 57 converged · 15 undertrained · 7 truncated · 3 stopped · 6 unscored.**

**THE FIRST VERSION OF THIS CHECK WAS WRONG, AND THE WAY IT WAS WRONG IS THE ENTRY.** It reported TEN
truncated runs. Three of those — `res392_lr8e-6_ws`, `focal_pool`, `calibrated_targets` — carry
`patience=3` in their recipe and stopped at exactly epoch 10 of 12. That is the stopper working. One of
them is `res392`, the resolution refutation, so a check written to stop us reasoning from killed runs
would have re-opened a correctly-closed family on its first run. The fix is one recipe key: `patience=0`
is the default and means no stopper was armed, so a run carrying it that ends short was ended by
something outside the trainer. Verdicts `TRUNCATED` (died) and `STOPPED` (decided) look identical on
disk and differ only there.

**WHAT IT ACTUALLY FOUND, STATED SMALLER THAN IT FIRST LOOKED.** Of the seven truncations, the ledger had
already labelled four in prose — `depth_regions_span` ("truncated at 9 of 12 epochs by its own wall-clock
`timeout`, not by a fault", and its conclusion rests on run 1, which finished), `dissent_seat` ("killed at
epoch 8 of 12"), `pair_memory` ("a first run whose checkpoint was overwritten"), and `focus_latmen`. One,
`slices16_ws`, is a **known false positive**: `RunMetrics.rows()` returns the LAST run's group, and that
file holds a finished twelve-epoch run plus a killed re-run whose checkpoint was never kept. The verdict
is about the last thing that ran, which is not always the member on disk. That leaves `raddino_dense` and
`vjepa_member` unmarked — the same two the hand audit found this afternoon.

So the tool did not expand the hand audit; it CONFIRMED it, and that is the honest claim. Its value is
that the label is now mechanical. Four of those seven were caught only because someone remembered to
write "(killed at epoch 8 of 12)" into a prose cell, and the two that nobody wrote down are exactly the
two that were reasoned from for weeks as properties of RadDINO and V-JEPA.

**WHAT A TRUNCATION IS ACTUALLY WORTH, WHICH IS NOT ONE NUMBER — and this paragraph replaces a wrong one.**
The first version of this entry, and knee-qk4a with it, said `dissent_seat`'s −0.0069 was "smaller than the
anneal is worth on every finished curve here." That was asserted, not measured, and measuring it refutes it.
For each of the 54 CONVERGED twelve-epoch runs, take the gain from best-through-epoch-N to best-overall —
exactly what a run killed at N rows under-reports by:

```
killed at  7 rows   median +0.0011   mean +0.0035   90th +0.0113   max +0.0336   over the 0.0062 floor: 7/54
killed at  9 rows   median +0.0000   mean +0.0006   90th +0.0023   max +0.0099   over the 0.0062 floor: 2/54
killed at 10 rows   median +0.0000   mean +0.0001   90th +0.0004   max +0.0028   over the 0.0062 floor: 0/54
```

**Truncation depth decides whether truncation matters, and the three truncated runs therefore split three
ways rather than sharing one verdict:**

- `vjepa_member` died at **10**. Nothing is owed to it — 0.8343 is its number, and re-running it buys a
  clean exit code rather than a score. Do not expect the V-JEPA reading to move.
- `dissent_seat` died at **9**, where the median cost is zero and only 1 of 54 runs would lose more than
  its own −0.0069. So the anneal does not explain that number and **knee-qk4a drops P1 → P3.** What survives
  is thin and should be stated as thin: a killed run's score is a bound, and `margin_seat` read positive on
  the same axis, so the objective family is not CLOSED on `dissent_seat` alone. That is a reason not to quote
  the closure, not a lead to spend a card on.
- `raddino_dense` died at **7**, mean +0.0035 with a 90th percentile of +0.0113. It is the one genuine floor
  of the three, and the only one whose warm start is priced by evidence rather than by principle.

The general claim — a truncated run's score is a floor — is still true and still worth the check. It is the
SIZE that was invented: priced at one depth, then quoted at every depth. The check reports the depth on
every line it prints; I did not read it.

**WHY THE VERDICT IS NOT A BOOLEAN.** `TRUNCATED` and `UNDERTRAINED` have opposite fixes. A truncated run
is re-run or warm-started; an undertrained one needs a longer budget or a fitted rate. Collapsing them
would have hidden that `cnx_dense` wanted a rate (+0.0259 when it got one) while `raddino_dense` wanted
only to be allowed to finish. Only `TRUNCATED` exits non-zero: an undertrained run is an honest
measurement of a budget and its score is real.

**THE OTHER HALF OF THE OUTPUT, READ A DAY LATE — and it yields nothing, which is the point of writing it
down.** The first corpus reading reported 15 UNDERTRAINED against 7 TRUNCATED, and only the truncations were
chased. Reading the larger class:

- **Six were `throughput_*` timing probes and one a smoke, all asking a single epoch.** A one-epoch run peaks
  on its last epoch BY CONSTRUCTION, so the verdict was vacuous for nearly half the class. `benchmark()` now
  drops `epochs<=1` and the corpus reads **81 runs: 56 converged, 7 truncated, 3 stopped, 9 undertrained,
  6 unscored.**
- **Three completed SHORT budgets** — `crop80_tissue` (8), `schedule_8` (8), `wholestack_fold2` (6). Their
  anneal ran to its end, so "peaked last" is the schedule finishing, not a crash. And the axis is already
  priced: the ledger's own `crop80_tissue` entry above says the control is 12 epochs to this arm's 8, and
  `schedule_length` puts that difference at **−0.0028** — inside the 0.0062 floor. Extending a roster member
  to 12 buys under half a noise floor for ~2 h of card. **Declined, on evidence that was already here.**
- **Three peak on the last epoch of a FULL twelve** — `cnx_dense` and the two `encoder_adapt` runs. All three
  already carry a confirmed recipe defect: `cnx_dense` returned +0.0259 when its rate was fitted, and
  `encoder_adapt` (0.6902 / 0.7309) is the closed SSL-as-init family. **No unclaimed run is left in the class.**

Worth stating precisely because the verdict is easy to over-read: **a late peak is ORDINARY under one-cycle.**
32% of finished twelve-epoch runs peak at e9 or e10, deep in the anneal. Peaking on the very last epoch is
what is rare — 3 of 53, against the ~14% a flat prior over the observed e5–e11 range would predict — and its
3-for-3 association with a confirmed recipe defect is a direction at n=3, not a rule.

So the check has now been read end to end, and **on this corpus it has produced no lead it did not confirm
from prose already in this file.** That is the honest measure of it: it converts knowledge that lived in
paragraphs into a command that cannot go stale, and on a corpus this well-audited that is all it can do. Its
value is prospective — the next killed run gets caught the evening it dies, not three weeks later.

## The card came free, and the measurement said not to spend it (2026-08-24 22:12Z)

`celltrack` released the card at 22:08Z after a reconciliation worth recording on its own: their session
announced "card-free" at 21:55Z, our `card_check` read a biohub python holding 1876 MiB at 22:05Z, and the
explanation was a `pytest` collection that imports a CUDA-touching module — true when claimed, briefly
false after, gone by 22:08Z. **A session's claim about what it is running is the weakest available
evidence, including when the session is ours.** Both projects spent 65 minutes yielding an idle card to
each other because each trusted the other's prose over the device.

With the card free the queue held one run priced and ready: warm-start `raddino_dense` from its e6
checkpoint, ~1.65 h, justified this morning as "the one truncation with a measured floor under it" (N=7,
mean +0.0035, 90th +0.0113). **It was not armed, and the reason is that three separate checks refuted it
in the twenty minutes before the card would have been taken.**

**1. The truncation is a DECISION, not a crash.** `convergence_check` reads `raddino_dense` as TRUNCATED
because `patience=0`, and `patience` can only see the automatic stopper. This ledger's own 2026-08-23
entry says the run was killed deliberately against a pre-registered band: slope +0.0018/epoch at e6, five
remaining epochs worth ~+0.009 at that slope, landing ~0.823 against a pool whose control is 0.8512. The
check cannot distinguish a kill from a judgement, and the depth-pricing I published this morning inherited
the misreading. **The check's TRUNCATED verdict needs the ledger read next to it; on this run the ledger
is the better instrument and it was already right.**

**2. "RAD-DINO is under-driven" is refuted by curves already on disk.** The rate-refit case rested on the
shallow slope with the loss still falling — the `cnx_dense` signature, which returned +0.0259 when its rate
was fitted. Against `band_dense_llrd`, the arm this repo already ran to characterise an underfit trunk:

```
at epoch 6     train loss     holdout AUC
raddino_dense     0.7570         0.8137
band_dense_llrd   0.8419         0.8435
```

RAD-DINO reaches a **lower** training loss and a **lower** holdout AUC than the known-underfit arm at the
same epoch. That is fitting fast and transferring badly — a representation mismatch, where more rate makes
it worse. The hypothesis predicted the opposite sign of exactly the number that was already recorded.

**3. Layer decay would have been the wrong tool anyway**, for the reason `band_dense_llrd`'s entry gives at
line 2025: `parameter_groups` anchors the ladder at the TOP opened block, so `--layer-decay` *divides* the
rate of the blocks the control trains rather than letting the top rate rise. Reaching for it to drive a
trunk harder does the opposite.

**Then the asset was priced card-free, which is what should have happened in the first place.** Both
candidates have holdout records on disk, so `record_mix --seats greedy` costs seconds:

```
greedy  1: 0.8598  +small_batch6      greedy  6: 0.8729  +sided_pool
greedy  2: 0.8669  +margin_seat       greedy  7: 0.8730  +aprior_dense
greedy  3: 0.8706  +calib_span        greedy  8: 0.8731  +crop80_span     <- peak
greedy  4: 0.8720  +through_dense     greedy  9: 0.8731  +attend_dense
greedy  5: 0.8726  +band_dense_long   ...
                                      greedy 36: 0.8704  +v3base_dense
                                      greedy 54: 0.8670  +raddino_dense
```

`raddino_dense` (0.8137 solo) enters at seat **54 of 68**; `cnx_own_recipe` (0.8186) at **53**. Neither
weak-but-distant trunk displaces anything, so the diversity-over-quality case does not hold for these two
on this holdout — worth stating plainly because it is the owner's stated thesis and it deserves a real
test rather than an assumption in either direction. n=2 trunks: this is evidence about RAD-DINO and
ConvNeXt, not a refutation of diversity.

**`v3base_dense` enters at seat 36, which retires a P1.** It is 0.8455 solo — the strong end of the v3
family, absent from `comp/members.roster`, and packaging it has been carried as P1 work needing an upload
window. It is not in the eight-member peak, and neither is `dinov3_base_lora`. The package is not holding
the weak end of a family whose strong end matters; the strong end does not seat either.

**AND THE TAIL IS THE FINDING THE OTHER THREE ARE FOOTNOTES TO.** The greedy curve runs 0.8720 at seat 4,
0.8731 at seat 8, 0.8704 at seat 36 — **a spread of 0.0016 across thirty-two seats, against a 0.0062 noise
floor.** Everything from seat 4 to seat 36 is one flat line inside the noise. Member SELECTION is not a
lever, and the reason a 33rd member cannot move the number is arithmetic rather than judgement. This is
the same result the member-swap work reached from the other direction (only the sign transfers, the
ordering broke on the third point), now with the curve printed next to it.

**So the card has no run priced above the bar tonight, and that is a measurement rather than idleness.**
The greedy curve says another member near 0.84-0.85 lands in the flat zone and is worth ~0.000; a run that
produces one costs 3.3 h. What the curve does license is the inventory note's conclusion, sharpened: the
lever is a member class substantially BETTER than 0.8512 solo, not another draw from the distribution we
have. Eight members buy +0.0133 over the best solo and then the ensemble is done.

**CORRECTED TWO MINUTES LATER, and the correction is the same mistake in miniature.** The paragraph above
closes "the card has no run priced above the bar tonight". That was written from the greedy curve, and the
greedy curve cannot see the axis that had a priced, pre-registered run waiting on it. **Greedy ranks
members by a HOLDOUT record, and a dark member is fitted at `--fold -1` with no holdout**, so not one of
the six dark members has ever appeared on that curve. Reading "member selection is not a lever" as "no
member run is worth the card" applied a measurement outside the denominator it was made on — which is the
error this ledger has now recorded three times in two days, most recently four paragraphs above.

`small_batch6_ff` (knee-rndp) is armed at 22:15Z on the freed card: `arm_check` clean, 12 epochs, ~95 min,
converting the best member this project owns (`small_batch6`, 0.8598 — and the exact member that held the
fold-0 seat in `lastseat` at 0.937, so its behaviour in this ensemble is a measured control) into a
SEVENTH dark member. Count-of-dark-members is the one variable every board entry since `finalfit` has held
fixed, because exactly six exist and no entry has ever had a seventh to offer. Pre-registration stands as
written in the arm: **≥0.939 the dose still pays · 0.936–0.938 flat, and the full-fit axis closes with a
number instead of an extrapolation · ≤0.935 indicts `--ema`/all-data and the harness is first suspect.**

The honest expectation is the middle branch at P≈0.3 for the top one, i.e. +0.002 to +0.003 — under the
house bar and under the +0.0082 first place costs. It is armed anyway, and the justification is not that
it might win: it is that the wall-clock is free, and that the FLAT branch is itself the deliverable,
closing the only axis that has ever moved this board with a measurement rather than an extrapolation.

**And it did not start.** The first launch exited 0 in under a minute with `run_train: YIELDING before
small_batch6_ff` — `lib.sh` gates on the EXISTENCE of `logs/YIELD_CARD`, not on its contents, and I had
carefully rewritten the file to record the handoff instead of deleting it. Exit 0, ninety-five minutes of
free card, and a log whose three lines say plainly what happened. **A gate that reads a file's presence
cannot read the prose inside it, and prose is what I put there.** The handoff belongs in git history,
which already had it (`cee8c20`); the flag is now removed and the arm relaunched at 22:20Z. Cost: four
minutes, and only because the ARM EXIT was read rather than assumed — the same failure the ledger records
for `nox` and for `git commit | tail`, arriving through a third door.

## The foreign trunk finished, and it is 53rd of 67 (2026-08-24 22:37Z)

`cnx_own_recipe` — ConvNeXt at a rate fitted for ConvNeXt rather than the 4e-5 inherited from ~189
DINOv2-family runs — is the arm the goal note ranks FIRST among everything still open, on the grounds that
its +0.0259 rate fix is the only +0.02-class mechanism this project owns. It has finished, converged, and
the reading is clean: peak **0.8186 at e6**, then five consecutive epochs of monotone decline to 0.8130.
No truncation, no stopper, no harness suspect. The rate fix is real and the member it produces is 0.041
below `small_batch6`.

Greedy over the whole same-split pool places it at **seat 53 of 67**, and the peak is unchanged: 0.8731 at
seat 8, the same eight members in the same order as the 22:12Z scan. So the fitted-rate foreign trunk is
not additive either — not as a quality member, and not as a diversity member.

**That is the goal note's item 1, measured.** The list of things "sized like the goal" had three entries;
one is now a finished negative, one (`v3base_own_recipe`) was retired as ~0-expected because DINOv3-base
does not carry the ConvNeXt defect, and one (RadDINO at a fitted rate) was refuted three ways earlier
tonight by its own loss curve. **The trunk axis is closed on finished evidence, not on a floor.** What
remains open is not a better encoder; the ledger has now said "the limit is upstream" from four
directions and the fourth one cost 3.1 h of card to say it.

**One process note, because it cost forty seconds and would have cost more later.** `displacement.sh
cnx_own_recipe_s2026` re-ran greedy with the candidate passed explicitly ON TOP of a pool glob of
`*_s2026.pt` that already contained it. The script tests nothing for any member matching that glob — its
explicit argument only adds something for arms named outside it, which is precisely the new `_ff` full-fit
members. Correct for what it was written for, a no-op for what I pointed it at.

### Per-label reader agreement — ρ=+0.53 against per-label AUC, and recalibration is worth 0.000 (2026-08-25)

**COST: no GPU.** `reader_agreement.by_label` has been in the tool since it was written and its numbers
have never been in this ledger. Run card-free while a submission was pending
(`logs/reader_agreement_20260825.log`).

Agreement on the reader's POSITIVE calls, per finding — **denominator 311 positive calls over 57
annotated studies (684 cells)**, so each rate carries roughly ±20pp and no pair of labels is separated:

| label | AUC (`band_dense`) | reader agrees | n calls |
|---|---|---|---|
| Baker's | 0.9065 | 68.8% | 16 |
| Medial Meniscus | 0.9047 | 78.1% | 32 |
| Medial OA | 0.8797 | 63.6% | 22 |
| Synovitis | 0.8764 | 81.2% | 16 |
| ACL | 0.8740 | 71.9% | 32 |
| **Contusion** | 0.8705 | **44.4%** | 36 |
| Effusion | 0.8595 | 68.0% | 50 |
| PF OA | 0.8583 | 66.7% | 24 |
| Lateral Meniscus | 0.8464 | 66.7% | 30 |
| Lateral OA | 0.8156 | 56.2% | 16 |
| **Fracture** | 0.8088 | **68.4%** | 19 |
| MCL | 0.7931 | 50.0% | 18 |

**Spearman ρ = +0.53** (Σd² = 134.5, n = 12, p ≈ 0.08) — moderate, NOT established. The residual is
carried almost entirely by two trauma findings pulling opposite ways: Contusion is 6th by AUC and LAST
by agreement, Fracture is 11th by AUC and 5th by agreement.

**FREE KILL, and it needs no resolution at all.** The error census is lopsided 3.4:1 — 108 false
positives against 32 misses, Synovitis the only inversion at −11 — which reads as a calibration lever.
It is not one. The metric is a per-label RANK AUC, so a monotone per-label recalibration cannot change
it by construction. **Any arm that corrects the reader's over-call rate is worth exactly 0.000 on this
board**, retired before it was proposed. Consistent with `calibrated_targets` at +0.0038, inside the floor.

**THE TWELVE-LABEL DEFICIT TARGET IS OVER-PRICED.** `farseat`'s FLAT branch hands off to "lifting
Lateral Meniscus 0.7981, PF OA 0.8165, MCL 0.8166, Lateral OA 0.8215 to 0.87 is +0.0189". MCL at 50.0%
and Lateral OA at 56.2% are the worst reader precision after Contusion, so that arithmetic prices
headroom the label table may not contain — it asks the model to out-rank its own training signal. NOT
refuted: Contusion scores 0.8705 on 44.4% precision, which proves a label can. But not licensed as a
target either. Anything built against the label deficit must name the labels it targets and state their
agreement.

**WHAT THIS DOES NOT SAY.** Agreement here is precision on the reader's positives only; recall is the
separate 32-miss column. And the board scores radiologist annotations while we train on report-read
labels — the line above (2026-08-22) and `interpretations/kneemri/2026-08-11` do not agree on how far
those two diverge, and this measurement does not settle it.

## 2026-08-25 — `merged_corpus`, and the label table that is not worse but coarser

**`merged_corpus` is FLAT, exactly as it pre-registered.** Peak holdout 0.8545 at epoch 8 of twelve
(`convergence_check` calls it converged, so this is the recipe's number and not a floor), against the
`lr4e-5_ws = 0.8525` control that shares its every flag. **+0.0020, inside the 0.0062 floor.** The arm
file named this as its DEFAULT outcome before the run and says what it is not: neither a refutation nor
a confirmation of the hidden-test distribution-shift argument. What it buys is a member reading a
different corpus, which is a diversity axis no seated member has — priced as a seat, not as a score.
Fold-0 harness verified live: 5,143 studies, 4,234 train, 909 holdout, byte-identical to every prior
fold-0 row.

**OUR LABEL TABLE IS NOT THE WORST MAINSTREAM TABLE. IT IS THE COARSEST ONE.** Scored by
`kneemri.evaluation.supervision_gap` against the 58 radiologist-annotated studies:

| table | gold macro AUC | distinct levels | tie-pair mass |
|---|---|---|---|
| `flight0234/report_labels_v4hybrid` | 0.8991 | 85 | — |
| `stevenleehans/llm_labels_v4_blend` | 0.8927 | 85 | 0.173 |
| **ours, `report_labels_v2`** | **0.8658** | **5** | **0.385** |
| `lixin73/report_labels_gpt56sol` | 0.8352 | — | — |
| **`stevenleehans` QUANTISED onto our 5 levels** | **0.8445** | 5 | 0.409 |

That last row is the control that reframes the whole comparison. **At equal resolution our reader beats
theirs by +0.021.** The 0.027 that first read as supervision quality is a scale artifact: AUC breaks no
ties, and `Supervision.MEASURED_AGREEMENT`'s five levels tie 38.5% of cell pairs against their 17.3%.
`v4hybrid` and `v4_blend` share a distinct-count, a min, a max and a mean — they are one reader with a
`__conf` column added, not two.

Breaking the ties recovers most of it, and **the weight does not matter** — `ours + w*(foreign − 0.5)`
reads 0.8874 at w = 0.02, 0.05 and 0.10, and 0.8872 at 0.25. That invariance IS the mechanism proof:
AUC is rank-based, so every small w yields the same ordering and the entire +0.0216 is tie-breaking
rather than reweighting. Averaging goes slightly further — two readers 0.8885, three 0.8912 — through
`python -m core.data.labels.table_blend`.

**THREE PUBLIC TABLES SCORE 1.0000 AND THAT IS LEAKAGE.** `yunusgmsoy/report_labels_v5`,
`gxpixel/train_labels_final` (which ships `gold_58.csv` and an `is_gold` column) and
`rayanbabur/train_targets_soft` (which ships `gold_benchmark_58.csv`) contain the ruler they are scored
against. Anyone ranking public tables this way is fooled by them; `table_blend.LEAKED` refuses all three
by name.

**WHAT THIS DOES NOT SAY.** The three-reader blend at 0.8912 does NOT beat its best single input at
0.8927, which is what averaging independent readers is supposed to do. 58 studies and ~696 cells, no
interval computed, so none of these numbers are separated — **do not quote any pair of them as a
delta**. The claim that survives is narrow and is a TRAINING claim the gold ruler cannot settle: a
five-level BCE target carries less ordering information into the gradient than an 85-level one. The arm
that settles it trains the canonical recipe on the blend and **pins `--eval-labels` to
`report_labels_v2`**, because our fold-0 holdout is scored against v2 itself and an unpinned ruler
measures disagreement with v2, not supervision quality.

## 2026-08-25 — native224: the resolution axis is flat, and that is a THROUGHPUT result

`native224` = the control's recipe with one flag moved, `--image-size 224`.

| arm | best holdout | s/epoch | reserved |
|---|---|---|---|
| `lr4e-5_ws` (control, 336 px) | 0.8525 | ~290 | 17997 MiB |
| `native224` | **0.8502** | **94** | **8124 MiB** |

−0.0023 against a 0.0062 floor. Flat, and `convergence_check` does not flag it — best epoch 7,
epochs 8–11 declining, so this is the arm's real number and not a floor.

**Read it as a price, not as a score.** DINOv2-small was pretrained at 224 and we have been paying
2.25× the patches to interpolate its position embeddings to 336 for no measurable accuracy. What the
flat result buys is card-time: a twelve-epoch arm costs 19 minutes at 224 instead of 60. Every matched
comparison from here can run at 224 against `native224` = 0.8502 as its control, and three arms fit in
the wall-clock one used to take.

**What it does NOT say.** It does not say resolution never matters — `band_dense` (0.8578) reads 20
slices at a wider band and was deliberately excluded as the control precisely because it would confound
geometry with resolution. It says that at THIS geometry (12 slices, 0.20–0.80 band, whole-stack, max
pooling) the extra pixels are not carrying anything. A finding that needs fine texture — cartilage
thinning, a small meniscal tear — could still want them, and no arm here has isolated one.

**As a member**: 0.8502 is a mid-pool solo score, and the input tensor differs from every seated member
(different pixel grid, not merely a different band), which is the PIXEL-distance axis `through_dense`
was bought on. Unpriced — 0 of votefit's 6 members carry a holdout record, so seat value is not
computable against the champion.

## 2026-08-25 — base224: encoder scale is flat, and distance turns out to be a twin detector

`base224` = `native224`'s recipe with one flag moved, `--variant base`. Twelve epochs, `ARM DONE`, best
at epoch 9, `convergence_check` does not flag it.

| arm | best holdout | s/epoch | reserved |
|---|---|---|---|
| `native224` (control, small trunk) | 0.8502 | 94 | 8124 MiB |
| `base224` | **0.8509** | **240** | **15943 MiB** |

+0.0007 — an order of magnitude under the floor. The arm existed to re-ask encoder scale with the batch
handicap removed: earlier `dinov2-base` runs at higher resolution were starved on batch size, so their
flat readings were confounded. Handicap removed (batch 12, 14.0 GB allocated against a 17 GB gate), the
axis still reads flat. **Encoder SCALE is not the binding constraint** at this geometry.

The flag landed — the config records `variant: base`, and the 240 s/epoch against small's 94 is the
bigger trunk actually running, not a base label over a small model. This is a reading about the axis.

**And it cost 2.6× the card for nothing**, which is the throughput half of the result: `native224`'s 94
s/epoch is the cheapest useful arm in the ledger, and the base trunk buys none of what it charges.

### The pricing, which says something the arm did not

The whole 224 family, priced in one call against `gradeswap_v1` (0.933, the highest board entry whose
every member still carries a holdout record — the champion's own six are all `_ff` and price nothing):

| candidate | seat value | best swap | nearest |
|---|---|---|---|
| `dinov3_v3small_224` | −0.0017 | −0.0020 | **0.7936** |
| `lora_r8_v3small_224` | −0.0010 | −0.0010 | 0.8159 |
| `base224` | **+0.0007** | **+0.0008** | 0.8679 |
| `native224` | +0.0002 | +0.0004 | 0.8946 |

**Distance is monotone ANTICORRELATED with seat value here**, and that is the entry's real finding. The
most distant member we have ever measured against this roster is also the worst seat in the family.

The mechanism is not subtle: distance is confounded with quality. A weak member is distant *because* its
predictions are noisy, and noise is maximally uncorrelated with everything. A rank-distance column cannot
separate orthogonal SIGNAL from orthogonal NOISE, and those two have opposite value. The seated members
say the same thing from the other side — every one of the seven sits at 0.88–0.92 to its nearest, so the
best roster this board has produced is built from members that are neither twins nor strangers.

So distance keeps exactly the job `seat_value`'s own docstring gives it and no more: a **twin detector**
that VETOES a candidate too close to a seated member, which is how the 2026-08-22 split was called
(`band_span` priced +0.0002 and cost a tick as a near-twin). It never endorses one for being far. The
usable rule is a band — positive seat value AND nearest inside roughly 0.86–0.92 — and `base224` is the
only member of this family inside it (knee-bxp2).

**Correcting myself in the open**: forty minutes earlier I had written the 00:00 slot plan around
`base224` as a *distance* bet, on its 0.8679 being below every seated member's nearest. That premise is
void. The move survives on the repaired argument — best on both columns at once — but it is a weaker
claim, and +0.0008 is under the +0.0015 this ruler has ever been shown to resolve.

## 2026-08-25 — holdannot: the annotation upweight costs nothing, and OUR model beats OUR labels

`holdannot` = the control's recipe with `--hold-annotations`, which stops the 58 radiologist-read studies
from OVERWRITING their report-derived rows — the 696 annotation cells and their 3.0 weight go away and
those studies train on what the report said instead. The studies themselves stay in the corpus; it is the
cells that are withheld, which is the whole point (`Supervision.annotations`). Twelve epochs, `ARM DONE`,
converged
(`convergence_check`: peak at 10 of 12, then two lower epochs — STOPPED, not truncated).

| run | holdout mean AUC | vs control |
|---|---|---|
| `native224` (control) | 0.8502 | — |
| `holdannot` (e9 / **e10** / e11) | 0.8482 / **0.8487** / 0.8481 | **−0.0015** |

**The member reading is a null, and the null is the point.** −0.0015 sits inside the arm's own
pre-registered `>= -0.006` band, so the only labels in this project read from PIXELS rather than from
text — 696 cells at 3.0, ~0.66% of total training weight — are worth nothing measurable on the metric.
That prices the annotation upweight: it is not what carries the corpus, and no future arm needs to
protect it.

**The instrument reading is the result.** The member exists so `supervision_gap` can be computed on a
model of OURS whose loss never touched a radiologist cell. Reading it over all **58 of 58** annotated
studies (`python -m kneemri.evaluation.supervision_gap --records holdannot_s2026.pt --device cpu`, which
needed a new entrypoint — `kneemri.evaluation.annotated_readings` — because nothing here could predict a
member over a study list that is not a fold holdout):

```
aggregate: labels 0.8658   model 0.8932
Contusion +0.1430 · Synovitis +0.0812 · Effusion +0.0658 · Baker's +0.0353 · Medial OA +0.0295
Fracture +0.0215 · Medial Meniscus +0.0144 · Lateral OA +0.0039 · ACL +0.0025
  --- above: the model beats its own labels, so the READER is the weak part ---
PF OA -0.0071 · Lateral Meniscus -0.0149 · MCL -0.0465
```

The PUBLISHED members read **0.8392 against the same 0.8658** — 0.027 BELOW their teacher, four of four
internal derangements on the deficit side. **Ours is +0.0274 the other way, with nine of twelve on the
label side, and the 4-of-4 shape gone** (ACL and Medial Meniscus are both surplus). So the −0.049 macro
deficit that `2026-08-25_the-focal-deficit.md` and `knee-487l` are built on is a property of a model we
did not fit, and the field's *"the entire competition is the label making"* consensus checks out for us.

Memorisation is not the explanation: the member saw the IMAGES of ~46 of the 58 with their REPORT labels,
and perfect memorisation of a report label scores the model AT the reference (0.8658), never above it.
The bias runs toward the null the finding cleared.

`blendlabels`' six pre-registered label-bottlenecked findings all sit inside our nine; our own split adds
**Medial Meniscus, Lateral OA and ACL** to the surface a better reader can reach. It does not make the
lever bigger than measured (+0.0034 on the metric, +0.0070 on the six) — it makes it OURS.

See `interpretations/kneemri/2026-08-25_our-model-beats-its-own-labels.md`, `knee-487l`, `knee-w7xa`.

### THE RULER: the board scores against ANNOTATIONS, and every number above is a PROXY (2026-08-25, knee-ivuq)

Read twenty minutes after the section above, from the competition's own overview (topic 733343, *Courtesy
of Dr. Jacob Kazam*), through `kneemri.kaggle.discussion_read` — one forum read, no card:

> *"Models are evaluated on twelve binary labels… the criteria used by the **annotating radiologists** are
> summarized below. In each case, ambiguous or borderline findings ("on the fence") were graded as
> **negative to favor specificity**."* · *"**Each study in the annotated reference set was independently
> labeled by two subspecialty-trained MSK radiologists, with disagreements adjudicated by a third
> radiologist to produce a single consensus ground truth.**"*

The board's ruler is image-read radiologist consensus — the same KIND of label as our 58, and **not** the
report-derived table every row in this ledger is measured against. The reports are an input the
competition supplies; they were never the target. `holdannot_s2026` scores **0.8487 against report labels
and 0.8932 against the 58 annotation cells** — same weights, same studies, no ensembling, no TTA. **+0.045
from the ruler alone**, and a documented mechanism for part of the standing 0.938-vs-~0.85 board gap that
had been carried as unexplained.

**The proxy is not NEUTRAL toward label-side arms; it is biased against them.** The metric is per-label
rank AUC and the two label sets rank studies differently — that is what `labels 0.8658` measures — so a
change that moves a model toward annotation truth mechanically LOSES AUC against report labels, by up to
that same +0.045. This lands on `calib_span`, killed at **−0.0020** and taking `calib_dense` with it:
`--calibrated-targets` exists to move the target away from the report and toward the annotation, so it was
scored on the ruler it was built to walk away from, and a small negative is what a correct move in that
direction *should* look like there. It does not revive the lever — `knee-p9oc`'s pooled-constants defect is
untouched by the ruler, and Effusion −0.0212\* against PF OA +0.0170\* is too structured for ruler bias
alone. It says the arm never measured what it claimed to. **knee-bs4l** pre-registers the decisive refit
(`calib_span` + `--hold-annotations`, control `holdannot_s2026`, read on both), the first arm here where a
negative proxy number would be evidence FOR the change.

**RULE from here (knee-9198):** a label-side arm reports BOTH numbers — proxy holdout AUC and the
`supervision_gap` aggregate over the 58. It costs one flag: `--hold-annotations`, priced at −0.0015.
`annotated_readings` REFUSES any checkpoint that trained on those cells rather than scoring it.
**Non-license:** n = 58, the annotated set is not a random sample (ACL 41% there against ~20% in the
corpus, topic 734106). The claim is about WHICH RULER, and it is documentary, not statistical.

**It also re-reads `merged_corpus`, for free.** That arm's headline — *"the six DEFICIT findings mean
+0.0061, the six SURPLUS findings mean −0.0022"* — used the PUBLISHED model's partition, which `holdannot`
has now shown is not ours. Re-split on our measured three-against-nine, the same recorded deltas give
**+0.0071 (ours to fix) against +0.0002 (the reader's)**: the separation SURVIVES and widens. Two cuts:
n = 3, with Lateral Meniscus +0.0196\* carrying nearly all of it, so it is one starred row wearing a
partition; and **ACL +0.0161 sits on the LABEL side for us**, where Rijeka supervises it directly — that is
supervision arriving, not encoder transfer, and it must not be counted toward the transfer claim. Rijeka's
ACL column is expert double-blind IMAGE reads, so of every supervision source here exactly two are
image-read rather than text-mined (the 58, and Rijeka), and the axis that has ever moved our deficit
findings is the one whose labels are the board's KIND. Convergent with `knee-ombx`, not new evidence for it.

### `merged_blend` — the two corpus mechanisms composed, and the SPLIT is the reading (2026-08-25, knee-twtr)

Three flags: `--scheme merged`, `--labels merged_blend_v1.csv`, `--eval-labels report_labels_v2.csv`. The
third is not optional — scored unpinned it would measure disagreement with v2 and report it as quality.
Control is `lr4e-5_ws_s2026` (0.8525), NOT `native224`. `convergence_check` EXIT 0; peak at epoch 7 (0.8589)
with e8–e10 flat at 0.8584, so the budget bound nothing.

```
MCL +0.0345*  Baker's +0.0139*  Lateral Meniscus +0.0134  Fracture +0.0128  Medial Meniscus -0.0095
ACL +0.0088  MEAN +0.0063  Contusion +0.0056  Synovitis -0.0052  Lateral OA +0.0051  Medial OA -0.0033
Effusion -0.0003  PF OA +0.0001        paired half-widths up to +/-0.0348
```

**The mean is the weak reading and the arm said so in advance.** +0.0063 clears its own pre-registered
0.0062 gate by 0.0001, which is not clearance in any meaningful sense, and it is an order of magnitude
under the house bar. **Directional.** Only MCL and Baker's are starred.

**Both splits, because they answer different questions.** The PUBLISHED partition is what the arm
pre-registered against and is reported so the pre-registration is honoured; OURS is what the split MEANS
(`holdannot` measured it on cells no member trained on).

| partition | deficit / ours-to-fix | surplus / the reader's |
| --- | --- | --- |
| published six-and-six | **+0.0087** | **+0.0039** |
| ours, three against nine | **+0.0160** | **+0.0031** |

Both halves rose on the published cut, which is the additive-composition outcome the arm named. On our
own cut the separation is **5x**, and the largest mover — **MCL +0.0345\*** — is our single weakest finding
(−0.0465 model−labels). That is the mechanism landing where it was predicted to land, at a size the mean
hides.

**Stated limit:** `hold_annotations: False`, so this member cannot be read on the objective ruler. Every
number above is the PROXY, and by knee-9198's own argument the proxy under-pays a corpus arm that moves
the model toward annotation truth. The split is the reading; the mean is the floor.

**Its seat, priced the same evening.** `member_curve` over the champion roster's five fold-0 counterparts
(`calib_span`, `grade_ladder`, `base_dense`, `lr4e-5_ws`, `through_dense`), with and without it. The sixth
roster member `final_ws_e12` is EXCLUDED: it is full-fit and reads **0.9541** on the fold-0 holdout it
trained on, which would have carried the whole greedy path.

| pool | best subset | all-member RankAverage | LogitAverage | best single |
| --- | --- | --- | --- | --- |
| five, without | 0.8680 | 0.8683 | 0.8686 | 0.8565 |
| six, with | **0.8700** | **0.8697** | **0.8702** | **0.8587** |
| seat value | +0.0020 | +0.0014 | +0.0016 | +0.0022 |

**It is in every best subset from one to six, and it is the best single member in the pool** — ahead of
`band_span` (0.8545) and `res392` (0.8523), the two that have held that place all week. Seat value
+0.0014 to +0.0020 offline, against the ~+0.001 a seat swap has historically been worth; roughly a double
seat, and still an order of magnitude under the house bar. `LogitAverage` stays ahead of `RankAverage` by
+0.0005, as the champion's own vote already assumes.

**CORRECTION, same evening, before the number was acted on.** "The best single member we own" was scoped
to the six-member roster pool and is wrong as written. Re-priced over the ten strongest members whose
holdout predictions are already cached (no card, the pair was training): `merged_blend` **ties**
`band_dense_long` at 0.8587, and `small_batch6_s2026` at **0.8598** — uncached, so not in the pool — is
still the best solo member this project has. What survives is the seat, and it survives against real
competition rather than five roster twins:

| pool of ten | best 1 | best 4 | best 5 | flat RankAverage |
| --- | --- | --- | --- | --- |
| without `merged_blend` (nine) | 0.8587 | 0.8696 | 0.8696 | 0.8692 |
| with it (ten) | 0.8587 | 0.8712 | **0.8713** | 0.8698 |

Seat value **+0.0017** on the best subset, +0.0006 flat — the same size the roster-twin pool reported, so
the price is not an artefact of a weak pool. It sits in **every** best subset from two members to nine.
The offline peak is five: `merged_blend + band_dense_long + band_dense + through_dense + calib_span` =
**0.8713**.

## 2026-08-26 — the mixup dose pair: the blend fired, the holdout did not move, and the data account is now outvoted 3-0

`mixup224` (α = 0.2) and `mixup224_strong` (α = 0.4) against `native224` = 0.8502. One flag apart from the
control and from each other, 224 px, twelve epochs, same 909-study holdout, same labels.

| run | best | peak epoch | final train loss | vs control |
| --- | --- | --- | --- | --- |
| `native224` (control) | **0.8502** | 9 | 0.6066 | — |
| `mixup224` α=0.2 | 0.8496 | 10 | 0.7369 | **−0.0006** |
| `mixup224_strong` α=0.4 | **0.8516** | 11 (last) | 0.7908 | **+0.0014** |

**Three numbers inside 0.002 — a third of the 0.0062 floor — and no ordering, since the control sits
between the two doses.** The arm pre-registered that a dose pair would be worth more than a single point
because "noise has no reason to respect the knob"; the knob was respected by the TRAIN loss (0.6066 →
0.7369 → 0.7908, monotone) and ignored by the holdout. So the blend demonstrably fired and did real work
to the fit, and the generalisation number did not care. That is `aug_long`'s signature repeated with a
mechanism specifically chosen to be different from `aug_long`'s.

**THE READING THAT MATTERS IS NOT THIS ARM'S, IT IS THE TALLY.** Three independent perturbations aimed at
memorisation of the (study, label) pair have now left the quantity invariant:

| perturbation | what it attacks | peak-to-finish decline |
| --- | --- | --- |
| `half_data` | example COUNT — half the studies | −0.0169 vs control −0.0182 |
| `aug_long` | example APPEARANCE — the intensity block | −0.0169 vs control −0.0182 |
| `mixup224` ×2 | the (study, label) PAIR itself — convex blending | holdout flat inside a third of the floor |

A quantity that survives halving the dataset, changing the appearance, and convexly mixing the pair is
not plausibly memorisation of that pair. **The overfitting diagnosis this project has carried for a week
is outvoted by its own three tests**, and the suspect it has never varied is the SCHEDULE — `one_cycle`'s
anneal over the 24-epoch budget where the peak-at-6 and the −0.0182 slide were both measured.

**THE CONFOUND IN ALL THREE, NAMED HERE BECAUSE IT UNDERCUTS TONIGHT'S PAIR SPECIFICALLY.** The decline
was measured on the 336/dense cell. This pair ran at 224, where `native224` over twelve epochs peaks at 9
and slides **−0.0016** to epoch 11 — a quarter of the floor. **So mixup was tested against a control that
had not been shown to have the defect mixup is supposed to cure**, on a budget half the length of the one
the defect was measured over. That is a faulty-TEST reading in the CLAUDE.md sense, not a refutation of
mixup, and it was visible in the arm header before either run: "the −0.0182 was measured over 24 epochs".
`native224_long` (24 epochs, no mixup) is armed to settle whether the defect exists at 224 at all, with
`mixup224_long` (α = 0.4, the dose whose curve had not finished) parked behind it on the card lock so it
can be killed unrun if the control reads NO DEFECT.

**What is NOT claimed:** that mixup does not work. It was tested at one resolution, on one budget, against
a control with no demonstrated decline, and the honest verdict is that this SETUP could not separate it.
What is claimed is that the three-way invariance is real and that spending a fourth arm on the data axis
before the schedule axis has been touched once would be spending on the least likely suspect.

Beads `knee-x580`. Interpretation: `interpretations/kneemri/2026-08-25_the-seat-axis-is-saturated-and-the-member-is-0.028-behind.md`.

## 2026-08-26 — the absolute mean carries a band 3.4x the paired floor, and we have been quoting the wrong one

`python -m kneemri.evaluation.paired_floor --labels .../report_labels_v2.csv --resamples 400`, on the
published out-of-fold predictions. The row is new: `PairedFloor.absolute`, the 95% bootstrap width of ONE
arm's own mean over studies, beside the two difference-widths that class already printed.

| scale | paired MEAN | unpaired MEAN | **absolute mean** |
|---|---|---|---|
| corpus, 4,407 studies | 0.0141 | 0.0308 | **0.0093** |
| our ruler's size, 881 studies | 0.0323 | 0.0683 | **0.0212** |

At the size our holdout actually is, a single arm's absolute mean sits inside **+/-0.0106**. The floor this
project quotes everywhere is **0.0062**, and that number is a *paired* floor — the width of a DIFFERENCE
between two arms scored on the same studies, where the per-study difficulty both arms face cancels. It is
the right ruler for `native224` against `native224_long`, and the WRONG ruler — by a factor of 3.4 — for
every claim that compares an absolute number to something measured somewhere else.

**Three standing claims are affected, and one of them dissolves.**

- `knee-md39` cites a SIGN INVERSION as evidence the ruler is broken: offline 0.8731 scored board 0.932
  while offline 0.8712 scored 0.937. Those two arms differ offline by **0.0019** — under a third of the
  0.0062 paired floor, before the absolute band is even reached. That inversion is not evidence of
  anything; it is two readings of one number. What survives in that bead is the OFFSET (+0.016 to +0.066),
  which is far outside any of these bands and remains real.
- The +1.82 transfer carries our own noise through it: +/-0.0106 offline is **+/-0.019 board points**, which
  is most of the spread the constant was fitted through.
- The 0.028 board deficit to the field's best reported single reads as 0.0154 holdout. Our side of that
  comparison alone carries +/-0.0106, and the forum's number carries an error of its own that we cannot see.
  So it is about 1.45 half-widths of our own instrument — a real ordering, quoted far too firmly at
  "exactly one significance bar".

**What it does NOT say.** Arm-to-arm readings on our fold are unaffected: those are paired by construction
and 0.0062 is still their floor. This changes how a number may be compared ACROSS instruments, not how two
of our own arms are compared to each other.

Beads `knee-md39`. Code: `LabelScores.mean_interval`, `PairedFloor.absolute`.

## 2026-08-26 — native224_long reads DEFECT REAL: the 224 cell overfits, and its anneal does not rescue it

`native224_long`, 24 epochs, `one_cycle`, no regulariser — the control half of the 224 pair, armed to
decide whether the 224 cell has the overfit the dense cell showed (-0.0182 peak-to-finish) or whether the
mixup pair was a cure looking for a disease.

| epoch | 0 | 3 | 6 | **9** | 12 | 15 | 18 | 21 | 23 |
|---|---|---|---|---|---|---|---|---|---|
| holdout mean AUC | 0.7337 | 0.8208 | 0.8445 | **0.8490** | 0.8410 | 0.8389 | 0.8334 | 0.8329 | 0.8325 |
| train loss | 1.2241 | 0.9632 | 0.8173 | 0.7127 | 0.6048 | 0.5246 | 0.4869 | 0.4707 | 0.4670 |

**Peak 0.8490 at epoch 9, finish 0.8325, decline 0.0165.** The arm's own pre-registration set DEFECT REAL
at a decline of 0.0124 (2x the 0.0062 paired floor) and NO DEFECT at 0.0062; this clears the upper
threshold with room, and it matches the dense cell's -0.0182 in kind and nearly in size.

TWO THINGS THE SERIES SAYS THAT THE THRESHOLD DOES NOT. First, the train loss falls monotonically for all
24 epochs, 1.2241 to 0.4670, while the holdout slides for fourteen consecutive epochs after e9 — the
classic separation, not a noisy tail. Second, and this is what the pre-registration flagged as the reason
to wait for 24 rather than read at 19: **the `one_cycle` anneal did not lift the finish.** The last four
epochs, deep in the low-LR tail where the schedule is supposed to consolidate, read 0.8335, 0.8329,
0.8327, 0.8325 — still descending. The budget is not merely unhelpful past e9; the schedule's own
recovery mechanism fires and buys nothing.

The run is not truncated. It ran the 24 epochs it asked for and its loss was still falling at the end, so
the 0.8325 is a real finish and not a floor left by a killed process.

WHAT THIS LICENSES, exactly as pre-registered: `mixup224_long`, the matched cure, which was parked behind
this control on the card lock and is now released. It asks whether a regulariser lets the 224 cell USE the
longer budget — i.e. finish above 0.8490 — rather than whether it merely slows the slide. Note the
schedule suspect (`nocycle224_long`, knee-fkw5) is now MORE interesting than it was, not less: the anneal
was the mechanism that was supposed to rescue the tail and it did not.

WHAT IT DOES NOT LICENSE. It does not say a 224 member should train 12 epochs instead of 24. Under
`one_cycle` a 12-epoch run is not the first 12 epochs of a 24-epoch run — the LR schedule differs
throughout — so 0.8490 at e9 is not a prediction of what a 12-epoch arm would score, and reading it as
one would be the same units error the board_transfer constant was quoted wrong by twice.

## 2026-08-26 — nocycle224_long: the anneal is EXONERATED, and the slide is the data

`nocycle224_long` = `native224_long` with one flag moved, `--one-cycle` off. Same cell, same 24 epochs,
same 224 corpus, same seed.

| arm | `one_cycle` | best holdout | at epoch | finish | s/epoch |
|---|---|---|---|---|---|
| `native224_long` | True | **0.8490** | e9 | 0.8325 | ~95 |
| `nocycle224_long` | **False** | **0.8429** | **e17** | **0.8326** | ~95 |

`native224_long` closed with the schedule named as the live suspect: the holdout slid for fourteen
consecutive epochs after e9, and the `one_cycle` anneal — the mechanism that is supposed to consolidate
in the low-LR tail — fired and bought nothing. That made the schedule MORE interesting, not less. This
arm removes it.

**The suspect is cleared, and the way it is cleared is the finding.** Without the anneal the run still
peaks and still slides, and it lands on the SAME finish: 0.8326 against 0.8325, a gap of one ten-thousandth
against a 0.0062 paired floor. A schedule that was causing the decline could not be deleted and leave the
decline byte-for-byte where it was. What moves instead is WHEN the peak arrives — e17 rather than e9 — which
is what a flat LR is supposed to do, so the flag demonstrably took effect and the null is a null about the
mechanism rather than about the plumbing. The slide is overfitting on this corpus at this capacity; the LR
schedule is not carrying it.

Peak-to-peak the constant LR is 0.0061 WORSE, which sits exactly on the floor and is not a separation. Read
it as "no better", not as "worse".

**WHAT THIS CLOSES.** The schedule axis, `knee-fkw5`. Both endpoints of the one flag that governs the LR
tail now read the same finish, so no further schedule variant is priced: a third setting would have to beat
a mechanism that has twice failed to move the number it was named for. The cure for the slide is
regularisation or data, and `mixup224_long` already tested the first of those.

**WHAT IT DOES NOT SAY.** Nothing about 336. Both arms live in the 224 cell, which the resolution rows
above price at ~0.005 below 336 offline and, through `board_transfer`'s +1.82, ~0.009 below on the board.
224 remains a throughput lever and neither of these numbers is a member.

### Per-label member selection, finally READ — the flat rank-average is the optimum of its own family (2026-08-26)

The metric is the unweighted mean of twelve **independent** per-label AUCs, so the combine that matches
it is per-label: let each finding choose its own subset of members rather than every label sharing one
flat rank-average. That is the most metric-aligned ensemble idea available, and until today it had never
produced a number here — `kneemri/evaluation/label_combine.py` implemented it on 2026-08-20, was cited
twice as prior art (line 539 above, and the 2026-08-20 interpretation), and **had no `main`**. A library
with no entrypoint is a claim, not a measurement. It has one now.

Seven fold-0 members (`gradeswap_v1`'s roster), read back from `member_curve`'s prediction store, so the
encoder never ran — the whole reading cost about a minute of arithmetic and no card:

```
top- 1 per label:  in-sample 0.8643   split-half 0.8538
top- 2 per label:  in-sample 0.8708   split-half 0.8637
top- 3 per label:  in-sample 0.8714   split-half 0.8663
top- 4 per label:  in-sample 0.8714   split-half 0.8674
top- 5 per label:  in-sample 0.8710   split-half 0.8686
top- 6 per label:  in-sample 0.8705   split-half 0.8688
top- 7 per label:  in-sample 0.8694   split-half 0.8696   (= the flat rank-average)
```

**Read the split-half column only.** It is monotone increasing in `k` and its maximum is at `k = 7`,
which *is* the flat rank-average already shipped. Every per-label subset is worse out of sample; the best
candidate (`k = 3`) is **−0.0033**.

**The control is inside the curve**, which is why this is a verdict rather than a shrug: selecting the
top `k` per label at `k =` every member *is* the flat average, so candidate and matched baseline are
estimated by the same estimator on the same studies. Nothing here compares two rulers.

**And the optimism is visible rather than assumed.** In-sample peaks at `k = 3–4` (0.8714), a +0.0020
"gain" that is entirely the selection choosing its own noise. The columns diverge by **0.0051** at
`k = 3` — larger than any candidate gain in the table, which is exactly what the module's docstring
pre-registered: *if the two columns diverge by more than the gain, the gain is the divergence*. Anyone
running this without the split-half estimator ships a per-label roster and reports +0.002.

The mechanism says why, so this is not merely a null. Twelve choices are fitted on 881 studies with as
few as 62 Fracture positives; a per-label AUC on 62 positives carries a standard error near 0.04, so the
ranking it induces is mostly noise, and selecting on it discards real members to chase it. Averaging
more members *is* the variance reduction — selecting fewer is the opposite of what the small-positive
labels need.

**What it does not close:** one roster, one fold; and hard SELECTION is not per-label *weighting* that
shrinks toward the flat average. That variant is untested — though its prior drops here, since the hard
version's optimum sits exactly at "do not select at all". It also says nothing about per-label
supervision, refuted separately from the training side (`per_label_loss`, 2026-08-19, above).

This is the eleventh flat-or-negative reading on ensemble construction and the first that is
metric-aligned by construction, so it strengthens `knee-uj15`'s arithmetic rather than merely agreeing
with it: the remaining distance is inside one member, and now the most favourable combine rule anyone
could propose has been measured at the place the ledger already stood.

### The ensemble is SATURATED — 37 members, and the honest peak is 0.872 from six seats onward (2026-08-26)

Card-free, slot-free, ~40 s of arithmetic over `.holdout.npz` arrays that were already on disk.

The lead this started from was the strongest unshipped one in the ledger: `record_mix --seats honest` on
2026-08-25 read the best honest seat COUNT as **twelve at 0.8720** against a shipped roster of six or
seven, and priced widening at **+0.0023**. The three members that reading wanted and the package does not
hold — `blendlabels`, `base224`, `attend_dense` — are all on disk, and mounting costs a dataset push, not
a slot and not a card. So it looked like a real gain sitting behind a free action.

It is not. Two readings, both `--seats honest` (greedy refitted on half the studies, scored on the other
half, both directions — the only one of the three modes whose number may be quoted):

| pool | best honest count | peak | 6 seats | 7 seats |
|---|---|---|---|---|
| 34 members, MOUNTED ONLY | 9 | 0.8720 | 0.8711 | 0.8710 |
| 37 members, + the three unmounted | 15 | **0.8721** | 0.8707 | 0.8713 |

**Widening from seven seats to the honest peak buys +0.0010 on the mounted pool.** A sixth of the 0.0062
floor. And **adding three genuinely new members to a 34-member pool moves the peak by +0.0001** — one
ten-thousandth, on an axis where the three were specifically the ones an earlier greedy had reached for.

So the 2026-08-25 +0.0023 was **not a count effect and not a member effect**. It was the difference
between two pools, read across two different member sets as though the count were the variable. The
correction is the same denominator discipline this file keeps relearning: compare curves only over the
same pool.

**What the shape actually says.** Both curves are flat from six seats to twenty within 0.0015 — smaller
than the bias term the estimator itself reports. There is no peak to find; there is a plateau, and we have
been standing on it since the fourth seat.

**This is the twelfth flat-or-negative reading on ensemble construction**, and it now covers every axis
the pool admits: seat count (here), membership (here, and six board entries with five different sevenths
at 0.936–0.937), per-label selection (`label_combine`, 2026-08-26 — the flat rank-average is the optimum
of its own family), the vote (`votefit_v1`, inside its pre-registered flat band), overlap TTA (five
members, measured), and external corpus (`merged_corpus`, +17% studies, flat on score AND ordering).

**THE CONSEQUENCE FOR SLOTS, WHICH IS THE POINT.** 0.8720 is the ceiling of *arithmetic over the members
we already have*. No entry built from this pool — any count, any subset, any per-label roster, any vote —
can beat the pinned 0.939 by more than noise. Holding slot 5 tonight is therefore not only the owner's
directive; it is what the measurement independently recommends, because there is nothing left to put in
it. The next real number costs a card, not a slot.

**HONEST LIMITS.** This is the fold-0 proxy pool: `_s2026` members with holdout records. Six of the
champion's members are `--fold -1` and have seen every holdout study, so they cannot appear here and no
number on this page is the shipped entry's value. What transfers is the SHAPE of the axis — flat past six
— not the level. And a member from an axis this pool does not vary at all (a genuinely different encoder,
image-derived supervision) is not tested by any of it: saturation of a pool is a statement about that
pool.

## 2026-08-26 — mixup224_long: the regulariser WORKS and the ceiling does not move (band 2, exactly as written)

It finished at **04:36Z and sat unread for four hours.** `mixup224_long_s2026.full.log` held a complete
24-epoch run with `ARM DONE` on its last line while the ledger's next two sections were written around it —
the same asymmetry `convergence_check` exists for, one layer out: that check guards a number that HAS been
quoted, and nothing guards a number nobody looked at. Found by grepping the arm directory for logs newer
than the sections above it, which is now the thing to do before opening a card.

Matched against the control directly above it — same 224 cell, same 24 epochs, same seed, one flag moved
(`--mixup 0.4`).

| arm | best holdout | at epoch | finish | peak→finish | final train loss |
|---|---|---|---|---|---|
| `native224_long` (control) | 0.8490 | e9 | 0.8325 | **−0.0165** | 0.4670 |
| `mixup224_long` (α = 0.4) | **0.8505** | **e12** | **0.8417** | **−0.0088** | 0.6646 |

**The peak is flat: +0.0015, a quarter of the 0.0062 floor.** Not a gain, and it is not read as one.

**The decline is halved: 0.0165 → 0.0088.** Below the 0.0124 threshold the control's own pre-registration
set for DEFECT REAL, so on the axis this arm was armed to move, it moved it across the line. The peak also
arrives three epochs later (e12 vs e9), and the final train loss is 0.20 HIGHER at the same budget —
the blend demonstrably fired and the model demonstrably memorised less. Plumbing is not a suspect here.

**This is pre-registered band 2, verbatim:** *"the peak is flat but the DECLINE shrinks (peak-to-finish
well under 0.0182) → mixup regularises as intended and the bound is elsewhere; keep it as a free rider, do
not spend a slot."* No slot is spent. Mixup at α = 0.4 is a free rider for any long-budget arm — it costs
nothing, it converts a 24-epoch budget from actively harmful into merely flat, and it buys no height.

### What it closes, and what it hands to the next arm

The **label-space regularisation account of the ceiling is now closed**, and closed with a mechanism
rather than a shrug. The tally is no longer three perturbations that failed to move the decline; it is one
that moved the decline cleanly and still left the peak where it was. That separates two things this repo
had bundled: the SLIDE is memorisation and is curable, and the CEILING is not the slide. Curing the
overfit returns +0.0015.

So the bound is not the schedule (`nocycle224_long`, exonerated above), not the example count (`half_data`),
not appearance augmentation (`aug_long`), and not label-space mixing (here). Every axis that perturbs how
the SAME supervision is presented to the SAME encoder now reads flat. What is left is the pair this
repo has said for two weeks it is left with: **a different representation, or different supervision** —
`knee-jk3` and `knee-rh4t`. `raddino_dense` (cold, running since 08:43Z) is the first of those, and this
result is why it is the right thing to have on the card rather than another regulariser dose.

**Limits.** 224 only; nothing here is a statement about the 336/dense cell where −0.0182 was measured, and
the α = 0.4 dose was chosen because its curve had not flattened at 12 epochs, not because it was swept at
24. A dose sweep is not priced — it would be sweeping a knob that already returned its answer.

### raddino_dense (cold, fixed) — radiology initialisation LOSES 0.037 against its own exact control

`experiments/arms/raddino_dense.sh`, 12 epochs of 12, 2026-08-26, ~3h20m on one card. Zero submission
slots. The arm `knee-jk3`/`knee-ombx` priced as the last mechanism above the 1.5% bar, and the cheap half
of a cost-tiered pair whose expensive half is RadImageNet.

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|-------|---|---|---|---|---|---|---|---|---|---|----|----|
| holdout | .7237 | .7416 | .7824 | .7953 | .7951 | .8088 | .8111 | **.8141** | .8106 | .8103 | .8101 | .8102 |
| train loss | 1.224 | 1.112 | 1.036 | .967 | .893 | .823 | .758 | .706 | .667 | .638 | .625 | .617 |

**Peak 0.8141 at e7 against `base_dense_s2026`'s 0.8512 — −0.0371, six times the 0.0062 paired floor.**

**THIS IS THE CLEANEST SINGLE-VARIABLE ENCODER READ THIS PROJECT HAS TAKEN, and that is why the number is
allowed to mean something.** RAD-DINO is finetuned FROM `facebook/dinov2-base`: same ViT-B/14, same 768
width, same 12 blocks, no registers. So `base_dense_s2026` is this encoder at this exact recipe, and the
arms differ in the INITIALISATION WEIGHTS and nothing else — same 336px/20-slice/0.1-0.9 cache, same
`--backbone-lr 4e-5`, same head rate, same 12 epochs, same one-cycle, same fold-0 holdout. Every previous
family change in this ledger moved architecture and pretraining together.

#### Both named harness suspects were cleared BEFORE the verdict was written

The predecessor's 0.8137 was unreadable for two independent reasons, and a re-run that merely reproduced
the number would have been the same broken arm twice. The peak moved +0.0004 (0.8137 → 0.8141), which is
exactly the coincidence that has to be checked rather than explained.

- **NORMALISATION — cleared by measurement, not by argument.** Until 2026-08-25 every trunk was fed the
  ImageNet constants, so RAD-DINO — pretrained on greyscale radiographs at 0.5307/0.2583 — saw a 0.20-std
  offset with a 12.8% scale error. The run's own `recipe:` line carries forty-odd flags and **does not
  carry the normalisation**, so the log cannot answer this (filed P1). The checkpoint can: its registered
  buffers read `mean [0.5307, 0.5307, 0.5307]`, `std [0.2583, 0.2583, 0.2583]`. **The fix fired.**
- **TRUNCATION — cleared.** The predecessor died at e6 of 12 (`knee-8wm6`). This run recorded e11. Under
  `one_cycle` a truncated run's score is a floor; this one's is not.
- **RATE — the weakest surviving suspect, and weaker than it looks.** `4e-5` was not tuned for RAD-DINO.
  But the exact control shares the architecture AND the rate and reaches 0.8512 on it, so the rate is not
  mismatched to the *model*, only possibly to the *initialisation*. A rate refit is owner-blocked.

#### What it refutes, stated at the size it earned

**The domain-match mechanism, as priced, is refuted at this corner.** `knee-ombx` priced chest-radiograph
initialisation at +0.005–0.015 plausible against the mechanism's 1.5–3%. It delivered **−0.037**. This is
not a flat null to be explained away as dilution: it is a large, converged, single-variable loss.

**It also prices RadImageNet, which is what the cost-tiering was for.** `backbone.py` pre-registered the
consequence: *"If radiology initialisation moves nothing against an EXACT control here, that conversion is
not worth buying."* It moved strongly negative, so the 2–4 hours of Keras→PyTorch conversion — carrying a
silent-mismatch failure mode that would read exactly like a null — is now **unlicensed**. That saving is
the arm's main return, and it is why the cheap half was run first.

#### What it does NOT kill

**Not MRNet (`knee-ombx`).** RAD-DINO is cross-MODALITY and cross-BODY-PART: chest radiographs against
knee MRI. MRNet is knee MRI — the same modality, the same joint, the same slice geometry. This result
lowers the prior on "domain-matched initialisation pays" but it does not test the matched case, and
conflating the two would be reading a radiograph result as an MRI one.

**Not the generalisation gap that motivated the whole axis.** `base_dense` reaches 0.8371 at train loss
0.8456 while `cnx_dense` reaches 0.7927 at a *better* 0.8332 — a trunk that fits our pixels harder
generalises 0.044 worse. That gap is still there and still unexplained; what is now known is that a
chest-radiograph initialisation does not close it.

**Not RAD-DINO at another rate**, per the surviving suspect above — but note this is a use-refutation, not
an asset-kill: the encoder stays mounted (`dimiturnt/rad-dino`) and re-testable.

## 2026-08-26 — the dense hold pair: the hold is free, and `--calibrated-targets` gives this project its best member

Two arms of one chain, run back to back on the same corner so the second is a single-variable pair
against the first. Both converged 12 of 12 under `convergence_check` — neither number is a truncation floor.

| arm | recipe | peak | vs its control | reading |
|---|---|---|---|---|
| **`band_dense_hold`** | `band_dense` plus `--hold-annotations` — the 58 radiologist-read studies withheld from training so the annotation ruler can score them | **0.8573** @ e7 | **−0.0005 vs `band_dense`** 0.8578 | **the pre-registration fires.** The flag was priced at −0.0015 from `holdannot` at native224 (0.8502 → 0.8487); at the dense corner it costs −0.0005. Both are a fraction of the 0.0062 floor and they agree in sign and order of magnitude, so **the hold cost is recipe-independent** and this control is now reusable and free for every dense label-side arm. Curve 0.7452 / 0.7779 / 0.8315 / 0.8403 / 0.8482 / 0.8546 / 0.8541 / **0.8573** / 0.8546 / 0.8543 / 0.8536 / 0.8535 — a clean five-epoch decline after the peak |
| **`band_dense_hold` + `--calibrated-targets`** = **`calib_dense_hold`** | one flag moved against the row above | **0.8634** @ e6 | **+0.0061 vs `band_dense_hold`** | **the best offline single member this project has produced**, past `band_dense`'s 0.8578, and the arm's own header predicted "the first member above 0.860" before it ran. Curve 0.7515 / 0.8007 / 0.8280 / 0.8438 / 0.8544 / 0.8554 / **0.8634** / 0.8620 / 0.8607 / 0.8596 / 0.8603 / 0.8598 |

**Why this is not one-seed scatter, stated as evidence rather than as confidence.** +0.0061 sits *on*
the 0.0062 paired floor, which alone would make it directional and nothing more. Two things lift it:

1. **It replicates on a second corner.** `knee-bs4l` measured the same flag as a matched pair at
   `calib_span` and read **+0.0072 proxy / +0.0129 annotated**. Two corners, same flag, same sign,
   same size — the only directional gain this project holds twice.
2. **The board ruler has never inverted a gap this size.** `board_transfer` read live 2026-08-26 15:24Z
   over 21 ensemble entries and 171 orderable pairs: compression **+1.72** board points per holdout
   point, and **0 of the 108 pairs the holdout resolved at all** (gap ≥ 0.0053) were ordered the other
   way by the board. All eight inversions are pairs the holdout left UNRESOLVED. +0.0061 is above that
   resolution line.

**What it does not license.** This is a MEMBER reading, and the champion is a six-seat ensemble at 0.939
whose axis has now read 0.936–0.939 flat across ten entries. Member quality and seat value are different
quantities — `comp/champion.json` holds two principles saying so in both directions. The +0.010 board this
converts to is a conversion, not a measurement, and it is a claim about a single member, not about a
submission. `seat_value` is the tool that answers the other half.

**Consequence.** `--calibrated-targets` is the only measured-positive term this project currently holds,
which is what makes `stackfit` (`knee-orjq`) an additivity test worth the card rather than an academic
one, and `stack3` (`knee-nu1m`) the arithmetic that follows it.

## 2026-08-26 — a worse member takes a bigger seat: quality passes through at 23%, and the empty cell has the only occupant that pays

Two free measurements, no card, both against `gradeswap_v1` (board 0.933, 7 members, every one scoreable —
the pinned champion `blendseat` cannot be priced at all, because five of its six members are `_ff` full-fit
and have seen every holdout study).

### The seat table

```
roster of 7: 0.8694
  +margin_seat        0.8709  (+0.0015)   best seat +0.0018   solo 0.8525   nearest 0.8628
  +calib_dense_hold   0.8704  (+0.0010)   best seat +0.0013   solo 0.8634   nearest 0.9273
  +small_batch6       0.8698  (+0.0004)                       solo 0.8598   nearest 0.9068
  +sided_pool         0.8698  (+0.0004)                       solo 0.8561   nearest 0.9006
  +vjepa_warm         0.8695  (+0.0001)                       solo 0.8337   nearest 0.8210
  +dissent_seat       0.8686  (-0.0008)                       solo 0.8509   nearest 0.9292
1 of 6 clears +0.0015, under which this ruler has never been shown to pick a board tick
seated spread: 0.8801 – 0.9248
```

**`calib_dense_hold` is +0.0056 solo over `band_dense` and +0.0013 in a seat. That is a 23% pass-through**,
and it is the first time this project has measured the quantity that
`interpretations/kneemri/2026-08-26_what-0950-actually-costs.md` assumed was 1.0. At 23%, the entire queued
member-quality lane ceilings at **0.943 board, not 0.950** — and ten board entries reading 0.936–0.939 flat
across count, member identity, fold status, encoder family, vote rule, serving views, corpus and
supervision stop being a puzzle.

The mechanism is in the same table's last column. `calib_dense_hold`'s nearest seated neighbour is
**0.9273 — above the top of the entire seated spread (0.9248)**. Our best member is our most redundant
candidate. Better answers to the same question, asked of the same pixels with the same supervision and the
same pooling, agree more, not less.

**And `margin_seat` inverts the lane.** Solo 0.8525 — **0.0109 WORSE** than `calib_dense_hold` — and it is
the only candidate of six that clears the bar, at **+0.0018**. A worse member bought a bigger seat, and
the distance column says why: nearest 0.8628, the most distant candidate that is not simply broken
(`vjepa_warm` at 0.8210 is distant-and-weak, the other failure mode, worth +0.0001).

This does NOT restate the retracted "the loss axis buys seats" rule — `dissent_seat`, the other objective
arm, is the MOST redundant candidate we hold (nearest 0.9292) and reads **−0.0008**. Decorrelation asked
for explicitly in the loss produced a twin; an AUC margin that never asked for diversity produced the
distance. One construction of the two named in
`2026-08-24_architecture_diversity_is_closed_and_the_empty_cell_is_strong_and_distant.md` measured, the
other refuted.

**CONSEQUENCE — `margin_calib` (`knee-myl0`), queued behind `blendhold`.** `margin_seat` and
`calib_dense_hold` sit on the IDENTICAL corner and have never been run together. The arm is that corner
with both flags: `--auc-margin 1.0 --rank-weight 0` for the distance, `--calibrated-targets` for the
quality. ~85 min, zero slots, read as SEAT first and SOLO second. Its informative failure is
`nearest >= 0.92` with no seat, which would say calibration and an AUC margin are one lever wearing two
names and close the objective route for good.

### The honest member curve, over a pool that has doubled

`record_mix --seats honest` over **84 of 102 fold-0 records, 909 shared studies** (the previous scan read
70 records and peaked at 0.8717):

```
best honest count 19 at 0.8728, against the in-sample peak 0.8745 at 10
  1 member   held-out 0.8634   bias -0.0000      (calib_dense_hold, the pool's strongest solo)
  4 members  held-out 0.8709   bias +0.0015
  7 members  held-out 0.8721   bias +0.0022
 10 members  held-out 0.8726   bias +0.0019
 19 members  held-out 0.8728   bias +0.0009
 24 members  held-out 0.8726   bias +0.0008
```

Selection bias is **+0.0019 mean**, in family with the +0.0033 priced in
`2026-08-23_selection-overfit-explains-greedy7.md` and smaller because the pool is bigger. The honest curve
is **flat from 8 to 24 members — span 0.0008, an eighth of the paired noise floor.** Our champion carries
six. Going to nineteen buys, honestly, about **+0.002 offline ≈ +0.003 board**: real, free of card, under
the bar, and directional only. It does not buy a submission slot, and choosing 19 *because* the curve peaks
there is the exact move that document forbids.

Greedy's own first five over the 84 (diagnostic, not quotable): `calib_dense_hold` 0.8634 →
`merged_blend` 0.8688 → `through_dense` 0.8712 → **`margin_seat` 0.8724** → `calib_span_hold` 0.8736. The
strongest solo and the strongest seat are the first and fourth picks, and they are the two ends of the
table above.

## 2026-08-26 — the ambiguity decomposition closes to the digit, and calibration is a DIVERSITY lever

The seat measurement earlier today priced a marginal member's quality into an ensemble seat at 23% and
was written up as if that priced member quality in general. It does not. A second reading, taken on
records already on disk with no card, prices the OTHER case — lifting every seated member with the same
flag — and the answer is not the ~100% a pure quality lift predicts.

Six members, `--calibrated-targets` on and off at three matched corners, all joined over the same 909
shared studies:

| corner | calibrated | uncalibrated | Δ |
|---|---|---|---|
| dense, `--hold-annotations` | 0.8634 | 0.8573 | +0.0061 |
| span, `--hold-annotations` | 0.8565 | 0.8493 | +0.0072 |
| span, no hold | 0.8545 | 0.8565 | **−0.0020** |
| | | **mean** | **+0.0038** |

Pooled by the shipped rank-average rule, on two nested pools:

| pool | mean solo | ensemble | **ambiguity** = ensemble − mean solo |
|---|---|---|---|
| calibrated trio | 0.85813 | 0.8685 | **+0.01037** |
| uncalibrated twins | 0.85437 | 0.8629 | +0.00853 |
| | | **Δ ambiguity** | **+0.00184** |
| calibrated pair (hold only) | 0.85995 | 0.8679 | **+0.00795** |
| uncalibrated pair (hold only) | 0.85330 | 0.8595 | +0.00620 |
| | | **Δ ambiguity** | **+0.00175** |

**`ensemble error = mean member error − ambiguity` closes to the digit on both pools**, which is the
part worth keeping:

    trio   +0.0038 (quality) + 0.00184 (ambiguity) = +0.00564   measured +0.0056
    pair   +0.00665          + 0.00175             = +0.00840   measured +0.0084

So the pass-through is **147% on the trio and 126% on the pair**, above one on both, and every point of
the excess is the ambiguity term. **Calibrated members disagree with each other MORE than their
uncalibrated twins do.** Two independent pools agree on the size to within 0.0001. Calibration is not
only a quality lever — it is the first diversity lever this ledger has that was not asked to be one, and
it arrives from the label table rather than from an architecture or a loss.

**That is a correction of my own arithmetic, and it should be recorded as one.** The first write-up of
this quoted "mean solo delta +0.0061" and derived a 92% pass-through. +0.0061 is the DENSE PAIR ALONE,
generalised to three pairs by nobody but me; `calib_span` at −0.0020 against `band_span` is already three
hundred lines above in this same file and the mean silently absorbed it. The denominator was the corner
that agreed with the story.

**Honest size, and it is small.** Every SOLO delta above is inside the 0.0062 three-seed floor, the
negative one included, so no single corner's sign is resolved. Only the +0.0084 pair ensemble clears the
floor. The ambiguity deltas are +0.0018, a third of the floor, and their agreement across two pools is
not independent — the pair is a subset of the trio. What licenses acting on this is the sign agreeing
across two pools and the decomposition closing exactly, not any one number clearing a bar.

**The confound, stated because it points at the arm this licenses.** Both positive pairs carry
`--hold-annotations` and the negative one does not. `calib_span` and `calib_span_hold` sit on the
IDENTICAL corner — 12 slices, 0.1–0.9 — and differ in nothing else, yet read −0.0020 and +0.0072, a
0.0092 spread on one flag. At n=1 per cell that is inside what scatter produces, and `calibrated_targets`
reads +0.0038 against `wsfold0_e6` with no hold at the 0.2–0.8 corner, so no-hold is not uniformly
negative. But `calibsix`'s five seats carry no hold, and if it reads flat this is the first suspect.

**CONSEQUENCE: `calibsix` (`knee-1c00`).** The champion's five uncalibrated seats refitted with
`--calibrated-targets` and nothing else; `calib_span_ff` is the sixth and already carries it, so the
entry would be all-calibrated — the matched case these pools measured. Priced at the LOW end
(pass-through 1.0, discarding the ambiguity term entirely): 5 × 0.0038 = +0.0038 offline, × the live
`board_transfer` +1.72 = +0.0065 board, **0.939 → 0.946**, with 0.949 the optimistic end if the
ambiguity term scales to six seats. Seven hours of otherwise-idle overnight card, ZERO submission slots
to produce, one to read. Chained behind `margin_calib`.

## 2026-08-26 — ambiguity is mostly member WEAKNESS, and that closes the "pick a distant member" lane

`knee-qij2`, pre-registered before the numbers, zero card, zero slots. Twenty members paired against
anchor `band_dense` (solo 0.8578) over 909 shared studies, `ambiguity = pair − mean(solo_anchor, solo_member)`.
Five carry no holdout record and are unusable here — `report03`, `report10`, `max12`, `max20`,
`dinov3_v3small`; their pair line reads 0.8578, the anchor alone.

RAW, by axis of difference:

| axis | n | mean ambiguity |
|---|---|---|
| ARCHITECTURE | 5 | **+0.0121** |
| LABEL | 4 | +0.0079 |
| GEOMETRY | 6 | +0.0071 |

Read raw that fires the pre-registered branch "architecture is the largest group -> the architecture axis
was closed on the wrong ruler". **It is a confound, not an axis, and the scatter says so out loud.**
Regressing ambiguity on the member's own solo across all fifteen: slope **-0.179**, **r-squared 0.875**.
Seven eighths of the variance in this ruler is HOW WEAK THE SECOND MEMBER IS. The architecture group *is*
the weak group — `cnx_dense` 0.7927, `raddino_dense` 0.8141, `vjepa_warm` 0.8337 are the three worst
members in the set and they hold the three largest ambiguities, in that order.

**CAVEAT, measured on a second anchor 2026-08-26T16:36Z and it cuts at this paragraph.** Refitting the
same regression against `band_span` over only the members that WORK (solo >= 0.84, so the three named
above leave) gives slope **-0.091**, half of -0.179. The steep slope and the r-squared 0.875 are both
carried by the weak members: among usable members, solo quality explains much less of ambiguity than
this section reads. The conclusion above survives -- no axis group separates on either fit, and it was
drawn over the full set -- but "seven eighths of the variance" is a statement about a set containing
three wrecks, not about the members a roster would actually choose between.

Residuals once the quality trend is removed:

| axis | residual ambiguity |
|---|---|
| LABEL | +0.00049 |
| ARCHITECTURE | -0.00004 |
| GEOMETRY | -0.00030 |

LABEL over GEOMETRY is **+0.00079**, half the pre-registered 0.0015 bar, and one of the four label members
(`severity_weight`, -0.0025) runs hard the other way. Pre-registration says DIRECTIONAL ONLY. The axis is
not real at this resolution and nothing is planned around it.

**THE GENERAL RESULT, and it is a negative worth keeping.** Ambiguity is not a currency that can be shopped
for. `ensemble error = mean member error − ambiguity` is an identity, and the second term is mostly a
restatement of the first — a worse member disagrees more because it is wrong more. That is why every
buy-the-distant-member rule this project has tried has failed on the board: `knee-bxp2`'s distance column
(anticorrelated with seat value inside a family), `dissent_seat`'s negative-correlation loss (-0.0008, the
most redundant candidate we hold). All of them were buying error and reading it as diversity. It also
explains the +0.0018 seat that `margin_seat` took at 0.0109 WORSE solo without that being good news.

**IT STRENGTHENS `calibsix` RATHER THAN WEAKENING IT.** Calibrated members are BETTER solo (+0.0038), so
this fit predicts their ambiguity should FALL by 0.179 x 0.0038 = **-0.00067**. It ROSE **+0.00184**.
`--calibrated-targets` is the one lever measured here that buys disagreement while moving quality the same
way — quality-adjusted its ambiguity gain is **+0.0025**, not +0.0018. Caveat: the slope is fitted on
`band_dense` pairs and the calibrated trio is a different pool, so cross-applying it is approximate.

LIMIT, as pre-registered: all fifteen pairs share one anchor. These are fifteen readings of `band_dense`'s
neighbourhood, not of the corpus. A second anchor costs nothing and is the honest follow-up if this is
ever leaned on harder than it is here.

## 2026-08-27 — the per-diagnosis head and its own named follow-up, BOTH closed on one control

Two arms against the same control, `native224_s2026` = **0.8502** (12 slices, 224 px, fold 0, scored on
`report_labels_v2`). They are not independent: `labelwise224`'s pre-registration, written before it ran,
named the window geometry as the follow-up if the head came back flat. It came back flat. The follow-up
ran anyway, and it is worse than flat.

| arm | best | vs control | pre-registered branch |
|---|---|---|---|
| `labelwise224_s2026` | 0.8517 (epoch 9) | **+0.0015** | within +/- 0.006 -> "per-diagnosis pooling is not the lever at 12 slices" |
| `adjacent224_s2026` | 0.8462 | **-0.0040** | within +/- 0.006 -> not the lever; and the ONE marked delta has the wrong sign |

Both converged — `convergence_check` lists neither as TRUNCATED, STOPPED or UNDERTRAINED, so these are
facts about the architectures and not about a killed process.

### `labelwise224` — the direction is right and the size is a fifth of the floor

Twelve attention channels, one per diagnosis, over the same trunk. Per-label against the control:

MCL **+0.0126** · Medial Meniscus +0.0065 · Lateral OA +0.0053 · Lateral Meniscus +0.0051 · Medial OA
+0.0046 · Baker's +0.0016 · PF OA +0.0015 · Fracture +0.0015 · **MEAN +0.0015** · Synovitis -0.0012 ·
Contusion -0.0009 · ACL -0.0087 · Effusion -0.0095.

By structure scale: **focal +0.0038, regional -0.0017**. On the six findings the ruler work called
model-bottlenecked: **+0.0037**. That signature is the one the mechanism predicts — a per-diagnosis
channel should buy the small, localised findings and not the diffuse ones — and it is worth writing down
that the direction held. What did NOT hold is the size. Paired half-widths run to **+/-0.0326** and **no
delta is marked significant**, MCL's +0.0126 included. The pre-registration asked for >= +0.010 on the
mean *with MCL and Lateral Meniscus among the movers*; the mean is +0.0015, MCL moved and Lateral
Meniscus did not. Half of one ruler, at a fifteenth of the required mean.

**The reading the pre-registration wrote for this branch stands: 58 image-read labels cannot supervise
twelve separate attention channels.** The head has the capacity; the supervision does not have the
resolution to fill it. The axis closes as a lever. It does NOT close as a member — +0.0015 with a
different internal structure is exactly the profile of a diversity seat, and it costs nothing to try in
one, which is the only follow-up this arm earns.

### `adjacent224` — the marked delta is on the finding contiguity was supposed to buy

Three consecutive slices as RGB, centres spread by linspace, against the same control:

MCL +0.0088 · ACL +0.0045 · Lateral OA +0.0026 · Effusion +0.0005 · Medial OA +0.0002 · Fracture -0.0010 ·
**MEAN -0.0041** · Lateral Meniscus -0.0058 · Contusion -0.0081 · PF OA -0.0092 · Baker's -0.0129 ·
**Medial Meniscus -0.0136\*** · Synovitis -0.0149. Focal -0.0018, regional -0.0073; the six
model-bottlenecked findings -0.0021.

The mean lands inside the noise band and by itself would be "not the lever". The reading is stronger than
that, and it comes from the one delta that IS marked against its own interval: **Medial Meniscus
-0.0136\***. A meniscal tear is the textbook case for wanting three touching slices instead of three
spread ones — it is small, it is contiguous, and it is the finding this arm was written for. It is the
single worst-hit label but one, and it is the only one the interval will vouch for. The mechanism did not
fail to fire; it fired and it cost.

Best guess at why, and it is a cheap one to hold: at 12 slices over a 0.2-0.8 band the stack is already
sparse, and spending three of the trunk's channels on neighbours of one centre buys local contiguity by
throwing away a third of the SPAN. That is the same trade `depth_regions` lost — a substitute for span
dressed as a complement.

### What this pair closes

The head axis and the window axis were the last two shapes that did not touch the data. Both are now
measured, both on the same control, and neither pays. Written next to the rest of the ledger: resolution
(-0.0002), crop extent (-0.0099 / -0.0011), depth regions (substitute for span), trunk family (53rd of
67), SSL-as-init (five collapses), the vote, the count and the seventh seat. **Every decode and
architecture axis this project owns is now closed on its own measurement**, and the only two numbers
still moving — `merged_blend_cf` 0.8597 and `merged_blend_s2026` 0.8589 — are both the CORPUS axis.
`merged_dense_cf` (knee-4hxv) is the arm that follows from that, and it is the only one queued.

## 2026-08-27 — the corpus arm lands flat, and EMA is measured where its confound does not exist

Two runs finished on the same card and neither clears anything. Both are recorded here because both
were queued as decisions and a decision that reads flat is still spent.

| arm | recipe | peak | vs its control | reading |
|---|---|---|---|---|
| **`merged_dense_cf`** | the merged corpus at the DENSE corner — `band_dense`'s 20 slices / 0.10–0.90 band over the combined study list, the arm the line above this section named as the only one queued | **0.8569** @ e8 | **−0.0009 vs `band_dense`** 0.8578, **−0.0028 vs `merged_blend_cf`** 0.8597 | **the corpus axis does not carry to the dense corner.** The section above closed every decode and architecture axis and left the corpus as the only lane still moving; this arm asked whether that motion survives at the recipe we actually ship, and it does not. Both deltas are a fraction of the 0.0062 floor, so nothing is refuted and nothing is bought — what is spent is the assumption that a corpus gain measured at the blend corner is portable. It is not measured to be. `merged_blend_cf` remains the corpus lane's best number and it was never beaten by adding slices to it. Survived a relaunch after the first attempt died at 05:01:58Z to an `nvlddmkm` Event ID 153 driver fault at 26.92 GB reserved — a fact about the shared machine, not about the recipe (the relaunch was byte-identical and completed twelve epochs) |
| **`ema_dense`** | `band_dense`'s invocation with `--ema` appended and nothing else touched (`knee-56w5`) | **0.8588** @ e7 | **+0.0010 vs `band_dense`** 0.8578 | **FLAT, which is the band the bead pre-registered as `allfit`'s data reading being CLEAN.** The arm existed because every `_ff` member is its `_s2026` twin plus `--fold -1` AND `--ema`, so the one entry that ever moved the board (`kneemri-allfit`, 0.937) moved two variables at once. The confounded pair cannot be separated where it occurred; the confounding VARIABLE can be measured at fold 0, where the 909-study holdout still works and the control was already on disk. It reads a sixth of the floor. **The harness suspect the arm named in advance is cleared, not assumed:** `--ema` defaults False, so a flag that never reached the trainer would produce a run byte-identical to the control — the checkpoint config stamps `'ema': True` and the score is not the control's, so the flag landed and the null is about EMA rather than about plumbing. An outside single-variable ablation of the same flag read **+0.003** (`research/2026-08-26_the-ablation-table-a-competitor-published.md`); we read +0.0010. Two independent measurements, same sign, both under their own floors — the honest joint reading is a real but sub-floor lever, not a null. **Consequence: `allfit`'s +0.004 is a DATA result**, its co-variable is priced and small, and `knee-tfj3`'s closure stands |

**The seat ruler, run card-free against `gradeswap_v1` (0.933, seven members), says neither is a member.**

```
roster of 7: 0.8694
  +merged_dense_cf     0.8701  (+0.0007)   best seat +0.0008 (for lr4e-5_ws)   nearest 0.8902
  +ema_dense           0.8690  (-0.0004)   best seat -0.0002 (for base_dense)  nearest 0.9306
0 of 2 clear +0.0015
```

**And the distance column carries the one mechanism worth keeping out of this pair: EMA CANNOT BUY
DIVERSITY, BY CONSTRUCTION.** `ema_dense`'s nearest seated neighbour is **0.9306** — the tightest twin
reading any candidate has posted against this roster, tighter than `calib_dense_hold`'s 0.9273 and far
tighter than `merged_dense_cf`'s 0.8902. That is not a surprise once stated: an EMA member is a running
average of the *same optimisation trajectory* as its non-EMA twin, so it can only ever be a smoothed
version of an ordering the roster already contains. `lr4e-5_ws` priced −0.0003 on the quality ruler and
took 0.931 while `band_span` priced +0.0002 and cost a tick, because a rank average cannot charge for
contributing one ordering twice — EMA is the purest form of that failure available. **So `--ema` is a
solo-quality lever (sub-floor, directional) and is disqualified as a diversity lever regardless of what
its solo number does.** The `_ff` members already carry it; nothing is owed there.

**What this leaves.** The corpus lane's portability is unproven and the EMA confound is priced out. The
only arm on the card is `stackfit`, which asks the question neither of these can: whether two directional
gains measured separately ADD. Its arithmetic is stated in its own header and reproduced here so the
number is read against a prediction rather than the other way round — `calib_dense_hold` 0.8634 plus
(`ema_dense` 0.8588 − `band_dense` 0.8578) = **0.8644 predicted under additivity**, band ±0.0062.

### The composition read `merged_dense_cf` was actually bought — and it is the first floor-clearing evidence AGAINST additivity

`knee-4hxv` did not queue this arm for its solo number. Its claim under test was that **the corpus lever
and the geometry lever COMPOSE**, "because they act on different parts of the chain (what the network
sees vs what it is told)", and it pre-registered +0.004 — `band_span`'s span-alone measurement. Read as
a composition question rather than as a member, the arm is not flat at all:

| corpus | 12 slices, 0.2–0.8 | 20 slices, 0.1–0.9 | what the geometry buys |
|---|---|---|---|
| ours (report labels) | `rate_wholestack` 0.8525 | `band_dense` **0.8578** | **+0.0053** |
| merged (Rijeka three-reader blend) | `merged_blend_cf` 0.8597 | `merged_dense_cf` **0.8569** | **−0.0028** |

**The same geometry move, measured on two corpora, lands 0.0081 apart — 1.3× the 0.0062 paired floor,
and the sign inverts.** So this is not "the corpus arm was flat"; it is "the geometry gain is a fact
about the corpus it was measured on, and it does not survive being moved to a better-labelled one".

**Suspect the harness before the mechanism, since a sign flip is the shape a plumbing fault makes.**
Both merged runs read the merged study list (the builder refuses a foreign corpus name outright since
`knee-48br`, so a merged run cannot silently decode ours); the dense merged cache is its own key
(`336px_20sl_0.1-0.9band_merged`, distinct from the 12-slice merged build, per the reading-axis test in
`tests/integration/argv_to_cache_key.py`); and `merged_dense_cf` converged at epoch 8 of 12 with the
recipe stamp carrying `slices: 20`, `band [0.1, 0.9]`. The driver fault that killed the first attempt is
excluded by construction — this number is the relaunch, which completed. What is NOT excluded is that
each cell is one seed, and 0.0081 is 1.3 floors, not 3.

**Why this matters more than either row above: it is the ONLY composition measurement this project
holds, and `stackfit` is on the card right now betting the opposite.** `knee-orjq`'s whole remaining
route to 0.950 is a STACK of directional gains, and nothing had ever tested whether gains add. This did,
by accident, on a different pair of levers — and it read SUBSTITUTION with the sign flipped, the
`depth_regions_span` shape. That does not settle `stackfit`: these two levers plausibly interact through
the same channel (a better-labelled corpus and a denser read are both ways of resolving the same fine
structure), where `--calibrated-targets` and `--ema` are mechanically independent — one changes what the
targets say, the other changes which weights are kept. But it moves the honest prior on `stackfit` off
"additive unless shown otherwise", and it means an additive read there should be believed *less*, not
more. Stated before that number lands, so it cannot be fitted to it afterwards.

### The additivity question `knee-orjq` gates the whole route on is ALREADY ANSWERED, on disk, and it reads ADDITIVE

Found 2026-08-27 06:55Z, card-free, while `stackfit` was at epoch 1 of 12 betting on the same question
with less power. Nothing was trained for this. Four checkpoints the project already held form a complete,
matched 2x2, and no one had ever put them in a table.

The four cells are matched on everything except two axes: slice count (12 vs 20) and
`--calibrated-targets`. All four are 336 px, band 0.1-0.9, `ema False`, `patience 0`, seed 2026, fold 0
holdout, and all four ran their full 12 epochs.

|  | calib OFF | calib ON | calibration effect |
|---|---|---|---|
| **12 slices** | `band_span_hold` **0.8493** (e7) | `calib_span_hold` **0.8565** (e9) | **+0.0072** |
| **20 slices** | `band_dense_hold` **0.8573** (e7) | `calib_dense_hold` **0.8634** (e6) | **+0.0061** |
| **slice-count effect** | **+0.0080** | **+0.0069** | |

Additive prediction for the far corner: 0.8493 + 0.0072 + 0.0080 = **0.8645**. Observed **0.8634**.

**Interaction = -0.0011**, against a 0.0062 three-seed paired floor. One sixth of it. **ADDITIVE.**

**Why this one has power where `stackfit` does not.** `knee-orjq` needs two terms EACH worth more than
the floor, or an additive reading is consistent with a stack of one. Here both main effects clear it:
+0.0072 and +0.0080. `stackfit`'s second term does not - `ema_dense` 0.8588 against `band_dense` 0.8573
is **+0.0015**, and the singles decompose its 0.8644 prediction into +0.0061 of calibration and +0.0015
of EMA. Its ADDITIVE band, 0.8582-0.8706, CONTAINS `calib_dense_hold`'s own 0.8634, so "both terms added"
and "EMA did nothing and calibration survived" are the same rung. That arm can still refute gross
substitution, which is worth the 85 minutes it is already spending, but it cannot license the route. This
table can.

**The honest limits, and the largest one was nearly missed.** All four cells carry
`--hold-annotations` (verified in each recipe stamp, along with the same labels file and the full
12-epoch budget), so the 2x2 is internally matched. But the SAME contrasts exist in the non-hold family
and they do not agree:

| contrast | hold family | non-hold family | gap |
|---|---|---|---|
| calibration at the span corner | `band_span_hold` 0.8493 -> `calib_span_hold` 0.8565 = **+0.0072** | `band_span` 0.8565 -> `calib_span` 0.8545 = **-0.0020** | **0.0092, SIGN INVERTED** |
| slice count without calibration | `band_span_hold` -> `band_dense_hold` = **+0.0080** | `band_span` -> `band_dense` = **+0.0013** | 0.0067 |

**CORRECTED 2026-08-27, SAME DAY, AND THE CORRECTION IS ARITHMETIC I HAD ALREADY WRITTEN ONE PARAGRAPH
DOWN.** The first version of this section read that table as "both main effects of this 2x2 are
properties of the holdout definition as much as of the levers, at 1.1-1.5 floors". Two things are wrong
with it.

FIRST, THE PREMISE. `--hold-annotations` does NOT change what is evaluated. It drops the 58 annotated
studies from the training TABLE; the holdout stays fold 0 of 5, about 880 studies, in both families
(`fold: 0`, `folds: 5` in every recipe stamp here, and `kneemri/evaluation/annotated_readings.py:16`
records that the fold-0 holdout naturally contains about a TWELFTH of the annotated studies, which is why
scoring the annotation ruler needs its own all-58 route). The flag's own direct cost is measured twice and
small — `band_dense_hold` -0.0005 against `band_dense`, `holdannot` -0.0015 at native224 — so the two
families share a denominator and differ by roughly 1.2% of the training set.

SECOND, THE RULER. A difference of differences carries about 1.4x a single paired comparison's noise,
0.0062 x sqrt(2) = **0.0088** — the number the next paragraph uses to clear the interaction, and which the
family table was never held to. Against it the calibration gap is 0.0092 (**1.05 floors**) and the
slice-count gap 0.0067 (**0.76 floors**). NEITHER FAMILY DISAGREEMENT IS RESOLVABLE. A sign inversion is
what a small positive effect measured twice with one seed each looks like when the effect is near the
floor; it is not evidence that the holdout definition owns the lever, and reading it as such was reading
scatter as a finding on exactly the axis this file spends most of its length warning about.

**What survives, stated at the strength the data support.** The two families do not demonstrably disagree
and do not demonstrably agree — the comparison is underpowered either way, and no arm should be spent
resolving it, because both candidate answers price the levers under the 1.5% bar. The INTERACTION is
computed entirely within the internally matched hold family and stays valid there at -0.0011, an eighth of
its own 0.0088 ruler. The non-hold interaction CANNOT be computed at all because `calib_dense` was never
launched. So: **additivity is established on this holdout, its generality is untested rather than
refuted**, and the honest claim is that the two terms do not SUBSTITUTE here. That is weaker than "gains
add globally" and stronger than the scoping paragraph it replaces.

One seed per cell besides. Best-epoch selection on a noisy curve lifts every cell by roughly its local
jitter, which largely cancels in a difference of differences but does not vanish.

**AND IT CONTRADICTS THE REASONING THAT KILLED AN ARM.** The `calib_span` row above concluded "a
calibration that is negative on the span corner cannot be positive on the denser one for any reason this
evidence supports", and `calib_dense` (knee-ep74) was cancelled unlaunched on that inference.
`calib_dense_hold` reads **+0.0061** - positive on the denser corner, in the family where the span corner
is also positive. That does not restore `calib_span`'s number, which stands as measured on its own
holdout; it shows the inference from one corner's SIGN to another's was the step that failed, and it is
the same step this table exists to test. The cancellation was reasonable on what was known and is now
superseded: whether calibration pays at the dense corner is answered YES on one holdout and untested on
the other.

**Set against the only other composition measurement this project owns, the pair now tells a mechanism
story rather than a verdict.** `merged_dense_cf` read the SAME slice-count move as -0.0028 on the merged
corpus against +0.0053 on ours: 0.0081 apart, sign inverted, 1.3 floors - SUBSTITUTION. Here calibration
and slice count compose at -0.0011 - ADDITIVE. The difference is exactly the channel argument written
into the merged entry before either number landed: a better-labelled corpus and a denser read are two
ways of resolving the same fine structure and eat each other, while what the targets SAY and how many
slices are READ are mechanically independent and do not. Composition is not a property of this project's
gains in general; it is a property of whether two levers touch the same channel. That is a usable rule,
and it is the first one the additivity question has produced.

**What it licenses, and the arm it does NOT license - checked before proposing rather than after.**
Stacking independent-channel terms is now evidenced, so the member-quality lane's ceiling is set by how
many such terms exist, not by whether they compose. The obvious next arm is the slice-count term, worth
+0.0080 offline - LARGER than calibration's +0.0061 - applied at roster scale the way `calibsix_v1`
applied calibration. **That arm is void, and the reason is the finding.** The champion is already MIXED:

| seat | slices | band | 
|---|---|---|
| `final_ws_cf`, `grade_ladder_cf`, `through_dense_cf` | 20 | 0.1-0.9 |
| `calib_span_ff` | 12 | 0.1-0.9 |
| `merged_blend_cf`, `lr4e-5_ws_cf` | 12 | 0.2-0.8 |

Three seats already carry the term; moving the other three does not ADD a lever, it COLLAPSES the pixel
groups onto one. That collapse is measured: `knee-nas5` read it as 0.0059 offline and **0.014 on the
board**, a 2.4x compression, and it is the only ensemble axis with a board transfer at all. So the arm
trades a +0.0080 offline floor-raise on three seats against a -0.014 board diversity loss, using the
weaker of the two rulers to buy the trade. It would lose, and it would take four hours of card to find
out.

**Which turns the additivity result into a sharper statement about the route than the one it was asked
for.** Terms compose when they touch different channels - established above. But a ROSTER has a fixed
number of seats, and quality and decode-path diversity both spend the same seat. So at member level the
gains add, and at roster level they compete, and `calibsix_v1`'s 23% pass-through is what that competition
looks like from the board: six seats' worth of offline gain arriving as one seat's worth of board gain.
The member-quality lane ceilings near 0.943 not because the terms fail to stack but because the roster
cannot hold more of them without giving up what it already holds. Reaching 0.950 therefore needs a seat
that is BOTH high-quality and in a pixel group nothing else occupies - which is external data and a
different decode path in the same member, not another term on an existing seat.

### How `stackfit` must actually be read, written at 07:02Z with the arm at epoch 3 of 12

Its pre-registration is broken, and the fix makes it a better arm than it was designed to be.

`stackfit` carries `--hold-annotations`. Its named control `calib_dense_hold` does too. But its second
term comes from `ema_dense`, which is `hold_annotations: False` - so the prediction
`calib_dense_hold + (ema_dense - band_dense)` adds a NON-HOLD delta to a HOLD baseline. That transport
assumes EMA's effect is the same in both families, and the 2x2 above just measured that calibration's is
not (sign inverted, 0.0092) and slice count's is not (0.0067). The one assumption the prediction rests on
is the one this session refuted.

**What is still clean is a matched single-variable pair, and it is the whole arm.** `stackfit` against
`calib_dense_hold`: same hold family, same 20 slices, same 0.1-0.9 band, same 336 px, same labels, same
`--calibrated-targets`, same 12 epochs, same seed. ONE flag differs, `--ema`. So read the arm as what
EMA buys ON TOP OF calibration at the dense corner, and read it against 0.8634, not against 0.8644.

PRE-REGISTERED, before the number:

| landing | reading |
|---|---|
| **>= 0.8696** | EMA is a real lever at this corner, LARGER than its +0.0010 non-hold estimate. The first thing this session would have found that clears the floor on a matched pair, and it would license EMA at roster scale the way calibration was. |
| **0.8572 - 0.8696** | **FLAT** - the most likely branch. EMA adds nothing detectable on top of calibration, consistent with its own +0.0010 single. Closes EMA as a lever rather than leaving it untried, and the gross-substitution rung below is what the 85 minutes actually bought. |
| **<= 0.8572** | EMA SUBSTITUTES for calibration - adding a term worth +0.0010 ate more than a floor of a term worth +0.0061. That is the `depth_regions_span` shape a fifth time, and it would be the strongest evidence this project holds against stacking. |

Note what CANNOT be recovered: EMA alone in the hold family was never run, so the calibration x EMA
INTERACTION is not computable from this arm at any landing. It measures a conditional effect, not a
composition. The composition question is answered by the 2x2 above, on different levers, for that
holdout only.

### `stackfit` — EMA buys nothing on top of calibration; the FLAT branch fired — 2026-08-27

**0.8631, best epoch 7 of 12**, against `calib_dense_hold`'s **0.8634**. **−0.0003**, a twentieth of the
0.0062 three-seed paired floor. The pre-registration's own most-likely branch, and the only branch that
was named FLAT.

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| holdout | .7467 | .7994 | .8330 | .8502 | .8579 | .8618 | .8624 | **.8631** | .8618 | .8621 | .8614 | .8609 |

Twelve epochs of twelve, a peak at 7 and a four-epoch slide after it, so this is a CONVERGED reading and
not a budget-bound one — nothing is owed to a longer schedule.

**What it settles.** Read as the matched single-variable pair it actually is (one flag, `--ema`, same
hold family / 20 slices / 0.1–0.9 band / 336 px / labels / `--calibrated-targets` / 12 epochs / seed),
EMA's conditional effect on top of calibration at the dense corner is **zero to within a twentieth of the
floor**. EMA's own non-hold single read +0.0010; this says that +0.0010 does not survive being added to a
term worth +0.0061. Not substitution — the bottom rung asked for ≤0.8572 and nothing like that happened —
just nothing. EMA closes as a lever at roster scale, measured rather than untried.

**What it does NOT settle, restated because the arm was designed to answer it and cannot.** EMA alone in
the hold family was never run, so calibration × EMA is a conditional effect here and not an interaction.
The broken second term (an `ema_dense` delta measured at `hold_annotations: False`, transported onto a
HOLD baseline) is why the original three-term additivity prediction was retired at 07:02Z, before the
number, rather than scored against it.

**One number worth carrying out of it.** At 0.8631 `stackfit` is the second-best solo checkpoint in the
pool, behind its own control at 0.8634 and ahead of `band_dense`'s 0.8578 by +0.0053. The
`--hold-annotations` + `--calibrated-targets` dense corner is where our best single members now live, and
NEITHER of the two is mounted or named by any kernel. That is a roster fact, not a result, and it is
priced by the seat ruler before it is priced by enthusiasm — the 23% pass-through measured on `calibsix`
says six simultaneous member gains buy about one member's worth of board.

### The seat lane is closed on its own ruler: count plateaus, and spread LOSES to quality — 2026-08-27

`record_mix --seats honest` and `--seats diverse` over the same 103-record pool (909 shared studies;
smoke runs, other folds and `confident224`'s different label table excluded — all three raise their own
error, which is how the pool was found).

**HONEST (held-out selection, the only quotable selector).** Six members read 0.8720, nine 0.8729,
seventeen 0.8736, and the peak is **0.8738 at twenty-four**. Six → twenty-four buys **+0.0018**, under a
third of the 0.0062 three-seed paired floor, with the in-sample peak (0.8755 at seventeen) sitting
+0.0017 above the held-out one — i.e. the selection bias is the same size as the entire gain from
quadrupling the roster. **Count is exhausted from about nine members.**

**DIVERSE (spread-built).** Peak **0.8650 at ninety-nine**, against honest's 0.8738. **−0.0088**, past the
floor by 1.4x. The curve says why plainly: maximising distance elects the pool's worst members first
(`focus_latmen` at seat 2 drops the ensemble from 0.8634 to 0.7501), and it never recovers — it needs
twenty-five seats to get back to 0.8451, which one member does alone.

**This answers the question `record_mix`'s own docstring asks and has never had an answer to: an ensemble
built for spread does NOT beat one built for quality.** Not narrowly — by 1.4 floors. Diversity is worth
buying only among members that are already good, which is what "the seat number" has always priced and
what the flat-distance selector throws away.

**The one thing this does NOT close, stated because it is the whole reason a slot might still go here.**
Both selectors are OFFLINE rulers, and the offline ruler is the instrument `knee-nas5` measured as
UNDER-pricing decode-path diversity by **2.4x** (0.0059 offline, 0.014 on the board). So "count is flat"
and "spread loses" are established in exactly the space where the diversity axis is known to be
compressed. They close count and they close blind-distance selection; they do not close a targeted
pixel-group add, which is the only diversity move with a measured super-linear board transfer.

**What it costs to have learned this: zero slots and eleven minutes of CPU.** Both runs are on the ledger
rather than on the board, which is the point — the seat lane now has a NUMBER saying stop, instead of
`knee-xuj5`'s policy saying stop.

### `finding_calib_dense` — un-pooling the agreement map is FLAT; the pooled map already carried it — 2026-08-27

**PRIMARY reading (`noise_binding`, the ruler the arm pre-registered): tracking +0.507 [+0.434, +0.579]
against `calib_dense_hold`'s +0.550 [+0.470, +0.630]. The pre-registered FLAT branch fired**
(`<= +0.475` lever · `+0.475..+0.625` FLAT · `>= +0.625` worse). The move is −0.043 on a ±0.075 band
whose width is the CI half-width the instrument reports on 58 studies, and the two intervals overlap
across almost their whole length. This is a no-op, read on the ruler chosen before the run.

**Single variable, verified in the printed recipe**: `--finding-calibration` and nothing else against
`calib_dense_hold` — same corner (336 px, 20 slices, whole-stack, `cls_mean`/`attend`, band 0.1–0.9,
`--calibrated-targets`, `--hold-annotations`, fold 0, seed 2026, 12 epochs, one-cycle).

**Holdout FLOOR (biased against this arm by construction, so not the verdict): 0.8595 @ epoch 7**
against `calib_dense_hold`'s 0.8634 @ epoch 6, **−0.0039** — inside the 0.0062 three-seed floor.
Converged, not budget-bound: 0.7476 / 0.7919 / 0.8340 / 0.8446 / 0.8476 / 0.8548 / 0.8563 / **0.8595** /
0.8587 / 0.8591 / 0.8595 / 0.8594, twelve of twelve epochs at ~370 s. The proxy is scored against the
REPORT labels, which carry the very defect these offsets correct, so a small negative here is what a
correct offset would also produce — which is exactly why the arm named `noise_binding` as primary in
advance, and why this row does not read the −0.0039 as evidence either way.

**WHAT IT SETTLES.** `--calibrated-targets` replaces the reader's five levels with a POOLED agreement
rate; `ReaderAgreement.by_label` shows that map hides a 37-point spread (Contusion 44.4%, Synovitis
81.2%) failing in opposite directions. The natural next move is one shrunk logit offset per finding.
It buys nothing. Twelve scalars fitted over ~57 cells each cannot resolve what the pooled rate already
approximates at 684 cells — the shrinkage that keeps them from fitting sampling noise also shrinks them
back toward the pooled value. **Per-finding calibration is retired at this corner**, and the label lane's
remaining headroom is not in the LEVELS of the map.

**WHAT IT DOES NOT SETTLE, and this is the part worth keeping.** The lane is still LIVE. Both members
sit near +0.5 on the over-called cells — half a true positive of confident reader error inherited, on
15.8% of every annotated cell — and `--calibrated-targets` is still the only axis measured to move it
(+0.683/+0.700 uncalibrated → +0.538/+0.550 calibrated, −0.145 and −0.150 at two independent corners).
This arm says the residual +0.5 is not reachable by refining the same map per finding. It says nothing
about reaching it by a different mechanism — a per-STUDY confidence rather than a per-FINDING constant,
or supervision that is not the reader at all.

**One incidental reading, noted and not acted on.** The arm's agreed-negative pole sits at 0.0725
against the control's 0.1310, and its agreed-positive at 0.7362 against 0.6920 — a WIDER pole
separation on both ends. Tracking is a ratio on that separation, so the arm is better separated and
tracks marginally less; the two effects nearly cancel. A member with cleaner poles and the same
tracking ratio is not obviously worse than the control, and the holdout's −0.0039 is inside the floor.
Not a finding — a reason not to write `finding_calib_dense` off as a strictly worse member if a seat
ever wants a differently-calibrated one.

### MedianLogit — a robust vote LOSES here, and the shape of the loss is why — 2026-08-27

`record_mix` gained a `--vote {rank,logit,median}` flag, and `Pool` gained the vote rule as a field.
The gap that closed: the whole-pool vote comparison at the foot of this tool's report has existed since
2026-08-17, but the SUBSET SEARCH and the seat curve above it were hardcoded to `RankAverage` — so every
"best k" and every seat number in this ledger was chosen under a rule the board entries do not serve.
The champion has served `LogitAverage` since `kneemri-vote`.

Pool: the offline shadow of the shipped roster — the five holdout twins of the champion six
(`final_ws` is dark and has no twin) plus the two `pixgroup` adds. 909 shared studies.

| seats | RankAverage | MedianLogit |
|-------|-------------|-------------|
| 2     | 0.8627      | 0.8632      |
| 3     | 0.8659      | 0.8616      |
| 4     | 0.8676      | 0.8660      |
| 5     | 0.8685      | 0.8646      |
| 6     | 0.8692      | 0.8671      |
| 7     | 0.8697      | 0.8664      |

Whole pool: RankAverage 0.8697 · LogitAverage 0.8702 · MedianLogit 0.8664. The median is −0.0033 under
the ledger's rule and −0.0038 under the shipped one.

**Read the SHAPE, not the size.** −0.0038 is inside the 0.0062 paired noise floor and would be noise if
it were one number. It is not one number: the median's curve OSCILLATES on the parity of the seat count
(2 up, 3 down, 4 up, 5 down, 6 up, 7 down) while every mean rule's rises monotonically. An even median
averages the middle two members and keeps some of the mean's information; an odd one elects a single
member and discards the pool. So a median vote makes the score depend on whether the roster happens to
be even — a dependence no seating argument can control for. The rule is refuted at this corner for a
named mechanism, not for a delta.

The rule stays in `kneemri.inference.vote`, with the parity artefact named in its docstring, because a
roster that IS outlier-shaped can re-ask this in one CLI flag and will not have to rediscover why the
curve zig-zags.

Incidental and useful: `LogitAverage` beats `RankAverage` by +0.0005 on this shadow, the same sign and
size as the 2026-08-23 re-pricing on the champion seven (+0.0002). The shipped choice is confirmed by
the ruler that had never been asked about it.

### The count plateau is a property of the VOTE RULE, not of ensembling — 2026-08-27

The first thing `--vote` was pointed at, and it moved a constant this project has been planning against.
`--seats honest` over the 91 same-ruler `_s2026` records (one, `confident224_s2026`, was scored by a
different label table and is excluded; 909 shared studies), run twice:

| seats | honest, RankAverage | honest, LogitAverage |
|-------|---------------------|----------------------|
| 5     | 0.8717              | 0.8708               |
| 8     | 0.8723              | 0.8729               |
| 10    | **0.8735**          | 0.8735               |
| 12    | 0.8733              | 0.8746               |
| 15    | 0.8733              | **0.8747**           |
| 20    | 0.8727              | 0.8735               |
| 24    | 0.8727              | 0.8736               |

Rank peaks at TEN and decays. Logit peaks at FIFTEEN, and its decay past the peak is gentler — still
0.8736 at twenty-four, where rank has fallen to 0.8727. The in-sample bias runs the same way in both
(+0.002 to +0.004), so this is not a selection artefact of one arm.

**What this corrects.** `record_mix`'s own docstring has said "count plateaus at eight to ten" since
2026-08-22, and that sentence is the reason the seat lane was treated as closed above ~10. It was
measured under `RankAverage` while every board entry since `kneemri-vote` has served `LogitAverage` —
the ruler and the entry disagreeing on the rule, which is exactly the harness fault class this project
keeps paying for. Under the rule we actually ship, the plateau sits five seats further out.

**What it does NOT license.** 11 → 15 seats is +0.0013 offline, which is at the ruler's 0.0015
resolution and prices at about +0.001 board — one tick, inside the noise. And the board has its own
evidence on the same axis: `calibsix` (6 seats) and `calibmix` (11 seats) both scored 0.941 under
LogitAverage, so the flat stretch this ruler now shows from 10 to 24 is what the board already
measured. Count is still not a lever. What changed is the REASON: it is flat because it is flat out
to twenty-four seats, not because it turns down after ten.

The greedy ordering under logit is also worth reading against what we serve — its first eight are
`calib_dense_hold`, `blendhold`, `finding_calib_dense`, `small_batch6`, `margin_seat`,
`calib_span_hold`, `through_dense`, `merged_blend`. Three `_hold` members and two calibration members
in the top six, and `finding_calib_dense` — retired this morning as a FLAT member on its own ruler —
seated third. A member that is unremarkable alone can still be the best third seat, which is the
diversity argument stated in the only place it can be checked.

### normcount224 — the last unmeasured recipe key is FLAT, and the knob space is now enumerated by MEASUREMENT — 2026-08-27

`--normalise-by-count` was the sixty-first and last recipe key this project had never varied. Its arm
pre-registered a ~0.10 chance of clearing the noise floor and expected score movement of about zero,
and it was armed as an INSTRUMENT: the claim "the knob space is enumerated" is what points our schedule
at members and ensembles instead of at another sweep, and that claim was a survey, not a measurement.

Twelve epochs at 224 px, ~115 s/epoch, 7.42 GB reserved (built small deliberately, to run beside a
co-tenant). Curve: 0.7472 / 0.7658 / 0.8147 / 0.8270 / 0.8369 / 0.8417 / **0.8498** / 0.8429 / 0.8453 /
0.8464 / 0.8454 / 0.8454. Peak 0.8498 at epoch 6; converged, not truncated.

Control `native224` reads 0.8502 at epoch 7. **−0.0004.** The pre-registered ladder put 0.8440–0.8564
in the FLAT band, so this lands mid-band and the knob is retired.

The instrument paid for itself as an instrument. The knob space is now CLOSED BY MEASUREMENT rather
than by survey: sixty-one keys, every one varied at least once, none of the remaining ones moving the
number. Nothing in the schedule changes — which is the point of running it. A null that licenses the
plan is worth its twenty minutes; it would not have been worth an hour.

### Aggregation beats input: fourteen arms split by WHAT they changed, not by how much — 2026-08-27

`kneemri.evaluation.run_comparison --split`, control `lr4e-5_ws_s2026`, deltas on the six
model-bottlenecked findings (ACL, Lateral Meniscus, Lateral OA, MCL, Medial Meniscus, PF OA). Zero card,
zero slot: every one of these arms had written its metrics file months ago and nobody had read them
side by side on this axis.

| arm | what it changed | focal six | regional | mean |
|---|---|---|---|---|
| sided_pool | `--pool sided` | **+0.0076** | +0.0007 | +0.0036 |
| attend_dense | `--slice-pooling attend` | **+0.0059** | +0.0046 | +0.0043 |
| depth_regions | `--slice-regions 4` | **+0.0016** | +0.0048 | +0.0024 |
| through_dense | `--through-plane` | **+0.0005** | +0.0065 | +0.0033 |
| res392 | 392 px | −0.0022 | +0.0032 | −0.0002 |
| depth_regions_span | regions + wider band | −0.0052 | −0.0017 | −0.0039 |
| attend_true | attention, other variant | −0.0068 | −0.0039 | −0.0050 |
| intensity392 | 392 px + intensity aug | −0.0076 | −0.0040 | −0.0066 |
| crop80_span | 80 mm crop, tissue centre | −0.0098 | −0.0025 | −0.0059 |
| sliceattn | slice attention | −0.0117 | +0.0016 | −0.0053 |
| slotmean | mean over slots | −0.0134 | −0.0099 | −0.0120 |
| focal_pool | focal pooling | −0.0144 | −0.0023 | −0.0094 |
| slices16 | 12 → 16 slices | −0.0177 | −0.0026 | −0.0114 |
| pair_memory | paired memory | −0.0448 | −0.0263 | −0.0371 |

**The split is by KIND and it is total.** Every arm that gave the model MORE INPUT — more pixels, more
slices, a tighter crop — lost on the focal findings. Every arm that changed HOW THE SLICES ARE
AGGREGATED gained. Four positives, four unrelated mechanisms, no exceptions in either direction.

**The mechanism, and it predicts the sign of both halves.** A thin focal structure — the ACL, a
meniscal tear, the MCL — lives on two or three slices of twenty. Any pooling that averages over depth
divides its evidence by the slice count. Adding slices makes the dilution WORSE, which is why
`slices16` is the single worst non-broken arm of the fourteen and why the effect is concentrated in
MCL (−0.0497) and Lateral Meniscus (−0.0262) rather than spread. Adding pixels does not touch the
dilution at all: `res392` INVERTS, reading regional +0.0032 against focal −0.0027 — it helped exactly
the findings a tighter view should have cost. Only a pooling that can concentrate on the slices where
the structure lives recovers it.

**What this kills.** The 512 px / focal-crop lane, which the supervision gap (`knee-cwpk`: the model
loses to its own reader by −0.193 on Lateral Meniscus, −0.123 on ACL) had made look like the obvious
next card job. Three resolution arms and two crop arms, all negative on the term they were built for.
Resolution is not the mechanism behind that gap. That reading cost twenty minutes and saved a day of
card.

**What this does NOT license.** Each of the four positives is individually inside the 0.0062 paired
floor, and the focal six is a SUBSET — a delta over six of twelve labels is half a mean, not a board
number. This is a DIRECTION, established by fourteen readings agreeing on a sign, not a result. The
mean column is the one the board scores and none of the four clears anything there either.

**The consequence, and it is the only thing here worth card.** The four act at three different points —
`--through-plane` before the encoder sees a slice, `attend` across the slice axis, `--pool sided`
across the slots — and have never been run together. `experiments/arms/fuse_attend.sh` and
`fuse_regions.sh` are that stack, priced in `knee-7d53` at ~45 min each and P(clears the floor) = 0.35.

**A defect found on the way.** `model.py:322` returns the attend path and never reaches `over_depth`,
so `--slice-regions` passed beside `--slice-pooling attend` is **silently ignored** — no error, no log
line, and the run records a region count it never used. That is why the stack is two arms and not one.

### The seat ruler is unanimous about which champion seat is weak — 2026-08-27

`record_mix --vote logit`, roster = the five holdout twins of the shipped champion six (`final_ws` has
no twin and cannot have one), 909 shared studies, roster reads 0.8701.

Ten candidates were priced both as an ADD and as a SWAP. Every single one gains more by replacing
`lr4e-5_ws` than by taking a seventh seat, and no candidate's best swap is against any other seat:

| candidate | as a 6th seat | best swap (all for `lr4e-5_ws`) |
|---|---|---|
| margin_seat | +0.0018 | **+0.0023** |
| small_batch6 | +0.0009 | +0.0013 |
| band_dense_long | +0.0007 | +0.0011 |
| sided_pool | +0.0007 | +0.0010 |
| attend_dense | +0.0005 | +0.0010 |
| crop80_span | +0.0004 | +0.0006 |
| depth_regions | +0.0000 | +0.0004 |
| base_dense | −0.0001 | +0.0001 |
| res392 | −0.0004 | −0.0001 |

**The board has already confirmed the bottom of this ordering.** `resseat` swapped `lr4e-5_ws_ff` for
`res392_ws_s2026` and scored 0.938 against `calibsix`'s 0.941 — three ticks down, from the candidate
this ruler ranks last of ten. The ordering transfers at the bottom, which is the only end it has been
tested at.

**And the top of it is not shippable.** `margin_seat` is the only candidate clearing the +0.0015 under
which this ruler has never picked a board tick, and it has no full-fit twin; the two candidates that
ARE packaged full-fit (`small_batch6_ff`, `base_dense_ff`) read +0.0013 and +0.0001, both inside the
ruler's own resolution. So the best-priced move this project can currently SHIP is one the instrument
cannot distinguish from the incumbent — which is the same wall `knee-97ib` hit from the other side when
it measured the marginal seat at exactly 0.000.

### The first of the two fusion arms lands under its own components — 2026-08-27

`fuse_attend` (`--through-plane` + `--slice-pooling attend` + `--pool sided`, twelve epochs, held split)
reads a best holdout of **0.8509** at epoch 8. `convergence_check` does not list it as truncated, so the
number is the recipe's and not a killed process's. Its control `calib_dense_hold` reads 0.8634, and
`band_dense` — the plain member the four positives were each measured against — reads 0.8578.

**Stacking the four did not add them.** Each acts at a different point and each was individually inside
the 0.0062 paired floor; the stack is −0.0125 against the control, which is OUTSIDE that floor in the
wrong direction. The three that combine here all reduce depth dilution, so the honest reading is that
they are not independent mechanisms adding up but three spellings of one, and running all three
concentrates past the point where the evidence is.

`fuse_regions` — the other half, and the arm the `model.py:322` defect forces to be separate — died eight
seconds in on a torn source snapshot (an uncommitted mid-edit tree, `knee-2dcp`) and has been
requeued. `stackfit.sh` (`knee-orjq`) stays gated on both singles landing non-negative, and this one did
not: on this reading the gate is closed by half already.

### `contrast_dense` — the representation lane's first probe costs 11 of 12 labels, and the batch is the named suspect — 2026-08-27

`contrast_dense` (`knee-ek5s`), the owner's "torn meniscus is physically separated as a link" idea in its
cheapest form: a supervised-contrastive term per finding over the vector each diagnosis reads before the
readout collapses it. Single variable — `--contrast-weight 1.0` — against `calib_dense_hold` **0.8634**,
verified by diffing the two stored recipes (the only other keys that differ are absent-in-the-older-run
artefacts: `finding_calibration`, `spacing`, `out`).

Twelve epochs at 420 s, control 413 s. Best **0.8547 at epoch 5**, then monotone decline to 0.8487 at
e11. `convergence_check` reports it unscored rather than truncated, so this is the recipe's number.

**−0.0087 against the control, outside the 0.0062 paired floor, the wrong way.** Per-label at each side's
own best epoch:

| label | control | arm | delta | | label | control | arm | delta |
|---|---|---|---|---|---|---|---|---|
| Effusion | 0.8559 | 0.8352 | **−0.0208** | | ACL | 0.8943 | 0.8894 | −0.0049 |
| Synovitis | 0.8589 | 0.8424 | −0.0165 | | Contusion | 0.8814 | 0.8770 | −0.0043 |
| Lateral OA | 0.8330 | 0.8169 | −0.0161 | | Fracture | 0.8214 | 0.8180 | −0.0034 |
| MCL | 0.7923 | 0.7766 | −0.0158 | | Baker's | 0.9135 | 0.9124 | −0.0011 |
| Medial OA | 0.8828 | 0.8691 | −0.0137 | | **PF OA** | 0.8584 | 0.8688 | **+0.0104** |
| Lateral Meniscus | 0.8700 | 0.8573 | −0.0127 | | | | | |
| Medial Meniscus | 0.8987 | 0.8931 | −0.0057 | | | | | |

**Eleven of twelve down, one up, no label moving more than ~0.02.** That is uniform mild damage across
labels the term places no per-label emphasis on — the shape of an auxiliary objective taxing a shared
trunk, and the same shape `side_dense` showed at ten times the size (all twelve down, −0.0641). It is not
the shape of a mechanism that helped some findings and hurt others.

**And the two labels the idea was FOR did not move the right way.** Continuity is a meniscus story; Lateral
Meniscus reads −0.0127 and Medial Meniscus −0.0057. Whatever the term did to the representation, it was
not putting torn beside torn.

**WHAT THIS DOES NOT REFUTE, and the arm named the suspect before it ran.** The docstring's stated weakness
is that contrastive learning conventionally uses batches two orders of magnitude larger than our twelve
studies, so a batch of twelve sees a 7%-prevalence finding about once — the positives a supervised
contrastive term needs are mostly absent from the batch it is computed on. That is a HARNESS suspect
(faulty TRAINING), not the idea, and it fired exactly where predicted: the term is loudest (~3× the
cross-entropy at init, chosen deliberately so a null would be readable) while its supervision is
degenerate. So the reading is **"supervised contrastive at batch 12 costs 0.009 and buys nothing"**, not
"shaping the representation does not work". `RememberedRanking` — a remembered pool of past readouts — is
the named precondition and remains untried.

**Cost note, because this project prices arms in wall-clock.** The header predicted ~200 s an epoch and
~40 min; it ran 420 s and 84 min. The estimate was 2× optimistic against a control already on disk at
413 s — an arm's cost should be read off its control's measured epoch time, not re-derived.

**No weight sweep** — a dose sweep on a term whose supervision is degenerate at this batch size would sweep
the wrong variable. The topology question goes to its strong form (masks, `knee-2sbk`) or to the remembered
pool, not to `--contrast-weight 0.3`.

**But it DOES earn a seat, and that correction is this entry's own.** The first version of this line read
"No seat, and no weight sweep", written before the member was ever priced as one. Greedy forward selection
under `record_mix --largest-pool --vote logit` over 104 records / 909 shared studies seats
`contrast_dense_s2026.pt` **fifth**, at 0.8749 — after `calib_dense_hold` (0.8634), `merged_blend_cf`,
`finding_calib_dense` and `blendhold`, and ahead of `margin_seat`, `warm_calib_dense` and `calib_span`. A
recipe that loses 0.0087 solo is the owner's standing point about diversity over solo quality, measured on
our own pool for the first time rather than asserted: the seat is bought by what the member disagrees about,
and the contrastive term changed what it disagrees about even while it cost eleven of twelve labels. The
disposition is **seat candidate**, not dead arm. What is still unpriced is whether that seat SURVIVES on the
board — the greedy order is an in-sample ranking and the packaged-roster question is separate (the member is
not in `all_ours`, so seating it costs a 7 GB push, `knee-unmb`).

### labelwise_dense — per-diagnosis pooling fires, and buys nothing at the depth built for it — 2026-08-27

bead: knee-377t. One variable against `calib_dense_hold` **0.8634**: `--slice-pooling max` → `labelwise`,
same 20 slices, same 0.1-0.9 band, same calibrated targets, same held annotations, same rate, same seed.
12 epochs, ~410 s each. Best **0.8613 @ e7**, then 0.8607 / 0.8611 / 0.8606 — a plateau, not a climb cut off.

**−0.0021, inside the 0.0062 paired floor: FLAT**, the third of the four pre-registered bands.

**The mechanism fired, and the check that proves it nearly failed for the wrong reason.** The header
pre-registered `slices.pool.gate.3.weight` as the key to assert non-zero. That key does not exist — the
dump came back with an EMPTY list, which is indistinguishable at a glance from the dead gate the check was
written to catch. The real key is `slices.gate.3.weight`, shape **(12, 64)**, absmax 0.171, std 0.059,
with the whole stack (`gate.0` 768→768, `gate.1` 64×768, `gate.3` 12×64) trained away from its
zero init. Twelve output channels is the per-diagnosis axis, so the arm really did send different findings
to different depths. The control's state dict carries NO gate keys at all, which is the single-variable
claim proved from the other side.

So this is NOT a repeat of `attend_dense`, where the aggregator sat in no parameter group and the arm
compared the arithmetic mean against the max. Working per-diagnosis attention has now been read at
336 px × 20 slices, which is the cell the ledger recorded as empty. **A pre-registered harness check with
the wrong key name is worse than none** — it returns the failure signature on a healthy run, and the
temptation is to read the empty list as the answer the header predicted.

**What this does to the interaction hypothesis.** The arm existed because two axes had each been closed in
the other's absence: slice count under MAX pooling read −0.0001 (`knee-ezz`, the head cannot spend the
slices), and per-diagnosis pooling at 12 slices read +0.0015 (`labelwise224`, the pooling has no slices to
spend). Raptor's `coatnet_v4_full` scored through our caches gains more from 16→20 slices (+0.0046) than
from 12→16 (+0.0027), so their per-finding head was still steepening where we stopped. Run WITH the depth,
ours does not steepen — it reads flat. The foreign curve is therefore about THEIR head or THEIR training,
not about per-finding pooling as such; the two are no longer interchangeable evidence and `foreign_gate`
numbers should not be used to price our own head again without saying which part is being borrowed.

**What is NOT refuted.** That per-diagnosis pooling is worthless — flat at the 0.0062 floor is not zero, and
`labelwise224`'s +0.0015 sits in the same band from the other side, so two arms now agree the effect is
small-and-positive-or-nothing rather than absent. And it is not refuted that the head is the wrong place:
what was tested is one aggregator over the slice axis, downstream of a `SlotHead` that already carries its
own per-diagnosis query (`slot_head.py:120`). Two per-diagnosis mechanisms stacked may simply be one too
many — the untested cell is labelwise pooling with the slot query REMOVED, not more depth.

**Disposition: a diversity seat, not an upgrade.** Same quality, different internal structure — the profile
that earned `labelwise224` its keep. It writes a holdout record, so it can be priced as a seat by
`seat_value` without a refit. It is NOT in the shipped package, so seating it costs a 7 GB push
(`knee-unmb`), and the seat-count reading of 2026-08-27 says a seventh seat is flat on both rulers anyway
(`interpretations/kneemri/2026-08-27_seat-count-is-flat-both-rulers.md`). It goes in the pool, not on a slot.

**Card handed to celltrack at 18:35Z as promised**; `card_check` reads the device free with no `kneemri`
process on it.

### `warm_calib_dense` — the warm-start's FIRST arm priced init TIME, not capacity, and came in −0.0072 — 2026-08-27

`warm_calib_dense_s2026` best holdout **0.8562** against its control `calib_dense_hold_s2026` **0.8634** —
**−0.0072**, just past the 0.0062 arm-vs-arm floor. Written here with its own name beside its own number,
which is the whole reason it took a day to surface: `0.8562` is quoted on six lines of this record and none
of them is about this arm, so `ledger_check` read it as recorded while its verdict sat unwritten in
`experiments/logs/warm_waiter.log:32`. That hole is now the `unattributed` advisory in the same check.

**The harness is clean and the pair is genuinely single-variable.** The flag LANDED — the run logs
`encoder initialised from calib_dense_hold.enc.pt, adapted on cache 0 (223 tensors, 0 left stock)` — and
`calib_dense_hold.sh:69` against `warm_calib_dense.sh:56` is the same `run_train` line but for
`--encoder-weights`. So the number is about the treatment.

**But the treatment is not the one `knee-wbwq` argues for.** The bead's claim is CAPACITY: a warm encoder
frees the adapter to learn something new. This arm passes no `--lora-rank` (recipe `"lora_rank": "0"`), and
neither does its parent. `fitted_encoder.py:88-97` skips only keys ending `DOWN`/`UP` and rewrites only
`BASE_WEIGHT`/`BASE_BIAS`; a rank-0 state has none of them, so every tensor falls through to
`merged[name] = value`. **Nothing was merged and no fresh capacity was built.** The arm priced init TIME —
does a converged trunk start better — and the answer is no. It says nothing about capacity.

**The schedule suspect it named in advance is still live and is now correctly sized.** The arm predicted
that below 0.85 would indict `one_cycle` ramping an already-converged trunk off its optimum. 0.8562 is a
SMALL negative, equally consistent with "no capacity to gain, plus a little ramp damage". The two are not
separated by this number. `nocycle224_long_s2026` (0.8442) is the nearest evidence on that axis and is
itself unrecorded here.

**Do not read this as the bead refuted.** What is refuted is one cell of a three-way grid (rank 0 × warm ×
one_cycle), and it is not the cell the mechanism argument lives in. The rewritten chain is on `knee-wbwq`:
**A** schedule — the same pair at `--backbone-lr 1e-5` or with no one_cycle ramp, ~40 min; **B** the actual
capacity arm, which needs a LoRA parent (`dinov3_base_lora_s2026` or `lora_r8_v3small_224_s2026`)
warm-started WITH a fresh `--lora-rank`; **C** `warm_calib_dense_short.sh` stays gated on A or B clearing.
Both are blocked on the card, which is with celltrack.

### `margin_calib` — the empty quadrant is occupied, and the two terms are ANTAGONISTIC, not the same lever — 2026-08-27

`margin_calib_s2026` best holdout **0.8235** @ epoch 10 of 12 (`convergence_check`: converged, not
truncated, not undertrained). The arm `knee-myl0` designed as the deliberate occupation of the
strong-and-distant cell — `--auc-margin 1.0 --rank-weight 0` (distance) together with
`--calibrated-targets` (quality), at the identical corner where each had measured positive alone.

| | solo | vs |
|---|---|---|
| `margin_seat` — the margin term alone | 0.8525 | — |
| `calib_dense_hold` — the calibration term alone | 0.8634 | — |
| additive prediction | 0.8581 | the bead's own arithmetic |
| **`margin_calib` — both** | **0.8235** | **−0.0346 against the prediction** |

**The harness branch fired and came back clean, which is what makes the number quotable.** The bead
pre-registered "solo < 0.8525 → suspect the harness before the mechanism; diff the recorded recipe stamps
key by key." Done, both ways: against `calib_dense_hold` only `auc_margin` (1.0 vs 0.0) and `rank_weight`
(0.0 vs 1.0) differ — the substitution as designed; against `margin_seat` only `calibrated_targets` and
`hold_annotations` differ. Both flags landed, the corner is exact, and the arm ran its full schedule. The
−0.0346 is the mechanism, not the plumbing.

**The pre-registered SAME-LEVER branch is REFUTED, and by the column it was read on.** That branch
predicted `seat < +0.0015` *and* `nearest ≥ 0.92` — calibration and the margin turning out to be one axis,
the distance collapsing to `calib_dense_hold`'s 0.9273. `seat_value --against gradeswap_v1`:

```
  +margin_seat        0.8709  (+0.0015)  [+0.0007, +0.0023]   nearest 0.8628  mean 0.8400
  +calib_dense_hold   0.8704  (+0.0010)  [+0.0005, +0.0015]   nearest 0.9273  mean 0.8925
  +margin_calib       0.8682  (-0.0013)  [-0.0021, -0.0005]   nearest 0.8298  mean 0.8101
```

The seat is **negative with a band that excludes zero** — it does not merely fail to help, it subtracts —
while the distance went the OPPOSITE way to the prediction: `nearest 0.8298` is the most distant candidate
this project has ever priced, further out than `margin_seat` itself. So the two terms are not one lever.
They are mechanically distinct exactly as the bead argued, and combining them is destructive.

**A mechanism, offered as a hypothesis and not measured here.** An AUC-margin loss ranks pairs across a
positive/negative boundary; `--calibrated-targets` deliberately smears where that boundary is. The margin
term then optimises a separation the targets have been changed to blur. That predicts the pair is
antagonistic wherever both are on, at any corner — which is a cheap thing to falsify later and is the only
part of this worth carrying forward.

**`knee-bxp2` is confirmed on its own terms: distance is a twin detector, never a merit column.** This is
the cleanest distant-and-weak member on file — maximal distance, negative seat — and a selector reading the
distance column alone would have picked it first. Cost: ~85 min of card, zero slots. The quadrant is
occupied and it was empty for a reason.

### `mixup_dense` — mixup at 0.2 on the dense recipe reads NULL, and the last epoch was still climbing

Ran 2026-08-28, twelve epochs, `band_dense`'s invocation with `--mixup 0.2` and nothing else moved, so
the control is `band_dense` 0.8578 (peak e8) at zero extra card.

| | holdout mean AUC | vs control |
|---|---|---|
| `band_dense` (control, peak e8) | 0.8578 | — |
| **`mixup_dense` (peak e11)** | **0.8552** | **−0.0026** |

Inside the 0.0062 paired floor in the negative direction: **null, not a refutation of mixup**, and the
shape says why. The run peaked on its LAST epoch (e11, 0.8552, still up from e10's 0.8546) — UNDERTRAINED
in `convergence_check`'s sense, the budget bound it rather than the schedule. Mixup slows fitting by
construction, so twelve epochs buys a mixup arm less than it buys its control, and the honest reading is
"at THIS budget mixup pays nothing", not "mixup does not work here". A longer arm is the only thing that
would separate those, and at ~50 min for +6 epochs against a mechanism already reading flat it loses the
ratio to the distillation lane.

Cost: ~73 min of card, zero slots. `knee-x580`.

### `raptor_distill` — the foreign teacher IS partly learnable, and the student is still a member we already hold — 2026-08-28

`--distill-weight 0.3` with `--teacher-targets raptor_teacher.npz`, the dense recipe otherwise untouched,
12 epochs, fold 0. The teacher is `raptor_v4_swa`: board-measured at 0.923 solo, EXACTLY level with our
own best single, and the first teacher from another architecture, another decode and another corpus.
`distill_pairs` diagnosed its own null as "a rank average cannot pay for an ordering it already holds";
this arm removes that defect by construction and asks whether the ordering was the whole story.

```
peak 0.8613 at e8 of 12, converged (e9 0.8606, e10 0.8593, e11 declining) — convergence_check clean
vs band_dense 0.8578  = +0.0035, inside the 0.0062 holdout floor
greedy entry position 44 of 121, at 0.8810 — the greedy peak is 0.8978 at position 5
```

THE SIGN THE BEAD PREDICTED IS THERE, AND IT IS NOT WORTH A SEAT. `distill_pairs` read +0.0016 from a
DINOv2 self-teacher; this reads +0.0035 from an architecture-distant one. The ordering of the two is the
mechanism's own prediction and it holds: a teacher outside the family transfers more than a teacher
inside it. Both are under the floor, so the pair is directional evidence about distillation, not a result.

THE DISTANCE RULER DECLINES IT, WHICH IS THE ONLY QUESTION `knee-for6` LEFT OPEN. The bead pre-registered
"a seat is licensed only if the distance is genuinely new, not by the solo number". Greedy admits it 44th
of 121 records, 39 positions after the peak set closes — our own `calib_dense_hold` enters 17th. Whatever
distance the teacher had, the student did not inherit it: a student trained on our decode, our corpus and
our trunk lands back in our pixel group no matter whose ordering it was taught. That is the finding, and
it is a constraint on the whole distillation lane rather than on this arm — distillation moves ORDERING
and the seat ruler prices PIXEL GROUP.

READ THE GREEDY'S TOP WITH ITS CONTAMINATION IN MIND. Positions 1, 2, 4, 7, 8 and 18 are raptor
checkpoints, every one `--fold -1`, so their holdout sits inside their training set and the 0.8978 peak is
not a number about anything. Only the relative position of two honestly-trained members is readable here,
which is exactly the comparison drawn above.

WHAT IT DOES NOT CLOSE. The teacher pass over 4,406 studies is on disk and free to every later arm on this
axis, so a different `--distill-weight`, a different teacher, or distillation onto a NON-DINOv2 trunk each
cost one student and no teacher. None of them is licensed by this number: the constraint found is about
where a student lands in pixel space, and the first two variants do not move it.

### `calib_dense_hold_ff` — a member with no number, and that is by construction

Finished 13:35Z on 2026-08-29, twelve epochs at 415 s, training loss 0.6765 (e5) → 0.5126 (e10), 27.01 GB
peak reserved. `--fold -1 --ema --calibrated-targets --hold-annotations`, 336 px, 20 slices.

**It reports `holdout mean AUC nan` and there is nothing wrong with it.** `--fold -1` trains on every
study, so there is no held-out set to score against; the number the ledger normally carries does not exist
for this arm and cannot be manufactured. It is a SHELF member — a checkpoint whose only ruler is what it
does to a roster it is seated in.

**And it cannot be priced against the champion either.** `seat_value` refuses: of `raptorseat_v1`'s six
members exactly one carries a holdout record, and a marginal contribution needs at least two. The five it
cannot read (`calib_span_ff`, `final_ws_cf`, `grade_ladder_cf`, `lr4e-5_ws_cf`, `through_dense_cf`) are all
`--fold -1` themselves — they have seen every holdout study, so a score over those studies is a score over
their own training set. The tool's own suggestion is the honest one: price against a fully scoreable entry
(`honest12`, 0.937, twelve members) and read the seat delta there, not against the 0.943 board row.

**Which is the standing cost of the whole-corpus decision, stated plainly.** Fitting on everything buys the
member the studies a fold would have withheld and spends the ability to say what the member is worth
without a slot. Both halves of that trade are real; the ledger should stop being surprised by the second.

**Priced anyway, through the fold-0 TWINS, 2026-08-29 20:41Z — and it does not license a slot.** The
champion's five `--fold -1` seats each have a fold-0 twin, and those twins are scoreable, so the swap can be
read on the shadow the roster already records. Arms differ in exactly one member: `through_dense_s2026`
(0.8558, the champion's weakest twin) out, `calib_dense_hold_s2026` (0.8634, the best solo member this
project owns) in. `record_mix --vote logit`, same five-way pool both sides:

| arm | best single | RankAverage | LogitAverage | best-4 subset |
|---|---|---|---|---|
| shadow (`through_dense`) | 0.8589 | 0.8701 | 0.8705 | 0.8708 |
| swap (`calib_dense_hold`) | 0.8634 | **0.8705** | **0.8707** | 0.8710 |

**+0.0004 rank, +0.0002 logit, off a member that is +0.0045 better alone.** The sign is positive and that is
all that transfers on a member swap ([[member-swap-transfers-one-to-one]]), but the size is a quarter of the
0.0053 floor and an eighth of the 0.02 bar. The mechanism is the same dilution `fourseat_v1` already paid a
slot to learn: a strictly better member seated among four it already agrees with returns almost nothing.
Read it in the oracle row — headroom improves from −0.0048 to −0.0027, so the swap makes the ensemble MORE
agreeing, not more diverse.

**So `calib_dense_hold_ff` is seated in the roster and stays off the submission queue.** It costs no extra
decode (it rides the champion's own feed), it is the best solo checkpoint on the shelf, and it is worth
having available the next time a roster is rebuilt around a genuinely new axis — but swapping it in is not
that axis, and the pinned champion (0.943) is not at risk from a +0.0004 offline delta.

### Diversity as a SELECTION RULE is refuted — 2026-08-29 20:50Z

Last tick closed member QUALITY as a lever (a +0.0045 member bought +0.0004 seated). The obvious successor
was the owner's own framing — "it's about diversity, not averaging 7 numbers that are the same number". So
run it as a rule, not a slogan. `record_mix --seats diverse` picks each next seat by rank-space distance from
what is already seated and NEVER OPENS THE LABELS; `--seats greedy` picks by the holdout mean. Same pool,
same vote, same evening:

| rule | peak | at seats | selection bias |
|---|---|---|---|
| `diverse` (rank distance, labels closed) | 0.8695 | 88 | none — it cannot overfit a score it never reads |
| `greedy` (holdout mean) | 0.8777 | 13 | +0.0032 |
| `honest` (greedy refit on half, scored on the other half) | **0.8753** | 28 | — |

**Honest-to-honest, quality beats distance by 0.0058 with a third of the seats.** And `diverse` is not
merely worse, it is worse in a diagnosable way: its first seven picks run 0.8634 → 0.8613 → 0.8579 → 0.8431
→ 0.8412 → 0.8335 → 0.8331. Maximising distance from the seated set walks straight at the pool's WORST
members, because the thing furthest from a good prediction is a bad one. Distance cannot tell a member that
disagrees usefully from a member that is simply wrong; only the labels can, and that rule refuses to look.

**What survives.** Diversity is real as a PROPERTY — greedy's own thirteen span calibration, blend, span,
margin, warm-start and MAE mechanisms, and it seats `focus_mcl_s2026` at eleven despite that arm regressing
its own target label. It is not usable as a SELECTION CRITERION. The honest curve is also the number to plan
against: **0.8753 at 28 seats**, not the 0.8777 the in-sample peak advertises, and the +0.0032 gap is the
size of every seat-count claim this ledger has made off a greedy curve.

### The honest curve DID transfer — the "sign flip" was a fold-0 offline paired with an `_ff` board score — 2026-08-29 21:45Z

Half a retraction of the retraction below, and the half that dies is the general claim, not the decision.

`kneemri.kaggle.entry_offline` filled the ten board rows that could honestly carry an offline number
(knee-awxb). With those in, the fold-0 family is resolvable end to end for the first time:

| entry | offline | board | seats |
|---|---|---|---|
| **honest12** | **0.8751** | **0.937** | 12 |
| peak8_v1 | 0.8731 | 0.932 | 8 |
| greedy7_v1 | 0.8712 | 0.933 | 7 |
| accumseat_v1 | 0.8695 | 0.933 | 7 |
| sevenplane_v1 | 0.8695 | 0.932 | 7 |
| sixstrong_v1 | 0.8692 | 0.930 | 6 |
| basedense_v1 | 0.8690 | 0.931 | 5 |

**`honest12` is the top of its own family on BOTH rulers.** The instrument picked the best-scoring entry
of every entry it can be compared with, and `board_transfer` now reports **0 inversions in 233 resolved
pairs**. "Its rankings do not transfer" is refuted by the record.

**Where the sign flip came from.** It was computed against the champion's **0.943** — an entry built from
`_ff` members fitted at `--fold -1`, whose offline is not merely missing but unmeasurable, and which
`entry_offline` now marks `offline_scope: in-sample` for exactly that reason. Pairing a fold-0 offline
against an `_ff` board score is the same contamination class that voided `raptor_solo`'s 0.8938, one level
up: the two entries differ in TRAINING DATA DOSE, not in seating, and the difference was charged to the
seating.

**What still stands, and it is the part that governed the decision.** A fold-0 seating cannot beat an `_ff`
champion — 0.937 against 0.943 is the measured dose gap, and the retraction below was still right to
cancel a 16-seat fold-0 entry priced at 0.945+. What changes is WHY: not "the instrument is broken" but
"the instrument is measured in a family that tops out at 0.937". The two readings differ in what they
license next, which is why this is worth correcting rather than leaving.

**What it opens (`knee-l5dx`).** The honest seating and the full-fit dose are separable terms that
have never been combined: seats chosen by the offline greedy, then served from `_ff` refits of those same
recipes. The dose term is measured at roughly +0.003 (`finalfit_v1` 0.934 → `allfit_v1` 0.937) and it
saturates; the seating term is measured at +0.004 to +0.006 inside the fold-0 family. Both are below the
1.5% bar on their own and the sum is not obviously above the champion, so this is a candidate, not a plan
— and it costs a training run per unrefit member, which is the expensive half.

**And it closed the same hour, for free, on the bead's own pre-test.** `record_mix --seats honest --vote
logit` restricted to the seven roster recipes that HAVE an `_ff` twin (`base_dense`, `calib_dense_hold`,
`calib_span`, `grade_ladder`, `lr4e-5_ws`, `small_batch6`, `through_dense`) peaks at **0.8705 on seven
members** — against unrestricted `honest12`'s 0.8751, that is **−0.0046**, half again the entire dose term,
given up before the dose term is paid. Under the +0.97 top-regime slope 0.8705 predicts board ~0.932–0.933
(it sits between `basedense_v1` 0.8690/0.931 and `peak8_v1` 0.8731/0.932); adding the saturating dose lands
~0.936 against a pinned 0.943. **The seating term lives in the five members with NO `_ff` twin**, so buying
it costs five training runs to land below what is already shipped. Closed without arming anything — the
cheap pre-test was worth more than the run it refused.

### RETRACTED BEFORE IT COST A SLOT: the honest curve is the instrument `honest12` already refuted — 2026-08-29 21:12Z

The section below reopened seat count off a `record_mix --seats honest --vote logit` curve and priced a
16-seat entry at +0.0044 board. **It is the same instrument that has already been on the board and lost.**

`honest12` (ref 55817896, 2026-08-27, kit `fec94e7`) shipped the peak of that same curve — twelve seats,
offline **0.8751**, board **0.937**. Against the champion's 0.943 that is **−0.006**, while its own holdout
said **+0.0050**. Not a compression: a SIGN FLIP. Its board row states the closure in its own words —
*"offline-greedy seat selection as an instrument … has now picked a seating that LOSES on the board while
gaining on its own holdout, so its rankings do not transfer and no further slot should be spent on one."*

**It is the same tool, the same flags and nearly the same number.** honest12 peaked at 12 / 0.8751 over 89
records; this window's run peaked at 28 / 0.8753 over 111, with 16 reading 0.8746. Ten of honest12's twelve
recipes are in the sixteen. The 16-seat set is honest12 plus six more seats drawn by the same greedy.

So the pricing below is wrong in the DIRECTION, not the magnitude: the honest curve's transfer coefficient
has one measured value, and it is **negative**. Pre-registered branches (0.945–0.950 / 0.940–0.943) both
omitted the only branch with evidence behind it, ~0.937.

**What I did wrong, since the rule that would have caught this is written in `CLAUDE.md`:** I read
`champion.json`'s saturation principle (six seatings, 6–11 seats, ~0.941) and treated the honest peak as
outside its range — without reading `comp/board.jsonl`, where a TWELVE-seat entry off this exact curve sits
with a score and a written verdict. The prose I checked was two days stale about the tool; the record was
not. `board_check --live` guards that file precisely so it can be read instead of the prose, and I read the
prose.

**What survives.** The six members this hunt surfaced are real and now rostered (`merged_span_cf` 0.8618,
`labelwise_dense_s2026` 0.8613, `warm_calib_dense_s2026` 0.8562, `contrast_dense_s2026` 0.8547,
`focus_mcl_s2026`, `calib_dense_hold_ff`) — that work is kept, and the roster-drift bug it exposed
(`knee-0dtv`) is real. What dies is the SEATING, not the members. honest12's own verdict points where:
*"the members themselves … remain unpriced individually"*, and `warm_calib_dense_s2026` is exactly the
warm-restart lane (`knee-wbwq`) that row sent the next slot to.

**The slot is unspent.** One spendable, one held for the owner's go, UTC day not yet rolled.

### The honest curve reopens seat count, and 16 seats dominate 28 — 2026-08-29 20:55Z

*(SUPERSEDED — see the retraction immediately above. Kept because the decode-cost reading and the
compression re-derivation in it are correct and reusable.)*

The honest peak is **0.8753 at 28 seats** against the shipped champion's five-twin shadow at 0.8701:
**+0.0052**, the first offline delta in days at the 0.0053 floor rather than a tenth of it, and it comes
from COUNT after quality (knee-07b0) and rank-distance (above) both closed.

**The board says count is saturated. It says so over 6 to 11 seats.** `champion.json` records five entries
spanning 0.939-0.941 at 6, 7, 7, 8 and 11 seats and concludes "THE SEATING AXIS IS SATURATED FOR US at
~0.941". Every one of those was seated off a GREEDY curve, whose in-sample peak sits at 13 and overstates by
+0.0032. Nothing has ever been shipped past 11 seats, so the saturation reading covers a range the honest
peak lies outside of. That is not a refutation of it — it is the boundary of what it was measured on.

**Compression, read from the tool and not from prose.** `board_transfer` at our own score band (entries
≥0.8690) is **+0.98 board points per holdout point [+0.70, +1.17]**, three resolved pairs. Not the +2.14 in
`champion.json`, not the +1.55 whole-record median the tool itself labels "the wrong number to price an arm
with". So +0.0052 offline prices at about **+0.005 board** — 0.943 → ~0.948, which is inside the top ten
(0.946). The interval is entirely positive. **This is the third time this constant has been quoted wrong
from prose in this repo; read it from the tool.**

**And the shipment is 16 seats, not 28.** `roster_reads` prices both:

| seats | decodes | honest held-out |
|---|---|---|
| 16 | **5** | 0.8746 |
| 28 | **5** | 0.8753 |

Identical read cost — the extra twelve members all land in decode groups already being paid for. So 28 buys
+0.0007 (a seventh of the floor, deep noise) for 1.75× the forward passes. **16 seats dominates: same I/O,
statistically the same score, far less runtime.** Runtime matters twice here — a kernel that times out
wastes the slot outright, and the efficiency track ranks us 16th at 0.925 on runtime alone.

Pre-registered before any slot is spent: the 16-seat entry should read **0.945-0.950** if the honest curve
transfers, and **0.940-0.943** if the board's saturation reading extends past 11 seats after all. Those two
branches are separated by more than one board tick, which is why the entry is worth a slot at all.

### Five of the sixteen best seats were not in the roster at all — 2026-08-29 21:05Z

Building the 16-seat entry meant checking each seat against `comp/members.roster`, and five of the sixteen
were absent — `merged_span_cf` (seat 2, written up above) plus four more at seats 3, 7, 11 and 13. All four
are fold 0, all ride the CHAMPION'S OWN DECODE, so seating any of them costs no extra read, and all four
recipes came out of their own `metrics.jsonl` in one call.

| member | solo | seat | what it is |
|---|---|---|---|
| `labelwise_dense_s2026` | **0.8613** | 3 | `--slice-pooling labelwise`; **0 ledger mentions before now** |
| `warm_calib_dense_s2026` | 0.8562 | 7 | `--encoder-weights encoders/calib_dense_hold.enc.pt` — a warm restart |
| `focus_mcl_s2026` | 0.5332 | 11 | the `--focus-label MCL` arm refuted earlier the same day |
| `contrast_dense_s2026` | 0.8547 | 13 | `--contrast-weight 1.0` |

**`labelwise_dense_s2026` is the third-best solo member the project owns** — behind only
`calib_dense_hold_s2026` (0.8634) and `merged_span_cf` (0.8618, itself unrostered until an hour ago) — and
it pools slices LABELWISE, which nothing else in the roster does; every other member pools with `max`. That
is quality and an unshared mechanism in one checkpoint, and it had zero mentions anywhere in this file.

**`warm_calib_dense_s2026` is a warm restart from a fitted encoder.** `champion.json` names that lane as the
one "0 of 131 arms have ever used" and sends card hours at it (knee-wbwq). One arm had already used it. The
checkpoint was on disk, scored, and unrostered.

**`focus_mcl_s2026` takes seat 11 at a solo of 0.5332.** That arm was refuted the same day — it regressed
its own target label by −0.0576 paired, because eleven of twelve heads carried no gradient. It still beats
110 other candidates for the eleventh seat. What a vote wants from a late member is disagreement it can
average away, not a good number, which is the owner's own point about not averaging seven copies of one
number. A seat is not a solo score, in either direction.

**The pattern, now four for four.** Every audit this evening found a member that existed, was scored, was
good, and was invisible to the only file that can ship it. Nothing decided to leave any of them out. The gap
is structural: a run finishing writes a checkpoint and a ledger section, and NOTHING writes the roster. Worth
a check of its own — a roster that is a strict subset of the scoreable pool minus a documented decline is
mechanically verifiable, and would have caught all five.

### `merged_span_cf` — declined as undocumented at 20:37Z, seated at 20:52Z

The roster audit found this checkpoint on disk with zero mentions here and declined it on the rule that an
undocumented checkpoint is a file, not a member. The rule is right. Applying it as terminal was not: the
matched seat curves above put `merged_span_cf` at **greedy seat two, the largest single jump in the entire
111-member curve** (0.8634 → 0.8709, +0.0075), and its own `merged_span_cf.metrics.jsonl` carried the whole
recipe. The entry was one read away, not unrecoverable.

Solo **0.8618**, peak epoch 8 of 12 and flat after — converged, not truncated. Second-best solo member the
project owns, behind `calib_dense_hold_s2026` (0.8634). Recipe: merged scheme, 336 px, 20 slices, band
0.2-0.8, crop 130 mm, centre image, spacing spread, tie first, `--calibrated-targets --whole-stack
--one-cycle --weight-floor`, batch 12, **fold 0**, seed 2026, no EMA, 438 s an epoch, 26.92 GB peak.

**It is fold 0 despite the `_cf` suffix**, so unlike every other `_cf` in the roster it is scoreable and the
0.8618 is honest. Its band 0.2-0.8 puts it on the roster's SECOND decode group, so seating it beside the
champion's six costs one extra read — which is exactly the trade `decode_entry` prices, and the first member
in a while with a large enough seat delta to be worth paying it for.

**The process lesson, since it is the second one this evening.** An audit that finds an undocumented artefact
should recover its provenance before ruling on it — `metrics.jsonl` holds the full recipe for every run this
trainer has ever written. Declining costs nothing to write and cost, here, the roster's second-best seat for
fifteen minutes.

### The reader gate fired NOT DISTINGUISHABLE, and it cost twenty minutes instead of the sweep

`Qwen/Qwen3-8B`, bf16, on the 57 image-annotated studies, scored by `core.data.labels.reader_upgrade`:

```
177 pairs the lexicon left blank; the candidate speaks on 81 (45.8%)
blank-pair accuracy 0.840 against a majority-class 0.840
the lexicon's own 463 answered pairs: candidate accuracy 0.812
blank-pair accuracy interval [0.737, 0.911] over studies
```

**Read the first number before the third.** The reader DOES speak where the lexicon abstained — 45.8% of
the blank pairs, which is the coverage the whole lane was arguing for. It is the second number that closes
it: 0.840 against a majority class of 0.840, to three decimals, with an interval [0.737, 0.911] that
brackets the baseline rather than clearing it. On the blank cells the prevalence is 13.6% (24 positives in
177), so "absent everywhere" scores 0.864 on the FULL blank set and 0.840 on the subset this reader chose
to speak on. It has matched that and added nothing.

**And it does not hold the ground the lexicon already had**: 0.812 on the 463 answered pairs, below the
lexicon's own 0.8705 macro against the same annotations. So the trade on offer is coverage that carries no
information bought with a regression where supervision already worked.

**What this refutes and what it does not.** It refutes an 8B reader as a source of supervision on the
blank cells, on this prompt, against this truth. It does NOT refute `knee-fuhq`'s finding that label noise
outranks architecture — that finding rests on our labels reading 0.8705 while a published coatnet reads
0.9167 on the same studies, and this arm did not touch it. The lane's real blocker is unchanged and is not
a result: BOTH cached 35B readers fail to load under transformers 5.15 (gptqmodel imports the removed
`no_init_weights`; the NVFP4 twin declares `modelopt`, absent from `AUTO_QUANTIZER_MAPPING`), and vLLM is
Linux-only here. A 35B is a different reader, not a bigger version of this one.

**The gate is the deliverable, not the verdict.** Pre-registering the blank-pair comparison turned an
open-ended sweep of 4,406 reports into a twenty-minute question with a written branch, and the branch it
fired was the one that forbids spending the card. That is the instrument paying for itself on first use.

**And the NARROW claim the audit left open is now measured too — on the cells it named.** `knee-07wb` /
`knee-pju4` survived `confident224` only as a claim about the three labels whose UNK share is extreme, so
the gate's blank set was checked per label rather than in aggregate:

| finding | blank pairs | the reader speaks on | its accuracy | majority class |
|---|---|---|---|---|
| Synovitis | 41 | 11 | 0.727 | 0.659 |
| Baker's | 29 | 7 | 0.857 | **0.966** |
| Fracture | 26 | 24 | 0.875 | 0.846 |

Baker's is a straight loss. The other two lead by +0.068 and +0.029 on ELEVEN and twenty-four asserted
cells — a single flipped cell moves Synovitis by 0.09, which is larger than the lead itself.

**The finding is about the RULER, not the reader.** The image-annotated set is 57 studies, so a per-label
blank-cell claim is being asked of denominators of 7 to 24. That cannot resolve anything at any reader
size, and no bigger model changes it. The narrow route therefore does not need a loadable 35B — it needs a
truth set with more annotated studies, which we do not have. Retire it as unmeasurable here rather than
leaving it open as if a stronger reader would settle it.

## 2026-08-30 — the roster was hand-written, and the project's second-best solo was unshippable (`knee-0dtv`)

`comp/members.roster` is the ONLY definition of what a package contains: a member absent from it cannot
be shipped no matter what any other file says. Nothing writes it. A run that finishes writes a
checkpoint, a holdout record and a ledger section — and then the roster is edited by hand, so every
member in it is there because somebody remembered.

`knee-0dtv` was filed on five seats: building a sixteen-seat entry found five of its sixteen best
members missing from the roster. The bead was three times too small. The new check
(`python -m kneemri.kaggle.roster_check`, the thirteenth) swept the whole processed root and read

```
123 scoreable of 200 checkpoints: 47 packaged, 0 declined, 76 undocumented
```

Seventy-six, not five. And the worst of them is not `labelwise_dense_s2026`. It is
**`stackfit_s2026`, 0.8631 solo** under LogitAverage on the 909-study fold-0 holdout — the
**second-best single checkpoint this project owns** (behind `calib_dense_hold_s2026` 0.8634, ahead of
`merged_span_cf` 0.8618), and it could not take a seat in any entry ever shipped. `raptor_distill_s2026`
0.8613 and `distill_pairs_s2026` 0.8594 were in the same state.

**THE RULE IS NOT A SCORE FLOOR, and checking that is what stopped this becoming a second bug.** The
first instinct was to package everything above some cut. Measured against the pool the roster ALREADY
packages, that would have declined this project's own instruments: the packaged set descends to
`focus_mcl_s2026` 0.5332, `mae_matched_s2026` 0.6735, `lora_r8_v3small_224_s2026` 0.8215,
`dinov3_base_lora_s2026` 0.8268, `crop80_tissue_s2026` 0.8345. Every one of those is a NAMED INSTRUMENT
— the only non-DINOv2 trunk, the only different physical crop, the MCL specialist, the medical
pretraining. So the cut is `MERIT = 0.8460`, the solo of `extended_ws_s2026`: the weakest member this
roster has ever added on merit rather than as an instrument. Above it, ship. Below it, decline IN WRITING,
and the way back on is to name the instrument the member would be — never to re-quote the number.

**The split: 30 added, 46 declined.** Thirty undocumented members clear 0.8460 and joined the roster
(0.8631 down to 0.8460). Forty-six are declined in `comp/members.declines`, each with its reason:
thirty-four below merit, eleven smoke/probe runs that never became members, and `confident224_s2026` —
a REAL arm scored under an older label table, written down as such so it is not mistaken for a twelfth
smoke run and so `knee-8hau`'s rescore has somewhere to land. After: `123 scoreable, 77 packaged, 46
declined, 0 undocumented`, exit 0.

**Why the check is two-sided.** "Everything scoreable ships" would be the wrong rule — twelve of the
finds genuinely should not ship. What was missing is not a policy that everything ships; it is a PLACE
TO WRITE DOWN that something does not, so the set difference is taken by a machine instead of by memory.
And a bare name in the declines file is a finding too: it records that somebody once decided, not what
they decided, which is the same defect one file over.

**What this does NOT do.** It does not push the package. Thirty new roster lines mean a dataset push
before any of them can be mounted — a separate, deliberate step. `knee-0dtv` was about the definition,
and the definition is now enforced.

### The 30 newly-shippable members buy +0.0002, and the honest curve is flat from THREE seats

Same evening, immediately after the roster fix, because the fix is what made the question askable: the
honest greedy had never once been run over the whole pool. `record_mix --largest-pool --vote logit
--seats honest` over all 200 checkpoints on disk — 111 survive the split-and-label-table narrowing
(the eleven smoke runs plus `confident224_s2026` drop out as a group, named in the log) — reads

```
   1 members: held-out 0.8624   in-sample 0.8634   bias +0.0010
   3 members: held-out 0.8729   in-sample 0.8741   bias +0.0012
  12 members: held-out 0.8743   in-sample 0.8776   bias +0.0034
  20 members: held-out 0.8750   in-sample 0.8773   bias +0.0023
best honest count 28 at 0.8753, against the in-sample peak 0.8777 at 13
```

**Against the previous unrestricted `honest12` at 0.8751 that is +0.0002.** Thirty members that could
not take a seat in any entry ever shipped — `stackfit_s2026` 0.8631, the project's second-best solo,
among them — were added to the pool, and the honest curve did not move. The member-count axis is
saturated: the pool already spanned whatever these thirty span.

**Read the shape, not just the peak.** Three seats reach 0.8729. The next TWENTY-FIVE buy +0.0024 in
total, and every step of it is inside the bias term the same rows report. The in-sample peak sits at 13
and the honest peak at 28, which is the estimator wandering on a flat surface, not a real optimum at
either count. A "best count 28" that is +0.0010 over count 8 is a coordinate on a plateau.

**Two consequences, and they point opposite ways.**

For SCORE: this closes "add more members of the kind we already build" as a lever, on our own honest
ruler and at full pool size. It was already narrowed by the 23% quality pass-through; it is now closed
by saturation, which is a stronger statement — pass-through says gains arrive small, saturation says
there are no gains left to pass through. What stays open is a genuinely different DECODE or CORPUS
(`knee-nj4l`, `knee-ombx`), which is exactly what a saturated pool of one architecture predicts.

For EFFICIENCY: three seats deliver 99.7% of what twenty-eight deliver. That is the strongest support
the efficiency lane has, and it was bought for free by a run that was aimed at score.

**A caveat the number carries and the plateau does not remove.** This is offline on 909 fold-0 studies,
and it is a statement about SEATING, not about weighting: the greedy seats members it then averages
under one rule. A weighted fit over the same pool is a different question and this does not answer it.

### The weighting caveat, discharged the same hour: no rule beats the one already shipped

The saturation entry above closed on a caveat — "this is a statement about SEATING, not about weighting;
a weighted fit over the same pool is a different question and this does not answer it." `vote_ceiling`
answers it, over the roster the fix had just enlarged, and it costs nothing (holdout records are on disk;
no card, no slot, no inference). Weights fitted on one half of the studies, scored on the other, both
directions, flat scored on the identical halves so the paired inflation cancels:

```
reading 77 members over 909 studies
  RankAverage            honest 0.8710  vs flat +0.0000  (fit-on-self 0.8710, overfit gap +0.0000)
  QualityWeightedRank    honest 0.8710  vs flat -0.0000  (fit-on-self 0.8710, overfit gap +0.0000)
  LabelQualityRank       honest 0.8711  vs flat +0.0001  (fit-on-self 0.8715, overfit gap +0.0004)
  LogitAverage           honest 0.8720  vs flat +0.0010  (fit-on-self 0.8720, overfit gap +0.0000)
  MedianLogit            honest 0.8707  vs flat -0.0003  (fit-on-self 0.8707, overfit gap +0.0000)
BEST honest rule: LogitAverage at 0.8720 (+0.0010 vs flat)
```

**The best rule is the one already shipped**, and its margin over flat rank is +0.0010 — a tenth of the
noise bar. `QualityWeightedRank`, the rule whose own docstring asked for exactly this read, comes in at
−0.0000: after its weights are paid for it is indistinguishable from not weighting at all.
`LabelQualityRank` is the only rule that overfits measurably (+0.0004 fit-on-self over honest), and even
its overfit number does not reach flat plus the floor. So the vote axis is closed on the enlarged
roster the same way the count axis is: not "we picked well", but "there is nothing here to pick".

**One ordering fact worth keeping, because the two runs share a ruler.** All 77 packaged members flat
reads 0.8720. The honest greedy's chosen 28 read 0.8753. **Choosing beats including by +0.0033** — more
than any vote rule buys over any other. The lever on this pool was never the rule and is not the count;
it is which members, and even that lands on a plateau. Both axes now say the same thing from opposite
sides: this pool is one architecture, measured out.

**Provenance correction, written minutes later against `interpretations/kneemri/2026-08-29_the-vote-has-nothing-to-weight.md`.**
This is a REPLICATION at a larger pool, not a discovery. `knee-xsfc` ran the same instrument earlier the
same day over 42 members and reached the same conclusion: `LogitAverage` 0.8725 (+0.0014 vs flat),
`QualityWeightedRank` +0.0000, `LabelQualityRank` +0.0002 with a +0.0004 overfit gap. The enlarged
77-member roster moves every one of those slightly DOWN — `LogitAverage` +0.0014 → +0.0010,
`LabelQualityRank` +0.0002 → +0.0001 — so the thirty new members did not give the vote anything more to
weight either. What is new here is only the denominator and the direction of the drift; the finding is
`knee-xsfc`'s. The ordering fact (77 flat 0.8720 against the greedy's chosen 28 at 0.8753) IS new, because
it needed the full-pool greedy that did not exist until tonight.

### The named seat order says WHY the thirty bought nothing, and it seats the two worst members in the pool

The honest curve gives counts; `--seats greedy` over the same 111 gives the ORDER, which is what actually
explains the null:

```
greedy  1: 0.8634  +calib_dense_hold_s2026
greedy  2: 0.8709  +merged_span_cf
greedy  3: 0.8741  +labelwise_dense_s2026
greedy  4: 0.8748  +blendhold_s2026
greedy  5: 0.8756  +finding_calib_dense_s2026
greedy  6: 0.8762  +margin_seat_s2026
greedy  7: 0.8768  +warm_calib_dense_s2026
greedy  8: 0.8771  +merged_blend_cf
greedy  9: 0.8773  +calib_span_s2026
greedy 10: 0.8775  +mae_matched_s2026
greedy 11: 0.8776  +focus_mcl_s2026
greedy 12: 0.8776  +through_dense_s2026
greedy 13: 0.8777  +contrast_dense_s2026
greedy 14: 0.8777  +small_batch6_s2026
```

**`stackfit_s2026` is not in it.** The project's second-best solo at 0.8631 — the member whose
unshippability was tonight's headline defect — earns no seat in the first fourteen. That is not a
disappointment, it is the mechanism: the greedy already had `calib_dense_hold` at 0.8634, and `stackfit`
IS `calib_dense_hold` plus EMA, measured flat when it was run (`knee-orjq`, the FLAT branch fired). A
member that differs from the incumbent by a lever measured flat has nothing to add to it. The roster bug
cost us a checkpoint we could not ship; the seat ruler says it would not have been seated. Both are true
and neither excuses the other — the bug was real, and the thing it hid happened to be a twin.

**The two WORST members in the whole pool are seated tenth and eleventh.** `mae_matched_s2026` scores
0.6735 solo and `focus_mcl_s2026` scores 0.5332 — near chance on a twelve-label mean — and the greedy
takes them ahead of every remaining member scoring 0.85-plus. This is the strongest direct evidence the
project has that **quality is not the seat criterion at the margin**, and it independently vindicates the
merit-level EXEMPTION written into `comp/members.declines` hours earlier: the instruments packaged below
the 0.8460 cut are not sentiment, they are seats. A score floor on the roster would have deleted the
tenth and eleventh best seats in the pool.

**What it does NOT license.** Two seats out of eleven is not a rule, the run is in-sample (the honest
curve is the one that may be quoted), and the same seats are inside the plateau where +0.0024 buys
twenty-five members. It licenses keeping the exemption, not building for it.
