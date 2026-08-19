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

**`--cache-studies N` — the smoke for a recipe whose cache does not exist yet.** `--studies` subsamples a
cache that is already on disk, so an arm on a reading nothing has decoded could not be smoked at all: the
only way to reach the first assertion was a two-hour full build. `--cache-studies N` decodes the first N
studies of the split, and it is in the cache NAME (`_first{N}`), so a partial decode can never be written
over the full arrays every member on disk was fitted on — they differ only in one axis length, with no
shape change to notice. Both CLIs take it, from `CacheFlags`, so the build and the run cannot disagree;
the run stamps it, so a member fitted on a truncated cache re-reads its own 60 studies rather than the
full 4407.

```bash
uv run python -m kneemri.corpus.build_cache --cache-studies 60 --band-low 0.1 --band-high 0.9 --workers 20
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
| `slice_attention` | attention over slices, not max | 0.8473 | +0.0016 | kept as a member |
| `slotmean` | pooling across SLOTS by mean instead of attention (`--slot-pooling mean`) | 0.8406 | −0.0051 vs `wholestack_s2026` | **found unledgered 2026-08-19** — a full 12-epoch arm on an axis this table records nowhere else, sitting on disk since it ran. Directional against attention, not a refutation, and the reasons are on the comparison rather than in it: the reading is on the **UNPAIRED** bound because the control kept no per-study predictions, and `run_comparison` reports the control as predating the `slot_pooling` option entirely — so "the control ran attend" is read off the era's default, not off its record. No label clears its interval; the largest rows are Lateral Meniscus −0.0187 and Baker's −0.0162, both regional and focal moving together (focal −0.0051, regional −0.0052), which is a uniformly slightly worse model rather than a redistribution. Attention over slots stays the default it already was; what this row buys is that the axis is now MEASURED once instead of assumed |
| `wholestack_folds` 2 | same recipe, fold 2 | **0.8518** | — | best single member; epoch-5 partial |
| `slices16_ws` | 16 cached slices, whole-stack | 0.8452 | −0.0005 | **refuted**, 33% slower for nothing. TWO runs share this name: the one quoted here (epochs 0–11, peak 0.8452 at epoch 11, and the checkpoint on disk is its epoch 11 at 0.84518) and a later re-run that was KILLED at epoch 8 having peaked at **0.8411**. Both sets of epochs are in one metrics file because runs used to append; every reader that took the last group quoted the dead run's 0.8411 for a member that scores 0.8452 (knee-1m2g). `RunMetrics.begin` now rotates a previous run's epochs aside, so no file written after 2026-08-19 can hold two runs |
| **`band_wide`** | slice band 0.20–0.80 → **0.10–0.90**, 16 slices at the control's mm-per-slice | **0.8553** | **+0.0028 vs `rate_wholestack`** | best single member yet; directional on the mean, and the per-label pattern is the finding |
| **`band_dense`** | the wide band at restored central density: 0.10–0.90, **20 slices** | **0.8578** | **+0.0053 vs `rate_wholestack`**, +0.0025 vs `band_wide` | best single member; **PF OA and Synovitis resolve** — first per-label rows in this family whose intervals clear zero. Dilution confirmed and symmetric: the centre comes back, the collaterals `band_wide` bought largely go. Pairs with `band_wide` at 0.8633 |
| **`depth_regions`** | region pooling (`--slice-regions 4`) on the wide dense band | 0.8549 | −0.0029 vs `band_dense` | mean is a non-result under the 0.0062 floor; the per-label read is the finding. Two labels clear their interval and disagree: **PF OA −0.0193\***, which is `band_wide`'s −0.0194 to four decimals (the patellofemoral loss is the BAND, not the pooling), and **Lateral Meniscus +0.0165\*** against `band_wide`'s +0.0007 (the only thing attributable to the pooling). Its own written prediction — centre holds, periphery recovers — is wrong on both clauses. [interpretation](../interpretations/kneemri/2026-08-19_depth_regions.md); `depth_regions_narrow` is the single-variable follow-up |
| **`depth_regions_narrow`** | the same region pooling on the band every OTHER member reads (0.20-0.80, 12 slices) | 0.8554 | +0.0029 vs `lr4e-5_ws` | the transfer question, ANSWERED IN TWO PARTS. The PF OA loss was the BAND: on a narrow band it does not merely vanish but reverses, **+0.0067**, confirming `depth_regions`' decomposition. The Lateral Meniscus gain does NOT transfer at strength: **+0.0056 [-0.0112, +0.0206]** here against +0.0165\* there, and read directly against `depth_regions` it is **-0.0112**. Both arms are flat on the mean, which is this arm's own pre-registered refutation condition for the pooled-budget argument. Seat value **+0.0006** against the shipped four. See below |
| **`band_density`** | the fourth corner of the span × density square: 20 slices at the CONTROL's 0.20–0.80 span, so density moves alone | 0.8540 | **+0.0014 vs `lr4e-5_ws`** | **the additive decomposition of `band_dense` is refuted.** Density alone buys +0.0014 — a quarter of the floor — and the per-label signature is not a weaker version of `band_dense`'s, it is a different one. `band_dense` resolved **PF OA +0.0192\*** and **Synovitis +0.0218\***; holding the span, those two rows *fall* (PF OA −0.0076, Synovitis −0.0140) and the two rows that clear their intervals are **Lateral Meniscus +0.0191\*** and **ACL +0.0179\***. So the central/OA gain is the SPAN or the span×density interaction, never the extra slices, and the arm's own pre-registered refutation condition — a collateral gain — is what landed. Consistent with the fixed representational budget: slices added inside an unchanged span redistribute toward focal ligament and meniscus rows at the centre's expense. Seat value **+0.0002** against the shipped four, the smallest any same-split candidate has posted. `band_span` (0.10–0.90 at 12 slices) is the matching span-alone corner |
| **`encoder_layer 9`** | the head reads ViT block 9 of 12 instead of the last one | 0.8230 | -0.0227 vs `wsfold0_e6` | 3.7x the floor, converged (flat over its last three epochs), flag verified in the recipe. Seat value **-0.0034**, the largest negative any same-split candidate has posted, so diversity does not rescue it. The linear-probe convention turns out to be right for this head. See below |
| **`focal_pool`** | pooled vector gains a third part: the mean of the top eighth of each channel over the patch grid | 0.8432 | -0.0025 vs `whole_stack` | Flat (inside the floor). Seat value -0.0012. Pre-registered clause half-holds: focal group -0.0004 vs diffuse -0.0065, the predicted direction, but carried by the diffuse group losing rather than the focal group gaining. Fracture +0.020, directional only (one crossing in twelve). See below |
| **`unfreeze_4`** | four of the encoder's twelve blocks adapt instead of six — the +25% throughput arm, run at length | 0.8320 | −0.0137 vs `wsfold0_e6` | **refuted**, 2.2x the floor, flag verified (`unfreeze_last 6→4`). Peaked at epoch 6 and flat for five epochs after, so it is converged, not short. Eleven of twelve labels negative, six clearing their interval; by structure scale **focal −0.0170 vs regional −0.0091**, which is the arm's own stated risk — the fine findings are the ones that needed the deeper blocks. Seat value **−0.0017**, so diversity does not rescue it either. The default stays six; the speed was real and is not worth 0.0137. See below |
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
Resolution at CONSTANT field of view is untouched (`res392`, 0.8523, still our best single member); what
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
