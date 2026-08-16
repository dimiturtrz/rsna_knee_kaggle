# Submissions

Kernels are named for the METHOD, not for the attempt: `kneemri-<source>-<variation>`. A new version of
one slug is that same method retrained or re-tuned; a different source or aggregation is a different
slug. The slug is the claim, so a grafted score can never quietly become ours.

Budget: **5 submissions per day**. Kernel *pushes* are unlimited — probe there, submit only when the
question needs the leaderboard to answer it.

## The method tuple

A slug alone cannot say what was run: `kneemri-trained` has meant five windowed members with overlap
TTA and three whole-stack members without it. Every entry is therefore a tuple over four closed axes,
and the tuple — not the slug — is what the coverage grid below is indexed by.

| axis | values | what it decides |
|---|---|---|
| **source** | `ours` · `theirs` · `fusion` | whose weights produced the prediction; `fusion` means both, decoded once |
| **members** | `1` · `2` · `3` · `5` · `20` · `23` | how many members are rank-averaged |
| **read** | `windowed` · `whole` | one group of slices per pass, or the whole stack |
| **passes** | `1/study` · `n/study` · `overlap` | test-time cost, and whether windows overlap |

`ours×5×windowed×overlap` is a complete method. Any two entries differing in one coordinate are the
only pair that isolates that coordinate, which is what makes a spend decision arguable rather than a
preference.

## Coverage grid

Five slots expire unspent at 00:00 UTC daily; they cannot be banked. This grid exists so a slot is
never spent on a cell that already has a number.

| tuple | status | LB |
|---|---|---|
| `ours×1×whole×1/study` | scored | 0.868 |
| `ours×2×whole×2/study` | scored | 0.882 |
| `ours×3×whole×3/study` | scored | 0.882 |
| `ours×5×windowed×overlap` | scored | **0.888** |
| `theirs×20×windowed×overlap` | scored | **0.888** |
| `fusion×23×windowed×overlap` | scored | 0.894 |
| `fusion×24×windowed×overlap` | scored | 0.898 |
| `ours×2×whole×2/study` (post-LoRA pool) | scored | **0.919** — best |
| `ours×5×whole×5/study` (post-LoRA pool) | scored | **0.911** |
| `fusion×23×whole×n/study` | **null** — T4 OOM, sample table scored | 0.500 |
| `ours×2×whole×2/study` (res392 + calibrated) | **null** — stale kit, no `Member.precision` | 0.500 |
| `ours×5×whole×5/study` | submitted 2026-08-13, result not recorded here | — |
| `ours×2×whole×2/study` re-run on kit `5ba86ee` | submitted 2026-08-13, result not recorded here | — |
| `fusion×25×windowed×overlap` (our 5 + their 20) | **untried** | — |
| `ours×5×whole×5/study`, five DIFFERENT recipes | **untried** | — |

A `null` is not a floor: neither of those two cells has ever been measured, and re-running one is a
legitimate spend. A cell with a number is not, unless the members underneath it changed.

Through 2026-08-15 every scored pair said the same thing — the best single model this project has
produced (holdout 0.8518) sat 0.020 BELOW five weaker ones, a third member of an identical recipe bought
nothing, and a different recipe bought +0.006 — so diversity was the axis with a number behind it.

**2026-08-16 bounds that reading rather than extending it.** Diversity earned its number while the
members were WEAK and while half of them were somebody else's. Against the post-LoRA all-ours pool, two
of our members score 0.919 and five score 0.911, while twenty-four members drawn from two packages score
0.898 — so above some member quality, adding members is a cost. The rule that survives both eras is
about the SPREAD, not the count: members worth adding are ones that fail differently and score nearly as
well, and the published CC0 members no longer clear the second half of that.

## Ledger

| date | kernel | v | method | local | LB | note |
|---|---|---|---|---|---|---|
| 2026-08-10 | `kneemri-probe` | 2 | passthrough `sample_submission` | — | not submitted | environment + corpus probe |
| 2026-08-10 | `kneemri-graft-verify` | 6 | fingerprint check only | 20/20 @ 4.65e-06 | not submitted | proves the architecture port |
| 2026-08-10 | `kneemri-graft` | 1 | **GRAFT** — 20 published CC0 members, our data layer, rank-averaged | their holdout 0.8398 | **0.888** | **their result, not ours** |
| 2026-08-10 | `kneemri-trained` | 1 | **OURS** — 5 folds fitted by `kneemri.training.train` on report-derived labels | holdout mean 0.8162 | **0.868** | trained with augmentation silently disabled |
| 2026-08-11 | `kneemri-trained` | 3 | **OURS v2** — augmentation actually applied + OneCycle + overlap TTA | holdout mean **0.8269** | **0.888** | **equals the 20-member graft with 5 of our own** |
| 2026-08-11 | `kneemri-trained` | 4 | **OURS** — 2 whole-stack members (fold 0 + fold 1) | 0.8481 | **0.882** | +0.021 per member on the holdout, −0.006 on the board |
| 2026-08-12 | `kneemri-trained` | 5 | **OURS** — 3 whole-stack members | 0.8493 | **0.882** | a third member of the same recipe bought NOTHING |
| 2026-08-12 | `kneemri-solo` | 1 | **OURS** — one model, no ensemble (our strongest member) | **0.8518** | **0.868** | the price of dropping ensembling: −0.020 |
| 2026-08-12 | `kneemri-fusion` | 1 | **FUSION** — our 3 whole-stack + their 20, one decode, rank-avg | mixed | **0.894** | best entry; 20 of 23 models are theirs |
| 2026-08-12 | `kneemri-fusion` | 2 | same 23, whole-stack members read WHOLE instead of windowed | mixed | **0.500** | T4 OOM → no submission written → the deleted `fallback` scored the sample table (knee-d16) |
| 2026-08-13 | `kneemri-efficient` | 1 | 2 strongest whole-stack members, 2 passes/study | 0.8509 | **0.500** | kit 54 commits stale, so the bf16 member ran fp16 and the image budget was missing |
| 2026-08-13 | `kneemri-trained` | 5* | **OURS** — strongest 5 whole-stack members, stamped kit | 0.8482 | *(no score returned)* | one of the three "Notebook Threw Exception" entries (knee-b6p) |
| 2026-08-13 | `kneemri-efficient` | 2 | the same 2 members on the fixed kit | 0.8509 | *(no score returned)* | second of the three exceptions |
| 2026-08-14 | `kneemri-fusion` | ~8 | 22 members, one decode, fp16 serving, a failing member drops itself | mixed | **0.895** | **best entry** — supersedes the 0.894 |
| 2026-08-15 | `kneemri-fusion` | 10 | 24 members (our 6 + the published 20 less the v3 and the nine-slot member) | mixed | **0.898** | superseded by the 0.919 efficient v4 — our own share doubled from 3 members to 6; +0.003 on the 22-member fusion, directional |
| 2026-08-16 | `kneemri-fusion` | 11 | 25 members — the expired-v3 exclusion fixed, so `lora_r8_v3small_224` serves | mixed | *pending* | submitted 05:05 UTC |
| 2026-08-16 | `kneemri-trained` | 8 | **OURS** — the strongest five all-ours whole-stack members, no published weights | 0.8610 | **0.911** | **THE ARM THAT SETTLES THE 0.919.** Ours-only, five members, +0.013 on the 24-member fusion — so the jump belongs to the MEMBER POOL, not to the two-member configuration: two different all-ours entries both clear the mixed fusion by more than the 1.5% bar. Published CC0 members are now a COST, not a contribution. It also puts the ruler and the board on opposite sides: local holdout ranks these five ABOVE the pair (0.8610 vs 0.8581, +0.003) and the board ranks them BELOW it (0.911 vs 0.919, −0.008). |
| 2026-08-16 | `kneemri-efficient` | 4 | **OURS** — `strongest(2)` whole-stack members, 2 passes/study, no overlap | — | **0.919** | **best entry, and the first that is entirely ours** — +0.021 on the 24-member fusion, so it clears the 1.5% bar as a RESULT, not a directional move. TWO of our own members beat twenty-four mixed ones at a twelfth of the forward passes, which inverts the diversity reading every earlier pair supported. Read it as a QUESTION before a finding: this kernel has never returned a valid score before (v1 scored 0.500 on a stale kit, v2 threw), so the 0.919 is the first measurement of the cell, and what changed underneath is the MEMBERS — `strongest(2)` now ranks over the post-LoRA pool, not the August-13 one. The comparison that settles it is the pending `kneemri-trained` v8 (all-ours, more members) against this. **SETTLED the same day: v8 scored 0.911.** The post-LoRA all-ours pool is what carries the jump — both all-ours entries clear the fusion — and the two-member configuration is worth a further −0.008 on top, which is directional, not a result. |

## What counts as ours

The graft row is the field's floor measured on our harness. It is **not a result**: the weights are
`pilkwang/rsna-knee-weights` (CC0), and the only thing of ours in it is the data layer and the ensemble.
It exists so that every later number has something to be compared against, and so the submission path
was proven before a model depended on it.

`kneemri-trained` is ours end to end — our slot routing, our ordering, our laterality, our crop, our
head, our fitting. It starts below the graft, which is the honest starting position for a replication
plus one day's work.

## The two numbers, side by side

| | members | epochs each | per-member holdout | public LB | rank of 1061 |
|---|---|---|---|---|---|
| graft (their weights) | 20 (5 folds x 4 seeds) | 60 | 0.8398 | 0.888 | 349 |
| ours (one day) | 5 (5 folds) | 16 | 0.8162 | 0.868 | 393 |

**0.020 apart, from an ensemble four times smaller trained for a quarter as long.** The public median
is 0.819, so both clear it; the leader is 0.942, so neither is close. Rank 349 with the field's own
weights is the tell that a graft scores where forks score — 86 teams sit within 0.005 of it.

The comparison is only this clean because both sides train on the SAME public label tables and run
through the SAME data layer. Members and epochs are the two axes we have not spent, and they are cheap:
27 s per epoch on the 5090 over the memory-mapped cache. Spending them is the obvious next move and
also the LEAST informative one — it buys rank without teaching anything. The labels are where the
un-shared edge is.

## What 0.888 bought

The graft is not only a floor — it **prices our data layer in leaderboard units**. The weights are the
same CC0 members the public forks run, so the difference between our 0.888 and what those weights score
in other hands is our pipeline's reading of the pixels.

**Correction to the first reading of this.** 0.899 was taken from a public notebook's title and called
"the plateau"; on the actual leaderboard 0.899 is about 100th place and the frontier is 0.942. So the
0.011 gap is to 100th place, not to the people running these weights best, and the real distance to the
top is 0.054 — far too large to be the slot tie-break. Something structural separates the leaders from
the public stack.

That agrees in direction and rough size with the 0.025 out-of-fold residual measured locally, which for
three days made the slot tie-break the lead: when two candidate series have equal slice counts, a stable
`sort_values` picks the first in FRAME order and our `max()` picks the first in LIST order.

**The ORDER is refuted (2026-08-14), by one CSV read** — and that is less than it first appeared.
`Study.from_directory` probes `sorted(directory.iterdir())` and the directory names ARE the series UIDs;
`train_series.csv`'s frame order is that same sorted-UID order for **4,407 of 4,407 studies**. So the two
pipelines see a slot's candidates in the same order, and order cannot be the source of a divergence.
`SlotTies.disagreeing_order` measures it and the CLI prints it first.

**That was written up as "the tie-break is refuted", which overstates it.** The step from "same order" to
"same choice" assumes their rule is *first maximal by slice count* — an inference recorded here, never
verified, because **we have their checkpoints and their config but not their router**. A rule that breaks
ties by anything else (the last maximal, SeriesNumber, acquisition time) diverges no matter how the
candidates are ordered.

And the rebuilt oracle's own covariates argued it does: over 24 studies the residual correlated **+0.664
with contested slots**, the strongest of four routing covariates, surviving control for series count
(**partial +0.496, p ≈ 0.009**). The population whose divergence was declared impossible was the one that
best predicted where the two pipelines disagree.

**Their rule is LAST-maximal; ours is first.** Decoding the same studies under each rule and scoring a
published member against its OWN recorded predictions:

| | 16 studies WITH a contested slot | 40 studies, unselected |
|---|---|---|
| `tie=first` (ours) | 0.1277 / 0.0236 | 0.1205 / 0.0094 |
| `tie=last` | **0.0252 / 0.0061** | **0.0671 / 0.0062** |

Replicated on **4 of 4 published members across 3 seeds**. Once the rule is corrected the routing
signature disappears entirely — `corr(residual, series_count)` falls from **+0.647 to −0.070** — so the
series-count correlation *was* the tie-break and nothing else.

**It is one contributor, not the cause.** On the unselected sample the mean falls by a third while 32 of
40 studies still disagree above 0.01, against the 1e-6 a faithful pipeline reaches.

**Ten axes are now ruled out and our decode matches theirs on every one**: slice order (their
`rules.order = 'normal'` is what we already do), percentile values and scope, DICOM windowing (they apply
none — VOI LUT is far worse), crop rounding, band sampling and its dedupe, the resize filter, and the
disjoint-window inference policy. Several of those were assumptions recorded in this file and never
tested.

The rule is plumbed as a per-member reading axis (`Routing(scheme, tie)`), because our members were
FITTED under first-maximal — a global switch would fix the graft and break everything of ours. The cost
of serving each side correctly is that `Ensemble.reads` groups by it, so a mixed entry decodes the test
set twice.

**And no declared reading axis differs either.** The published members' own checkpoints record
`crop_mm 130.0`, `band 0.20,0.80`, `img 336`, `slices 12`, `group 3`, `pool cls_mean`, the identical slot
ORDER, and byte-for-byte the `rules` dict `member_record.py` hardcodes. So the residual lives in the
IMPLEMENTATION of those rules — decode, percentile normalisation, resize — not in any recorded policy.
`kneemri.evaluation.oof_oracle` is rebuilt and pointed at exactly that, reporting per-study residuals against free
covariates so a defect that concentrates says so.

## What the six scored entries say, read together

| entry | members | per-member holdout | public LB |
|---|---|---|---|
| one model, no ensemble | 1 | **0.8518** (our best ever) | 0.868 |
| whole-stack | 2 | 0.8481 | 0.882 |
| whole-stack | 3 | 0.8493 | 0.882 |
| ours v2, old windowed recipe | 5 | 0.8269 | **0.888** |
| the published CC0 members | 20 | 0.8398 | **0.888** |
| **ours 3 + their 20** | **23** | mixed | **0.894** |

Built to isolate one variable, and they do. **The best single model this project has produced loses to
five much weaker ones by 0.020** — more than whole-stack (+0.0147) and more than every knob combined.
And a third member of our OWN recipe bought nothing (0.882 → 0.882) while a different recipe bought
+0.006, so it is DIVERSITY and not count. Members that fail the same way do not help each other.

That is also why the holdout cannot pick an entry alone: the 5-member set is +0.021 WORSE per member
than the 3 whole-stack set and scored +0.006 better.

## v2: our five members equal their twenty

| | members | public LB | rank of 1077 |
|---|---|---|---|
| graft (their weights) | 20 | 0.888 | 367 |
| ours v1 | 5 | 0.868 | 412 |
| **ours v2** | **5** | **0.888** | **367** |

**+0.020 on the leaderboard, 45 places, from three defects in our own code** — augmentation never
called, no learning-rate schedule, and the TTA default left off. No new idea, no bigger model, no
extra data.

The graft has since stopped being a floor and become a *control*: same data layer, same ensembling,
same submission path, differing only in whose weights and how many. Matching it with a quarter of the
members says our per-member training is now roughly on par with the published one — which is what the
holdout was already saying (0.8269 against their reported 0.8398, on different splits) and is now
confirmed on the hidden test, where neither of us chose the studies.

The remaining 0.054 to the leader is untouched by any of this, and nothing measured so far explains
it. It is not the schedule, the augmentation, the labels, the slice count or the TTA.

## What the ablations say (fold 0, matched; ruler held at pilkwang)

|  | pilkwang labels | v4_blend labels |
|---|---|---|
| geometric augmentation | **0.8202** | 0.8194 |
| + intensity augmentation | **0.8221** | 0.8203 |

Both imported ideas are null: intensity augmentation +0.0014 on average, better labels −0.0013, no
interaction. Set against what fixing our own code was worth — augmentation never called +0.0110, the
missing schedule +0.0061, the TTA default +0.0035 — **repairing what we had beat importing what others
had by an order of magnitude.**

That is not an argument against reading the field: the reading is what found the schedule, and the
`.903` notebook is what made label quality checkable at all. It is an argument against assuming a
published gain transfers. Every imported idea here either needed adapting or died on a mechanism
specific to this architecture — channel-stacked slices, side-defined targets, a ViT that is not
translation-equivariant, readers that are not independent.

Intensity augmentation stays behind `--intensity-aug` rather than being deleted: the holdout is drawn
from the same corpus as training, so it is structurally blind to the cross-scanner axis that transform
exists for. It is worth one submission slot when a slot is being spent anyway — a claim that is only
honest because it was written down before the result.

## The two oracles

Anything claiming to reproduce the published pipeline has to pass both, because they check different
things and neither implies the other:

- **Fingerprint** (`kneemri-graft-verify`) — each member's output on a seeded synthetic bag. Proves the
  ARCHITECTURE is the function those weights were fitted to. Blind to the pixels: it feeds noise.
- **Out-of-fold agreement** (`python -m kneemri.evaluation.oof_oracle`) — our pipeline against the member's own
  stored predictions on the studies it held out. Proves the PIXELS: routing, order, laterality, crop,
  percentile, resize. Last measured max|diff| 0.025, per-study correlation 0.99, under disjoint windows.

The residual 0.025 is an OPEN QUESTION, not a rounding error — a faithful pipeline would agree to about
1e-6, as the fingerprint does. The tie-break explanation is **refuted** (above), and so is every
recorded reading rule: the published checkpoints declare the same crop, band, slot order and rules we
use. What is left is the IMPLEMENTATION of those rules.

**This oracle was lost and rebuilt, which is why it is named as a module and not a script.** It lived in
`scratchpad/oof_oracle.py`, was never committed, and vanished with the weights it ran on — so for three
days two beads argued about a number nobody could recompute, and both were settled by facts sitting
inside those checkpoints the whole time. It reports per-study residuals against free covariates now,
because a max|diff| says a defect exists and nothing about where.

## Provenance warning on the live entry

The 0.888 submission's five members were trained and epoch-selected against a report-derived label
table that is **no longer reproducible** — it was not on disk anywhere and had no fetch step (see
`comp/README.md`). The members themselves are fine and the leaderboard score is real; what cannot be
reproduced is the holdout number that chose their epochs, or any comparison between them and an arm
trained today.

Anything retrained from here uses `pilkwang/rsna-knee-llm-labels` → `report_labels_v2.csv`, which is
fetched rather than remembered. Do not compare a new member's holdout against the old ones'.

## 2026-08-13: the all-ours weights finally reached Kaggle

Three nights running, the ledger recorded no submission at all — five slots expire unused at 00:00 UTC
every day, and they cannot be banked. The reason was a gate misapplied: an entry is supposed to be
justified offline by `member_curve` first, but that gate needs a GPU pass to produce predictions, and
"the GPU is busy" was treated as if it blocked SUBMITTING. It never did. The scored run happens on
Kaggle's T4.

| entry | members | passes/study | LB |
|---|---:|---:|---|
| `kneemri-efficient` | 2 (res392 0.8523 + calibrated_targets 0.8495) | 2 | **0.500 — null, see below** |
| `kneemri-trained` v6 | 5 all-ours whole-stack | 5 | pending |
| `kneemri-efficient` v2 | the same 2, RE-RUN on kit 5ba86ee | 2 | pending |

The members dataset had been stuck at 2026-08-11 — `res392`, `calibrated_targets`, `sliceattn`,
`intensity392` and `extended_ws` existed only on this machine. It now carries all six (498 MB), which is
also why `strongest(2)` selects a different, untested pair rather than repeating the 0.882 fold pair.

THREE SLOTS SPENT, TWO LEFT, and `kneemri-fusion` v4 was still running when the machine slept — its
output can be submitted whenever, so it becomes the next day's first entry rather than a loss. Which is
the operating fact worth keeping: a run is not tied to the day it started, only a SLOT is.

**That 0.500 is a NULL RESULT, not a floor, and the stale kit is the whole cause.** The OOM fix that
bounds the forward pass by IMAGES rather than studies landed at 2026-08-12 13:11 UTC (`9f31b35`); the
mounted kit was built at 09:46 UTC the same day, before it. So the run served whole-stack members
without the budget split, blew the T4's 16 GB exactly as the 08-12 entry did, and the fallback wrote a
constant table. The members were never measured. The same stale kit also cost `Member.precision`:

**Read that entry as a floor, not a score.** The mounted kit was built 2026-08-12 09:46 and HEAD was 54
commits ahead, so it lacked `Member.precision` — and `calibrated_targets` is a bf16 member, served under
fp16 on a T4 whose exponent range is narrower. Found mid-flight, too late to stop the run. The kit is
rebuilt and `build_kit` now stamps its commit into `kit.json`, so the drift is visible next time.

**Two kernels were one push away from spending a slot on a crash.** The package now contains a DINOv3
LoRA member, while `kneemri-trained` ran EVERY member and `kneemri-fusion` fused every member — both
mount DINOv2 only, with internet disabled. That is exactly how the 0.500 entry was spent. `trained` is
capped at `strongest(5)` and `fusion` now drops variants whose weights are not mounted.

## Gotchas paid for in full

- **Submission is notebook-only.** `kaggle competitions submit -f submission.csv` returns 400. The
  working form is `-k <owner>/<kernel> -v <version> -f submission.csv`.
- **Search for the artefact, not the container.** Three kernel runs died on mount-shape assumptions:
  the competition sits under `/kaggle/input/competitions/<slug>`, datasets under
  `/kaggle/input/datasets/<owner>/<name>`, and a `--dir-mode zip` upload is expanded on mount with its
  wrapping directory REMOVED.
- **Probe what you IMPORT, not only what you plan to use.** The runtime probe checked pydicom, timm and
  transformers (all present) and never checked `jaxtyping`, which our own modules import at module top.
- **A published checkpoint is a training artefact**, not a bare state dict: weights sit under `model`
  beside epoch, fold scores, config, out-of-fold predictions and its own fingerprint.
- **bfloat16 is EMULATED on the T4, and it costs about 5x.** `torch.cuda.is_bf16_supported()` returns
  True there, which reads as "fine" and is not: its own source returns True immediately for compute
  capability >= 8 and otherwise falls through to an emulation check. Turing has fp16 tensor cores and no
  bf16 ones, so bf16 matmuls leave the fast path. Measured on the card, a chunk of 48 studies at 336 px:
  **12 s in fp16 against 58 s in bf16**, same architecture, same resolution. Serve bf16 members in fp16
  on anything below sm_80 — `Ensemble.arithmetic` does, and every entry that has ever scored ran fp16.
- **A kernel that only LOADS is not a submission and costs no slot.** `kneemri-loadcheck` mounts the kit
  and the members, then loads, forwards and sends a 48-study chunk per member. Every runtime fact that
  cost a slot on 2026-08-13/14 — a class mismatch from a transformers version, an emulated dtype, a
  memory ceiling — is answerable there in twenty minutes for free. Run it after any kit rebuild.
- **Kaggle allows two concurrent GPU sessions ACROSS ALL PROJECTS.** On 2026-08-13 both were held by
  the sibling celltrack campaign, so kneemri runs serialised behind them: submissions are limited by
  SESSIONS, not by the five daily slots. A completed kernel version can be submitted at any later time,
  so a long run started before a machine sleeps is not wasted — it is submitted the next day.
- **READ THE ERROR BODY. The CLI throws it away, and a whole evening was spent on the theory it left
  behind.** `kaggle competitions submit` prints `400 Client Error ... CreateCodeSubmission` and nothing
  else. The body says:
  `{"error":{"code":400,"message":"This competition requires an output FileName for Notebook Submissions."}}`
  — the submit was missing `-f submission.csv`, which for a CODE submission names the notebook's OUTPUT
  file, not a local path to upload. Every 400 on 2026-08-15/16 was that one argument.
  I diagnosed it instead as "one code submission may be PENDING at a time, per competition" — the 400s
  did coincide with a pending entry, a second kernel got the same 400, and the story explained every
  observation. It was still wrong, and four of that day's five slots expired at 00:00 UTC (slots do not
  roll over) waiting for a queue that was never blocking. Thirty retries at 110 s apart tested the same
  wrong argument thirty times. **A retry loop is not a diagnosis** — if the first attempt fails without
  naming a cause, get the cause before automating the attempt:
  ```python
  api.competition_submit_code(None, "probe", "<comp>", "<owner>/<kernel>", <version>)  # raises with .response.text
  ```
  What IS measured about the queue: Kaggle reruns the notebook on the hidden test set before scoring, so
  a `PENDING` entry stays pending for HOURS (the 21:06 entry scored at about 05:00, ~8 h). And three
  entries submitted back to back at 05:05 were all accepted, so a pending entry blocks nothing.
- **`machine_shape: NvidiaTeslaT4` gives two T4s and only four CPUs.** On a decode-bound pipeline the
  CPU count binds first: measured 12 ms per DICOM slice off the mount.
