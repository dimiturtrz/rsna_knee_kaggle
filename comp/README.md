# Kaggle submission — kneemri

Competition: [`rsna-knee-abnormality-detection`](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).
Deadline **2026-10-22**. Score is the **unweighted mean of twelve per-label ROC AUCs**.

This directory is named `comp/`, not `kaggle/`, on purpose: a top-level `kaggle/` directory shadows the
installed `kaggle` package on `sys.path[0]`, so `import kaggle` and `python -m kaggle` resolve to the repo
instead of the library. The sibling project hit that and had to call the venv entrypoint by path forever
after.

## Provenance

`survey.py`, `publish.py`, `build_kit.py`, `verify_kit.py` were copied from `biohub_kaggle/kaggle/` and
retargeted at four constants (kit dataset id, packages, kit path, competition slug). They are a **copy on
purpose** — the second consumer is what reveals which parts are genuinely competition-agnostic, and the
extraction into an installed `sdlc-kaggle` package waits on that evidence. Fix a bug in both, and note it
here, until they are one thing.

## The label table is an external dependency, and it needs fetching

We do not train on `train.csv`'s annotations — only **58 of 4,407** studies carry them. We train on a
report-derived table with a confidence per (study, target), published as a Kaggle dataset:

```bash
kaggle datasets download pilkwang/rsna-knee-llm-labels -p "<processed>/labels" --unzip
```

That produces `report_labels_v2.csv`, whose `<target>` / `<target>__conf` columns are exactly what
`Targets.load` reads. **Run this before any training run**; a missing table is a hard failure at
`pd.read_csv`, not a degraded run.

This step was missing, and the cost was real: the whole measured ablation ledger was scored against a
table that no longer existed anywhere on disk, so its ruler could not be reproduced and no new arm could
be compared against it. An external input with no fetch step is not a dependency, it is a memory.

## The two auth paths

- `survey.py` speaks the REST API directly with a `KGAT_*` access token from `KAGGLE_TOKEN` (env-only —
  never written to `~/.kaggle/`, never committed). `publish.py` did too and was **retired 2026-08-12**:
  its transport had been dead since the token expired on 2026-07-23, and its one irreplaceable
  behaviour — refusing to push while the kit differs from the repo — now lives in `push.sh`, on the
  path we actually use.
- The official `kaggle` CLI reads `~/.kaggle/kaggle.json` and works for reads (`competitions files`,
  `kernels list`, `kernels pull`). Both are live on this machine; pick by which credential you have.

## Spending a submission

**Every entry is justified OFFLINE first.** `kneemri.evaluation.member_curve` scores every subset of members on
the shared fold-0 holdout and costs no slot; every arm is fitted on that same split, so any candidate
ensemble is rankable before it is submitted. A slot is spent only when the offline number predicts a
gain over **the best entry already on the board (0.919, 2026-08-16)**, or when the question cannot be
answered offline at all — holdout-to-LB transfer for a recipe that has never been there.

**The two numbers have MERGED, and that is the finding.** Ours-best and board-best used to differ —
0.888 from our own weights against 0.895 from a 22-member mostly-grafted fusion — so a candidate could
beat ours and still move us down. Since 2026-08-16 the best entry on the board IS all-ours (0.919 from
two members, 0.918 from four), and both clear the mixed fusion by more than the 1.5% bar. Published CC0
members are now a COST, not a contribution; there is one bar to beat and it is 0.919.

One caveat on the tool itself: `member_curve`'s subset ranking was partly decided by corpus order until
2026-08-13, because the AUC broke ties by array position and rank-averaging ties ~20% of studies at two
members. Fixed (midranks) and re-run on 2026-08-16 over eight cached members, which is where the 0.8612
four-member joint pick comes from. The pre-fix output, including the 0.8601 five-member peak, is dead —
it was ranked on the broken tie-break and no number from it should be quoted.

Two mechanics make this cheaper than it sounds:

- **Unused daily slots do not roll over.** Five per day, reset at 00:00 UTC. Abstaining banks nothing;
  it only avoids waste. The limit that actually binds is the **two Final Submissions** that get judged
  (rule 2.2.b), and those are chosen deliberately at the deadline, never by "the last two submitted".
- **The count curve is already measured**, so the leaderboard no longer has to tell us that more
  members help — one model at 0.8518 scored 0.868 against five weaker ones at 0.888 (`knee-9pl`). The
  open question is WHICH members, and the offline curve answers exactly that.

## Method (lib-as-dataset + light kernel)

1. **`kneemri-kit` dataset** (`dimiturnt/kneemri-kit`) — our `kneemri`/`core` source tree, trained weights,
   and wheels for whatever is missing from Kaggle's base image. No code is duplicated into the kernel.
2. **A light script kernel** mounts the kit + competition data, `pip install --no-index --no-deps *.whl`,
   adds the source root to `sys.path`, imports the pipeline, writes `/kaggle/working/submission.csv`.

```bash
uv run python comp/build_kit.py --wheels <dir>         # rebuild FIRST, every time kneemri/ changes
uv run python comp/verify_kit.py                       # fails if the kit drifted from the repo
cd comp && kaggle datasets version -p kit -m "<message>" --dir-mode zip
```

**Rebuild the kit whenever `kneemri/` changes, not only when the weights change.** A kernel run was
lost to `ModuleNotFoundError: No module named 'kneemri.kaggle.submission'`: the kernels had been refactored
to import a newly extracted module, the MEMBERS dataset was pushed, and the kit was not.
`verify_kit.py` compares the kit against the repo and catches this in a second — it was skipped
because the change felt like it was "only in the kernels", which is backwards. The kernels are a
bootstrap that imports the kit, so a change to either is a change to both.

**`build_kit.py` rebuilds the kit directory from scratch, so run it WITH `--wheels`.** Running it
without produced a kit whose source was perfectly in sync and which could not import anything, because
the jaxtyping wheels were gone — and `verify_kit.py` called that "safe to submit", since it only
compared source. It now checks the required wheels too, which is what makes the gate below mean
something.

**Which credential actually works (2026-08-11).** Two live in `~/.kaggle/`:

| file | what it is | state |
|---|---|---|
| `access_token` | the `KGAT_*` bearer token `publish.py` uses | **expired 2026-07-23** |
| `kaggle.json` | username + key, for the official CLI | **works** |

An expired bearer token gets Kaggle's marketing HTML rather than a 401, so `publish.py` fails with a
wall of markup and exit 1. The official CLI authenticates fine, which makes `publish.py`'s founding
claim — that the CLI "cannot run here at all" — false today. See `knee-cws`; do not delete that module
reflexively, its value was the stale-kit refusal, not the transport.

```bash
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
bash comp/push.sh dataset "$(uv run python -c \
  "from kneemri.inference.members import MemberPackage; print(MemberPackage.shipped_root())")" "<notes>"
bash comp/push.sh kernel  kernels/kneemri-trained
cd comp && kaggle competitions submit rsna-knee-abnormality-detection \
  -k dimiturnt/kneemri-trained -v <version> -f submission.csv -m "<what changed>"
```

**A code submission needs an explicit kernel `-v <version>`.** Omitting it — which the Python API
allows, `competition_submit_code(..., kernel=...)` with no `kernel_version` — answers **403 Forbidden**
from `CreateCodeSubmission`, not a message naming the missing field. The same 403 shape also comes back
for a competition slug that does not exist, so check the slug before assuming a permission problem:
ours is `rsna-knee-abnormality-detection`, and a wrong guess made `ListSubmissions` 403 and
`kaggle competitions list -s <guess>` return "No competitions found" while the credential was fine.

**Never hand the dataset CLI a path with a separator in it.** `-p comp/members` failed with a mangled
temp path (`comp/members_manifest.json.json`) — the CLI concatenates directory and filename on Windows.
`push.sh` is why the absolute package path above is safe: its dataset branch `cd`s to the package's
parent and passes the bare directory name.

**There is exactly ONE members directory**, `MemberPackage.shipped_root()` — `<processed root>/all_ours`. The in-repo
`comp/members` was deleted on 2026-08-18: two directories stamped the same dataset id, only one was ever
uploaded, and the other aged to seven members while the live package held nine, which made
`mount_check`'s default invocation refuse a valid kernel (knee-n0w0).

**A member's SLICE AGGREGATION travels with it, and this was verified by breaking it (2026-08-11).**
`members.py` reconstructed every member with `StudyDesign(pool=...)` and never passed `slice_pooling`,
so a member fitted with `--slice-pooling max` came back with the default channel stacking. Checked
against a real checkpoint (`max12_s2026.pt`):

| reconstruction | max&#124;diff&#124; vs its stored fingerprint | outcome |
|---|---|---|
| with `slice_pooling` (fixed) | 6.32e-06 | passes, against a 2e-3 tolerance |
| without it (the old path) | **2.25** | refused |

A 2.25 discrepancy is not numerical noise — the two aggregations are different functions on different
pixels. The fingerprint gate did its job, so the failure was loud rather than a wrong submission; but
it made every member of the current best recipe unsubmittable until the accessor existed.

**This is now a gate, not advice.** `publish.py` runs the drift check in `main`, before authenticating
and on every subcommand, so a stale kit cannot reach Kaggle by either route — including a members-only
dataset push, which is how it escaped the first time. It fired on its first real run.

## Open questions — settle before building around them

- ~~**Is submission notebook-only?**~~ **Settled: yes.** `kaggle competitions submit -f submission.csv`
  returns 400 here too; the working form is `-k <owner>/<kernel> -v <version> -f submission.csv`. The
  kit-as-dataset machinery is therefore load-bearing, not optional. Recorded in `SUBMISSIONS.md`.
- ~~**The competition rules page is JavaScript-only**~~ **Settled 2026-08-13, from the rules saved
  verbatim to `COMPETITION_RULES.txt` on 2026-08-12.** The clause was unreadable by any tool here for
  three days and then sat unread for one more because the issue still said it needed a browser.
  - **External data and models are PERMITTED.** §2.6.a allows External Data that is "publicly available
    and equally accessible to use by all Participants … at no cost"; §2.6.b says use "is acceptable
    unless specifically prohibited by the Host", and the Competition-Specific Rules contain no such
    prohibition. Everything we rely on qualifies: DINOv2 (free, and the competition mounts it itself),
    DINOv3 (free behind a click-through Meta licence), and pilkwang's two datasets (CC0-1.0). No forum
    disclosure of external data is required by these rules.
  - **Nothing here conditions a SUBMISSION.** The obligations attach to winners (§2.8, §8), which is why
    the entry that went out while this was open was never at risk — only a prize claim was.
  - **The winner licence is CC-BY-NC 4.0**, and §2.5.a carves out exactly our awkward case: "in the event
    that input data or pretrained models with an incompatible license are used … you do not need to grant
    an open source license for that data and/or model(s)". DINOv3's licence is covered.
  - **The one real obligation, and it is not yet met.** §8.a: a winner must deliver model weights "as a
    public kaggle dataset so it is both publicly accessible and linked to the inference/submission code".
    All three of ours are **private** — `kneemri-members`, `kneemri-kit`, `dinov3-hf`. That is correct for
    now: §5.d and §6.a forbid private sharing during the Competition Period, and publishing early would
    hand the solution to the field. **If we place, `kneemri-members` and `kneemri-kit` must be flipped to
    public before the winner deadline.**

## The Efficiency Prize, measured rather than assumed (2026-08-11)

Its evaluation page is JS-only like the rules, but `ryanholbrook/rsna-knee-abnormalities-efficiency-lb`
publishes the leaderboard as a notebook output, and `kaggle kernels output` pulls it. From the snapshot:

- **905 teams are ranked**, which is essentially everyone — participation is automatic, not opt-in, and
  costs us nothing to leave alone.
- **Runtime is weighted hard.** The best score anywhere on that board, **0.940, sits at efficiency rank
  638**; rank 1 scores 0.934 and rank 100 scores 0.836. It is not a re-sort of the accuracy leaderboard.
- **Our accuracy is already worth a top-30 efficiency place if it runs cheaply.** The three teams scoring
  exactly the 0.888 we held at that snapshot sit at ranks **29, 33 and 50** — and we have since scored
  0.919, above every one of them, so the placement that number buys has only gone up.

So the decision is not "target it or not" — it is one number: what do members two through five actually
buy? We currently pay 5 members x 10 TTA windows = 50 forward passes per study.

**THAT NUMBER NOW EXISTS, and it was answered on the leaderboard rather than the holdout** (the
member-count curve was void — computed across split-seeds and leaked):

| entry | members | passes/study | public LB |
|---|---:|---:|---:|
| solo, our strongest member | 1 | 10 | 0.868 |
| **whole-stack** | **2** | **2** | **0.882** |
| whole-stack | 3 | 3 | 0.882 |
| ours v2, windowed | 5 | 50 | 0.888 |
| one pixel group, all-ours (2026-08-16) | 3 | 3 | 0.904 |
| joint-chosen, disjoint windows (2026-08-16) | 4 | 4 | 0.918 |
| **whole-stack pair, all-ours (2026-08-16)** | **2** | **2** | **0.919** |

The curve is NOT flat at the bottom — one member costs −0.020 against five — so a one-member entry is
out. But **two whole-stack members is the efficiency entry**, and it is already scored: 0.882, six
thousandths under our own best, for **2 forward passes against 50**. Twenty-five times cheaper.

The reason it is that cheap is measured and specific: **a whole-stack member reads ONE window**, because
its slot embedding is already the elementwise max over every slice. Window TTA is worth +0.0146 to a
windowed member and exactly **0.0000** to a whole-stack one, so the 10x TTA multiplier simply does not
apply. The accuracy work and the efficiency work turn out to want the same member.

**That risk is now settled, and the answer is favourable.** Joining the efficiency leaderboard to the
public one on team name (897 teams in common): 61.3% have the same score on both, and **38.7% — 347
teams — carry an efficiency score strictly BELOW their public best**, by a median of 0.018. It is never
above. The efficiency entry is therefore not forced to be the team's best-accuracy submission: teams
demonstrably hold both, including the efficiency leader (public 0.939, efficiency entry 0.934).

So a cheap submission **adds** an efficiency candidate rather than displacing the accurate one, and the
track costs us nothing but a submission slot. Get both leaderboards with
`kaggle competitions leaderboard <slug> -d` and `kaggle kernels output ryanholbrook/rsna-knee-abnormalities-efficiency-lb`.

## Measured on 2026-08-10 by `kernels/kneemri-probe`

- **The competition mounts at `/kaggle/input/competitions/<slug>`**, not `/kaggle/input/<slug>`. Probe v1
  assumed the latter and died before any other probe ran.
- **`machine_shape: NvidiaTeslaT4` really does give two T4s** — `sm_75`, 15.6 GB each, torch 2.10.0+cu128
  whose `arch_list` includes `sm_75`. The host has only **4 CPUs** and 32.9 GB RAM: on a read-bound
  pipeline the CPU count, not the GPU, is the binding constraint.
- **Almost nothing needs to ship as a wheel — but "almost" is not "nothing".** Present offline: pydicom
  3.0.2, timm 1.0.26, transformers 5.0.0, polars, cv2, SimpleITK, nibabel, einops, sklearn, scipy, PIL.
  Absent: monai, zarr, **and `jaxtyping`** — which `kneemri.models.model` imports at module top, so the kit
  ships `jaxtyping` + `wadler_lindig` wheels and the kernel `pip install --no-index --no-deps`s them
  before importing anything of ours. The probe asked which libraries the *pipeline* needs and missed the
  ones *our own source* imports: **probe what you import, not only what you plan to use.**
- **12 ms per DICOM slice** off the mount (186 slices in 2.31 s).
- **The base-variant submission path is verified locally** (2026-08-11): a DINOv2-base member packages,
  reports `variant=base`, resolves to `facebook/dinov2-base` at width 768 and the
  `metaresearch/dinov2/PyTorch/base/1` mount, loads, and passes its fingerprint check. Worth doing on
  the host rather than discovering in a queued kernel — the previous variant defect (`Member.variant`
  did not exist while the kernel already called it) would have surfaced only as an AttributeError
  minutes into a run.

## Gotchas that cost real time (inherited, verify each still holds)

- **T4, not the default P100** — set `"machine_shape": "NvidiaTeslaT4"` in `kernel-metadata.json`. Kaggle's
  default GPU is a P100 (`sm_60`, Pascal), which modern torch builds (compiled for `sm_70`+) refuse with
  *"no kernel image is available for execution on the device"*. The `--accelerator` CLI flag does **not**
  override it; only the metadata field does. Valid tokens: `NvidiaTeslaT4`, `NvidiaTeslaP100`, `Tpu1VmV38`.
  **Do not invent one** — an unrecognised token is not rejected, it silently degrades to the P100, and the
  only symptom is that `sm_60` error thrown hours later from the first forward.
- **`NvidiaTeslaT4` gives you TWO T4s** — `device_count() == 2`, each `sm_75` with 14.6 GB. There is no
  separate "x2" shape to ask for; shard the work yourself.
- **fp16 inference, not bf16** — the T4's `sm_75` has no bf16 at all. bf16 is a *training*-only choice
  (gradient range); a forward has no gradients, so it buys nothing and costs mantissa.
- **Mount every variant a member might need.** The trained kernel mounts DINOv2 small AND base, and
  picks the mount matching each member's own recorded variant. Mounting one and loading the other is not
  a loud failure: widths differ (384 vs 768), so it surfaces as a shape error during load if we are
  lucky and as a silently wrong model if the head accepts it.
- **No internet** (`enable_internet: false`) — every dependency must be a wheel in the kit, and every
  pretrained backbone must be a mounted Kaggle model. Every public notebook for this competition mounts
  `metaresearch/dinov2/PyTorch/small/1` for exactly this reason.
- **GPU kernels queue** — a run is not instant; poll `kernels status`.
- **Probe cheaply, early.** Kernel *pushes* are unlimited; *submissions* are 5/day. Accelerator and
  environment questions are answerable in a minute — do not learn them six hours into a run.
