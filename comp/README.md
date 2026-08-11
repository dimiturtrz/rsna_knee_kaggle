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

## The two auth paths

- `survey.py` / `publish.py` speak the REST API directly with a `KGAT_*` access token from `KAGGLE_TOKEN`
  (env-only — never written to `~/.kaggle/`, never committed). They exist because the official CLI
  authenticates with a username + key pair, which is a *different* credential.
- The official `kaggle` CLI reads `~/.kaggle/kaggle.json` and works for reads (`competitions files`,
  `kernels list`, `kernels pull`). Both are live on this machine; pick by which credential you have.

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
lost to `ModuleNotFoundError: No module named 'kneemri.submission'`: the kernels had been refactored
to import a newly extracted module, the MEMBERS dataset was pushed, and the kit was not.
`verify_kit.py` compares the kit against the repo and catches this in a second — it was skipped
because the change felt like it was "only in the kernels", which is backwards. The kernels are a
bootstrap that imports the kit, so a change to either is a change to both.

## Open questions — settle before building around them

- **Is submission notebook-only?** The sibling competition rejected file uploads with
  `400 FAILED_PRECONDITION "only accepts Submissions from Notebooks"`, which is the entire reason the
  kit-as-dataset machinery exists. Unverified here. If RSNA accepts a CSV upload, most of this is optional.
- **The Efficiency Prize track** is a separate leaderboard with its own scoring. Decide whether to target it
  before optimizing inference cost by accident.

## Measured on 2026-08-10 by `kernels/kneemri-probe`

- **The competition mounts at `/kaggle/input/competitions/<slug>`**, not `/kaggle/input/<slug>`. Probe v1
  assumed the latter and died before any other probe ran.
- **`machine_shape: NvidiaTeslaT4` really does give two T4s** — `sm_75`, 15.6 GB each, torch 2.10.0+cu128
  whose `arch_list` includes `sm_75`. The host has only **4 CPUs** and 32.9 GB RAM: on a read-bound
  pipeline the CPU count, not the GPU, is the binding constraint.
- **Almost nothing needs to ship as a wheel — but "almost" is not "nothing".** Present offline: pydicom
  3.0.2, timm 1.0.26, transformers 5.0.0, polars, cv2, SimpleITK, nibabel, einops, sklearn, scipy, PIL.
  Absent: monai, zarr, **and `jaxtyping`** — which `kneemri.model` imports at module top, so the kit
  ships `jaxtyping` + `wadler_lindig` wheels and the kernel `pip install --no-index --no-deps`s them
  before importing anything of ours. The probe asked which libraries the *pipeline* needs and missed the
  ones *our own source* imports: **probe what you import, not only what you plan to use.**
- **12 ms per DICOM slice** off the mount (186 slices in 2.31 s).

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
- **No internet** (`enable_internet: false`) — every dependency must be a wheel in the kit, and every
  pretrained backbone must be a mounted Kaggle model. Every public notebook for this competition mounts
  `metaresearch/dinov2/PyTorch/small/1` for exactly this reason.
- **GPU kernels queue** — a run is not instant; poll `kernels status`.
- **Probe cheaply, early.** Kernel *pushes* are unlimited; *submissions* are 5/day. Accelerator and
  environment questions are answerable in a minute — do not learn them six hours into a run.
