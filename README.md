# kneemri

[![ci](https://github.com/dimiturtrz/rsna_knee_kaggle/actions/workflows/ci.yml/badge.svg)](https://github.com/dimiturtrz/rsna_knee_kaggle/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Twelve findings from one knee MRI study.** Entry for RSNA's
[knee abnormality detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
competition (deadline 2026-10-22): given a study — several MRI series across sagittal, coronal and axial
planes at different weightings — predict twelve binary findings (ACL, MCL, medial/lateral meniscus,
medial/lateral/patellofemoral OA, effusion, synovitis, Baker's cyst, contusion, fracture).

**Where it stands.** Six entries scored. Our own best is **0.888** (five members, rank 367/1077 at the
time); the highest number on the board is **0.894**, and it is *mostly not ours* — twenty of its
twenty-three models are CC0 weights someone else trained. The full measured account, including what was
refuted, is in
[interpretations/kneemri/](interpretations/kneemri/) — what actually moved the number, what the
leaderboard taught us, [what the labels
contain](interpretations/kneemri/2026-08-13_what-the-labels-actually-contain.md), and [the day the
ruler itself was
wrong](interpretations/kneemri/2026-08-13_the-ruler-was-wrong-and-the-witness-earned-its-keep.md).
What the public frontier does is in [research/](research/2026-08-11_public-notebooks.md).

Two numbers govern how the rest is read. The run-to-run **noise floor is 0.0062** (three seeds, one
recipe, one split), which puts every knob-sized change out of reach at n=1 — and it is the holdout's own
*sampling* error, so more seeds cannot lower it.

And **ensembling is the largest effect here**, which took the leaderboard to establish and contradicts
what an early three-member curve suggested. One member scored **0.868**; five weaker ones scored
**0.888**. But it is DIVERSITY rather than count: a third member of our own recipe bought *nothing*
(0.882 → 0.882), while adding our three to twenty differently-trained ones bought +0.006 over either
component alone. Members that fail the same way do not help each other.

The changes that moved the number most were not ideas imported from anywhere; they were our own
defects, and finding them is most of what this project has done. Augmentation was written,
unit-tested, and never called by the training loop (+0.0110). The fold
split was derived from the training seed, so two seeds never shared a holdout and cross-scoring them
leaked 717 training studies into a validation number. Of six ideas taken from the strongest public
notebooks, five were refuted or landed inside the noise floor.

**What the task actually is.** The score is the unweighted mean of twelve per-label ROC AUCs, so only rank
order is read — calibration is worth nothing and ensembling averages *ranks*, not probabilities. A label
left at chance forfeits ~0.029 of the final score however well the other eleven do, which makes the rare
findings worth more attention than the common ones, not less. Only a subset of training studies carry the
twelve annotations; every study carries a radiology report, and `test.csv` has no `Report` column — so the
reports are a source of training targets, never an inference-time input.

## Layout

- `core/`, `kneemri/` — your packages. The guardrails target all of them; organise modules however the project needs.
- `comp/` — Kaggle submission machinery (kit build, drift check, REST publish/survey). Named `comp/` rather
  than `kaggle/` because a top-level `kaggle/` shadows the installed `kaggle` package on `sys.path[0]`.
- `tests/` — `unit/`, `integration/`, `e2e/`.
- `devtools/` — the guardrail tools (`graph.py` arch-fitness, plus the class-shape explorers).

The set of packages the gates lint/test/graph is the `packages` answer (comma-separated) — add more
without touching any gate file; every gate renders over the list. Directional layer contracts between
packages are **opt-in** (add import-linter yourself — see `devtools/README.md`), never imposed.

## Quickstart

```bash
git clone https://github.com/dimiturtrz/rsna_knee_kaggle
uv sync --extra dev --extra devtools
uvx nox -s lint test cov
```

## Gates

`uvx nox -s lint test cov` (or `uvx nox -s gates`) runs exactly what CI runs:

- **lint** — ruff check (pinned) + ruff format --check (advisory) + vulture + arch-fitness (`graph.py --assert`) + ast-grep + jscpd.
- **test** — pytest.
- **cov** — pytest --cov with a `--fail-under=90` floor.

## Architecture site

`uvx nox -s archmap` writes `docs/architecture/graph.json` (committed, diffable — nodes + weighted import edges)
plus a self-contained interactive viewer (`index.html`, gitignored, rebuilt on demand). **Commit the
graph.json diff** — it's the architecture-erosion record.


Publishing to GitHub Pages — one repo = one Pages site, so pick ONE of:

- **Sole owner** — this repo has no other Pages deployer. Answer `archviz_pages` yes at scaffold time; the
  generated `pages.yml` owns a staged site (`/architecture/` = main, `/architecture/preview/` = dev, `/`
  redirects) and enables it. Nothing else to do.
- **Compose** — this repo ALREADY deploys Pages (a docs/demo site). Leave `archviz_pages` off and fold the
  archmap build into that existing workflow as an `/architecture/` subpath. Build it isolated (no project
  sync, no touch to the pinned devtools) — `write_viewer` only needs the vendored template + libs:

  ```yaml
  - uses: astral-sh/setup-uv@v5
  - name: Architecture view (/architecture/)
    run: |
      uv run --no-project --with "sdlc-devtools @ git+https://github.com/dimiturtrz/sdlc-scaffold.git@v1.27.1#subdirectory=sdlc-devtools" \
        python -c "from devtools.graph.archmap import Archmap; Archmap(['core', 'kneemri']).write_viewer(project='kneemri')"
      mkdir -p _site/architecture
      cp docs/architecture/graph.json docs/architecture/index.html _site/architecture/
  ```

  graph.json stays committed (the diff-truth); regen it locally with `uvx nox -s archmap` when the import
  structure changes. (`../synthscape` is the reference — archmap folded into its `deploy-fit-explorer`
  workflow beside the fit-explorer views.)

## Development

Tooling + quality gates are provisioned by an in-house copier template (**sdlc-scaffold**). Refresh with
`uvx copier update`; `.copier-answers.yml` pins the template version.

### Local hooks

The same gates CI enforces are bound to git events via pre-commit. Install **both** stages:

```bash
pre-commit install                        # commit stage — fast static gates (ruff/vulture/arch-fitness/…)
pre-commit install --hook-type pre-push   # push stage — fast unit suite (tests/unit)
```

The pre-push hook runs `pytest tests/unit` so a change that lints clean but breaks a test **contract** (a
signature change a mirror test still calls the old way) is caught before the push, not after CI goes red. It
is deliberately push-only (not every commit) and unit-only (no integration/e2e); the coverage floor stays a
CI job. The default `pre-commit install` does **not** wire pre-push — run the second line once per clone.
