# kneemri

[![ci](https://github.com/dimiturtrz/rsna_knee_kaggle/actions/workflows/ci.yml/badge.svg)](https://github.com/dimiturtrz/rsna_knee_kaggle/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Twelve findings from one knee MRI study.** Entry for RSNA's
[knee abnormality detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
competition (deadline 2026-10-22): given a study — several MRI series across sagittal, coronal and axial
planes at different weightings — predict twelve binary findings (ACL, MCL, medial/lateral meniscus,
medial/lateral/patellofemoral OA, effusion, synovitis, Baker's cyst, contusion, fracture).

**Where it stands** (2026-08-20). The highest number on the board is **0.929**, from a four-member
entry that is *entirely ours* — no CC0 weights. It clears the 0.895 the 22-member mostly-grafted fusion
reached, so the published weights we started from are now a [**cost** rather than a
contribution](interpretations/kneemri/2026-08-16_the_grafted_members_became_a_cost.md). The last
measured RANK is older than the score: on 2026-08-17 an entry of **0.919** stood at **84 of 1,758** —
the 95th percentile, against a leader at 0.951 and a median at 0.874, with the gap to the frontier down
from 0.054 to **0.032**. The +0.010 since then has not been re-ranked, and is not claimed as one; four
days before that measurement we were 431 of 1,395 at the 31st percentile, *losing ground while
gaining*. The full measured account, including what was refuted, is in
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

Diversity, however, is not the same as *foreign* — which the board took until 2026-08-16 to say. Five
all-ours members reached 0.911 and two reached 0.919, both above the mixed 22-member fusion by more than
the 1.5% bar, so the gain belongs to the **member pool** and not to borrowed weights. And the holdout
prices diversity too cheaply: collapsing three members into one pixel group cost 0.0059 offline and
**0.014** on the board, 2.4x. Of the [ensemble's six design
axes](interpretations/kneemri/2026-08-17_the_ensemble_axes_and_which_are_levers.md), diversity is the
only lever left: [count plateaus at
eight](interpretations/kneemri/2026-08-22_the_pool_is_exhausted.md), per-label selection is dead (its
*oracle* sits below the plain average), and the vote itself is flat — rank, logit and quality-weighted
averaging land within 0.0005 of each other, an order below the board's noise floor. Diversity is the
lever left, and what is left of it is small rather than absent: priced card-free over **63 candidates**,
the best available eighth seat is **+0.0005** offline — which at the measured **+2.14 board points per
holdout point** is about +0.001 on the board, a tick rather than a step. The entry that took us to 0.933
priced **+0.0001** offline, so a seat number below the holdout's own resolution is not evidence of
nothing; it is the seat ruler running out of range.

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
  [`kneemri/README.md`](kneemri/README.md) indexes every `python -m` command the package has, generated from
  the modules by `python -m kneemri.commands` and asserted current in CI.
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

## Acknowledgements

**DINOv3** (Meta AI, `facebook/dinov3-vitb16-pretrain-lvd1689m`) is used under the DINOv3 licence, which
requires that the materials be acknowledged in any published result. A DINOv3-base member
(`dinov3_base_lora`, rank-8 adapters over the frozen encoder) has been named by a submitted entry since
2026-08-20, so the requirement is live rather than anticipated. **DINOv2** (Meta AI) is used under Apache
2.0 and carries no such requirement; it is named here because every other member is built on it.
