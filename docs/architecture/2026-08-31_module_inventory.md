# Module inventory — `core/` and `kneemri/`

Fact-gathering half of **knee-579v.1**. This is an INVENTORY, not a design: no pipeline
taxonomy is proposed here, no module is judged, nothing is moved. The taxonomy decision is
knee-579v.2's.

Generated 2026-08-31 by a throwaway AST scan (method below). Scope: every `.py` under
`core/` and `kneemri/`, including `__init__.py`. Excluded: `devtools/` (template-owned),
`tests/`, `comp/`, `experiments/`, `noxfile.py`.

## How every number here was derived

A single Python script (`ast`-based, run under `.venv/Scripts/python.exe`) — not grep, and
not `docs/architecture/graph.json`. graph.json was read first as the bead suggested; it holds
the right nodes and weighted import edges but only for `core.*` and `kneemri.*`, so it cannot
see importers in `comp/`, `experiments/` or `tests/`, and it cannot answer `main()`. The scan
below subsumes it.

- **Module list** — `rglob('*.py')` under `core/` and `kneemri/`, `__pycache__` dropped,
  nothing else filtered. `__init__.py` is a module and gets a row (13 of them).
- **`main()` (Y/N)** — a **top-level** `def main` / `async def main` in the parsed AST. A
  `main` nested inside a class or function does not count; neither does a bare
  `if __name__ == "__main__":` block with inline code.
- **Docstring line** — `ast.get_docstring(tree)`, first non-empty line, verbatim. Not one
  module was invented for: all 235 modules carry a module docstring.
- **Importer counts** — every `.py` under `core/`, `kneemri/`, `comp/`, `experiments/`,
  `tests/` and `devtools/` was AST-parsed and every `Import` / `ImportFrom` resolved,
  relative imports included (`from .x import y` resolved against the importing package).
  `from pkg.mod import Name` credits **both** `pkg.mod` and `pkg.mod.Name`, so a module
  imported through its package `__init__` is counted at both. Self-imports are dropped.
  Only importers that resolve to a module IN THIS TABLE are counted.
- **`imp` column = NON-TEST in-repo importers** (`core/`, `kneemri/`, `comp/`,
  `experiments/`, `devtools/`). **`t` column = test importers** (`tests/`, or any
  `test_*.py`), counted separately. The ORPHANS and IMPORT HUBS sections use `imp` only —
  a module imported solely by its own unit test is still nothing the pipeline calls.
- **Lines** — `len(src.splitlines())`, raw, blanks and docstrings included.
- **`doc?` (orphans section only)** — is the module's dotted name mentioned in any
  human-authored doc or script (`.md .sh .ps1 .toml .yaml .yml .cfg .txt`, dot-directories
  and `external/`, `logs/`, `smokes/` excluded; 833 files scanned). It is NOT evidence the
  module is scheduled — only that a human wrote its name down somewhere.

Anything not listed above was not measured. In particular: this document does **not** claim
to know what invokes an orphan at the shell, whether a module is dead, or which stage owns it.

## COUNTS

Modules per directory (a directory's own `__init__.py` counts in that directory).

| directory | modules |
| --- | ---: |
| `core/` | 10 |
| `core/data/` | 1 |
| `core/data/corpus/` | 31 |
| `core/data/labels/` | 17 |
| `kneemri/` | 6 |
| `kneemri/embeddings/` | 4 |
| `kneemri/evaluation/` | 51 |
| `kneemri/experiments/` | 18 |
| `kneemri/inference/` | 16 |
| `kneemri/kaggle/` | 29 |
| `kneemri/models/` | 17 |
| `kneemri/runs/` | 6 |
| `kneemri/training/` | 29 |
| **core/ total** | **59** |
| **kneemri/ total** | **176** |
| **TOTAL** | **235** |

- modules with a top-level `main()`: **109** of 235
- of those, with zero non-test in-repo importers (ORPHANS): **80**
- modules with zero importers of any kind, test included: **9**

Of those, **6** have NO `main()` either — nothing imports them and they are not
entrypoints. Listed here as a fact, with no claim about whether they are dead:

| module | lines | docstring first line |
| --- | ---: | --- |
| `core.data` | 7 | The data layer: what a study IS (`corpus`) and what it is LABELLED with (`labels`). |
| `core.data.labels` | 1 | labels. |
| `kneemri.embeddings` | 1 | embeddings. |
| `kneemri.experiments` | 1 | experiments. |
| `kneemri.models` | 1 | models. |
| `kneemri.runs` | 1 | runs. |

All six are package `__init__.py` files whose package is only ever reached through a
submodule (`from core.data.labels.targets import …`, never `from core.data import labels`).
This is a property of the import convention, not a loose module.

## ORPHANS

**80 modules** define a top-level `main()` and are imported by nothing in
`core/`, `kneemri/`, `comp/`, `experiments/` or `devtools/`. They are reachable only as
`python -m <module>` typed by a human. `t` = test importers (a non-zero `t` means a unit
test constructs it, not that anything schedules it). `doc?` = its name appears in a
committed doc or script.

| module | t | doc? | docstring first line |
| --- | ---: | :-: | --- |
| `core.data.corpus.band_coverage` | 1 | Y | What the central slice band actually discards: `python -m core.data.corpus.band_coverage`. |
| `core.data.corpus.build_cache` | 2 | Y | Build the decode-once training cache: `python -m core.data.corpus.build_cache [--slices N] [--workers N]`. |
| `core.data.corpus.build_merged` | 0 | Y | Merge two decoded corpora into one cache: `python -m core.data.corpus.build_merged`. |
| `core.data.corpus.build_mrnet` | 0 | n | Ingest the MRNet corpus: `python -m core.data.corpus.build_mrnet --archive PATH`. |
| `core.data.corpus.build_rijeka` | 0 | n | Ingest the kneeMRI corpus: `python -m core.data.corpus.build_rijeka --archive PATH`. |
| `core.data.corpus.crop_alignment` | 1 | Y | Do a study's slots agree WHERE the knee is: `python -m core.data.corpus.crop_alignment`. |
| `core.data.corpus.crop_laterality` | 1 | Y | Does the 130 mm crop cut the LATERAL compartment? `python -m core.data.corpus.crop_laterality`. |
| `core.data.corpus.decode_bench` | 1 | Y | Is decode I/O-bound or GIL-bound? `python -m core.data.corpus.decode_bench --studies 96`. |
| `core.data.corpus.diagnose_laterality` | 1 | Y | How was each study's side decided? `python -m core.data.corpus.diagnose_laterality --out <csv>`. |
| `core.data.corpus.diagnose_matrix` | 1 | Y | How many series hold slices of DIFFERENT matrix sizes? `python -m core.data.corpus.diagnose_matrix`. |
| `core.data.corpus.fetch_bench` | 1 | Y | How fast a batch reaches the trainer, per reader and per handle count: |
| `core.data.corpus.slot_geometry` | 1 | n | Do a study's slots image the same anatomy in patient space: `python -m core.data.corpus.slot_geometry`. |
| `core.data.labels.finding_calibration` | 1 | Y | Does the reader's scale mean the same thing for every FINDING, or is it shifted per finding? |
| `core.data.labels.level_agreement` | 1 | Y | Is one calibration constant right for all twelve findings? `python -m core.data.labels.level_agreement`. |
| `core.data.labels.overcall_clustering` | 1 | n | Is the reader's over-call a property of the STUDY, or twelve independent coin flips? |
| `core.data.labels.reader_disagreement` | 1 | n | Where independent readers of the same report disagree: `python -m core.data.labels.reader_disagreement`. |
| `core.data.labels.reader_upgrade` | 1 | Y | Does a new reader beat the lexicon ON THE PAIRS THE LEXICON LEFT BLANK? |
| `core.data.labels.report_reader` | 1 | Y | Read the reports with a model instead of a lexicon: `python -m core.data.labels.report_reader`. |
| `core.data.labels.rubric_boundary` | 1 | n | Where the HOST'S rubric and our report reader disagree: `python -m core.data.labels.rubric_boundary`. |
| `core.data.labels.rubric_gap` | 1 | Y | How many derived targets does the host's published rubric say are wrong? `python -m core.data.labels.rubric_gap`. |
| `kneemri.commands` | 1 | Y | Every `python -m` entrypoint the package has, generated from the modules themselves. |
| `kneemri.embeddings.build_embeddings` | 1 | Y | Encode a pixel cache into an embedding cache: `python -m kneemri.embeddings.build_embeddings`. |
| `kneemri.embeddings.report_embeddings` | 1 | Y | Embed each study's radiology report once, so the image model can be supervised by all of it. |
| `kneemri.evaluation.alignment_damage` | 1 | n | Does cross-slot misregistration cost AUC, and only on the findings small enough to care: |
| `kneemri.evaluation.attribution` | 1 | Y | What is the model actually using — measured by taking things away, not by drawing heatmaps. |
| `kneemri.evaluation.board_transfer` | 1 | Y | Does our holdout ORDER entries the way the leaderboard does? `python -m kneemri.evaluation.board_transfer`. |
| `kneemri.evaluation.competence_axis` | 1 | Y | Whether the whole member pool differs along ONE trade-off axis — the question a seat search cannot ask. |
| `kneemri.evaluation.encoder_drift` | 1 | Y | How far supervision actually moved the encoder away from the weights it was pretrained with. |
| `kneemri.evaluation.foreign_gate` | 1 | Y | The plumbing oracle for a foreign member: `python -m kneemri.evaluation.foreign_gate`. |
| `kneemri.evaluation.foreign_overlap` | 1 | Y | Did a foreign model train on our holdout? `python -m kneemri.evaluation.foreign_overlap`. |
| `kneemri.evaluation.foreign_ruler` | 1 | Y | A foreign model and ours on ONE ruler: `python -m kneemri.evaluation.foreign_ruler`. |
| `kneemri.evaluation.label_probe` | 1 | n | Whether a label interaction GENERALISES — the verdict half of the residual-coupling gate. |
| `kneemri.evaluation.label_stability` | 1 | Y | Which weak labels are reproducible and which are the holdout's noise: `python -m kneemri.evaluation.label_stability`. |
| `kneemri.evaluation.level_calibration` | 1 | Y | The report reader's confidence levels, re-mapped onto the rate at which they are actually positive. |
| `kneemri.evaluation.noise_binding` | 1 | Y | Does a member track the LEXICON or the RADIOLOGIST where the two disagree? |
| `kneemri.evaluation.oof_oracle` | 1 | Y | Does OUR pipeline hand a published member the same pixels its author did? |
| `kneemri.evaluation.ordinal_signal` | 1 | Y | Whether the graded YES levels in the report table carry ORDINAL information, or only certainty. |
| `kneemri.evaluation.paired_floor` | 1 | Y | How tightly a per-label delta can be read when both arms saw the SAME studies. |
| `kneemri.evaluation.pool_census` | 1 | Y | Is there a member in this pool that is STRONG and sees the knee DIFFERENTLY — or is that cell empty? |
| `kneemri.evaluation.prevalence_match` | 1 | Y | Is the medial-over-lateral deficit PREVALENCE, or is it vision? `python -m kneemri.evaluation.prevalence_match`. |
| `kneemri.evaluation.raptor_inventory` | 1 | n | What a directory of foreign checkpoints actually CONTAINS, read from the checkpoints themselves. |
| `kneemri.evaluation.ruler_check` | 1 | Y | Does the choice of evaluation ruler change WHICH recipe wins? |
| `kneemri.evaluation.ruler_transfer` | 1 | Y | Does a DELTA between two arms survive changing the ruler? `python -m kneemri.evaluation.ruler_transfer`. |
| `kneemri.evaluation.seat_bonus` | 1 | n | What a seat is worth: the board gain of an ensemble over the SOLO entry of its own first member. |
| `kneemri.evaluation.side_blindness` | 1 | Y | Can the model tell the two compartments apart at all? `python -m kneemri.evaluation.side_blindness`. |
| `kneemri.evaluation.stack_coverage` | 1 | Y | Does keeping 12 slices of a deep stack cost accuracy? `python -m kneemri.evaluation.stack_coverage`. |
| `kneemri.evaluation.table_separation` | 1 | Y | Are two label tables actually separated on the gold 58? `python -m kneemri.evaluation.table_separation`. |
| `kneemri.evaluation.teacher_distance` | 1 | Y | How far foreign members sit from EACH OTHER, measured with no labels and no submission slot. |
| `kneemri.evaluation.tie_divergence` | 1 | Y | When two series tie for a slot, do they SHOW the same thing? `python -m kneemri.evaluation.tie_divergence`. |
| `kneemri.evaluation.unsure_cost` | 1 | n | What an `UNK` cell is actually worth: `python -m kneemri.evaluation.unsure_cost`. |
| `kneemri.evaluation.vote_ceiling` | 1 | n | What is the VOTE worth, once the fitting is paid for? |
| `kneemri.evaluation.weights_gap` | 1 | Y | Where do THEIR weights beat OURS, per label, on studies both held out? |
| `kneemri.experiments.arm_check` | 1 | Y | Prove every committed arm's flags before a queue costs GPU: `python -m kneemri.experiments.arm_check`. |
| `kneemri.experiments.arm_read` | 1 | Y | Read a finished arm the way its own header said to: per label, against the control it declared. |
| `kneemri.experiments.budget_residuals` | 1 | n | What the VRAM cost model DERIVES against what the arms MEASURED: `python -m |
| `kneemri.experiments.card_check` | 1 | Y | Is the GPU held by exactly one of our jobs? |
| `kneemri.experiments.ledger_check` | 1 | Y | Every run that finished should appear in the ledger: `python -m kneemri.experiments.ledger_check`. |
| `kneemri.experiments.progress` | 1 | n | What the card is doing RIGHT NOW: `python -m kneemri.experiments.progress`. |
| `kneemri.experiments.queue_check` | 1 | Y | Is every queued arm still gated by a live process? |
| `kneemri.experiments.stale_gap` | 1 | Y | What `--patience` would have to be, derived from every run's own curve instead of chosen. |
| `kneemri.experiments.trajectory_check` | 1 | Y | Can a running arm still reach its branch: `python -m kneemri.experiments.trajectory_check`. |
| `kneemri.inference.foreign_holdout` | 1 | n | Give a published foreign checkpoint a holdout record: `python -m kneemri.inference.foreign_holdout`. |
| `kneemri.inference.holdout_backfill` | 1 | Y | Re-derive an already-finished run's holdout predictions: `python -m kneemri.inference.holdout_backfill`. |
| `kneemri.inference.serve_bench` | 1 | Y | Does compiling the encoder pay at SERVING time? `python -m kneemri.inference.serve_bench`. |
| `kneemri.inference.teacher_readings` | 1 | Y | Write an ensemble's ordering of the TRAINING studies for a student to be taught: |
| `kneemri.kaggle.decode_entry` | 1 | Y | The best roster a DECODE BUDGET can buy: `python -m kneemri.kaggle.decode_entry`. |
| `kneemri.kaggle.discussion_read` | 1 | Y | What are the other competitors saying? `python -m kneemri.kaggle.discussion_read --live`. |
| `kneemri.kaggle.dry_run` | 1 | Y | Run the SUBMISSION path over labelled studies: `python -m kneemri.kaggle.dry_run --studies 40`. |
| `kneemri.kaggle.efficiency_read` | 1 | Y | Where do we stand on the OTHER leaderboard? `python -m kneemri.kaggle.efficiency_read --live`. |
| `kneemri.kaggle.entry_offline` | 1 | Y | Why is a board entry's `offline` null? `python -m kneemri.kaggle.entry_offline [--write]`. |
| `kneemri.kaggle.mount_check` | 1 | Y | Does every kernel declare the mounts its members need? `python -m kneemri.kaggle.mount_check`. |
| `kneemri.kaggle.raptor_teacher` | 1 | Y | The raptor member's ordering over OUR corpus, written as a teacher file. |
| `kneemri.kaggle.roster_check` | 1 | Y | Is every scoreable member either packaged or explicitly declined? `python -m kneemri.kaggle.roster_check`. |
| `kneemri.kaggle.run_check` | 1 | Y | Did the kernel run the entry it was named for? `python -m kneemri.kaggle.run_check --log <path>`. |
| `kneemri.kaggle.seat_value` | 1 | Y | What a candidate member would add to the champion roster, card-free. |
| `kneemri.runs.metrics_recipe` | 1 | Y | Give a metrics file written before recipes existed the recipe its own checkpoint already carries. |
| `kneemri.training.fitted_encoder` | 1 | Y | A fitted member, read back as an encoder another run can start from. |
| `kneemri.training.head_trial` | 1 | Y | Fit the HEAD alone on cached encoder features: `python -m kneemri.training.head_trial`. |
| `kneemri.training.pretrain` | 1 | Y | Adapt the encoder to knee MRI without labels: `python -m kneemri.training.pretrain --epochs N`. |
| `kneemri.training.train` | 1 | Y | Fit one member: `python -m kneemri.training.train --fold 0 --epochs 8`. |

Orphans by directory:

- `core/data/corpus/` — 12 of 31
- `core/data/labels/` — 8 of 17
- `kneemri/` — 1 of 6
- `kneemri/embeddings/` — 2 of 4
- `kneemri/evaluation/` — 29 of 51
- `kneemri/experiments/` — 9 of 18
- `kneemri/inference/` — 4 of 16
- `kneemri/kaggle/` — 10 of 29
- `kneemri/runs/` — 1 of 6
- `kneemri/training/` — 4 of 29

## IMPORT HUBS

The ten modules with the most **non-test in-repo importers**. Ties broken alphabetically;
the table shows every module down to the tenth distinct count so a tie is visible.

| # | module | importers | test importers |
| ---: | --- | ---: | ---: |
| 1 | `core` | 108 | 1 |
| 2 | `core.cli` | 108 | 1 |
| 3 | `kneemri.kaggle.kernel_run` | 72 | 2 |
| 4 | `core.findings` | 52 | 75 |
| 5 | `kneemri.inference.vote` | 34 | 11 |
| 6 | `kneemri.evaluation.scores` | 30 | 12 |
| 7 | `kneemri.inference.ensemble` | 30 | 7 |
| 8 | `core.data.corpus.study_cache` | 28 | 21 |
| 9 | `kneemri.device` | 26 | 2 |
| 10 | `kneemri.inference.members` | 25 | 16 |
| 11 | `kneemri.models.backbone` | 24 | 18 |
| 12 | `core.data.corpus.slots` | 22 | 18 |

Three of these rows need a caveat, each checked by reading the importing statements:

- **`core` and `core.cli` are ONE fact, not two.** Their importer sets are byte-identical.
  The repo's entrypoint convention is `from core import cli`, and this scan credits both the
  package (`core`) and the imported submodule (`core.cli`) for that one line. So the real
  reading is: 108 modules open with `from core import cli` — the four-line entrypoint
  preamble `core/cli.py`'s own docstring describes.
- **`kneemri.kaggle.kernel_run`'s 72 importers are not library callers.** They are the
  per-entry kernel scripts under `comp/kernels/<entry>/<entry>.py`, each doing a deferred
  `from kneemri.kaggle.kernel_run import KernelRun` inside a function (`# noqa: PLC0415`).
  It is one call site copied across every submitted entry, not 72 independent dependents.
- Every other row is an ordinary in-package library import.

## SIZES

**36 modules** over 300 lines, largest first.

| lines | module | main() | importers |
| ---: | --- | :-: | ---: |
| 749 | `kneemri.models.backbone` | N | 24 |
| 748 | `kneemri.inference.members` | N | 25 |
| 735 | `kneemri.training.objective` | N | 7 |
| 698 | `kneemri.evaluation.member_curve` | Y | 2 |
| 694 | `kneemri.experiments.arm_check` | Y | 0 |
| 609 | `kneemri.inference.ensemble` | N | 30 |
| 587 | `core.data.corpus.study_cache` | N | 28 |
| 574 | `kneemri.models.model` | N | 14 |
| 533 | `kneemri.evaluation.run_comparison` | Y | 1 |
| 499 | `kneemri.training.device_budget` | N | 3 |
| 496 | `kneemri.training.flag_vocabulary` | N | 1 |
| 493 | `kneemri.training.trainer` | N | 5 |
| 484 | `kneemri.runs.member_record` | N | 3 |
| 484 | `kneemri.evaluation.attribution` | Y | 0 |
| 476 | `kneemri.evaluation.record_mix` | Y | 9 |
| 467 | `kneemri.training.run_setup` | N | 5 |
| 440 | `kneemri.kaggle.board_read` | Y | 2 |
| 429 | `kneemri.experiments.queue_check` | Y | 0 |
| 416 | `core.data.corpus.slot_reader` | N | 17 |
| 412 | `kneemri.evaluation.raptor_feed` | Y | 3 |
| 398 | `core.data.labels.rubric_gap` | Y | 0 |
| 394 | `kneemri.training.head_trial` | Y | 0 |
| 379 | `kneemri.kaggle.submission` | N | 6 |
| 377 | `kneemri.kaggle.mount_check` | Y | 0 |
| 376 | `core.data.labels.report_reader` | Y | 0 |
| 373 | `core.data.labels.reader_agreement` | Y | 8 |
| 371 | `kneemri.evaluation.competence_axis` | Y | 0 |
| 370 | `core.data.labels.targets` | N | 19 |
| 361 | `kneemri.evaluation.board_transfer` | Y | 0 |
| 361 | `core.data.corpus.slots` | N | 22 |
| 336 | `kneemri.runs.run_metrics` | N | 12 |
| 328 | `kneemri.kaggle.board_check` | Y | 1 |
| 328 | `kneemri.evaluation.supervision_gap` | Y | 8 |
| 327 | `core.data.labels.level_agreement` | Y | 0 |
| 312 | `kneemri.models.encoder_surgery` | N | 1 |
| 311 | `kneemri.evaluation.seat_bonus` | Y | 0 |

## FULL TABLE

One row per module, ordered by path. `main` = top-level `main()`. `imp` = non-test in-repo
importer count. `t` = test importer count. `importers` names the non-test importers (`-` if
none), **truncated to the first twelve** with a `+N more` count when there are more — the
count in `imp` is always complete. `lines` = raw line count. The statement is the module's
own docstring first line.

| module | lines | main | imp | t | importers | docstring first line |
| --- | ---: | :-: | ---: | ---: | --- | --- |
| `core` | 1 | N | 108 | 1 | `comp.verify_members`, `comp.witness_check`, `core.data.corpus.band_coverage`, `core.data.corpus.build_cache`, `core.data.corpus.build_merged`, `core.data.corpus.build_mrnet`, `core.data.corpus.build_rijeka`, `core.data.corpus.crop_alignment`, `core.data.corpus.crop_coverage`, `core.data.corpus.crop_laterality`, `core.data.corpus.decode_bench`, `core.data.corpus.diagnose_laterality`, …(+96 more) | Machine- and task-neutral infrastructure shared by every entrypoint: data locations, config, observability. |
| `core.bootstrap` | 140 | N | 13 | 1 | `core.data.corpus.band_coverage`, `core.data.corpus.crop_coverage`, `core.data.corpus.crop_laterality`, `kneemri.evaluation.alignment_damage`, `kneemri.evaluation.paired_floor`, `kneemri.evaluation.prevalence_match`, `kneemri.evaluation.ruler_transfer`, `kneemri.evaluation.scores`, `kneemri.evaluation.side_blindness`, `kneemri.evaluation.stack_coverage`, `kneemri.evaluation.supervision_gap`, `kneemri.evaluation.table_separation`, …(+1 more) | The resampling protocol every corpus measurement quotes its interval under. |
| `core.cli` | 79 | N | 108 | 1 | `comp.verify_members`, `comp.witness_check`, `core.data.corpus.band_coverage`, `core.data.corpus.build_cache`, `core.data.corpus.build_merged`, `core.data.corpus.build_mrnet`, `core.data.corpus.build_rijeka`, `core.data.corpus.crop_alignment`, `core.data.corpus.crop_coverage`, `core.data.corpus.crop_laterality`, `core.data.corpus.decode_bench`, `core.data.corpus.diagnose_laterality`, …(+96 more) | The four statements every `python -m` entrypoint in this repository opens with, written once. |
| `core.data` | 7 | N | 0 | 0 | - | The data layer: what a study IS (`corpus`) and what it is LABELLED with (`labels`). |
| `core.data.corpus` | 1 | N | 0 | 1 | - | corpus. |
| `core.data.corpus.acquisition` | 300 | N | 8 | 6 | `core.data.corpus.crop_laterality`, `core.data.corpus.diagnose_laterality`, `core.data.corpus.laterality`, `core.data.corpus.slot_geometry`, `core.data.corpus.slot_reader`, `core.data.corpus.slots`, `core.data.corpus.study`, `kneemri.evaluation.tie_divergence` | One MRI series, read as an acquisition: what it is, which way its slices run, and where it sits. |
| `core.data.corpus.anatomy` | 248 | N | 5 | 2 | `kneemri.evaluation.alignment_damage`, `kneemri.evaluation.competence_axis`, `kneemri.evaluation.run_comparison`, `kneemri.models.model`, `kneemri.training.head_trial` | Where each finding is actually read: the plane that shows it, and whether it needs fluid weighting. |
| `core.data.corpus.band_coverage` | 255 | Y | 0 | 1 | - | What the central slice band actually discards: `python -m core.data.corpus.band_coverage`. |
| `core.data.corpus.build_cache` | 100 | Y | 0 | 2 | - | Build the decode-once training cache: `python -m core.data.corpus.build_cache [--slices N] [--workers N]`. |
| `core.data.corpus.build_merged` | 78 | Y | 0 | 0 | - | Merge two decoded corpora into one cache: `python -m core.data.corpus.build_merged`. |
| `core.data.corpus.build_mrnet` | 69 | Y | 0 | 0 | - | Ingest the MRNet corpus: `python -m core.data.corpus.build_mrnet --archive PATH`. |
| `core.data.corpus.build_rijeka` | 58 | Y | 0 | 0 | - | Ingest the kneeMRI corpus: `python -m core.data.corpus.build_rijeka --archive PATH`. |
| `core.data.corpus.cache_flags` | 128 | N | 9 | 2 | `core.data.corpus.build_cache`, `core.data.corpus.build_merged`, `core.data.corpus.build_mrnet`, `core.data.corpus.build_rijeka`, `core.data.corpus.fetch_bench`, `kneemri.embeddings.build_embeddings`, `kneemri.training.flag_vocabulary`, `kneemri.training.head_trial`, `kneemri.training.pretrain` | The command-line flags that name a cache — one home, shared by every CLI that reads one. |
| `core.data.corpus.crop_alignment` | 235 | Y | 0 | 1 | - | Do a study's slots agree WHERE the knee is: `python -m core.data.corpus.crop_alignment`. |
| `core.data.corpus.crop_centre` | 169 | N | 9 | 11 | `core.data.corpus.band_coverage`, `core.data.corpus.cache_flags`, `core.data.corpus.crop_alignment`, `core.data.corpus.crop_coverage`, `core.data.corpus.crop_laterality`, `core.data.corpus.slot_reader`, `kneemri.inference.members`, `kneemri.runs.member_record`, `kneemri.training.options` | WHERE the physical crop is taken about — the image's middle, or the tissue's. |
| `core.data.corpus.crop_coverage` | 207 | Y | 3 | 3 | `core.data.corpus.crop_alignment`, `core.data.corpus.crop_laterality`, `core.data.corpus.slot_geometry` | Which crop centre actually keeps the knee: `python -m core.data.corpus.crop_coverage`. |
| `core.data.corpus.crop_laterality` | 240 | Y | 0 | 1 | - | Does the 130 mm crop cut the LATERAL compartment? `python -m core.data.corpus.crop_laterality`. |
| `core.data.corpus.decode_bench` | 120 | Y | 0 | 1 | - | Is decode I/O-bound or GIL-bound? `python -m core.data.corpus.decode_bench --studies 96`. |
| `core.data.corpus.diagnose_laterality` | 91 | Y | 0 | 1 | - | How was each study's side decided? `python -m core.data.corpus.diagnose_laterality --out <csv>`. |
| `core.data.corpus.diagnose_matrix` | 104 | Y | 0 | 1 | - | How many series hold slices of DIFFERENT matrix sizes? `python -m core.data.corpus.diagnose_matrix`. |
| `core.data.corpus.fetch_bench` | 143 | Y | 0 | 1 | - | How fast a batch reaches the trainer, per reader and per handle count: |
| `core.data.corpus.folds` | 147 | N | 11 | 4 | `core.data.corpus.build_cache`, `kneemri.evaluation.attribution`, `kneemri.evaluation.foreign_overlap`, `kneemri.evaluation.label_combine`, `kneemri.evaluation.member_curve`, `kneemri.evaluation.ruler_check`, `kneemri.inference.holdout_backfill`, `kneemri.inference.teacher_readings`, `kneemri.training.head_trial`, `kneemri.training.options`, `kneemri.training.train` | Splitting studies so a holdout number means something. |
| `core.data.corpus.laterality` | 63 | N | 2 | 1 | `core.data.corpus.diagnose_laterality`, `core.data.corpus.study` | Which knee was scanned, and how to map it onto a left-knee convention. |
| `core.data.corpus.merged_cache` | 146 | N | 1 | 1 | `core.data.corpus.build_merged` | Two decoded corpora as ONE cache, because the trainer reads exactly one. |
| `core.data.corpus.mrnet_corpus` | 279 | N | 1 | 1 | `core.data.corpus.build_mrnet` | The MRNet (Stanford) corpus, read into the arrays this project's trainer already loads. |
| `core.data.corpus.mrnet_labels` | 113 | N | 2 | 2 | `core.data.corpus.build_mrnet`, `core.data.corpus.mrnet_corpus` | Where one MRNet exam's ACL verdict comes from — two sources, because two layouts of this corpus ship. |
| `core.data.corpus.prefetch` | 59 | N | 1 | 1 | `kneemri.training.trainer` | Read the next batch while the current one is on the GPU, instead of in front of it. |
| `core.data.corpus.rijeka_corpus` | 235 | N | 1 | 1 | `core.data.corpus.build_rijeka` | The kneeMRI (Rijeka) corpus, read into the arrays this project's trainer already loads. |
| `core.data.corpus.slot_geometry` | 267 | Y | 0 | 1 | - | Do a study's slots image the same anatomy in patient space: `python -m core.data.corpus.slot_geometry`. |
| `core.data.corpus.slot_reader` | 416 | N | 17 | 20 | `core.data.corpus.band_coverage`, `core.data.corpus.cache_flags`, `core.data.corpus.crop_alignment`, `core.data.corpus.crop_coverage`, `core.data.corpus.crop_laterality`, `core.data.corpus.slot_geometry`, `core.data.corpus.study`, `core.data.corpus.study_cache`, `kneemri.evaluation.oof_oracle`, `kneemri.evaluation.tie_divergence`, `kneemri.inference.ensemble`, `kneemri.inference.members`, …(+5 more) | Turning one series into the fixed-shape uint8 stack the encoder eats. |
| `core.data.corpus.slot_ties` | 211 | Y | 1 | 1 | `kneemri.evaluation.oof_oracle` | How often two series compete for one slot — and whether that ambiguity is ever resolved differently. |
| `core.data.corpus.slots` | 361 | N | 22 | 18 | `core.data.corpus.anatomy`, `core.data.corpus.band_coverage`, `core.data.corpus.build_cache`, `core.data.corpus.build_merged`, `core.data.corpus.build_mrnet`, `core.data.corpus.build_rijeka`, `core.data.corpus.cache_flags`, `core.data.corpus.crop_alignment`, `core.data.corpus.crop_coverage`, `core.data.corpus.crop_laterality`, `core.data.corpus.laterality`, `core.data.corpus.slot_geometry`, …(+10 more) | Routing a study's series into the six acquisition slots the head attends over. |
| `core.data.corpus.study` | 122 | N | 11 | 2 | `core.data.corpus.band_coverage`, `core.data.corpus.crop_alignment`, `core.data.corpus.crop_coverage`, `core.data.corpus.crop_laterality`, `core.data.corpus.decode_bench`, `core.data.corpus.slot_geometry`, `core.data.corpus.slot_ties`, `core.data.corpus.study_cache`, `kneemri.evaluation.oof_oracle`, `kneemri.evaluation.tie_divergence`, `kneemri.kaggle.submission` | One study assembled into the bag of slot images the model reads, plus its presence mask. |
| `core.data.corpus.study_cache` | 587 | N | 28 | 21 | `core.data.corpus.band_coverage`, `core.data.corpus.build_cache`, `core.data.corpus.cache_flags`, `core.data.corpus.crop_alignment`, `core.data.corpus.crop_coverage`, `core.data.corpus.crop_laterality`, `core.data.corpus.decode_bench`, `core.data.corpus.diagnose_laterality`, `core.data.corpus.diagnose_matrix`, `core.data.corpus.merged_cache`, `core.data.corpus.mrnet_corpus`, `core.data.corpus.prefetch`, …(+16 more) | Decode every study once, into a memory-mapped uint8 cache the trainer can read many times. |
| `core.data.corpus.window_reader` | 124 | N | 2 | 3 | `core.data.corpus.fetch_bench`, `core.data.corpus.study_cache` | How a batch of studies reaches the trainer: by indexing the cache, or by reading its file outright. |
| `core.data.labels` | 1 | N | 0 | 0 | - | labels. |
| `core.data.labels.confident_labels` | 74 | Y | 1 | 2 | `kneemri.evaluation.unsure_cost` | Write a label table with the reader's unsure calls removed. |
| `core.data.labels.finding_calibration` | 215 | Y | 0 | 1 | - | Does the reader's scale mean the same thing for every FINDING, or is it shifted per finding? |
| `core.data.labels.label_power` | 104 | Y | 2 | 2 | `kneemri.evaluation.paired_floor`, `kneemri.evaluation.run_comparison` | How precisely each label's AUC can be measured at all: `python -m core.data.labels.label_power`. |
| `core.data.labels.level_agreement` | 327 | Y | 0 | 1 | - | Is one calibration constant right for all twelve findings? `python -m core.data.labels.level_agreement`. |
| `core.data.labels.overcall_clustering` | 150 | Y | 0 | 1 | - | Is the reader's over-call a property of the STUDY, or twelve independent coin flips? |
| `core.data.labels.reader_agreement` | 373 | Y | 8 | 3 | `core.data.labels.finding_calibration`, `core.data.labels.overcall_clustering`, `core.data.labels.reader_upgrade`, `core.data.labels.rubric_boundary`, `core.data.labels.rubric_gap`, `core.data.labels.severity_boundary`, `core.data.labels.table_blend`, `kneemri.evaluation.noise_binding` | Does the label reader agree with the radiologist equally in every language? |
| `core.data.labels.reader_disagreement` | 119 | Y | 0 | 1 | - | Where independent readers of the same report disagree: `python -m core.data.labels.reader_disagreement`. |
| `core.data.labels.reader_upgrade` | 155 | Y | 0 | 1 | - | Does a new reader beat the lexicon ON THE PAIRS THE LEXICON LEFT BLANK? |
| `core.data.labels.report_language` | 172 | Y | 4 | 1 | `core.data.labels.reader_agreement`, `core.data.labels.rubric_boundary`, `core.data.labels.rubric_gap`, `core.data.labels.severity_boundary` | What language each report is written in, and whether it grades its findings. |
| `core.data.labels.report_reader` | 376 | Y | 0 | 1 | - | Read the reports with a model instead of a lexicon: `python -m core.data.labels.report_reader`. |
| `core.data.labels.rubric_boundary` | 299 | Y | 0 | 1 | - | Where the HOST'S rubric and our report reader disagree: `python -m core.data.labels.rubric_boundary`. |
| `core.data.labels.rubric_gap` | 398 | Y | 0 | 1 | - | How many derived targets does the host's published rubric say are wrong? `python -m core.data.labels.rubric_gap`. |
| `core.data.labels.ruler` | 55 | N | 1 | 2 | `kneemri.experiments.arm_check` | Whether two label tables score the same holdout the same way — the ruler behind every recorded AUC. |
| `core.data.labels.severity_boundary` | 166 | Y | 1 | 1 | `core.data.labels.rubric_gap` | Does the 58-study ruler obey its own severity rule? `python -m core.data.labels.severity_boundary`. |
| `core.data.labels.table_blend` | 180 | Y | 1 | 2 | `core.data.labels.reader_disagreement` | Combine several report readers into one label table: `python -m core.data.labels.table_blend`. |
| `core.data.labels.targets` | 370 | N | 19 | 8 | `core.data.labels.finding_calibration`, `core.data.labels.level_agreement`, `core.data.labels.report_reader`, `core.data.labels.ruler`, `kneemri.evaluation.attribution`, `kneemri.evaluation.label_combine`, `kneemri.evaluation.member_curve`, `kneemri.evaluation.paired_floor`, `kneemri.evaluation.ruler_check`, `kneemri.evaluation.stack_coverage`, `kneemri.inference.holdout_backfill`, `kneemri.kaggle.dry_run`, …(+7 more) | The training targets: twelve graded scores per study, each with the confidence it was read at. |
| `core.denominator` | 42 | N | 11 | 1 | `core.data.corpus.study_cache`, `kneemri.evaluation.vote_ceiling`, `kneemri.experiments.arm_check`, `kneemri.experiments.convergence_check`, `kneemri.experiments.ledger_check`, `kneemri.experiments.trajectory_check`, `kneemri.kaggle.champion_check`, `kneemri.kaggle.decode_entry`, `kneemri.kaggle.kit_check`, `kneemri.kaggle.mount_check`, `kneemri.kaggle.roster_reads` | What a check MEASURED, and the refusal owed when that is nothing. |
| `core.findings` | 98 | N | 52 | 75 | `comp.kernels.kneemri-graft-verify.kneemri-graft-verify`, `core.data.corpus.mrnet_corpus`, `core.data.corpus.rijeka_corpus`, `core.data.labels.confident_labels`, `core.data.labels.finding_calibration`, `core.data.labels.label_power`, `core.data.labels.level_agreement`, `core.data.labels.reader_agreement`, `core.data.labels.reader_disagreement`, `core.data.labels.report_reader`, `core.data.labels.rubric_gap`, `core.data.labels.severity_boundary`, …(+40 more) | The twelve findings, in the ONE order everything downstream agrees on: `core.findings`. |
| `core.mounts` | 73 | N | 15 | 5 | `comp.kernels.kneemri-fleettwo.kneemri-fleettwo`, `comp.kernels.kneemri-graft-verify.kneemri-graft-verify`, `comp.kernels.kneemri-graft.kneemri-graft`, `comp.kernels.kneemri-halfseat.kneemri-halfseat`, `comp.kernels.kneemri-holdswap.kneemri-holdswap`, `comp.kernels.kneemri-onepair.kneemri-onepair`, `comp.kernels.kneemri-raptor.kneemri-raptor`, `comp.kernels.kneemri-raptorseat.kneemri-raptorseat`, `comp.kernels.kneemri-raptorweight.kneemri-raptorweight`, `comp.kernels.kneemri-spanseat.kneemri-spanseat`, `comp.kernels.kneemri-stackseat.kneemri-stackseat`, `comp.kernels.kneemri-vjepaseat.kneemri-vjepaseat`, …(+3 more) | Finding a file under `/kaggle/input` without walking the competition data. |
| `core.native_path` | 107 | N | 1 | 1 | `core.paths` | A data root written for one OS, read on another. |
| `core.paths` | 105 | N | 16 | 13 | `comp.build_kit`, `comp.verify_kit`, `core.cli`, `core.data.corpus.decode_bench`, `core.data.labels.reader_agreement`, `kneemri.evaluation.annotated_readings`, `kneemri.evaluation.foreign_overlap`, `kneemri.evaluation.vote_ceiling`, `kneemri.experiments.arm_check`, `kneemri.inference.holdout_backfill`, `kneemri.inference.members`, `kneemri.inference.teacher_readings`, …(+4 more) | Machine-local data locations, adapted to the OS the process is actually running on. |
| `core.sleep_policy` | 98 | N | 1 | 2 | `core.wakelock` | What each platform can be asked about its idle timer, and which policy this one gets. |
| `core.wakelock` | 36 | N | 12 | 1 | `core.data.corpus.build_cache`, `core.data.corpus.build_merged`, `core.data.corpus.build_mrnet`, `core.data.corpus.build_rijeka`, `kneemri.embeddings.build_embeddings`, `kneemri.evaluation.annotated_readings`, `kneemri.evaluation.attribution`, `kneemri.evaluation.member_curve`, `kneemri.inference.holdout_backfill`, `kneemri.inference.teacher_readings`, `kneemri.training.fitting`, `kneemri.training.pretrain` | Hold the machine awake for as long as a long job is running. |
| `kneemri` | 1 | N | 0 | 1 | - | Twelve findings from one knee MRI study: series ingest, encoding, study-level aggregation, submission. |
| `kneemri.commands` | 113 | Y | 0 | 1 | - | Every `python -m` entrypoint the package has, generated from the modules themselves. |
| `kneemri.device` | 25 | N | 26 | 2 | `comp.kernels.kneemri-fleettwo.kneemri-fleettwo`, `comp.kernels.kneemri-halfseat.kneemri-halfseat`, `comp.kernels.kneemri-holdswap.kneemri-holdswap`, `comp.kernels.kneemri-onepair.kneemri-onepair`, `comp.kernels.kneemri-raptor.kneemri-raptor`, `comp.kernels.kneemri-raptorseat.kneemri-raptorseat`, `comp.kernels.kneemri-raptorweight.kneemri-raptorweight`, `comp.kernels.kneemri-spanseat.kneemri-spanseat`, `comp.kernels.kneemri-stackseat.kneemri-stackseat`, `comp.kernels.kneemri-vjepaseat.kneemri-vjepaseat`, `comp.verify_members`, `comp.witness_check`, …(+14 more) | Which device this process trains, encodes or scores on — decided in ONE place. |
| `kneemri.embeddings` | 1 | N | 0 | 0 | - | embeddings. |
| `kneemri.embeddings.build_embeddings` | 100 | Y | 0 | 1 | - | Encode a pixel cache into an embedding cache: `python -m kneemri.embeddings.build_embeddings`. |
| `kneemri.embeddings.embedding_cache` | 223 | N | 2 | 1 | `kneemri.embeddings.build_embeddings`, `kneemri.training.head_trial` | Encode the corpus once, so every question that is not about pixels costs seconds instead of hours. |
| `kneemri.embeddings.report_embeddings` | 125 | Y | 0 | 1 | - | Embed each study's radiology report once, so the image model can be supervised by all of it. |
| `kneemri.evaluation` | 1 | N | 2 | 0 | `kneemri.evaluation.raptor_feed`, `kneemri.evaluation.raptor_model` | evaluation. |
| `kneemri.evaluation.alignment_damage` | 233 | Y | 0 | 1 | - | Does cross-slot misregistration cost AUC, and only on the findings small enough to care: |
| `kneemri.evaluation.annotated_readings` | 215 | Y | 2 | 1 | `kneemri.evaluation.noise_binding`, `kneemri.evaluation.supervision_gap` | Our own members read over the annotated studies: `python -m kneemri.evaluation.annotated_readings`. |
| `kneemri.evaluation.attribution` | 484 | Y | 0 | 1 | - | What is the model actually using — measured by taking things away, not by drawing heatmaps. |
| `kneemri.evaluation.board_transfer` | 361 | Y | 0 | 1 | - | Does our holdout ORDER entries the way the leaderboard does? `python -m kneemri.evaluation.board_transfer`. |
| `kneemri.evaluation.competence_axis` | 371 | Y | 0 | 1 | - | Whether the whole member pool differs along ONE trade-off axis — the question a seat search cannot ask. |
| `kneemri.evaluation.encoder_drift` | 185 | Y | 0 | 1 | - | How far supervision actually moved the encoder away from the weights it was pretrained with. |
| `kneemri.evaluation.foreign_gate` | 105 | Y | 0 | 1 | - | The plumbing oracle for a foreign member: `python -m kneemri.evaluation.foreign_gate`. |
| `kneemri.evaluation.foreign_member` | 226 | Y | 2 | 1 | `kneemri.evaluation.raptor_feed`, `kneemri.evaluation.raptor_model` | Serve a FOREIGN checkpoint through ITS OWN decode, and record the result as one of our members. |
| `kneemri.evaluation.foreign_overlap` | 110 | Y | 0 | 1 | - | Did a foreign model train on our holdout? `python -m kneemri.evaluation.foreign_overlap`. |
| `kneemri.evaluation.foreign_ruler` | 276 | Y | 0 | 1 | - | A foreign model and ours on ONE ruler: `python -m kneemri.evaluation.foreign_ruler`. |
| `kneemri.evaluation.gold_rows` | 83 | Y | 3 | 2 | `kneemri.evaluation.annotated_readings`, `kneemri.evaluation.level_calibration`, `kneemri.evaluation.supervision_gap` | The radiologist-annotated rows of train.csv, and a report table read onto them. |
| `kneemri.evaluation.holdout_metrics` | 87 | N | 1 | 1 | `kneemri.evaluation.run_comparison` | The scored epoch a run's KEPT PREDICTIONS reconstruct, for the runs that predate the recorder. |
| `kneemri.evaluation.label_combine` | 214 | Y | 1 | 1 | `kneemri.evaluation.member_curve` | Combine members PER LABEL, and say honestly what that is worth. |
| `kneemri.evaluation.label_probe` | 173 | Y | 0 | 1 | - | Whether a label interaction GENERALISES — the verdict half of the residual-coupling gate. |
| `kneemri.evaluation.label_stability` | 107 | Y | 0 | 1 | - | Which weak labels are reproducible and which are the holdout's noise: `python -m kneemri.evaluation.label_stability`. |
| `kneemri.evaluation.level_calibration` | 186 | Y | 0 | 1 | - | The report reader's confidence levels, re-mapped onto the rate at which they are actually positive. |
| `kneemri.evaluation.member_axes` | 181 | Y | 1 | 1 | `kneemri.kaggle.seat_value` | The three distance rulers read TOGETHER, because each alone names a different member as the good one. |
| `kneemri.evaluation.member_curve` | 698 | Y | 2 | 2 | `kneemri.evaluation.record_mix`, `kneemri.kaggle.seat_value` | What does each additional ensemble member actually buy? |
| `kneemri.evaluation.member_distance` | 107 | N | 3 | 2 | `kneemri.evaluation.member_axes`, `kneemri.evaluation.pool_census`, `kneemri.kaggle.seat_value` | How far a candidate's ORDERING sits from the roster's — the axis the holdout cannot see. |
| `kneemri.evaluation.member_divergence` | 181 | N | 3 | 1 | `kneemri.evaluation.member_axes`, `kneemri.evaluation.pool_census`, `kneemri.kaggle.seat_value` | How far apart two members are in CONFIDENCE — the axis a rank average throws away. |
| `kneemri.evaluation.member_mahalanobis` | 155 | Y | 1 | 1 | `kneemri.evaluation.member_axes` | Distance between members with the POOL'S OWN COVARIANCE divided out — far only along an unspanned axis. |
| `kneemri.evaluation.member_space` | 39 | N | 2 | 2 | `kneemri.evaluation.member_axes`, `kneemri.evaluation.member_mahalanobis` | The members as POINTS — the feature space every member-distance ruler is taken in. |
| `kneemri.evaluation.noise_binding` | 259 | Y | 0 | 1 | - | Does a member track the LEXICON or the RADIOLOGIST where the two disagree? |
| `kneemri.evaluation.oof_oracle` | 291 | Y | 0 | 1 | - | Does OUR pipeline hand a published member the same pixels its author did? |
| `kneemri.evaluation.ordinal_signal` | 246 | Y | 0 | 1 | - | Whether the graded YES levels in the report table carry ORDINAL information, or only certainty. |
| `kneemri.evaluation.paired_floor` | 172 | Y | 0 | 1 | - | How tightly a per-label delta can be read when both arms saw the SAME studies. |
| `kneemri.evaluation.pool` | 101 | N | 10 | 12 | `kneemri.evaluation.competence_axis`, `kneemri.evaluation.member_axes`, `kneemri.evaluation.member_curve`, `kneemri.evaluation.member_distance`, `kneemri.evaluation.member_divergence`, `kneemri.evaluation.member_space`, `kneemri.evaluation.pool_census`, `kneemri.evaluation.record_mix`, `kneemri.evaluation.vote_ceiling`, `kneemri.kaggle.decode_entry` | What several members predicted for the same studies — the input every offline reading joins on. |
| `kneemri.evaluation.pool_census` | 230 | Y | 0 | 1 | - | Is there a member in this pool that is STRONG and sees the knee DIFFERENTLY — or is that cell empty? |
| `kneemri.evaluation.prevalence_match` | 188 | Y | 0 | 1 | - | Is the medial-over-lateral deficit PREVALENCE, or is it vision? `python -m kneemri.evaluation.prevalence_match`. |
| `kneemri.evaluation.protocol_prior` | 191 | Y | 2 | 1 | `kneemri.evaluation.foreign_gate`, `kneemri.evaluation.foreign_ruler` | Does the ensemble already use ALL of the acquisition-protocol signal, or only part of it? |
| `kneemri.evaluation.published_ruler` | 108 | N | 6 | 1 | `kneemri.evaluation.paired_floor`, `kneemri.evaluation.prevalence_match`, `kneemri.evaluation.ruler_transfer`, `kneemri.evaluation.stack_coverage`, `kneemri.evaluation.supervision_gap`, `kneemri.evaluation.weights_gap` | The whole-corpus ruler the twenty published checkpoints add up to. |
| `kneemri.evaluation.raptor_feed` | 412 | Y | 3 | 2 | `kneemri.evaluation.raptor_model`, `kneemri.kaggle.raptor_run`, `kneemri.kaggle.raptor_teacher` | The raptor decode, reimplemented as OUR code from its published description. |
| `kneemri.evaluation.raptor_inventory` | 128 | Y | 0 | 1 | - | What a directory of foreign checkpoints actually CONTAINS, read from the checkpoints themselves. |
| `kneemri.evaluation.raptor_model` | 242 | Y | 13 | 4 | `comp.kernels.kneemri-fleettwo.kneemri-fleettwo`, `comp.kernels.kneemri-halfseat.kneemri-halfseat`, `comp.kernels.kneemri-holdswap.kneemri-holdswap`, `comp.kernels.kneemri-onepair.kneemri-onepair`, `comp.kernels.kneemri-raptor.kneemri-raptor`, `comp.kernels.kneemri-raptorseat.kneemri-raptorseat`, `comp.kernels.kneemri-raptorweight.kneemri-raptorweight`, `comp.kernels.kneemri-spanseat.kneemri-spanseat`, `comp.kernels.kneemri-stackseat.kneemri-stackseat`, `comp.kernels.kneemri-vjepaseat.kneemri-vjepaseat`, `kneemri.evaluation.raptor_inventory`, `kneemri.kaggle.raptor_run`, …(+1 more) | The raptor model, reimplemented as OUR code so a foreign checkpoint can be SERVED, not only measured. |
| `kneemri.evaluation.record_mix` | 476 | Y | 9 | 1 | `kneemri.evaluation.competence_axis`, `kneemri.evaluation.member_axes`, `kneemri.evaluation.member_mahalanobis`, `kneemri.evaluation.pool_census`, `kneemri.evaluation.side_blindness`, `kneemri.evaluation.vote_ceiling`, `kneemri.kaggle.decode_entry`, `kneemri.kaggle.entry_offline`, `kneemri.kaggle.seat_value` | Score ensembles of members that are already on disk, with no card and no inference pass. |
| `kneemri.evaluation.residual_coupling` | 127 | N | 1 | 1 | `kneemri.evaluation.label_probe` | How much label structure survives in a model's RESIDUAL — the gate on a label-correlation head. |
| `kneemri.evaluation.ruler_check` | 146 | Y | 0 | 1 | - | Does the choice of evaluation ruler change WHICH recipe wins? |
| `kneemri.evaluation.ruler_transfer` | 264 | Y | 0 | 1 | - | Does a DELTA between two arms survive changing the ruler? `python -m kneemri.evaluation.ruler_transfer`. |
| `kneemri.evaluation.run_comparison` | 533 | Y | 1 | 1 | `kneemri.experiments.arm_read` | Read several runs' metrics files side by side: `python -m kneemri.evaluation.run_comparison`. |
| `kneemri.evaluation.scores` | 194 | N | 30 | 12 | `kneemri.evaluation.alignment_damage`, `kneemri.evaluation.annotated_readings`, `kneemri.evaluation.attribution`, `kneemri.evaluation.competence_axis`, `kneemri.evaluation.holdout_metrics`, `kneemri.evaluation.label_combine`, `kneemri.evaluation.label_probe`, `kneemri.evaluation.member_curve`, `kneemri.evaluation.ordinal_signal`, `kneemri.evaluation.paired_floor`, `kneemri.evaluation.pool`, `kneemri.evaluation.pool_census`, …(+18 more) | Scoring predictions against targets — the competition's metric, and nothing else. |
| `kneemri.evaluation.seat_bonus` | 311 | Y | 0 | 1 | - | What a seat is worth: the board gain of an ensemble over the SOLO entry of its own first member. |
| `kneemri.evaluation.side_blindness` | 233 | Y | 0 | 1 | - | Can the model tell the two compartments apart at all? `python -m kneemri.evaluation.side_blindness`. |
| `kneemri.evaluation.stack_coverage` | 216 | Y | 0 | 1 | - | Does keeping 12 slices of a deep stack cost accuracy? `python -m kneemri.evaluation.stack_coverage`. |
| `kneemri.evaluation.supervision_gap` | 328 | Y | 8 | 2 | `kneemri.evaluation.foreign_gate`, `kneemri.evaluation.foreign_ruler`, `kneemri.evaluation.noise_binding`, `kneemri.evaluation.prevalence_match`, `kneemri.evaluation.ruler_transfer`, `kneemri.evaluation.run_comparison`, `kneemri.evaluation.table_separation`, `kneemri.evaluation.unsure_cost` | Is a finding held back by its LABELS or by the MODEL? `python -m kneemri.evaluation.supervision_gap`. |
| `kneemri.evaluation.table_separation` | 129 | Y | 0 | 1 | - | Are two label tables actually separated on the gold 58? `python -m kneemri.evaluation.table_separation`. |
| `kneemri.evaluation.teacher_distance` | 140 | Y | 0 | 1 | - | How far foreign members sit from EACH OTHER, measured with no labels and no submission slot. |
| `kneemri.evaluation.tie_divergence` | 203 | Y | 0 | 1 | - | When two series tie for a slot, do they SHOW the same thing? `python -m kneemri.evaluation.tie_divergence`. |
| `kneemri.evaluation.unsure_cost` | 180 | Y | 0 | 1 | - | What an `UNK` cell is actually worth: `python -m kneemri.evaluation.unsure_cost`. |
| `kneemri.evaluation.vote_ceiling` | 187 | Y | 0 | 1 | - | What is the VOTE worth, once the fitting is paid for? |
| `kneemri.evaluation.weights_gap` | 187 | Y | 0 | 1 | - | Where do THEIR weights beat OURS, per label, on studies both held out? |
| `kneemri.experiments` | 1 | N | 0 | 0 | - | experiments. |
| `kneemri.experiments.arm_check` | 694 | Y | 0 | 1 | - | Prove every committed arm's flags before a queue costs GPU: `python -m kneemri.experiments.arm_check`. |
| `kneemri.experiments.arm_diff` | 135 | N | 1 | 1 | `kneemri.experiments.arm_check` | One arm's flag line read against ANOTHER arm's — the two checks whose subject is a pair. |
| `kneemri.experiments.arm_gate` | 80 | N | 1 | 1 | `kneemri.experiments.arm_header` | An arm's PRECONDITION, as a line something can evaluate rather than a sentence in its header. |
| `kneemri.experiments.arm_header` | 147 | N | 3 | 3 | `kneemri.experiments.arm_check`, `kneemri.experiments.arm_diff`, `kneemri.experiments.arm_read` | What an arm's header DECLARES, read from the shell that printed the arm's name. |
| `kneemri.experiments.arm_read` | 119 | Y | 0 | 1 | - | Read a finished arm the way its own header said to: per label, against the control it declared. |
| `kneemri.experiments.arm_smoke` | 284 | N | 1 | 1 | `kneemri.experiments.arm_check` | Run every accepted arm for one epoch on a few dozen studies: `arm_check --smoke <dir>`. |
| `kneemri.experiments.arm_verdict` | 56 | N | 1 | 1 | `kneemri.experiments.arm_check` | What a pre-flight check SAYS, and whether it refuses the run — apart from what it detects. |
| `kneemri.experiments.budget_residuals` | 149 | Y | 0 | 1 | - | What the VRAM cost model DERIVES against what the arms MEASURED: `python -m |
| `kneemri.experiments.card_check` | 265 | Y | 0 | 1 | - | Is the GPU held by exactly one of our jobs? |
| `kneemri.experiments.convergence_check` | 290 | Y | 1 | 1 | `kneemri.experiments.trajectory_check` | Which runs actually FINISHED: `python -m kneemri.experiments.convergence_check`. |
| `kneemri.experiments.ledger_check` | 290 | Y | 0 | 1 | - | Every run that finished should appear in the ledger: `python -m kneemri.experiments.ledger_check`. |
| `kneemri.experiments.progress` | 101 | Y | 0 | 1 | - | What the card is doing RIGHT NOW: `python -m kneemri.experiments.progress`. |
| `kneemri.experiments.queue_check` | 429 | Y | 0 | 1 | - | Is every queued arm still gated by a live process? |
| `kneemri.experiments.stale_gap` | 152 | Y | 0 | 1 | - | What `--patience` would have to be, derived from every run's own curve instead of chosen. |
| `kneemri.experiments.switch_coverage` | 106 | N | 1 | 1 | `kneemri.experiments.arm_check` | Every training switch is armed, smoke-covered, or declared not a recipe: `arm_check --switches`. |
| `kneemri.experiments.trajectory_check` | 276 | Y | 0 | 1 | - | Can a running arm still reach its branch: `python -m kneemri.experiments.trajectory_check`. |
| `kneemri.experiments.vram_need` | 37 | Y | 1 | 1 | `kneemri.experiments.arm_smoke` | What one arm's flags will reserve, in MiB, for the shell that decides whether to start it. |
| `kneemri.inference` | 1 | N | 0 | 1 | - | inference. |
| `kneemri.inference.budget` | 81 | N | 11 | 4 | `comp.kernels.kneemri-fleettwo.kneemri-fleettwo`, `comp.kernels.kneemri-halfseat.kneemri-halfseat`, `comp.kernels.kneemri-holdswap.kneemri-holdswap`, `comp.kernels.kneemri-onepair.kneemri-onepair`, `comp.kernels.kneemri-raptorseat.kneemri-raptorseat`, `comp.kernels.kneemri-raptorweight.kneemri-raptorweight`, `comp.kernels.kneemri-spanseat.kneemri-spanseat`, `comp.kernels.kneemri-stackseat.kneemri-stackseat`, `comp.kernels.kneemri-vjepaseat.kneemri-vjepaseat`, `kneemri.inference.ensemble`, `kneemri.kaggle.kernel_run` | The wall-clock a Kaggle batch session gives a run, and whether the run is on course to overspend it. |
| `kneemri.inference.ensemble` | 609 | N | 30 | 7 | `comp.kernels.kneemri-blendseat.kneemri-blendseat`, `comp.kernels.kneemri-blendthree.kneemri-blendthree`, `comp.kernels.kneemri-calibmix.kneemri-calibmix`, `comp.kernels.kneemri-calibsix.kneemri-calibsix`, `comp.kernels.kneemri-fleettwo.kneemri-fleettwo`, `comp.kernels.kneemri-graft.kneemri-graft`, `comp.kernels.kneemri-greedyseat.kneemri-greedyseat`, `comp.kernels.kneemri-halfseat.kneemri-halfseat`, `comp.kernels.kneemri-holdswap.kneemri-holdswap`, `comp.kernels.kneemri-honest12.kneemri-honest12`, `comp.kernels.kneemri-jitterfit.kneemri-jitterfit`, `comp.kernels.kneemri-loadcheck.kneemri-loadcheck`, …(+18 more) | Predicting a study from many members, and combining them the way the metric actually reads. |
| `kneemri.inference.foreign_head` | 52 | N | 1 | 1 | `kneemri.inference.foreign_member` | The pooling head a published foreign checkpoint carries, rebuilt from the shapes in its own state dict. |
| `kneemri.inference.foreign_holdout` | 110 | Y | 0 | 1 | - | Give a published foreign checkpoint a holdout record: `python -m kneemri.inference.foreign_holdout`. |
| `kneemri.inference.foreign_member` | 213 | N | 2 | 3 | `kneemri.evaluation.foreign_gate`, `kneemri.inference.foreign_holdout` | A published foreign checkpoint, rebuilt from its own archive and read through OUR corpus. |
| `kneemri.inference.holdout_backfill` | 200 | Y | 0 | 1 | - | Re-derive an already-finished run's holdout predictions: `python -m kneemri.inference.holdout_backfill`. |
| `kneemri.inference.holdout_predictions` | 150 | N | 8 | 4 | `kneemri.evaluation.annotated_readings`, `kneemri.evaluation.attribution`, `kneemri.evaluation.label_combine`, `kneemri.evaluation.member_curve`, `kneemri.evaluation.oof_oracle`, `kneemri.evaluation.ruler_check`, `kneemri.inference.holdout_backfill`, `kneemri.inference.teacher_readings` | One member's probabilities for a fixed set of studies — the half of an analysis that needs a GPU. |
| `kneemri.inference.holdout_slab` | 60 | N | 7 | 6 | `kneemri.evaluation.attribution`, `kneemri.evaluation.oof_oracle`, `kneemri.inference.holdout_backfill`, `kneemri.inference.holdout_predictions`, `kneemri.inference.member_readings`, `kneemri.inference.teacher_readings`, `kneemri.inference.trainer_record` | The holdout's pixels, handed to a model one batch at a time instead of all at once. |
| `kneemri.inference.member_readings` | 73 | N | 4 | 1 | `kneemri.evaluation.annotated_readings`, `kneemri.evaluation.label_combine`, `kneemri.evaluation.member_curve`, `kneemri.evaluation.ruler_check` | Members grouped by the cache each one names, so every member is read from its OWN pixels. |
| `kneemri.inference.members` | 748 | N | 25 | 16 | `comp.kernels.kneemri-fusion.kneemri-fusion`, `comp.kernels.kneemri-graft-verify.kneemri-graft-verify`, `comp.kernels.kneemri-graft.kneemri-graft`, `comp.kernels.kneemri-loadcheck.kneemri-loadcheck`, `comp.verify_members`, `comp.witness_check`, `kneemri.evaluation.annotated_readings`, `kneemri.evaluation.attribution`, `kneemri.evaluation.label_combine`, `kneemri.evaluation.member_curve`, `kneemri.evaluation.oof_oracle`, `kneemri.evaluation.ruler_check`, …(+13 more) | Reading a package of trained members, and proving each one is the model it was. |
| `kneemri.inference.prediction_cache` | 129 | N | 7 | 3 | `kneemri.evaluation.annotated_readings`, `kneemri.evaluation.label_combine`, `kneemri.evaluation.member_curve`, `kneemri.evaluation.ruler_check`, `kneemri.inference.holdout_backfill`, `kneemri.inference.holdout_predictions`, `kneemri.inference.teacher_readings` | Probabilities already computed for a member on a fixed holdout — the store, not the reading. |
| `kneemri.inference.serve_bench` | 216 | Y | 0 | 1 | - | Does compiling the encoder pay at SERVING time? `python -m kneemri.inference.serve_bench`. |
| `kneemri.inference.teacher_readings` | 140 | Y | 0 | 1 | - | Write an ensemble's ordering of the TRAINING studies for a student to be taught: |
| `kneemri.inference.trainer_record` | 59 | N | 1 | 1 | `kneemri.inference.holdout_predictions` | The array a training run already wrote for its own holdout, offered in place of an inference pass. |
| `kneemri.inference.vote` | 265 | N | 34 | 11 | `comp.kernels.kneemri-blendseat.kneemri-blendseat`, `comp.kernels.kneemri-blendthree.kneemri-blendthree`, `comp.kernels.kneemri-calibmix.kneemri-calibmix`, `comp.kernels.kneemri-calibsix.kneemri-calibsix`, `comp.kernels.kneemri-fleettwo.kneemri-fleettwo`, `comp.kernels.kneemri-greedyseat.kneemri-greedyseat`, `comp.kernels.kneemri-halfseat.kneemri-halfseat`, `comp.kernels.kneemri-holdswap.kneemri-holdswap`, `comp.kernels.kneemri-honest12.kneemri-honest12`, `comp.kernels.kneemri-onepair.kneemri-onepair`, `comp.kernels.kneemri-pixgroup.kneemri-pixgroup`, `comp.kernels.kneemri-raptorseat.kneemri-raptorseat`, …(+22 more) | HOW members vote, as a family rather than as one hardcoded rule. |
| `kneemri.kaggle` | 1 | N | 1 | 2 | `kneemri.kaggle.efficiency_read` | kaggle. |
| `kneemri.kaggle.board_check` | 328 | Y | 1 | 1 | `kneemri.kaggle.entry_offline` | Is the submission record COMPLETE? `python -m kneemri.kaggle.board_check --live`. |
| `kneemri.kaggle.board_read` | 440 | Y | 2 | 1 | `kneemri.kaggle.champion_check`, `kneemri.kaggle.efficiency_read` | What are we actually behind? `python -m kneemri.kaggle.board_read --live`. |
| `kneemri.kaggle.champion_check` | 238 | Y | 2 | 2 | `kneemri.kaggle.priceable_entries`, `kneemri.kaggle.seat_value` | Can the best entry this project has still be rebuilt? `python -m kneemri.kaggle.champion_check`. |
| `kneemri.kaggle.decode_entry` | 173 | Y | 0 | 1 | - | The best roster a DECODE BUDGET can buy: `python -m kneemri.kaggle.decode_entry`. |
| `kneemri.kaggle.discussion_read` | 202 | Y | 0 | 1 | - | What are the other competitors saying? `python -m kneemri.kaggle.discussion_read --live`. |
| `kneemri.kaggle.dry_run` | 125 | Y | 0 | 1 | - | Run the SUBMISSION path over labelled studies: `python -m kneemri.kaggle.dry_run --studies 40`. |
| `kneemri.kaggle.efficiency_read` | 231 | Y | 0 | 1 | - | Where do we stand on the OTHER leaderboard? `python -m kneemri.kaggle.efficiency_read --live`. |
| `kneemri.kaggle.entry_offline` | 168 | Y | 0 | 1 | - | Why is a board entry's `offline` null? `python -m kneemri.kaggle.entry_offline [--write]`. |
| `kneemri.kaggle.kernel_closure` | 76 | N | 1 | 1 | `kneemri.kaggle.kit_check` | Which of our files a kernel actually loads, read from the imports rather than from the package list. |
| `kneemri.kaggle.kernel_run` | 246 | N | 72 | 2 | `comp.kernels.kneemri-accumseat.kneemri-accumseat`, `comp.kernels.kneemri-allfit.kneemri-allfit`, `comp.kernels.kneemri-augseat.kneemri-augseat`, `comp.kernels.kneemri-band.kneemri-band`, `comp.kernels.kneemri-basedense.kneemri-basedense`, `comp.kernels.kneemri-best4.kneemri-best4`, `comp.kernels.kneemri-blendseat.kneemri-blendseat`, `comp.kernels.kneemri-blendthree.kneemri-blendthree`, `comp.kernels.kneemri-calibmix.kneemri-calibmix`, `comp.kernels.kneemri-calibsix.kneemri-calibsix`, `comp.kernels.kneemri-calibsolo.kneemri-calibsolo`, `comp.kernels.kneemri-calibspan.kneemri-calibspan`, …(+60 more) | Everything a Kaggle kernel does once it can import us. |
| `kneemri.kaggle.kernel_script` | 81 | N | 2 | 1 | `kneemri.kaggle.board_check`, `kneemri.kaggle.mount_check` | Which members a kernel will ask for, read from the `KernelRun(...)` call in its script. |
| `kneemri.kaggle.kernel_sources` | 50 | N | 2 | 1 | `comp.verify_members`, `kneemri.kaggle.mount_check` | What a kernel declares it will have attached, read from its `kernel-metadata.json`. |
| `kneemri.kaggle.kit_check` | 163 | Y | 1 | 1 | `comp.build_kit` | Is the kit we are about to ship the code we are running? `python -m kneemri.kaggle.kit_check`. |
| `kneemri.kaggle.kit_wheels` | 58 | N | 2 | 1 | `comp.build_kit`, `comp.verify_kit` | Which wheels a kit MUST carry, and why each one decides whether a member loads at all. |
| `kneemri.kaggle.live_dataset` | 159 | N | 1 | 1 | `kneemri.kaggle.mount_check` | What a Kaggle dataset actually holds, asked of Kaggle rather than of the directory it was built in. |
| `kneemri.kaggle.mount_check` | 377 | Y | 0 | 1 | - | Does every kernel declare the mounts its members need? `python -m kneemri.kaggle.mount_check`. |
| `kneemri.kaggle.mounted_kit` | 80 | N | 1 | 1 | `kneemri.kaggle.kit_check` | The stamp on the dataset Kaggle will MOUNT, as opposed to the kit sitting on this disk. |
| `kneemri.kaggle.package_members` | 217 | Y | 2 | 3 | `kneemri.kaggle.champion_check`, `kneemri.kaggle.roster_check` | Package our trained members for Kaggle: `python -m kneemri.kaggle.package_members`. |
| `kneemri.kaggle.priceable_entries` | 82 | N | 1 | 1 | `kneemri.kaggle.seat_value` | Which scored entries can still price a seat, since the ones that scored best cannot. |
| `kneemri.kaggle.raptor_run` | 120 | N | 11 | 1 | `comp.kernels.kneemri-fleettwo.kneemri-fleettwo`, `comp.kernels.kneemri-halfseat.kneemri-halfseat`, `comp.kernels.kneemri-holdswap.kneemri-holdswap`, `comp.kernels.kneemri-onepair.kneemri-onepair`, `comp.kernels.kneemri-raptor.kneemri-raptor`, `comp.kernels.kneemri-raptorseat.kneemri-raptorseat`, `comp.kernels.kneemri-raptorweight.kneemri-raptorweight`, `comp.kernels.kneemri-spanseat.kneemri-spanseat`, `comp.kernels.kneemri-stackseat.kneemri-stackseat`, `comp.kernels.kneemri-vjepaseat.kneemri-vjepaseat`, `kneemri.kaggle.raptor_teacher` | Serving the raptor member on a Kaggle split, which our own runner cannot do. |
| `kneemri.kaggle.raptor_seat` | 95 | N | 9 | 1 | `comp.kernels.kneemri-fleettwo.kneemri-fleettwo`, `comp.kernels.kneemri-halfseat.kneemri-halfseat`, `comp.kernels.kneemri-holdswap.kneemri-holdswap`, `comp.kernels.kneemri-onepair.kneemri-onepair`, `comp.kernels.kneemri-raptorseat.kneemri-raptorseat`, `comp.kernels.kneemri-raptorweight.kneemri-raptorweight`, `comp.kernels.kneemri-spanseat.kneemri-spanseat`, `comp.kernels.kneemri-stackseat.kneemri-stackseat`, `comp.kernels.kneemri-vjepaseat.kneemri-vjepaseat` | Seating the raptor member beside our own, as a seventh seat in rank space. |
| `kneemri.kaggle.raptor_teacher` | 165 | Y | 0 | 1 | - | The raptor member's ordering over OUR corpus, written as a teacher file. |
| `kneemri.kaggle.roster_check` | 166 | Y | 0 | 1 | - | Is every scoreable member either packaged or explicitly declined? `python -m kneemri.kaggle.roster_check`. |
| `kneemri.kaggle.roster_reads` | 200 | Y | 1 | 1 | `kneemri.kaggle.decode_entry` | How many decodes would a roster cost, and which members share each? `python -m kneemri.kaggle.roster_reads`. |
| `kneemri.kaggle.run_check` | 129 | Y | 0 | 1 | - | Did the kernel run the entry it was named for? `python -m kneemri.kaggle.run_check --log <path>`. |
| `kneemri.kaggle.seat_value` | 208 | Y | 0 | 1 | - | What a candidate member would add to the champion roster, card-free. |
| `kneemri.kaggle.slot_fill` | 98 | N | 1 | 1 | `kneemri.kaggle.submission` | How much of each study actually reached the model, accumulated across a served split. |
| `kneemri.kaggle.submission` | 379 | N | 6 | 2 | `comp.kernels.kneemri-graft.kneemri-graft`, `comp.kernels.kneemri-loadcheck.kneemri-loadcheck`, `kneemri.kaggle.dry_run`, `kneemri.kaggle.kernel_run`, `kneemri.kaggle.raptor_run`, `kneemri.kaggle.raptor_teacher` | Predicting a whole split and writing the submission — the part both kernels do identically. |
| `kneemri.models` | 1 | N | 0 | 0 | - | models. |
| `kneemri.models.backbone` | 749 | N | 24 | 18 | `comp.kernels.kneemri-loadcheck.kneemri-loadcheck`, `comp.verify_members`, `comp.witness_check`, `kneemri.embeddings.build_embeddings`, `kneemri.evaluation.attribution`, `kneemri.evaluation.encoder_drift`, `kneemri.evaluation.oof_oracle`, `kneemri.experiments.budget_residuals`, `kneemri.inference.ensemble`, `kneemri.inference.holdout_predictions`, `kneemri.inference.members`, `kneemri.inference.serve_bench`, …(+12 more) | Which DINOv2 an encoder means, and how wide its features are. |
| `kneemri.models.depth_attention` | 70 | N | 1 | 1 | `kneemri.models.slice_pooling` | Attention pooling over a slot's slices that can first see ORDER along depth. |
| `kneemri.models.depth_window` | 43 | N | 1 | 1 | `kneemri.models.model` | How a slice is PRESENTED to the encoder's three input channels — copies of itself, or its neighbours. |
| `kneemri.models.ema` | 128 | N | 1 | 2 | `kneemri.training.trainer` | An exponentially-moved average of the weights, kept alongside the ones being trained. |
| `kneemri.models.encoder_surgery` | 312 | N | 1 | 1 | `kneemri.models.model` | What is DONE TO an encoder before it is fitted: adapters, freezing, compilation, recompute. |
| `kneemri.models.fingerprint` | 61 | N | 2 | 1 | `kneemri.models.model`, `kneemri.runs.member_record` | The portable identity of a fitted member: what it answers to a seeded synthetic question. |
| `kneemri.models.grid_fit` | 54 | N | 1 | 1 | `kneemri.models.model` | Whether a trunk must be TOLD that the read is not the size its position grid was fitted at. |
| `kneemri.models.labelwise_attention` | 64 | N | 1 | 1 | `kneemri.models.slice_pooling` | Attention pooling over a slot's slices with ONE query per diagnosis, not one shared by all twelve. |
| `kneemri.models.lora` | 77 | N | 1 | 4 | `kneemri.models.encoder_surgery` | Low-rank adaptation: the pretrained weights never move, and the model still adapts. |
| `kneemri.models.model` | 574 | N | 14 | 17 | `comp.kernels.kneemri-loadcheck.kneemri-loadcheck`, `kneemri.embeddings.build_embeddings`, `kneemri.evaluation.attribution`, `kneemri.evaluation.encoder_drift`, `kneemri.inference.ensemble`, `kneemri.inference.holdout_predictions`, `kneemri.inference.members`, `kneemri.runs.member_record`, `kneemri.training.flag_vocabulary`, `kneemri.training.landed`, `kneemri.training.options`, `kneemri.training.run_setup`, …(+2 more) | The study-level model: a shared image encoder under a per-diagnosis attention head. |
| `kneemri.models.slice_attention` | 59 | N | 3 | 2 | `kneemri.models.depth_attention`, `kneemri.models.labelwise_attention`, `kneemri.models.slice_pooling` | Learned pooling over a slot's slices, in place of an elementwise maximum. |
| `kneemri.models.slice_pooling` | 71 | N | 6 | 6 | `kneemri.embeddings.build_embeddings`, `kneemri.models.model`, `kneemri.models.volume_read`, `kneemri.training.flag_vocabulary`, `kneemri.training.head_trial`, `kneemri.training.options` | The slice-pooling vocabulary: what each NAME decides about how a slot's slices are read. |
| `kneemri.models.slot_head` | 174 | N | 4 | 6 | `kneemri.models.model`, `kneemri.training.flag_vocabulary`, `kneemri.training.head_trial`, `kneemri.training.options` | Per-diagnosis attention over the acquisition slots of one study. |
| `kneemri.models.token_pooling` | 164 | N | 6 | 5 | `kneemri.embeddings.build_embeddings`, `kneemri.models.model`, `kneemri.models.volume_read`, `kneemri.training.flag_vocabulary`, `kneemri.training.head_trial`, `kneemri.training.options` | The token-pooling vocabulary: what each NAME decides about how one image's tokens become one vector. |
| `kneemri.models.trunk_tokens` | 125 | N | 1 | 3 | `kneemri.models.backbone` | A trunk that does not speak the vision transformer's output contract, wearing it. |
| `kneemri.models.volume_read` | 111 | N | 1 | 2 | `kneemri.models.model` | What a VOLUMETRIC trunk changes about a read, and the guard that proves the change landed. |
| `kneemri.process_table` | 144 | N | 2 | 3 | `kneemri.experiments.card_check`, `kneemri.experiments.queue_check` | This machine's process table, read whichever way this machine allows. |
| `kneemri.runs` | 1 | N | 0 | 0 | - | runs. |
| `kneemri.runs.holdout_record` | 188 | N | 16 | 15 | `kneemri.evaluation.foreign_member`, `kneemri.evaluation.holdout_metrics`, `kneemri.evaluation.label_probe`, `kneemri.evaluation.ordinal_signal`, `kneemri.evaluation.record_mix`, `kneemri.evaluation.run_comparison`, `kneemri.evaluation.weights_gap`, `kneemri.inference.foreign_holdout`, `kneemri.inference.holdout_backfill`, `kneemri.inference.trainer_record`, `kneemri.kaggle.entry_offline`, `kneemri.kaggle.priceable_entries`, …(+4 more) | What a run predicted for EVERY study it held out, written beside the checkpoint that predicted it. |
| `kneemri.runs.member_record` | 484 | N | 3 | 6 | `kneemri.runs.recorded_run`, `kneemri.training.fitting`, `kneemri.training.train` | What a trained member records about itself, so a checkpoint can say what produced it. |
| `kneemri.runs.metrics_recipe` | 70 | Y | 0 | 1 | - | Give a metrics file written before recipes existed the recipe its own checkpoint already carries. |
| `kneemri.runs.recorded_run` | 64 | N | 5 | 1 | `kneemri.evaluation.run_comparison`, `kneemri.experiments.convergence_check`, `kneemri.experiments.ledger_check`, `kneemri.inference.holdout_backfill`, `kneemri.runs.metrics_recipe` | What a finished run recorded about ITSELF, from whichever of its two records can state each fact. |
| `kneemri.runs.run_metrics` | 336 | N | 12 | 10 | `kneemri.evaluation.holdout_metrics`, `kneemri.evaluation.label_stability`, `kneemri.evaluation.run_comparison`, `kneemri.experiments.budget_residuals`, `kneemri.experiments.convergence_check`, `kneemri.experiments.ledger_check`, `kneemri.experiments.progress`, `kneemri.experiments.trajectory_check`, `kneemri.inference.holdout_backfill`, `kneemri.runs.metrics_recipe`, `kneemri.runs.recorded_run`, `kneemri.training.fitting` | One machine-readable line per epoch, so runs can be compared without grepping prose. |
| `kneemri.spawn` | 77 | N | 6 | 1 | `kneemri.experiments.arm_check`, `kneemri.experiments.arm_smoke`, `kneemri.kaggle.kit_check`, `kneemri.kaggle.live_dataset`, `kneemri.kaggle.mounted_kit`, `kneemri.process_table` | Running a child process on a box that is also training. |
| `kneemri.teacher_file` | 49 | N | 4 | 4 | `kneemri.evaluation.teacher_distance`, `kneemri.inference.teacher_readings`, `kneemri.kaggle.raptor_teacher`, `kneemri.training.teacher_targets` | The teacher-readings npz — the one place its writer and its reader agree on what is inside it. |
| `kneemri.training` | 1 | N | 0 | 3 | - | training. |
| `kneemri.training.accumulation` | 66 | N | 1 | 1 | `kneemri.training.trainer` | Trading time for effective batch size, so the card's width stops choosing the recipe. |
| `kneemri.training.augmentation` | 190 | N | 3 | 2 | `kneemri.training.landed`, `kneemri.training.run_setup`, `kneemri.training.trainer` | What a training image is perturbed by before the model sees it, and how hard. |
| `kneemri.training.card_memory` | 90 | N | 1 | 2 | `kneemri.training.heartbeat` | What the card had free when a run started, against what it has now. |
| `kneemri.training.device_budget` | 499 | N | 3 | 2 | `kneemri.experiments.arm_check`, `kneemri.experiments.budget_residuals`, `kneemri.training.run_setup` | How much of the card one run may hold, and what a recipe will actually reserve. |
| `kneemri.training.fitted_encoder` | 145 | Y | 0 | 1 | - | A fitted member, read back as an encoder another run can start from. |
| `kneemri.training.fitting` | 234 | N | 1 | 2 | `kneemri.training.train` | The epoch loop: passes over the corpus, and the decision of which pass is KEPT. |
| `kneemri.training.flag_vocabulary` | 496 | N | 1 | 3 | `kneemri.training.options` | What the training command line ACCEPTS, apart from what a flag MEANS. |
| `kneemri.training.grades` | 137 | N | 10 | 8 | `kneemri.training.head_trial`, `kneemri.training.label_focus`, `kneemri.training.landed`, `kneemri.training.objective`, `kneemri.training.pair_memory`, `kneemri.training.remembered_contrast`, `kneemri.training.remembered_ranking`, `kneemri.training.run_setup`, `kneemri.training.side_ranking`, `kneemri.training.trainer` | What a readout IS, and how the loss reads the five-level reading that judges it. |
| `kneemri.training.head_trial` | 394 | Y | 0 | 1 | - | Fit the HEAD alone on cached encoder features: `python -m kneemri.training.head_trial`. |
| `kneemri.training.heartbeat` | 118 | N | 1 | 1 | `kneemri.training.trainer` | Progress within an epoch, so a slow run can be told apart from a wedged one. |
| `kneemri.training.label_focus` | 85 | N | 3 | 5 | `kneemri.training.landed`, `kneemri.training.objective`, `kneemri.training.run_setup` | Which of the twelve findings a run's loss is allowed to learn from: `LabelFocus`. |
| `kneemri.training.landed` | 300 | N | 2 | 2 | `kneemri.training.run_setup`, `kneemri.training.train` | Did the switches this run asked for actually reach the model? Checked before the card is spent. |
| `kneemri.training.mixup` | 127 | N | 3 | 3 | `kneemri.training.flag_vocabulary`, `kneemri.training.options`, `kneemri.training.trainer` | Two studies blended into one example, so a memorised study-to-label pair becomes actively wrong. |
| `kneemri.training.objective` | 735 | N | 7 | 7 | `kneemri.training.head_trial`, `kneemri.training.landed`, `kneemri.training.pair_memory`, `kneemri.training.remembered_contrast`, `kneemri.training.remembered_ranking`, `kneemri.training.run_setup`, `kneemri.training.trainer` | What the training loss actually optimises, as a policy the trainer holds. |
| `kneemri.training.options` | 297 | N | 7 | 15 | `kneemri.experiments.arm_check`, `kneemri.experiments.arm_smoke`, `kneemri.experiments.budget_residuals`, `kneemri.experiments.switch_coverage`, `kneemri.experiments.vram_need`, `kneemri.training.run_setup`, `kneemri.training.train` | What one training run's command line SAYS, as a value: `kneemri.training.options`. |
| `kneemri.training.pair_memory` | 68 | N | 2 | 1 | `kneemri.training.remembered_contrast`, `kneemri.training.remembered_ranking` | A few steps' worth of readouts, kept so a ranking pair can span batches. |
| `kneemri.training.patience` | 71 | N | 1 | 1 | `kneemri.training.fitting` | Stopping a run once the holdout has stopped improving, without shortening its schedule. |
| `kneemri.training.pretrain` | 289 | Y | 0 | 1 | - | Adapt the encoder to knee MRI without labels: `python -m kneemri.training.pretrain --epochs N`. |
| `kneemri.training.pretrained_encoder` | 112 | N | 3 | 2 | `kneemri.training.flag_vocabulary`, `kneemri.training.options`, `kneemri.training.run_setup` | Reading a domain-adapted checkpoint back into a stock backbone, loudly. |
| `kneemri.training.remembered_contrast` | 104 | N | 1 | 2 | `kneemri.training.run_setup` | The contrastive term over keys that span steps, rather than keys inside one batch. |
| `kneemri.training.remembered_ranking` | 69 | N | 1 | 3 | `kneemri.training.run_setup` | The ranking term over pairs that span steps, rather than pairs inside one batch. |
| `kneemri.training.report_supervision` | 74 | N | 2 | 3 | `kneemri.training.run_setup`, `kneemri.training.trainer` | Supervising a study by its whole report, as an auxiliary term beside the twelve labels. |
| `kneemri.training.run_setup` | 467 | N | 5 | 4 | `kneemri.experiments.arm_check`, `kneemri.experiments.arm_smoke`, `kneemri.experiments.budget_residuals`, `kneemri.experiments.vram_need`, `kneemri.training.train` | What a run's flags BUILD, separated from what they say. |
| `kneemri.training.side_ranking` | 107 | N | 3 | 5 | `kneemri.training.landed`, `kneemri.training.objective`, `kneemri.training.run_setup` | How the loss reads a compartment PAIR: as two independent columns, or as two sides to be ordered. |
| `kneemri.training.spectral_mae` | 246 | N | 1 | 2 | `kneemri.training.pretrain` | Masked image modelling with a spectral reconstruction term, for adapting the encoder to knee MRI. |
| `kneemri.training.teacher_targets` | 90 | N | 3 | 2 | `kneemri.training.landed`, `kneemri.training.run_setup`, `kneemri.training.trainer` | An ensemble's readings of the TRAINING studies, lined up with the corpus so a student can be taught them. |
| `kneemri.training.train` | 147 | Y | 0 | 1 | - | Fit one member: `python -m kneemri.training.train --fold 0 --epochs 8`. |
| `kneemri.training.trainer` | 493 | N | 5 | 5 | `kneemri.training.fitting`, `kneemri.training.flag_vocabulary`, `kneemri.training.landed`, `kneemri.training.options`, `kneemri.training.run_setup` | Adapting the encoder and fitting the head, on report-derived targets weighted by confidence. |

