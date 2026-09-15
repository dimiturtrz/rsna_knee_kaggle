# Config tree: hyperparameters as a frozen dataclass tree, recipes as TOML (knee-feb62.6)

Owner 2026-09-15: "too many argvs". This is the audit and the design. Nothing here is built yet.

## 1. The surface today (audited 2026-09-15, tree at 8d9c13b8 + hardness WIP)

| Thing | Where | Size |
|---|---|---|
| Train flags | `kneemri/training/flag_vocabulary.py` (`FlagVocabulary.parser`, :68; groups `_augmentation_arguments` :380, `_label_quality_arguments` :440, `_curriculum_arguments` :479, `_focus_arguments` :509, `_sampling_arguments` :537, `_optimiser_arguments` :593, `_teacher_arguments` :681) | 70 `add_argument` |
| Encoder flags | `kneemri/training/encoder_flags.py` (`EncoderFlags.add`, called at flag_vocabulary.py:364) | 15 `add_argument` |
| Parsed object | `kneemri/training/options.py:54` `TrainingOptions`, frozen, flat, built as `cls(**vars(parser.parse_args(argv)))` at :369 | 103 fields |
| Existing grouping | properties on `TrainingOptions`: `routing` :295, `crop` :300, `band` :310, `reading` :320, `cache` :324, `supervision` :336, `positions` :382 | 7 views |
| Stamp | `TrainingOptions.recipe` :248, a flat `dict[str,str]` of every field, logged at `train.py:41`, stamped into the checkpoint as `"options"` at `runs/member_record.py:435` | |
| Back-compat reader | `TrainingOptions.of_recipe` :264, which drops unknown keys and keeps defaults for missing keys. It already reads every historical checkpoint. | |

**Ceiling pressure, and the reason this is not cosmetic.** `flag_vocabulary.py` is 740 of 750 lines (`file_max`, pyproject.toml:522). The 5 `--hard-*` flags now being added will overflow it, and each future flag forces another split. `run_setup.py` is 732 of 750.

Group census of the ~85 train flags, read from the vocabulary group names (counts per group UNVERIFIED, not tallied flag by flag):
- data/reading: slices, band, crop, px, scheme, tie, whole-stack, cache
- model: pooling, slot dropout, stochastic depth, encoder (15)
- optim: lr, backbone lr, one-cycle, warmup, EMA, epochs, batch, patience, recompute
- aug: geometric, intensity, ghosting, mixup, `--aug-*` schedule
- sampling: sample alpha/spread, corpus oversample, curriculum/certainty ramp, `--hard-*`
- targets/labels: labels, eval labels, calibrated, hold annotations, rank weight, weight floor, label quality, focus
- teacher: distill weight, teacher targets/features
- runtime/io: fold, folds, split seed, out, paths, studies, cache studies

## 2. Consumers of the parsed object or its stamp

**Of `TrainingOptions` the type** (12 modules):
- `training/`: `options`, `flag_vocabulary`, `encoder_flags`, `run_setup` (:42, :89), `landed` (`Request`, :235, built from primitives by design), `train`, `trainer`
- `runs/member_record.py` (:372)
- `experiments/`: `arm_check` (:46, :296–309, parses every arm through the real parser), `arm_smoke`, `switch_coverage` (:91, :95, iterates `fields(TrainingOptions)`), `default_drift` (:49, `TrainingOptions().recipe`), `budget_residuals` (`of_recipe`), `vram_need`

**Of the flat stamp dict** (reads keys by string; these must keep working on OLD checkpoints forever):
- `inference/members.py:508` (`fold`)
- `inference/holdout_backfill.py:97–118` (`fold`, `folds`, `split_seed`, `eval_labels`, `labels`)
- `models/weight_soup.py:97–103` (architecture and split keys)
- `kaggle/raptor_features.py:116` (`fold`)
- `runs/recorded_run.py`, `runs/metrics_recipe.py`, `runs/run_metrics.py:340`
- `experiments/{convergence_check,ledger_check,geometry_check,control_check,progress,arm_smoke}`
- `kaggle/{champion_check,roster_check,seat_value,traversal_census}`, `inference/{vote,norm_adaptation,foreign_holdout}` (grep hits on "recipe"; the exact keys used are UNVERIFIED)

**Out of scope:** `foreign_run` has 2 flags of its own (:143–144) and does not touch `TrainingOptions`. The ~90 evaluation and kaggle CLIs have their own small parsers and are also out of scope.

**Existing frozen dataclasses to reuse as nodes (beside their producers):**
- `AugmentConfig` (augmentation.py:62)
- mixup `Blend` (mixup.py:47), a draw rather than config
- `ProgressSchedule` (progress_schedule.py:38)
- `Accumulation` (accumulation.py:14), `GradientClip` (gradient_clip.py:27), `LabelFocus` (label_focus.py:49)
- `Routing`, `Crop`, `Band`, `Reading`, `Supervision` (already returned by the options properties)

**How arms pass recipes:**
- `experiments/lib.sh:521` `run_train <name> <flags…>` passes argv straight to `kneemri.training.train`.
- `experiments/recipe.sh` holds the shared strings `RECIPE`, `THROUGHPUT` and `PATIENCE`.
- 374 arm scripts in total: 357 call `run_train` and 5 call `foreign_run`.

## 3. Design

### 3.1 Tree

Frozen dataclasses. Each node is defined in the module that consumes it, never in a `*_types.py` bucket. The root `TrainConfig` replaces `TrainingOptions` in `options.py`.

```
TrainConfig                      options.py
├── run: RunConfig               options.py       fold, folds, split_seed, out, paths, studies
├── data: ReadingConfig          (Reading/Crop/Band/Routing already exist; keep, nest)
├── model: ModelConfig           run_setup.py     pooling, slot_dropout, stochastic_depth, encoder: EncoderConfig (encoder_flags.py)
├── optim: OptimConfig           fitting.py       lr, backbone_lr, one_cycle, warmup, ema, epochs, batch, patience, clip: GradientClip, accumulation: Accumulation
├── aug: AugmentConfig           augmentation.py  (existing) + ghosting, mixup, schedule: ProgressSchedule
├── sampling: SamplingConfig     sampling.py      alpha, spread, oversample, hardness: ProgressSchedule
├── targets: Supervision         (existing)       labels, eval_labels, calibrated, hold, rank_weight, weight_floor, focus: LabelFocus
└── teacher: TeacherConfig       teacher_targets.py
```

- **One leaf, one legacy name.** Each leaf carries `field(metadata={"flag": "--slot-dropout", "help": …})`. The legacy flat key is the flag's dest name, which is unique today because `TrainingOptions` is flat. The dotted path (`model.slot_dropout`) is derived from the tree position, so no mapping table is written by hand.
- **Argparse generated from metadata.** It walks the tree and emits the parser. This deletes the 740-line `flag_vocabulary.py` and `encoder_flags.py` instead of splitting them a third time, and a new knob becomes one annotated field.

### 3.2 Recipe files and overrides

Recipe files are committed under `experiments/recipes/<name>.toml` and read with stdlib `tomllib` (Python ≥3.12 floor, pyproject.toml:8; no new dependency). Tables mirror the tree:

```toml
[model]
pooling = "labelwise"
slot_dropout = 0.40
stochastic_depth = 0.05

[aug.schedule]
start = 0.0
mid = 1.0
end = 0.3
ramp = 0.3
tail = 0.15
```

Precedence, lowest first:
1. dataclass defaults
2. `--config file.toml`
3. legacy `--flags` (the shim; every existing arm)
4. `--set model.slot_dropout=0.3` (repeatable)

The values are typed through the same `_READERS` that `of_recipe` uses (options.py:49), so a TOML string and a CLI string cannot parse differently.

### 3.3 Stamp and back-compat

- **Stamp.** It stays a flat `dict[str,str]` keyed by the LEGACY names, identical to today's `recipe`. One added key, `"config": "<recipe file name or none>"`, is informational only. Every consumer in §2 that reads keys by string keeps working with zero change, and `weights_only=True` loading is unaffected (still strings).
- **`TrainConfig.of_recipe`.** It walks leaves by legacy name, so every checkpoint on disk reads unchanged, renamed-key dropping included.
- **`TrainConfig.toml()`.** It dumps a run's config as a recipe file, so any historical run becomes a recipe with one command.

### 3.4 Proofs (the constraint that makes the refactor safe)

1. **Arm equivalence.** For every `run_train` invocation `ArmCheck.invocations()` yields (357 arms), `TrainConfig.of(argv).recipe == TrainingOptions.of(argv).recipe`, byte-identical dict. Run once against the old class before it is deleted, and freeze the result as a fixture.
2. **Round trip.** `TrainConfig.of_toml(cfg.toml()) == cfg` and `of_recipe(cfg.recipe) == cfg` over the arm corpus, using the equivalence classes defaults / every-leaf-set / None-able leaves / bools.
3. **Recorded stamps.** `of_recipe` over every `metrics.jsonl` recipe row on disk yields the same values as before. This is an integration test on a small committed fixture sample, since the unit tests may not read the data root.
4. **Landed.** `Request` is still built from primitives (landed.py:245 explains why); it takes the leaves off the tree. `Landed.missing` must stay empty on the smoke.
5. **`switch_coverage` and `default_drift`** iterate `leaves(TrainConfig)` by legacy name; the unit tests stay green without edits to COVERAGE.

### 3.5 Gates

- **Demeter depth 2.** Only chains that end in a call count. `config.aug.schedule.at(p)` is depth 3, so RunSetup hands the NODE down (`Augmentation(config.aug)`, `ProgressCurve(config.aug.schedule)`), and consumers call methods on the node they were given. Reading `config.model.slot_dropout` is data navigation and passes.
- **Fan-in.** The root replaces `TrainingOptions` one for one (12 importers today). Each node is imported only by its producer module and the root, so no new god-node appears. `bottleneck_degree = 8` (pyproject.toml:521) applies to fan-in AND fan-out, and `options.py` fan-out grows by the node modules. Check `graph.fitness --assert` at step 2. If it trips, the root moves to its own `config_tree.py` beside the metadata walker.
- **Pyrefly strict.** The leaf walker uses `dataclasses.fields` plus `get_type_hints`, which is already the pattern in `of_recipe`. `cast` is confined to the reader table.
- **Method mirror.** Each new public method gets a `test_<name>` in the mirrored file.
- **No bucket module.**
- **LCOM.** Today's docstring apology for LCOM4=6 (options.py:9–15) disappears, because the six responsibilities become six nodes.

## 4. Migration: ordered, each commit gated green alone

Every step starts after knee-p28we lands, since it touches the same files.

| # | Commit | Proof it lands | PRICE (CPU) |
|---|---|---|---|
| 1 | TOML recipe loader over the CURRENT flat `TrainingOptions`: `--config file.toml` with flat keys, precedence as §3.2, plus `experiments/recipes/grand.toml`. **Unblocks knee-feb62.2 immediately.** | round trip; precedence classes; arm_check reads `--config` | ~1.0 h |
| 2 | Freeze the arm-equivalence fixture from the old parser (all 357 arms → recipe dicts) | fixture committed; test reads it | ~0.5 h |
| 3 | Tree nodes with leaf metadata; `TrainConfig.of/of_recipe/recipe/toml`; argparse generated from metadata; delete `flag_vocabulary.py` and `encoder_flags.py`; `TrainingOptions = TrainConfig` alias for one commit | equivalence fixture byte-identical; `graph.fitness --assert` | ~2.5 h |
| 4 | Flip consumers to nodes: RunSetup hands nodes down, `Landed.Request` from leaves, switch_coverage/default_drift/arm_check/arm_smoke/budget_residuals/vram_need on leaves; drop the alias | full gates; `Landed.missing` empty on CPU unit smoke | ~2.0 h |
| 5 | Nested TOML tables plus `--set dotted.path=v`; grand.toml rewritten nested; `recipe.sh` RECIPE/THROUGHPUT/PATIENCE become `experiments/recipes/base.toml` that arms `include` via `--config base.toml --config grand.toml` (later files override) | precedence tests; arm_check over all arms | ~1.0 h |
| 6 | Docs: CLAUDE.md conventions line, experiments/README recipe section | — | ~0.3 h |

**Total ~7.3 h CPU, 0 card.** The bead said ~5 h; the audit raises it because of the generated parser (step 3) and the 12-consumer flip (step 4).

If only the grand run matters this week, **step 1 alone (~1 h) is enough** to launch knee-feb62.4 from a recipe file instead of argv. Steps 2–6 then follow without blocking the run, and the run's stamp is identical either way.

Risk is low: the stamp keys never change, so no historical checkpoint, ledger row or kaggle reader can break. The one real hazard is a flag whose argparse `default` differs from its dataclass default. Step 2's fixture catches exactly that, which is why it comes before step 3.
