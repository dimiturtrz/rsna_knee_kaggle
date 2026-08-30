# kneemri — agent instructions

Keep this file and `CLAUDE.md` IN SYNC (different harnesses read each; same content).
## Issue tracking — beads (bd)

This project uses **bd (beads)** for issue tracking. Run `bd init` once, then `bd prime` for the full
workflow. `.beads/issues.jsonl` is git-tracked; the Dolt engine is gitignored.

- `bd ready` — available work · `bd show <id>` — details · `bd update <id> --claim` · `bd close <id>`
- Use `bd` for ALL task tracking — not TodoWrite or markdown TODO lists.

## Build & Test

`uv` is the env + dependency manager; `nox` is the task runner (every session calls `uvx`/`uv run`
so local runs match CI exactly).

```bash
uv sync --extra dev --extra devtools   # create/refresh .venv
uvx nox -s lint test cov               # the merge gate, locally
uvx nox -s gates                       # same, one entrypoint
```

**`uvx nox`, never bare `nox`.** It is not on PATH here and is not a project dependency, so both `nox`
and `uv run nox` fail — and they fail in the way that costs you most: *silently*. `nox -s gates > log
2>&1; echo "EXIT $?"` reports the ECHO's status, so a missing binary reads as a clean pass while the
log holds one line saying `command not found`. This has now happened twice in one session (`pkill` was
the first).

**Verify a gate by its own success marker, not by an exit code you printed.** Grep the log for
`Session gates was successful`. Chain with `&&`, not `;`, so a missing binary short-circuits instead of
handing the next command a fresh exit status to succeed with.

**Verify a COMMIT the same way: by `git log`, not by the tail of its output.** The pre-commit hooks print
twenty-odd Passed lines, so `git commit ... | tail -3` shows three of them whether the commit landed or
not — the one Failed line is hundreds of lines above. It cost three silent no-op commits in one session
(a `ruff format` disagreement, a pyrefly `NDArray` type argument, an E501). Write the commit's output to
a log, then `git log --oneline -1`; on a failure grep that log for `Failed`.

**Finish every `bd` WRITE before `git add`, and run none while a commit is in flight.** A bd write ends
by exporting the working set to `.beads/issues.jsonl` *and* git-adding it — measured, and reads
(`bd show`/`list`/`ready`) do not. So a `bd close` fired next to a commit changes a STAGED file
underneath pre-commit, which reports `files were modified by this hook` and blames whichever hook was
running — always one of the three slowest, always with clean output, always green on the retry
(knee-hahw).

## Package structure

Your code lives in your declared packages: `core/`, `kneemri/` — organise modules however the project needs.
The scaffold ships the **guardrails**, not a layering. `graph.py --assert` keeps each package's
internal import graph honest (no cycles, no god-modules, no oversized files, no unmirrored logic modules);
thresholds live in `[tool.structure]`.

Directional layer rules (the kernel imports none of the others; a viewer never imports a trainer) are
enforced by **import-linter** — a one-way forbidden import is no cycle, so it's the axis `graph.py` can't
see. Contracts live in `[tool.importlinter]` (a `# >>> LOCAL-SLOT: import-contracts` region you extend),
run via `lint-imports` in nox/CI/pre-commit.

## Static-analysis gates — graduated, never suppressed

Each gate is engine + params + invocation, run identically in `noxfile.py` and CI. Philosophy: a check
starts advisory, and once the code is clean it **graduates** to enforced; from then on any regression
fails. Fix the finding, don't suppress it; suppressions stay minimal and meaningful (a `# noqa: RULE` with
no prose, never a blanket ignore). Thresholds are legislated values (file_max, CC 10), never seeded from
the repo's current state. The linter is a bug-finder, not cosmetics.

### Enforced — these BLOCK, in nox, CI and pre-commit alike

Every one of them is a rule you can write TO. Discovering a gate by failing it means the doc failed first.

- **ruff** (pinned `0.15.13`) — `check` on the full house select, enforced; `format --check` enforced too — pre-commit always ran it for real, so an advisory nox/CI meant the merge gate passed a tree that could not be committed.
  Encodes: logging not prints (T20), named constants not magic numbers (PLR2004), keyword-only bools
  (FBT), specific-exception-or-crash (BLE001/S), low complexity (C90/PLR), imports-at-top (PLC0415).
- **vulture** (pinned `2.16`) — dead-code at `min_confidence = 80`; scope is `[tool.vulture]`, not the CLI.
- **coverage** — pytest-cov with a `--fail-under=90` floor.
- **deptry** (pinned `0.25.1`) — undeclared (DEP001) / unused (DEP002) / transitive (DEP003) imports; config in `[tool.deptry]`.
- **pyrefly** (pinned `1.1.1`) — strict static types: every `def` annotated, plus the annotations check.
- **jscpd** — copy-paste over the threshold in the packaged `jscpd.json`.
- **import-linter** — directional forbidden-import contracts (`[tool.importlinter]`); the layer axis graph.py can't express.

The rest run in ONE process — `python -m devtools.run <packages> --gate <list>` — because each would
otherwise pay interpreter startup separately. All of them still answer `python -m devtools.<module>
<packages> --assert` individually, which is how you check ONE while fixing it:

- **arch-fitness** — `python -m devtools.graph.fitness`: god-module (fan-in/out), god-file, import-cycle,
  and the FILE-level test mirror (a logic module needs its `tests/unit/<pkg>/…/test_<name>.py`); thresholds
  in `[tool.structure]`. Advisory sub-checks: line-floor (off) + chokepoint. It is `devtools.graph.fitness`,
  NOT `devtools.graph` — that is a package and answers `No module named devtools.graph.__main__`.
- **demeter** — `python -m devtools.coupling.demeter`: CALLING a method through a field into a stranger.
  Ceiling `[tool.structure] demeter_max_depth` (default 2). Only chains ending in a CALL count — reading a
  deep config tree is data navigation, not coupling.
- **purity** — `python -m devtools.coupling.purity`: a `@property` that assigns to `self`. A read dressed as
  attribute access has to BE a read.
- **composition** — `python -m devtools.coupling.composition`: a `holds` cycle (A owns a B that owns an A),
  the intra-file case an import cycle cannot represent.
- **envy** — `python -m devtools.coupling.envy`: a METHOD more interested in another class than its own.
  Floor `[tool.structure] feature_envy_min` (default 4); value objects are never the target.
- **contracts** — `python -m devtools.coupling.contracts`: directional forbidden-USE rules over the resolved
  arrows (not imports). Empty by default — you add them.
- **ast-grep** — `python -m devtools.astgrep`: the packaged module-shape rule set, the syntactic axis ruff
  has no plugin API for.
- **method mirror** — `python -m devtools.hygiene.mirror`: for each public method `A.a`, the module's mirror
  test file holds a `test_a` that CALLS it and ASSERTS. STRICTER than graph's file-level mirror, which one
  smoke test satisfies for a module of twenty methods. Resolved through `[tool.structure] test_layout`
  (`off` disables both mirrors); per-method escape `# devtools-ignore: test-mirror`. See `docs/UNIT_TESTS.md`,
  which also states the case AGAINST the convention.
- **small** — `python -m devtools.hygiene.small`: a unit test touches nothing it did not create — no external
  data root, no network, no sleep, no unseeded RNG. UNIT ONLY; an integration test SHOULD read a real fixture.
- **shape-contracts** — `python -m devtools.shape_contracts`: a public array/tensor boundary must carry a
  **jaxtyping** shape (`Float[Tensor, "b c h w"]`), not a bare `np.ndarray`/`Tensor`. Aliases in
  `[tool.shape_contracts]`; make shapes live at runtime with a `@shapecheck` decorator (jaxtyping+beartype,
  see `devtools/README.md`). A fresh generation has 0 boundaries and passes — a bare boundary then fails.
  Here it has GRADUATED: nox, CI and pre-commit all pass `--assert`, and the tree measures 0 bare
  boundaries, so a new one BLOCKS.

### Advisory — these REPORT and never block

Reviewer signal. Reading them is how you find the next thing to fix, not a thing you must fix now.

- **class-shape smells** — `devtools.cohesion.state_candidates` (namespace-state), `devtools.cohesion.lcom`
  (LCOM4 cohesion), `devtools.cohesion.data_clumps` (Fowler clumps).
- **complexity ranking** — `devtools.cohesion.complexity` (radon; ruff C901 is the FIXED gate, this ranks).
- **magic literals** — `devtools.hygiene.magic_literals`: recurring literals that want a StrEnum/dataclass.
- **behavioural arrows** — `devtools.graph.arrows` and `devtools.graph.calls`: the resolved call/construct
  graph the architecture viewer draws.
- **ruff whole-tree** — the enforced select over `.` with `--statistics`, wider than the blocking `lint_paths`.
- **vulture at 60** and **coverage at 95%** — the same engines one notch stricter than their blocking bar.
- **archmap** (doc-gen) — `python -m devtools.graph.archmap <packages>` emits
  `docs/architecture/graph.json` (the committed, diffable architecture — nodes + weighted import edges) plus
  a self-contained interactive **cytoscape viewer** (`index.html`, regenerated + gitignored) that folds/expands
  packages to any depth and focuses a module's neighbourhood. Regenerate with `uvx nox -s archmap` and **commit the
  graph.json diff** — it's the architecture-erosion record. The pre-commit hook regenerates on commit; CI runs
  `--check` (advisory) to flag a stale graph.json. **Publishing** (one repo = one Pages site): if the repo has
  no other Pages deployer, opt in with `archviz_pages` — it owns a staged site (`/architecture/` = main,
  `/architecture/preview/` = dev, `/` redirects). If the repo ALREADY deploys Pages, do NOT enable it; fold
  archmap into the existing workflow as an `/architecture/` subpath instead (see README: "Compose archmap into
  an existing Pages site"). Doc-gen, not enforcement — import-linter is the directional gate.

## Scaffolding

Guardrails provisioned by **sdlc-scaffold** via copier — `.copier-answers.yml` pins the version. The gate
config, `devtools/`, and the nox/CI/pre-commit runners are **template-owned**: don't hand-edit them to pass
a gate — fix upstream in the scaffold and `uvx copier update`, or edit only within `# >>> LOCAL-SLOT`
regions. `copier update` pulls scaffold improvements as reviewable steps.

**Sync once with the git-shim in log-only mode after a `devtools_ref` bump.** A bump makes `uv` REBUILD
the `sdlc-devtools` git dep, and uv builds a git dependency by checking it out into its own cache and
running `git reset --hard <sha>`. The global shim blocks reset-hard, so the sync dies with `BLOCKED by
git-shim [reset-hard] in global scope` — mid-gate-run, after ruff/vulture/import-linter have already
printed clean, which reads as a gate regression. It is not one: the target is uv's cache, never this
tree. Build it once, then run gates normally (the build is cached, no further git op):

```bash
GIT_SHIM_LOG_ONLY=1 uv sync --extra dev --extra devtools   # log-only, not GIT_SHIM=off — still recorded
uvx nox -s gates
```

A path-scoped allowance in the shim would be the permanent fix and lose no protection (uv only ever
resets its own cache), but the shim is machine-global and outside this repo, so it is the owner's call
(knee-fo45).

**Don't test template-owned code, either.** The `devtools/` detectors are unit-tested in the scaffold
itself; a `tests/unit/devtools/` here would re-litigate every template change locally (it duplicates
upstream coverage of byte-identical code, and breaks on the next `copier update`). Trust the upstream
tests — this repo's tests cover only ITS packages. Consumer-specific detector *config* (e.g. that your
`omit` list exempts your shells) is proven through the detector's public behavior, not a copied unit test.

## Conventions

- **Fourteen checks, each for a failure that already happened** (only the smoke wants the card, for two minutes): `python -m
  kneemri.experiments.arm_check` before arming a queue (a flag typo costs a queue slot to discover),
  `python -m kneemri.experiments.ledger_check` after runs finish (five arms once sat in the ledger as "queued" for
  two days after running),
  `python -m kneemri.experiments.queue_check` whenever a watcher is launched or a session ends (the card sat idle
  for five minutes because three `after.sh` watchers had died with the session that launched them, and their logs
  still read `waiting for 'ARM DONE ...'` — the queue LOOKS armed either way, and only the process table knows;
  it now also blocks the OPPOSITE hole, two live watchers owing one arm, which would run it twice on one card),
  `python -m kneemri.experiments.card_check` before putting anything on the GPU (two identical `member_curve`
  scans ran on one card for twenty-five minutes after a `nohup` relaunch landed on a job that had NOT failed;
  the cost was the READING, not the card — contention stretched one member from three minutes to eleven, one
  step from being recorded as a fact about the member. It asks the DEVICE rather than guessing from argv, and
  blocks only on TWO OF OURS, since the other project shares this card by arrangement),
  `python -m kneemri.kaggle.kit_check` before shipping a kit and `--mounted` before a
  submission (two slots went to code the repo had already fixed; `--mounted` asks Kaggle what it will
  actually run, since uploading is manual), and `python comp/witness_check.py --device cpu` before a
  submission (it found three defects on first use), and `python -m kneemri.kaggle.mount_check` before a
  submission (it caught `kneemri-solo` ranking over a package that had gained a DINOv3 member while its
  metadata still declared DINOv2 only — a kernel that loads every member and then cannot build one), and
  `python -m kneemri.kaggle.champion_check` whenever the roster or a champion kernel is edited — the only
  one that guards a RESULT rather than a shipment: it asserts the entry pinned in `comp/champion.json` can
  still be rebuilt (its members still named by its kernel's `chosen` tuple, still packaged by the roster),
  because losing that path would leave every other gate green — and that the pin is still the board's best
  `ensemble` row, because it has now lagged twice with every other gate green. Read the entry and its score
  from that file, never from here: this line quoted 0.925 through two later champions.
  `python -m kneemri.kaggle.board_read --live` before planning against the GAP — the only one that looks
  at the competition rather than at us. It costs no slot, writes `comp/leaderboard.json` with the UTC
  instant of the read, and `champion_check` refuses one over two days old (five slots a day reset at
  00:00 UTC, so an older read describes a board that has since been spent). The pinned gap read 0.008
  while the API said 0.015 and nothing noticed, which is the same way `board_transfer`'s slope was quoted
  wrong from prose twice. And
  `python -m kneemri.experiments.trajectory_check --run <metrics path> --target <pre-registered score>` while an
  arm is STILL RUNNING — the only check that can end a job, and the only one whose subject is the card rather than
  a shipment or a number. It measures the gain from epoch 3 to the peak across every run on disk matched to the
  arm's own epoch budget (twelve epochs: median +0.0210, p90 +0.0380, max +0.0529 over n=96) and exits non-zero
  only when the arm plus the LARGEST such gain still misses the branch it was given before it started. It may
  only TRUNCATE — never promote, never rank, never close a family — and a truncated `one_cycle` score is a FLOOR,
  which is what `convergence_check` will then say about it. Its own first run found the corpus defect that would
  have disarmed it: `lora_r8_v3small_224` holds 16 rows against a 12-epoch recipe, one stitched series whose gate
  and peak are different runs, reading +0.1350 and lifting the ceiling two and a half times. See knee-gnzx. And
  `python -m kneemri.experiments.convergence_check` before quoting ANY run's score — the only one that guards a
  NUMBER rather than a shipment. It compares a run's best epoch, its last recorded epoch and the epochs its recipe
  ASKED for, and separates TRUNCATED (died early; under `one_cycle` the anneal never ran, so the score is a FLOOR)
  from STOPPED (`patience` fired, a decision) from UNDERTRAINED (peaked on its last epoch; the budget bound it).
  Only TRUNCATED exits non-zero. It exists because three runs entered the ledger as facts about architectures when
  they were facts about a killed process, and because its own FIRST run over-reported three patience stops as
  crashes — including `res392`, which would have re-opened a correctly-closed family.
  `python -m kneemri.kaggle.board_check --live` after any submission scores, and before quoting a board number
  or a transfer constant — the check that guards the RECORD. `comp/board.jsonl` held 34 of the competition's 58
  submissions until 2026-08-25: fifteen scored all-ours entries were absent, `basedense_v1` and `votefit_v1`
  (the entry `comp/champion.json` now pins) among them, and the compression
  constant `board_transfer` fits from that file read +1.82 against a complete set's +1.55. It joins on the Kaggle
  submission REF, never a name — asking Kaggle for the list spends no slot — so every submission carries a row,
  the ones this file will not price included (`scope: excluded` plus the reason). A row's `offline` must be a number
  read on studies the entry did NOT train on: on 2026-08-29 `raptor_solo`'s 0.8938 was voided because
  `foreign_overlap` shows it was 98.6% in-sample, and that one cell was carrying the whole "compression goes
  negative at the top" reading plus eleven of fifteen apparent ruler inversions.
  `python -m kneemri.kaggle.roster_check` after ANY run finishes, and before building an entry out of the pool —
  the check that guards what can be SHIPPED. Nothing writes `comp/members.roster`; it is hand-edited, so every
  member on it is there because somebody remembered. On 2026-08-29 five of a sixteen-seat entry's best seats were
  not on it, and the sweep that followed found **76 scoreable members in neither the roster nor anywhere else** —
  `stackfit_s2026` at 0.8631 among them, the second-best solo the project owns, unshippable for as long as nobody
  typed a line. It asserts every checkpoint with a holdout record is named in the roster OR in
  `comp/members.declines` WITH A REASON (a bare name is a finding too: it records that somebody decided, not what).
  Anything in neither is the defect. Thirty members joined the roster on that sweep and forty-six were declined in
  writing; a decline is an argument, and it is re-opened by answering it, never by re-quoting a number.
 A recipe is smoked with `arm_check --only <arm> --smoke <throwaway dir> --smoke-studies 60` (48 train / 12 holdout, 2 s an epoch) before it is queued for forty minutes. `--smoke` takes the DIRECTORY the throwaway checkpoints go to and `--only` takes the arm; `--studies`/`--epochs` are `train`'s flags, not this tool's, and passing the arm path to `--smoke` costs a run that dies on `FileExistsError` against the arm script itself.
  A recipe on a reading axis NOBODY HAS DECODED cannot be smoked until something writes the truncated cache, and
  `python -m kneemri.experiments.arm_check --smoke` is what writes it — sixty studies in under a minute, a distinct
  filename (`_first60`) so it can never overwrite a full array. Reach for that before hand-building one: on 2026-08-19
  I decoded `crop80_span`'s by hand and only then found the tool already had the step. See `experiments/README.md`.
- **Data lives out of the repo** (under a root pointed at by `paths.yaml`, gitignored). No datasets,
  weights, or run outputs in git.
- **No notebooks** (`.ipynb`) — a one-off run becomes a committed CLI / `python -m` entrypoint
  (reproducible + reviewable), never a REPL session.
- **Minimal comments** — prefer self-documenting names; a comment is a liability, not a default.
- **Config objects over long arg lists**; strategy pattern over `if`/type dispatch.
- **Doc layers** — `learning/<date>_<topic>.md` = the study ramp / general understanding;
  `research/` = external / field synthesis (theirs); `interpretations/<task>/<date>_<topic>.md` =
  sense-making of *our own* results (per task, `converging/` for cross-task). The build log is git history;
  portfolio READMEs carry the result + link out.
- **Before filing a finding, READ THE LEDGER AND THE INTERPRETATIONS** — `experiments/README.md` and
  `interpretations/kneemri/`. Beads are the work queue; those two are where measured results live, and a
  bead can be silent about a number that was settled a day earlier. On 2026-08-13 I raised two P0s in one
  session against things already recorded there with better evidence: that whole-stack members gain
  nothing from overlap TTA (five members, measured, plus bead knee-eeo, plus an interpretation), and
  that ensemble diversity beats member quality (six leaderboard entries built to isolate it). Both were
  filed as discoveries and then "corrected" by measurements the repo had already made. `bd search` will
  not find them; grep the ledger.




<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:3216161c -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
