# The general Kaggle playbook, checked against what this repo already ran

Position when written: board 0.946 (`tripleseat_v2`), first place ~0.011 above, deadline 2026-11-05.
Question from the owner: we optimise like a real-world product; what do competition specialists do
that we do not?

## The playbook

Two sources, one domain-specific and one general:

- `research/2026-09-02_qbp-top-1pct-vs-median-in-closed-medical-imaging-competitions.md` — nine top
  write-ups from RSNA 2022/2023/2024. Ranked separators: localize-then-classify (9/9), RNN/attention
  slice aggregation fitted on the metric (7/9), pseudo-labels / confident learning (4/9), auxiliary
  segmentation loss (4/9, +0.01-0.03 per model in RSNA 2023 1st), task-structure heads (5/9), large
  DIVERSE ensembles (9/9), external data (2/9).
- General grandmaster practice (NVIDIA GM playbook; Deotte): trust the CV-LB relation rather than
  either number alone; pseudo-label with out-of-fold soft targets, multi-round; distil OOF + test
  predictions into one stronger single model; multi-seed and full-data refits; diversity plus
  SELECTION beats raw member quality.

## Where each lever stands here

| lever | status here | where |
|---|---|---|
| localize-then-classify | tried heavily (44 ledger lines) | `research/2026-09-02_qbq-localizer-routes-for-knee-mri.md`, ledger |
| pseudo-labels / confident learning | run on train and on KMAR-50K cross-site; flat | `experiments/README.md:12376-12394`, `:14392`, `:16312-16335` |
| test-time pseudo-labels / TTA-style transduction | audited, knob-sized | `research/2026-09-01_transductive_test_time_adaptation_audit.md` |
| auxiliary objectives | twelve `--*-weight` terms at full budget, negative | `experiments/README.md:16620` (knee-evdfd) |
| distil into one model | `distill`, `dml_*`: inside the 0.0053 floor | knee-8bmi2 |
| full-data refit, multi-seed | done; refit term ~+0.005 | knee-yatne |
| CV-LB trust | `board_transfer`, ledger75; residual sd 0.0013 | CLAUDE.md, knee-f9aao |
| learned slice aggregator | trained gate, labelwise, depth, moments: flat, span 0.0019 | knee-7gyw, `experiments/README.md:15355` |
| auxiliary SEGMENTATION loss | no ledger line | no local masks; SKM-TEA is instrument-only by ruling |
| diverse ensemble | members are near-copies (one trunk, one recipe) | the solo-deliverable directive rules the axis out |

## Reading

Most of the playbook has already been run and read flat, the learned slice aggregator included: once
its gate was actually in the optimiser (knee-7gyw), max, mean and a trained gate spanned 0.0019, and the
ledger reads the aggregation stage as saturated, with the limit upstream in the per-slice feature. One
item has never been run: auxiliary segmentation supervision, which acts on exactly that per-slice
feature. It is blocked by the external-data ruling unless a mask source is licence-clean. The one separator every
top write-up shares, a diverse ensemble, is excluded by our own solo directive rather than by a
measurement, which is a choice worth restating whenever the gap is priced.

Nothing public was found on the named top-10 competitor. A hidden-test AUC gives little room for
metric hacking beyond overfitting the public split, which the private split punishes; repeated top
finishes are better explained by skill than by exploitation.

## Sources

- [NVIDIA — The Kaggle Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- [Playground S6E2 1st place — diversity, selection, trusting the CV-LB relation](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t)
- [NVIDIA — Winning a Kaggle competition with GenAI-assisted coding (Deotte)](https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/)
- [Deotte — Pseudo Labeling QDA](https://www.kaggle.com/code/cdeotte/pseudo-labeling-qda-0-969/comments)
