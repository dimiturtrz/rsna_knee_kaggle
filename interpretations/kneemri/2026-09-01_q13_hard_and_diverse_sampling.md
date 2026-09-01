# Q13 (knee-nyqc): hard and diverse example sampling

2026-09-01, Kimi. Read-only audit of what exists, what has been run, and what would need to be built.

## Headline

The repo is **not empty** on this axis. A disagreement-weighted sampler (`Disagreed`, `Ramped`) is implemented in `kneemri/training/sampling.py`, unit-tested in `tests/unit/kneemri/training/sampling.py`, and wired into `kneemri/training/run_setup.py:563`. A pre-registered arm script (`experiments/arms/disagreed_sampler.sh`) queues the uniform control vs. full-tilt pair. It has **not been run** — no row in `experiments/README.md` and no `disagreed_*_s2026.metrics.jsonl` on disk. What is genuinely missing is **diversity sampling** over the feature space, **prevalence-balanced batches**, and any **loss-based** hard-example miner (the project deliberately rejects loss as the signal).

## What already exists

| piece | location | state |
|---|---|---|
| Uniform baseline | `kneemri/training/sampling.py:60-83` | landed + tested |
| Disagreement-weighted draw with replacement | `kneemri/training/sampling.py:86-153` | landed + tested |
| Ramped disagreement curriculum | `kneemri/training/sampling.py:155-212` | landed + tested |
| Spread-file generator (`train_disagreement`) | `kneemri/evaluation/train_disagreement.py` | landed |
| CLI flags (`--sample-spread`, `--sample-alpha`, `--sample-ramp`) | `kneemri/training/flag_vocabulary.py:396-430` | landed |
| Wiring into trainer | `kneemri/training/run_setup.py:563` | landed |
| Pre-registered arm script | `experiments/arms/disagreed_sampler.sh` | landed, **unrun** |

The design rationale is documented in both `sampling.py` and `train_disagreement.py`: the project rejects loss-based hard mining because `uncertainty_split` puts ~87% of the residual on the aleatoric side, so high-loss studies are disproportionately mislabelled and mining them amplifies noise. Member disagreement is used instead because it is epistemic — high where the family is split, low where it agrees (including where it agrees on being wrong).

## What has actually been measured

Nothing. A grep of `experiments/README.md` for `disagreed_uniform` or `disagreed_full` returns no matches. There are only three production metrics files under `logs/` and none are from this axis. The spread file `train_spread_honest12.npz` is not present in the current working environment.

The bead's own notes already contain the pre-registered design and price: two runs of ~3.1 h each = **6.2 h of card**, alpha 1.0 vs alpha 0.0, with a control that uses the same with-replacement bootstrap draw so the only difference is the tilt.

## What is still missing

| missing sampler | why it is not the same as `Disagreed` | what it would take |
|---|---|---|
| **Diversity / coverage sampling** | `Disagreed` re-draws the same hard studies; it does not guarantee coverage of under-sampled regions of feature space | maintain a running feature centroid or use a cached embedding and sample to maximise minimum distance, or cluster and sample per cluster |
| **Prevalence-balanced batches** | current batches are random slices of the drawn positions; Fracture has ~62 holdout positives and is swamped in every batch | stratify each batch by label presence or by rare-label count; touches `Trainer.batches` or adds a batch builder |
| **Loss-based hard mining** | explicitly rejected as the primary signal, but could be a secondary filter after disagreement | needs per-example loss recorded per epoch (see Q10 / knee-97jg.2); then a `HardLoss` policy |
| **Small-loss / easy-example selection** | opposite of hard mining; useful if the model is overfitting to noise | same per-example loss dependency |

## Verdict

- The disagreement sampler is the right null to run first. It is ready to `arm_check --smoke` and queue.
- Diversity and prevalence balancing are separate, unimplemented axes. They should be designed only after the disagreement result is in, because a flat disagreement result changes where the residual is assumed to live (input/label side rather than draw side).
- Any loss-based sampler is blocked on per-example loss instrumentation (Q10 follow-ups).

## Touch points for whoever implements

- Smoke/queue: `experiments/arms/disagreed_sampler.sh`, `kneemri/experiments.arm_check --only disagreed_sampler --smoke <dir> --smoke-studies 60`.
- Diversity sampler: new class in `kneemri/training/sampling.py`, a cached embedding reader in `kneemri/inference/prediction_cache.py` or `kneemri/embeddings/`, and a new flag in `flag_vocabulary.py`.
- Prevalence-balanced batches: `Trainer.batches` (`kneemri/training/trainer.py:445-466`) or a new batch policy.
