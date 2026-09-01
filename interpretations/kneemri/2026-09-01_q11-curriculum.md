# Q11 (knee-6t4l): training curriculum beyond LR scheduling — none today; one candidate worth a card

2026-09-01, Kimi. Read-only audit, citations throughout.

## Is there a curriculum today? No — confirmed from the code

What varies across training, exhaustively: `OneCycleLR` (pct_start 0.15,
`kneemri/training/trainer.py:92,350-368`), per-batch window re-draw (`trainer.py:414`), constant
augmentation with no flips (`augmentation.py:39-41`), optional EMA and mixup (both OFF in the dense
recipe), resolution fixed at setup (`run_setup.py:536`). Until today sampling was a uniform
permutation (`trainer.py:393-397`); today's commit 59b89541 added the `Sampling` strategy
(`Uniform`/`Disagreed`, `kneemri/training/sampling.py`) — a per-epoch *weighting*, still not a
within-run time order. The bead's premise is accurate. The bead overstates the palette: EMA and
mixup are available but off.

## The four candidates, priced

Denominators: 3,498 training studies, 909 holdout, 52,872 cells. Floors: 0.0062 three-seed paired;
house bar +0.015; an arm costs 2 × ~3.1 h of card.

**(a) Prevalence-balanced sampling — refuted in substance, twice.** The same gradient-share lever
ran as `per_label_loss` and scored **−0.0043**, with the rarest label hurt most (Fracture −0.022);
the oracle line killed per-label weighting at ensembling; "per-label treatment is closed on both
sides" (`experiments/README.md:545-582`). AUC is prevalence-invariant. A sampler is this lever
through a third door. Do not test.

**(b) Progressive resolution 224→336 — axis priced flat at fixed sizes; the time-ordered variant is
unmeasured but weak.** Fixed 392 px at matched rate scored +0.0002 (ledger:296-297, 875-897).
`resample_to` makes it cheap to build, but the honest appeal is card-time savings, not score, and a
null closes nothing (the axis is already closed at both ends).

**(c) Easy-to-hard by per-cell report certainty — the only candidate with a measured mechanism
pointing forward.** The ingredient is already loaded at training time: weight =
floor + (1−floor)·trust, CONFIDENCE_FLOOR = 0.25 (`core/data/labels/targets.py:28-31,326`);
`confident.csv` exists. The adjacent arm refutes REMOVAL, not ordering: `confident224` deleted the
14,445 UNK cells (27.3% of 52,872) and lost **−0.0071** — the uncertain cells carry real signal
(ledger:326). Blendlabels (+0.0034 mean, +0.0070 on the six label-bottlenecked findings) says the
label lane is live and small. Nobody has ever measured *when within a run* supervision arrives.
Complementary to today's `disagreed_sampler` (knee-nyqc), which moves *which studies*, not *which
cells when*. Prior: P(clears +0.015) ≈ 0.10; P(clears the floor either way) ≈ 0.5.

**(d) Label-subset staging — the strong form is refuted, the staged form lacks a mechanism.**
`LabelFocus` exists (`label_focus.py:43-60`) but full-focus arms were catastrophic (focus_mcl
0.5332, focus_latmen 0.4747) — jointness is demonstrably load-bearing. Second choice at best.

## Recommendation: test (c), as `certainty_curriculum.sh`

- **Control**: the dense recipe exactly as `disagreed_sampler`'s, `Uniform()` sampling. If
  `disagreed_uniform_s2026` lands first it IS the control and the arm costs one run, not two.
- **Flag**: `--certainty-ramp <epochs>` — cell weights start at the floor for every cell and ramp
  linearly to full trust by epoch 6 of 12. Implementation: epoch-indexed blend of the weight table
  in `Targets`, not the objective — the `weights.sum()` normalisation keeps the loss scale matched.
- **Branches** (mean holdout AUC, arm − control): ≥ +0.015 → lever real, sweep ramp {3, 9};
  +0.005…+0.014 → directional, keep flag; ±0.004 → **informative null, time-order axis closed**;
  ≤ −0.005 → noisy-late ordering costs; read against confident224 before concluding.
- **Gate**: 0.0062 floor, +0.015 bar. Both checkpoints are ordinary members, seat-testable either
  way. **Cost**: 6.2 h, or 3.1 h against a landed disagreed control.

## Status caveat (for sequencing, not for the answer)

`disagreed_uniform` failed at launch today (trainer exit 127, `logs/disagreed_sampler.log:5`); both
`disagreed_*_s2026.metrics.jsonl` hold recipe rows only, no epochs. Q13's read should land before
this arm's prior is finalised.
