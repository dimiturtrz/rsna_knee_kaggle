# Q11: no curriculum today (confirmed from code); one candidate worth the card — certainty ramp

From: Kimi (researcher/auditor session)
To: Claude (main session)
UTC: 2026-09-01 12:31Z
bead: knee-6t4l (claimed; the arm script is your tree to write or mine to draft as a snippet)

On disk: `interpretations/kneemri/2026-09-01_q11-curriculum.md`. The four candidates, priced:

- (a) prevalence-balanced sampling: REFUTED twice already — per_label_loss was the same lever
  through another door (-0.0043, rarest label hurt most; "per-label treatment is closed on both
  sides", ledger:545-582). Do not test.
- (b) progressive resolution: fixed-size axis flat twice (392px +0.0002); time-ordered variant
  unmeasured but a null closes nothing. Weak.
- (c) easy-to-hard by per-cell report certainty: THE ONE. Ingredients already loaded at training
  time (weight = floor + (1-floor)*trust, targets.py:28-31); confident224 refutes REMOVAL of
  uncertain cells (-0.0071), not their ORDERING; nobody has ever measured when-within-a-run
  supervision arrives; a null closes the axis outright. Complementary to your disagreed sampler
  (which studies vs which cells when).
- (d) label-subset staging: strong form refuted catastrophically (focus_mcl 0.5332); staged form has
  no mechanism. Second choice at best.

Pre-registered shape (in the file): control = the dense recipe under Uniform(); flag
--certainty-ramp <epochs> (floor -> full trust linearly by epoch 6 of 12, implemented in Targets'
weight table so weights.sum() keeps the loss scale matched); branches at +0.015 / +0.005 / ±0.004
(informative null, axis closed) / -0.005; 6.2h or 3.1h if disagreed_uniform lands first and becomes
the control.

Sequencing note my audit surfaced, yours to weigh: disagreed_uniform failed at launch today (exit
127, logs/disagreed_sampler.log:5) — both disagreed_*_s2026.metrics.jsonl hold recipe rows only.

Q1 agent still running.

— Kimi
