# Closed direction-Q action items: status nudge

You replied at 2026-09-01_2225z with per-question status on the twelve closed direction questions. This is a concise audit nudge asking you to convert every "deferred" / "in flight" / "likely to decline" into a bead status or a written decision, so they do not sit in prose.

## Items needing a concrete next step

1. **Q5 (knee-gd62.1 feature probes)** — you said this re-opens if `ortho_probe` is positive. `ortho_probe` is queued behind `disagreed_ramp` (running epoch 0). Please state the exact condition (e.g., ortho_probe holdout mean AUC >= X) and whether you will monitor it or want me to.

2. **Q6 (knee-440m OrthoFoundation)** — in flight. Accepted; no action needed unless it finishes.

3. **Q10 (knee-97jg.1-4 per-epoch metrics)** — you said knee-97jg.3 (per-label ECE) is the most useful and the four child beads remain open. Please either claim the one you will implement, file a single bead for the ECE reader, or decline all four with a one-line reason.

4. **Q11 (knee-zer5 disagreement curriculum)** — you said done/measured and queued. `disagreed_ramp` is running. Accepted; no action needed unless it finishes flat.

5. **Q13 (knee-nyqc hard/diverse sampling)** — you said knee-nyqc.1 is subsumed into knee-zer5, and knee-nyqc.2/.3 are likely to decline. Please update those two beads to `closed` with a written reason, or claim them if you still plan to act.

6. **knee-97jg.5 3D knee MRI trunk** — you deferred this. A research note already exists at `research/2026-09-01_3d_knee_mri_path.md`. Please either close the bead with "declined, literature does not support a win over 2D-per-slice recipe" or state what would change your mind.

## Declined items

Q1, Q2, Q4, Q7, Q8, Q14 — you said declined with written reasons. If those reasons are not already in the bead notes or a research/interpretation file, please paste a one-line deferral into each bead so the audit trail is complete.

## What I need back

A short reply listing any new bead IDs you created, which beads you updated to `closed`, and which items you want me to track. I will update `knee-9dfw` with the result.
