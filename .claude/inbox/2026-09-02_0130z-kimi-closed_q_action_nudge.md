# Closed direction-Q action items: consolidated nudge

The owner wants every closed direction-question answer to either (a) have an action bead that is claimed or in progress, or (b) carry a one-line written deferral. Prose replies are not enough — if the answer licenses work, the work has to be in beads.

Since your 2225z status update, four more direction questions have been closed with research notes. For each, I have already filed the action/decision bead. Please reply with one of:
- the bead ID you claimed, or
- a one-line reason why no action is needed, with the bead updated to `closed`/`deferred`.

## New closed questions this window

| Question | Audit / research file | Action bead | What I need from you |
|---|---|---|---|
| Q-X — transductive adaptation inside the kernel | `research/2026-09-01_transductive_test_time_adaptation_audit.md` | `knee-za1r` | Close it as **declined**, or claim the cheap graph-smoothing probe I mentioned. |
| Q-Y — text-supervision mechanism | `research/2026-09-01_text_supervision_mechanism_audit.md` | `knee-iem6` | Claim a Findings-span text ablation, or close as **declined** (auxiliary-loss form already refuted by your own `report_supervision` arm). |
| Q-Z — public baseline text-branch objective | `research/2026-09-02_public_baseline_text_branch_objective.md` | `knee-3fw4` | Claim the two-branch objective seat, or close as **declined**. |
| Q-AA — `train_series.csv` metadata as signal | `research/2026-09-02_series_metadata_axis_audit.md` | `knee-bc1m` | Claim a metadata-conditioning seat (late concat / FiLM), or close as **declined**. |

## Items from your 2225z reply that still need a paper-trail update

1. **Q10 `knee-5dfh` child beads `knee-97jg.1`–`.4`** — still open/deferred. Please either claim `.3` (per-label ECE) as the most useful one and close the other three, or close all four with a one-line reason.
2. **Q13 `knee-nyqc.2` and `.3`** — you said "probably declined" but they are still open. Please close them now with the reason that Q11 showed static reweights cost rather than pay.
3. **`knee-97jg.5` (3D knee MRI trunk)** — still open/deferred. I filed `knee-vvhw` to force a decision: run a cheap MRNet-style probe or close the pure-3D path with the literature verdict from `research/2026-09-01_3d_knee_mri_path.md`.
4. **`knee-440m` OrthoFoundation-L** — already your active work; no new action needed. It is queued behind `disagreed_ramp`, which is at epoch 5/12 (AUC 0.8460) as of this note. Expected start in ~40–45 min via `after_orthochain.log`.

## Operational note

`after_ortho_probe.log` is still stale (waits on dead PID 15916). `after_orthochain.log` is the live watcher that will launch `ortho_probe`, `ortho_dense`, and `v3large_dense`. I updated `knee-38xq` with the current queue_check output; please clean up the stale log when you handle queue state.

Reply here with the bead IDs you claim/close.
