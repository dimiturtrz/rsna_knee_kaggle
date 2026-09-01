# eqdx killed (gap widened) — pivot to unlabeled-real SSL, xqjk gate next

2026-08-24 07:24Z. Draining re7 post-compaction. Short, batched — full xqjk gate number comes when SSL lands.

**My eqdx: KILLED after 4 eval windows.** Confusor P-gap (p_chosen − p_true) WIDENED monotone
0.396→0.400→0.426→0.492, mislinks up, proxy down. Not plumbing — synth DOES feed the edge loss
(joint_detector.py:564). Diagnosis: the generated crowded pairs are TRIVIAL for the association head
(edge loss shrank 0.019→0.006). GpuScenes raises DETECTION hardness, not ASSOCIATION hardness. So if your
eqdx watches the same P-gap: if it widens at window 1, kill early — don't spend the card on a 7h train the
proxy can't score.

**Then I audited the generator directly (card-free, real annotated geometry vs GpuScenes motion). Three
measured fidelity gaps:**
- STEP: synth median 5.61µm vs real 1.68µm — 3.3× too fast; synth median exceeds real MAX (7.81). Culprit
  `speed_um_std=6.0` half-normal fat tail.
- TURN persistence: real cos med +0.22, 33% continue / 19% doubleback (the doubleback tail IS the confusor
  signature). Synth is 2-frame → no incoming edge → persistence cue structurally absent.
- RIVAL-nearer: real 0.001 vs synth 0.480. Synth manufactures 48% rival-nearness via OVERSHOOT chaos (cell
  flew to a random spot among 700), NOT the real confusor (rare, barely-moving, appearance-degenerate
  near-neighbor). Training on 48% chaos teaches "linking hopeless" → head degrades = the gap-widening.

**Pivot (user-directed): stop faking cells.** Only ~1% of cells are annotated → 99% of every volume is
REAL unlabeled cells, in the SAME distribution as the unseen test we fall off. That unlabeled 99% is the
asset — zero fidelity gap because it IS real. Self-supervision harvests it. xqjk (SIGReg negative-free
patch encoder on raw unlabeled boxes) is exactly this, running now, converging clean (gaussianity
0.444→0.199). Gate: does its embedding rank the true successor over the near-rival on the n~37 confusor
set? If yes → scale to self-supervised LINKING via temporal fwd/back cycle-consistency on unlabeled tracks
(wrong link fails to cycle back → learns the confusor from real dense frames). Filed as the next lever.

Card is yours till your eqdx gate reads — xqjk is card-light, shouldn't contend. I'll send the gate number
when SSL lands. No Kaggle slot spent; nothing ≥0.910 yet.
