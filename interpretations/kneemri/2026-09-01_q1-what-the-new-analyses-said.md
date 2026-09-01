# Q1 (knee-5qpz): what the 2026-08-27 → 09-01 instruments actually told us

2026-09-01, Kimi. One page, numbers only; every row cites the module and the file on disk holding the
number. Harvested by a read-only agent re-reading each instrument's docstring and its recorded
output; nothing here is from memory.

## The bead's six

| instrument | what it measured | the number (denominator) | decision it licenses |
|---|---|---|---|
| `attribution_faithfulness` + `slot_probe` (kneemri/evaluation/, landed 860a99a2, 08-31) | is slot-drop occlusion faithful (infidelity/sensitivity, Yeh 2019); is head attention a cheap surrogate; readout vs the anatomy table | attention ≥ ablation on **12/12 findings** (infidelity 0.037–0.125 vs 0.052–0.188); readout: **7/12** findings read heaviest on the table-primary slot, mean off-table mass **0.259**, 0/12 degenerate (one member, attend_true_s2026, 200 studies) | head attention is a licensed cheap importance surrogate; 0.259 off-table mass is a measured fact about the model. No entry decision. **Run but never written up — numbers live only in logs** (`logs/wgaa_faith.log:4-18`, `logs/wgaa_readout200.log:38-41`) |
| `oof_oracle` (kneemri/evaluation/oof_oracle.py) | pipeline-vs-author pixel fidelity on published checkpoints' oof blocks | standing numbers max\|diff\| **0.025**, per-study corr 0.99; graft 0.888 vs public forks ~0.899 | **landed 2026-08-18, never re-run in the window** — its number is docstring-and-old-interpretation only (`oof_oracle.py:12-14`, `interpretations/kneemri/2026-08-14_their-router-not-ours.md:63`) |
| `reader_agreement` + overcall_clustering / overcall_audit / label_errors (core/data/labels/) | LLM-reader vs radiologist agreement; who owns the over-call; is it independently detectable | **140 wrong of 684 cells** (57 studies): 108 confident FP, 32 misses; English 66.7% vs non-English 64.2% (+0.025 [−0.076, +0.126]); dispersion **1.24 [0.83, 1.65]** (56 studies) brackets independence; extractor failure only **16/108 (14.8%)**; cleanlab detector lift **4.05×** at budget 25 | per-study label lane closes without card; loss sharpening contraindicated; confident-FP flagging viable as a gate (`logs/reader_agreement_20260825.log:22-25`, `interpretations/kneemri/2026-08-31_the-overcalls-are-mostly-not-the-extractor.md:21-46`, `logs/label_errors.log`) |
| `member_divergence` (landed d1fc9f05, 08-30) | JS divergence (bits) between members' Bernoulli beliefs — the confidence axis rank correlation cannot see | honest12 roster (12 members / 909 studies): margin_seat most divergent **0.0665 bits** at 0.8772 rank corr; mae_matched most rank-distant at 0.0598 bits — the columns top different members | **none — informational**; explicitly unvalidated as a predictor (`interpretations/kneemri/2026-08-30_confidence-divergence-column.md:22-71`) |
| `champion_uncertainty` + `uncertainty_split` (landed 78a54aef, 08-31) | epistemic/total split: the share of an entry's uncertainty a different member could still remove | honest12: pooled reducible **12.7%** (total 0.6490 / aleatoric 0.5668 / epistemic 0.0822 bits, 12 members / 909 studies); whole roster of 79: reducible **8.3%** — more members CONCENTRATED belief; pinned champion raptorweight_v1: **refuses** (1 of 6 scoreable) | adding members to this pool goes the wrong way; label quality is the untouched lever (`interpretations/kneemri/2026-08-31_the_pool_is_redundant_not_the_task.md:38-57,113-122`) |
| `label_combine` (window run 08-30) | per-label member routing with split-half honest estimator vs flat | pool of 113 / 909 studies, LogitAverage: flat 0.8666; per-label routing peak 0.8760 @ top-10 vs global greedy 0.8763 @ 19; top-1-per-label 0.8657 is the WORST point; all gaps far under the 0.0062 floor | **combiner lane closed** — no per-label specialists exist; selection (+0.0094) is real but routing supplies none of it (`interpretations/kneemri/2026-08-30_the-combiner-is-closed.md:52-68,105-115`, `logs/routing_logit.log`) |

## The instruments the bead's list missed (same window, all with numbers on disk)

| instrument | the number | decision |
|---|---|---|
| `competence_axis` (3c4d9e1b, 08-27) | **81.7–82%** of what separates 105 members (909 studies) is one scalar | the pool is one number deep (`interpretations/kneemri/2026-08-27_the-pool-is-82-percent-one-number.md`) |
| `foreign_gate` / `raptor_feed` (8daf5e85, 8c49c98e, 08-27) | same foreign checkpoint: −0.0348 through our feed, **+0.0281 (0.8915)** through theirs, vs our best 0.8634 (909 studies) | the feed was the whole shortfall (`interpretations/kneemri/2026-08-27_the-feed-was-the-whole-shortfall.md`) |
| `foreign_overlap` (e1f2109e, 08-29) | **896 of 909 (98.6%)** of our holdout sits in the foreign member's training table | voided raptor_solo's 0.8938 (`interpretations/kneemri/2026-08-29_the-foreign-member-trained-on-our-holdout.md`) |
| `teacher_distance` (4bdda3ad, 08-28) | 13 foreign checkpoints = **4 members and 9 copies** (nearest agreement 0.82–0.91, 400 shared studies) | the fleet is small (`interpretations/kneemri/2026-08-28_fleet-distance.md`) |
| `vote_ceiling` (54fe65d0, 08-29) | quality weighting buys **+0.0000 to +0.0002 honest** over flat (42 members / 909) | the vote has nothing to weight (`interpretations/kneemri/2026-08-29_the-vote-has-nothing-to-weight.md`) |
| `pool_census` (66cde5c7, 08-30) | all strong members are twins: nearest agreement 0.9011–0.9496 while the pool reaches 0.0688 | the strong-and-distant cell is nearly empty (`interpretations/kneemri/2026-08-30_the-strong-are-all-twins.md`) |
| `level_calibration` (0c85223f, 08-30) | reader orders well (per-finding rank AUC 0.91–0.99) and scales badly (696 cells / 58 studies) | trust the reader's order, not its levels (`interpretations/kneemri/2026-08-30_the-reader-is-right-about-order-and-wrong-about-value.md`) |
| `label_errors` (f5e1f808, 08-31) | independent cleanlab detector: lift **4.05×** on the 108 confident FPs | targeted label surgery is detectable, not blind (`logs/label_errors.log`) |
| `member_axes` / `member_mahalanobis` / `member_space` (b1c05b69, 08-31) | the TWIN/NOISY/DISTINCT/PUZZLE 2×2 join | adoption gate stands; **not yet validated against a board seat decision** (commit body) |
| `seat_bonus` (ec6dc769) | board ensemble bonus **saturates +0.014 at 6 seats / +0.020 at 7** (final_ws_e12 across 14 entries) | seat count past 6 is spent (commit body) |
| `entry_offline` / `selection_bias` (963e9a33, 0db425b6) | transfer slope **1.026 board per offline point**; bias-corrected offline priced at 0.0027 by the board | the board priced an offline constant (`interpretations/kneemri/2026-08-30_the-board-priced-an-offline-constant.md`) |
| `residual_split` (b3c04b52, 09-01) | **24.3%** of 896 pooled ensemble errors unanimous; **~5.4%** minority-recoverable — the combiner's maximum (honest12 / 909) | the residual's shape caps every combiner (`interpretations/kneemri/2026-09-01_a-quarter-of-the-residual-is-unanimous.md:24`) |
| `population_split` / `study_series` (1305286f, 09-01) | Fat_Suppression ≡ Fluid_Sensitive on **24,371/24,371** rows; per-positive error flat (0.482–0.555) across series-count strata | no acquisition axis exists (`interpretations/kneemri/2026-09-01_the-residual-has-no-acquisition-population.md:15-30`) |
| `feature_probe` (fbda236c, 09-01) | frozen trunks **tied** (DINOv2-L 0.7996, V-JEPA2-L 0.7985, DINOv2-S 0.7896 on the 909) but **0.019 apart fitted** | the trunk lane is a recipe question (`interpretations/kneemri/2026-09-01_the-trunks-are-tied-frozen-and-not-fitted.md:9-16`) |
| `train_disagreement` (59b89541 + 7398ea5c, 09-01) | member-spread bands: the quiet band (<0.05) holds **zero studies** (699/2175/624 across the higher bands; mean 0.1244) | drove the sampler tilt to α=1.0 (commit 7398ea5c body) |

## Correction 2026-09-01: the seating closure was misread

`seat_value --against honest12` over the ten strongest unshipped members shows that **solo strength and
distinctness are anti-correlated in the shipped pool**, not because a distant seat cannot help, but
because every distant member ever seated was weak. Two members are both DISTINCT and strong enough:
`aprior_dense_s2026` (rank 0.8811, js 0.0139, solo 0.8557) and `attend_dense_s2026` (rank 0.8916,
js 0.0128, solo 0.8569). The size bar for a distinct seat was priced on the board at >= 0.8521 solo,
and both clear it. `vjepaseat` failed because it was distant *and* weak, not because distance buys
nothing. So `kneemri-distinct14` (honest12 + these two) is the open test, with control honest12's
LANDED 0.937. This does not reopen the member-count lane (seat bonus still saturates at 6–7 seats),
but it does reopen the **which-member** lane for seats that are both distinct and strong.

## Correction 2026-09-01 (b): lensseat closes a sixth direction

`lensseat_v1` (ref 55940907) gave the champion's most-swapped seat to `labelwise_dense_s2026`
(`--slice-pooling labelwise`, third-best solo 0.8613) and read **0.944 vs 0.944** control — flat. A
pooling FUNCTION should be a distinct axis from the competence scalar `competence_axis` measures, but
the board says it is not: it lands on the same 81.7–82% one-scalar reading. This is a sixth independent
direction closing on the same scalar, not a qualification of it.

## The headline

Every window instrument that produced a decision pointed the same way: combiner / seating /
member-count / pooling-function lanes are closed from **six** independent directions (uncertainty
12.7%→8.3%; residual 5.4% minority; routing null; census twins; anti-correlated solo/distinctness in
the shipped pool; lensseat flat). The open levers are **label quality** (108 confident FPs, 4.05×
detectable, only 14.8% extractor failure), **non-monoculture trunks** (frozen-tied, fitted-apart; the
foreign fleet is 4 real members and 9 copies), and the pending **distinct-and-strong seat** test
(`kneemri-distinct14`). Two housekeeping findings: the attribution instruments were **run but never
written up**, and `oof_oracle` sits in the bead's list but landed before the window and was never
re-run in it.
