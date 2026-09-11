# RESEARCHER LANE STATE — lives at inbox/researcher_state.md (moved out of scratchpad 17:30Z per
# owner directive: scratchpad/ is retired for this lane; durable deliverables go to research/ and
# interpretations/kneemri/, instruments to research/scripts/, external pulls to research/external/).
# References to "scratchpad/researcher_state.md" in older notes mean THIS file.
# TIMESTAMP RULE (D338, from main's measured drift table): letter filenames AND letterheads are
# stamped from `date -u` RUN AT SEND TIME in the same command — never estimated. D326–D337 carry
# estimated stamps, drifting +1h→+4h; D337's letterhead (18:40Z) is local-mislabeled, file is 1540Z.

## LATEST (17:23Z) — M359 answered (D353): gold rescore built, INVERTS the v2 ruler; next free letter D354; main last seen M359 (17:19Z)
- Harness: research/scripts/adjudicated_rescore.py (npz + holdout469 key + verdicts; splits
  tear-277/nontear-192/contusion-120/all-469; two stems → paired McNemar).
- FIRST GOLD-RULER READ (v6tears at e2, EARLY — re-score at e12): tear-277 acc band_span 0.693 <
  v5blend 0.762 < **v6tears 0.866** (McNemar 39–10); all-469 0.625→0.693→0.795. On the v2 ruler the
  same arms read 0.8565 → 0.8113 → 0.755 — MONOTONE WORSE with more correction. **The v2 ruler's
  anti-correction gradient is measured as artefact on gold: it penalises each arm in proportion to
  how much of v2's error it unlearned.** The table chain must be read on the adjudicated rescore +
  T1, never the v2 ruler.
- Anomaly to re-check at e12: v6tears also jumps on NON-tear (0.693) and Contusion (0.675) cells
  despite changing only tear labels — shared-feature liberation or early-epoch artefact.
- Chain (his card): v5blend_span DONE (peak 0.8113 v2-ruler), v6tears_span running, v7nontear_span
  queued. 469/334 studies/1.4 cells per study; tear split one-sided by construction.
- OPEN: re-score v6tears at e12 + add remainder-control (agreement with npz v2-truth outside the
  1,379 corrected cells) to the harness.

## PRIOR LATEST (17:10Z) — M358 verified + answered (D352): Simpson's paradox confirmed
- Main's re-cut reproduces exactly on my join: Contusion hi-margin cluster 0.780 / lo 0.861 (INVERTED);
  OA-family hi 0.783 / lo 0.294. My D351 aggregate was Contusion's 0.833 base rate leaking through a
  shared margin threshold — recorded as my error.
- THE OPERATION: Contusion WHOLESALE (556 cells, ~0.833 → ~463 fix/~93 break), OA family MARGIN-GATED
  ≥0.6 (147 cells ~0.78), Effusion/Synovitis/571 left (low-margin cluster is 0.70 not 0.50 — leave for
  a RATE reason). Net ~+453 vs my rule's +241. Sequencing: NOT in the v6tears table; the ~93 breakage
  goes in the table header before banking.
- V6 PROMPT REPRICED: bar moved from beat-0.167 to beat-0.833 — the prompt's repair value is nearly
  dead; its residual instrument question = "does the qualifier recover the substitution's ~93 break
  cells (fracture-adjacent true bruises)?" — measurable on LocalReader.annotated(58) gate before any
  full read. Substitution first, prompt second, never stacked.

## PRIOR LATEST (17:06Z) — M357 answered (D351): the soft margin separates
- v6tears BUILT + banked (main, f189c68e): 1,371 cells changed, guards in code; arm v6tears_span
  queued (P(clears 1.5% alone) ~0.1, displaces depth24; v2 ruler should read AT/BELOW v5blend, T1 at/above).
- M357's three candidates measured on 72 OA-family verdict cells: **(a) soft margin WINS** — high
  tercile (median |Δ| 0.66): v2 acc 0.17 / cluster 0.83; lo/mid = coin flip. Spearman −0.323, p=0.006.
  (b) cluster spread second (loose cluster predicts, but it's the margin in disguise). (c) language
  untested — subsumed if margin holds on fresh cells.
- Rule: substitute cluster on non-tear dissent with margin ≥0.6 = **365 cells** (Contusion 204, LatOA
  66, MedOA 37, PF OA 31, Baker's 13, Eff 10, Syn 4), expect ~303 fixes / ~62 breaks; leave 571
  contested cells alone. Contusion overlaps the v6 qualifier — sequence, don't stack.
- Caveat: hypothesis on 72 cells; fresh OA adjudication batch before banking.

## PRIOR LATEST (16:51Z) — M356 answered (D350): substitute wholesale on tears
- 469 CLOSED (main, all blind): 32/469 affirmed (6.8%). v2 correct 51/469; cluster 418/469. On TEAR
  dissent cells (277 cells / 223 studies, ALL v2=1): **v2 accuracy 0.000, cluster accuracy 1.000**.
  OA columns opposite: v2 right ~half (LatOA 0.632) — leave OA alone in any v6.
- SAMPLING ANSWER: 469 = selector ∩ fold-0 holdout; folds.py:61 assigns folds by HASH OF REPORT TEXT —
  no label-side variable in the path; tear cells 20.1% in holdout vs 20.6% expected; clustering mild
  (1.4 cells/study). 277/277 is a population statement.
- WHOLESALE SUBSTITUTION BANKED: 1,379 corpus tear-dissent cells (1,097 studies) — substitute cluster
  value, expected ~1,379 fixes ~0 breaks. Guards: only the selector's cells (uncontested mass errs
  1.48%), and cluster's 1.000 rests on 223 studies (~1% residual invisibility ≈ 14 cells).
- Contusion qualifier (v6, CONTUSION_CLARIFIED) priced as INSTRUMENT by main: ~60 cells of one column,
  gated by LocalReader.annotated(58) before the full read.

## PRIOR LATEST (16:40Z) — M355 answered (D349): ALL-TWELVE
- Main's M354 error (vision_prompt.py is the IMAGE probe, not the blocker) self-corrected in M355;
  the seam is built: ReadingPrompt.clarifications tuple + CONTUSION_IMPACT_QUALIFIER verbatim from my
  D348 sentence, unit-tested (CONTUSION_CLARIFIED = v6 prompt).
- My answer to contusion-alone vs all-twelve: ALL-TWELVE — (1) attribution (single-finding prompt is a
  different prompt, two variables), (2) THE QUALIFIER'S NOUNS ARE THE OA COLUMNS' VOCABULARY — the
  degenerative/traumatic distinction is where the (c) under-call dip lives; all-twelve measures that
  interaction free, (3) over-generalisation risk to the tear columns visible. TUNE chunks 1–8 (spent),
  REPORT on 9–10 across ALL TWELVE columns.
- Main: chunk 9 landed (31/423 affirmed 7.3%); two of his blind calls were my named exemplars (h0330,
  h0446) — rubric cross-check working.

## PRIOR LATEST (16:30Z) — M354 answered (D348)
- CONTUSION SUB-TAXONOMY (30 cells, report lines read): **Class 1 = degenerative/OA marrow edema read
  as traumatic contusion, >60%** — v5 fires on any "bone marrow edema" phrase, never checks the cause;
  criterion says "edema-like signal FROM IMPACT". Same defect shape as degeneration-vs-tear one column
  over. Class 2 = fracture-entity boundary, BOTH directions (SIF/stress/OCD; the 4 over-corrections
  live here). Class 3 = self-contradictory/negation-scope (3–4). Prompt sentence nameable TODAY:
  "subchondral/degenerative edema adjacent to cartilage loss is NOT contusion; contusion requires
  trauma context (impact, pivot-shift, fracture association)" — protects class 2's over-correct half.
- (b) table_blend.py = MEAN/tiebreak over base+FOREIGN readers; does NOT own column substitution. Keep
  hand-rolled pandas; name the op in the table header.
- (c) under-call dip NOT Contusion: v5's uc errors = PF OA 4, Contusion 4, MedOA 3, LatOA 2 — OA-lean,
  second small thread, not chased.
- Main unfroze the ledger writes after the merge audit. Card: depth20_b10 → v5blend_span → depth24.

## PRIOR LATEST (16:18Z) — M353 answered (D347)
- v5blend (v5 everywhere + v3 Effusion, main's construction): prices EXACT — 0.830/0.696 on 282
  verdicts (= v5) and gold Effusion acc 0.737 / AUC 0.8299 (= v3). report_labels_v5blend.csv trains
  on card now (v5blend_span). Effusion positives 1,342→1,492.
- MERGE-BUG AUDIT: only D345's v3/v5 rate table + residual split were corrupted (retracted, superseded
  by D346 clean). D343/D344/T1a/reseat/flips/κ/gold all traced SAFE (pre-chunk-3 or tables-only).
- v5 residual on 282: **Contusion owns it** — 26 of 38 persistent over-calls + 4 of 6 over-corrections
  (0.559 right on 68 cells). A v6 prompt = single-column commission, boundary-specific
  (bruise/fracture/OA-edema), both directions — not sterner, not gentler.
- Main's aggregator now reads the answer column dynamically; my pipeline normalises both spellings.
- Chunks 7–10 pending; 282 verdicts, affirmed 6.4% vs pre-registered ~25%.

## PRIOR LATEST (16:12Z) — D346 URGENT CORRECTION sent
- **D345(1) WAS COMPUTED ON A CORRUPTED MERGE** (chunks 3+ write `answer`, chunks 1–2 `verdict`; my
  concat NaN'd 47+ rows and counted them all wrong). CLEAN numbers on 235 verdicts (216 oc / 19 uc):
  v2 0.051/0.737, v3 0.685/0.737, **v5reader 0.810/0.737 — McNemar p=7.1e-04, v5 WINS**. D345's
  "coin flip, no re-queue" is retracted; v5reader is the measured better table. Merge normaliser:
  read both column names (`verdict`/`answer`) — the bug class is in the state file permanently.
- M352: (1) v5 rungs {0.08,0.28,0.68,0.82} separate error halves; 0.5 cut already optimal. (2) best
  free table = v5@0.5; v3+v5 avg ≈ v5. (3) Effusion on gold (57/34 pos): v2 TP34/FP16/FN0 AUC 0.8299;
  v3 TP21/FP2/FN13 AUC 0.8299 — deletion swaps recall for precision at flat AUC; **host annotators
  label small effusions positive despite the criterion**; v5 worse on effusion (0.7449) → v5 wins on
  tear columns, v3 on effusion; a per-column table choice is supported.
- Main queued v3_labels_span behind depth20_b10 — D346 tells him v5reader is the verdict-supported
  table if the swap is still cheap.

## PRIOR LATEST (15:58Z) — M351 answered (D345)
- CHUNK-3 CORRECTION to my D344 headline: v3 over-call right 0.709 (n=86) → **0.473 (n=129)**; v5
  0.488 — statistically indistinguishable, no re-queue case. v2 0.039. Under-call n=12: v3 0.583 /
  v5 0.500. v3-vs-v5 corpus disagreement 3,062 cells (5.79%, balanced) — a real next-arm variable.
- Residual: 68/129 wrong under v3 = 37 persistent over-call + **31 over-CORRECTIONS** (verdict=1,
  v3=0). Contusion worst (0.351). A v4 prompt buys ≤half the persistent 37 and risks growing the 31 —
  not worth card yet.
- EFFUSION ANSWERED: criterion = "moderate or large amount of fluid". All 1,599 flips are v2-soft
  0.68 (hedge rung) → v3 0.38: small/hedged effusions graded negative = criterion-line CORRECTION not
  bias. Explains gold-58 Effusion weakness: blends inherited the lexicon's threshold error, so
  κ~0.90 agreement was agreement ON the same error.
- Main queued v3_labels_span (band_span recipe, v2-pinned eval, T1 read clears at ≥0.8742). Chunks
  4–10 still coming; every rate provisional.

## PRIOR LATEST (15:46Z) — M350 all four parts DONE (D344): THE FIX EXISTS (v3)
- MECHANISM (1): report_labels_v2.csv written 2026-08-11 = LEXICON-era pipeline. Criteria entered the
  reader 2026-08-27 (ec4d3f72; grade-wording fix 229101d9 same day). The criteria explicitly grade
  degeneration negative — but no prompt held them when v2 was made. Blast radius 12.1% of positives.
- THE FIX (2): report_labels_v3.csv ALREADY EXISTS (criteria-era Qwen reader). On main's 94 verdicts:
  over-call right 0.709 vs v2's 0.058, under-call preserved 0.875. v5reader similar (0.733/0.750).
  v4/v6/v7 are v2-lineage (0.058). Corpus-wide: v3 flips 6,461 v2=1→0 (ZERO reverse), deleting 40% of
  v2's positives (16,129→9,668); agrees with cluster on 1,420/2,315 selector cells.
- COST (3): using v3 = zero. New v4 per-finding prompt = card-hours on the Qwen reader, not card-free.
- PRIZE (4): pre-register band_span-recipe on v3, TWO-ruler read (v2 ruler punishes it by construction;
  T1 is the honest primary). If it clears control on T1, the lexicon table is the project's largest
  measured defect. D330 silence layer untouched by any prompt.
- Card (his): labelwise_depth20_b10 (~16:35Z), depth24 behind. Slot mark 15:57Z passing with the
  licence-clean six building as a kernel.

## PRIOR LATEST (15:36Z) — M349 answered (D343): the bias is ONE-SIDED
- 469 interim read (94 cells): over-call (v2=1 vs cluster 0, n=86) = **94.2% v2-WRONG**; under-call
  (v2=0 vs cluster 1, n=8) = **87.5% v2-RIGHT**. Mechanism: v2 reads degeneration as tear on TEAR
  columns (ACL 0/15, MedMen 0/18, LatMen 0/17) but correctly as OA on OA columns (LatOA 4/4 right).
  **The correction should be ONE-SIDED (flip only the 2,080 v2=1 cells)** — two-sided writes ~205
  wrong flips into the 235 under-call cells. v2_corrected arm flipped both ways.
- Corpus contamination priced: ~1,955 false-positive labels = **12.1% of all 16,128 positive labels**
  in report_labels_v2, systematic degeneration→tear, concentrated MedMen/Contusion/LatMen/ACL.
  Prompt-addressable (reports CONTAIN the words; the extractor maps them wrong).
- T1a (94 verdicts substituted): T1-six 0.8840 vs T0-six 0.8814 → reseat holds at +0.0026; one-sided
  ruler says the same (+0.0025). The reseat's case is robust to every correction variant.
- Main moved audit400 + holdout469 to the data root (D:/data/.../audit400/) — the durable home.
  His answers land there as answers_<i>of10.csv; chunks 3–10 pending.

## PRIOR LATEST (15:30Z) — M348 licence-clean reseat DONE (D342)
- Licence-clean pool 277 (18 dropped: 6 roster `# licence:` tags + 12 declines-sourced; tags vs names
  agree; tag gate covers ROSTERED only — 12 unrostered tainted live only in declines prose).
- Licence-clean six: stack_increments, chainfour, labelwise_extended, blendhold, stack_increments_base,
  resnet_224_long → T1 0.8843 / T0 0.8763. Constraint cost −0.0007 (vs 0.8850 unconstrained).
- VERDICT: T0-era greedy six (same pool) = T0 0.8773 / T1 0.8817. T1-six does NOT beat T0-six on T0
  (−0.0010, inside floor) → "ship tonight" arm of M347's outcomes FAILS; on T1 +0.0026 → ruler-aligned
  move, STAND-IN until the 469 verdicts. foreign_labels_dense licence-clean but NOT selected.
- Scripts: research/scripts/rescore_t1_ruler{,_licence_clean}.py (+ .out.txt).

## PRIOR LATEST (15:14Z) — M347 all three parts DONE (D341)
- (a) 369 npz → 312 kept / 57 excluded (named). Full two-ruler table: research/scripts/rescore_t1_ruler.py
  + .out.txt. VOID-INSAMPLE flags on raptor published (896/909) + DINOv3 folds (703/909). Nearly all
  honest members gain ~+0.003 under T1. **v2_corrected: 0.8392 → 0.8555 (+0.0163, pool's largest) vs
  band_span 0.8592 on T1 → −0.0037, a TIE not a kill — repair route NOT retired.** 8 of 469 selector
  cells pre-masked in npz (no-op flips).
- (b) Honest-pool greedy six on T1: stack_increments, merged_labelwise_cf, labelwise_extended,
  merged_span_cf, blendhold, foreign_labels_dense → T1 0.8850 / T0 0.8762. ZERO overlap with champion;
  champion 0/6 scorable (2 ff fold−1 + 4 no-npz) so the comparison is VACUOUS — champion is fully blind,
  and the corrected-ruler six can only be tested on the board.
- (c) holdout469 pack built (audit400/holdout469/, 10 chunks, key shut): pre-registered v2-right ~25%
  (~117/469). Awaits main's blind adjudication tonight.
- Card (his): labelwise_depth20_b10 then depth24 (instrument arms). Label axis off the card.

## PRIOR LATEST (14:54Z) — M346 answered (D340): corrected-ruler measurement
- THE EVALUATION CHANNEL, MEASURED: 469 v2-lone-dissent cells in the 909-study holdout (4.30%);
  flipping them in the RULER to cluster_value lifts every member (+0.001 to +0.014). calib_dense_hold
  vs stackfit: 0.0003 apart on v2 ruler → 0.0038 apart ORDER-FLIPPED (stackfit over) on corrected —
  the ledger75-exposing pair's answer moves with label content, not just cut. discord_mask recovers
  most (+0.0140) — never trained on those cells. Eval-on-own-generator inflation priced.
- v2_corrected (on card, tracking 0.8392 at e9 vs 0.8565) MUST be read on both rulers — pre-registered
  in D340: band-3 on v2 ruler AND no recovery on corrected ruler = repair route retires for real.
- Honest ruler = adjudicate the 469 cells (tonight-scale, 402 already done once); cluster vote is the
  cheap approximation with ~25% backward cells.
- M346 process: main's "subtraction route closed" retracted to ONE arm (confident_span) in the ledger;
  new rule in ledger: state class composition of any subset before acting on it.

## PRIOR LATEST (14:44Z) — M345 answered (D339): masking non-monotonicity = positive content
- discord_mask (13.94% internal-four discordant) = 0.8148 vs control 0.8565 (−0.0417); confident_span
  (27.3% UNK) = −0.0116. MEASURED mechanism: discordant cells are **96.8% v2-positive** (UNK cells
  0.0%, soft 0.280) — the arm masked **7,131 positives = 44.2% of all positive supervision**; UNK arm
  masked zero hard positives. Per-column damage tracks per-column positives masked at r=−0.65 (MCL
  65% of positives → −0.1058). All 12 columns fell. Main's "hard positives" survives second-order
  (audit: contested positives, 15.6% error) but the 7× needs no hardness.
- CLAUDE.md now carries the timestamp rule (main's edit, cites D338).
- v2_corrected on his card (~14:15Z start, pre-registered ≥0.8665 / 0.8515–0.8665 / ≤0.8515).

## PRIOR LATEST (18:05Z) — M343 TASK DONE (D336): selector CSV on disk
- `labels/v2_lone_dissent.csv` written: 2,315 rows (study_uid, column, v2_value, cluster_value,
  v2_soft, cluster_soft); selector = 3 foreign tables unanimous AND v2 differs. Four paths cited in
  D336. Main builds report_labels_v2_corrected.csv and arms `v2_corrected` tonight behind
  discord_mask.
- My answer on correct-vs-mask: CORRECT, but write cluster SOFT mean not hard vote (v2's targets are
  confidence-graded; hard flips smuggle a second variable; the 25% tail gets a lean not a flip).
- Process bug of mine: a temp draft named *.tmp.md in inbox/main/ got archived by his listener as a
  letter — .part convention requires the FINAL suffix after .part. Caught and repaired in minutes.
- discord_mask prediction on record (both sides): lands near control — 86% of its cells are clean.

## PRIOR LATEST (17:50Z) — M342 answered (D335): the selector is v2-lone-dissent
- Audit chain closed: B=0.0787 five-way / 0.0714 four-way, REFUTED both ways; M341 phantom retracted
  by main. Decomposition: 52 steven_full-only cells err 1.9%; 12 four-way-discordant cells err 75%.
- THE SELECTOR (D335): "≥2-of-4 dissent" = 93 cells corpus-wide (useless). The real structure: the 3
  foreign tables vote as ONE cluster (flight/steven_blend lone-dissent 0.0%); lone dissent ≈ **v2 vs
  the cluster** = **2,315 cells (4.38%)**, 90% v2-says-1 (over-call), concentrated MedMen 567 /
  Contusion 556 / LatMen 431 / ACL 298. Purity 75% measured TWICE independently (pack 9/12; gold-58
  24/32). ≈1,700 probable label errors pre-identified. Interpretation:
  interpretations/kneemri/2026-09-11_the-label-error-selector-is-v2-vs-the-field.md
- confident_span (mask 27.3% UNK) = 0.8449 vs control 0.8565 (−0.0116): subtraction by CONFIDENCE
  fails; the v2-lone-dissent set is subtraction by measured ERROR. discord_mask (circular 13.94%
  selector) running on his card.
- LANE HYGIENE (owner directive 17:30Z): scratchpad/ retired for this lane. State file =
  inbox/researcher_state.md; instruments = research/scripts/ (build_audit400.py, score_audit400.py,
  paired_boot.py, attn_full.py); external pulls = research/external/pubkernels/. Letter drafts no
  longer written (inbox/main/ is the record). audit400 pack stays in session Temp until the audit
  closes, then belongs under the data root beside overcall_audit.csv.

## PRIOR LATEST (17:20Z) — M341 phantom claim answered (D334)
- The audit's 0.6934 is REAL for the five M326 tables (13,774 unanimous-POSITIVE cells — a phantom
  caps that at ~57); main's phantom-combo exact match at 36,664 is a COINCIDENCE (11-col perturbation:
  35,962 vs 32,844). His search space (internal labels dir) never contained the foreign tables. His
  pack-split argument (338/402) was void — that split was my design parameter.
- THE REAL DEFECT IS MINE: the voter set. `llm_labels_full` (steven_full) is a calibration outlier —
  corpus positive rate 0.556 vs ~0.27 (Synovitis 0.97, Fracture 0.50), lone-dissents on 25.7% of
  cells. Four-way unanimity without it = **0.9505** (note: ~ main's withdrawn 93.9%). The REFUTED vs
  CONDITIONAL verdict swings entirely on whether it votes.
- M341's B=0.0487 is invalid as stated: weights and strata must share ONE voter set (his weights, my
  strata = mixed populations). Audit-as-run: **B=0.0787 REFUTED stands as the result of record.**
- Constructive path delivered: 52 of 64 D cells were steven_full-only dissents; 12 remain discordant
  four-way (row_ids in D334). Main can re-tabulate k_U'/390 and k_D'/12 from his EXISTING verdict
  file — no new adjudication. B' = 0.9505·(k_U'+3)/390 + 0.0495·(k_D'+3)/12.
- manifest.json corrected at source with provenance + the episode recorded.

## PRIOR LATEST (16:45Z) — M338/M339/M340 answered (D333)
Mailbox mechanics, NEW PATTERN (15:20Z): the listener runs as a HARNESS BACKGROUND TASK
(`python inbox/listen_for_mail.py --box researcher 86000`, run_in_background, disable_timeout) — its
exit IS the delivery notification. Delivery exits 1 (harness shows "failed" — cosmetic); a clean
exit-0 after 86000s is the timeout. On ANY completion notification: re-arm a fresh listener first,
then `ls -lat inbox/researcher/read/`. The external supervisor is RETIRED: three stale copies of
scratchpad/researcher_mail_supervisor.sh plus an inbox/watch.sh chain from another session were
respawning listeners that killed each other via _kill_existing_listener (last-writer-wins, fresh
heartbeat) — that kill loop, not delivery failure, caused the day's "dead listener" mysteries.
All killed 12:43Z. Verify no external watchers: powershell Get-CimInstance matching
'researcher_mail_supervisor|box researcher' should show ONLY the harness task's python.
STILL DO the direct `ls inbox/researcher/` + `read/` check on resume — notifications cover mail
arrival, not session restarts.
- D327 (M333-Q3, the audit — DESIGNED, PRE-REGISTERED, BUILT): five-way unanimity at ≥0.5 measured at
  **69.34%, NOT M326's 93.9%** (irreproducible under ≥0.5/≥0.25/≥0.1; 4-of-5=96.8%; asked main for his
  definition). Pack: 402 cells = 338 unanimous self-weighted over column×sign + 64 discordant sentinel
  (∝ column discordance; Synovitis 84% discordant = D20's 84% UNK). Pre-registered decision rule on
  B = 0.6934·(k_U+3)/338 + 0.3066·(k_D+3)/64: <3% LICENSED / 3–6% CONDITIONAL / >6% REFUTED (Northcutt
  flip threshold). Rubric: affirmed=1, negated-or-unmentioned=0. Scope: bounds EXTRACTOR error only.
  Pack at Temp/.../3dc4cc46.../scratchpad/audit400/ (pack + 8 chunks + key + score_audit.py + manifest,
  seed 20260911). Builder: research/scripts/build_audit400.py (parametric — rebuilds under another unanimity
  definition in minutes). Awaits main's adjudication (M200-style).
- D328 (M333-Q1/Q2): ONLY study-denominated medical AUC law = Zhang 2210.02189 (β=0.21–0.25, patients;
  measured 2000→6000 = +0.011 @ ~0.81). Mahmood 2207.01725 has NO medical task and NO tabulated
  exponents. Price of +0.010 from 0.944: 2.2–2.7× IF ceiling c=0; c=0.02 → 3.7×@β.25; c≥0.046 →
  unreachable at any multiple. Ceiling unmeasured anywhere ≥0.90 baseline. Corpora: NOTHING with
  label-overlap is competition-usable — OAI (3 OA cols at scale) DUA-blocked, CoPAS (6–7 cols) private
  MTA, fastMRI+ 4 clean cols images-DUA-blocked, MRNet 1.5 cols RUA, Rijeka CC BY-NC-ND, SKM-TEA
  no-derivatives, RIDER Knee RETIRED. Highest-value null: no public corpus pairs knee MRI with reports —
  our own corpus is the only annotatable one. Synthesis: power-law ceiling and audit residual are the
  same unknown; labels are the scarce resource, not images.
- Research file: research/2026-09-11_data-volume-power-law-and-knee-mri-corpora.md

- D329 (M335-Q4): meniscus-residual notebooks dissected. The "repro" is BYTE-IDENTICAL (cmp clean) —
  one measurement, not two. Applied mechanism = one-column rank overlay (MedMen = 0.30·Transformer +
  0.60·Raptor + 0.10·fullfit0033 bag, rank space, other 11 cols byte-copied from 0.936 parent) worth
  +0.001 board ≈ +0.012 on the column — per-column class bounded at 0.0057, dropped. "Residual" in
  the title = TokenResidualPool gated-head ARCHITECTURE in the DINO leg; the post-parent residual
  blend exists but is disabled (1.00/0.00). Bag model = private bundle (hash-bound runtime stealing
  the DINO input cache via a numpy.zeros monkeypatch), meniscus-only output, training labels
  unverifiable. Weak label = LLM soft report labels on 4,349 studies (our own label family; frontier
  shares our ceiling). Specialist-supervision hypothesis confounded with blend geometry, unpriced.
- M336 in flight (agent 42): annotator-disagreement vs attainable-AUC ceiling; report-vs-pixels MSK
  miss rates. Main is adjudicating the audit pack (chunks 1-2 done, 34/102 affirmed).
- M336 §1: main RETRACTED 93.9% unanimity unprompted; 69.34%/30.7% adopted.
- Pulled kernels live at research/external/pubkernels/{renta937,hyakuman937}.py

- D330 (M336): eval-label noise price is EXACT and linear — Fogliato 2003.13808 Prop B.1 (from Menon
  ICML 2015): symmetric ρ → AUC_obs=(1−2ρ)AUC*+ρ; one-sided FN ρ → (1−ρ)AUC*+ρ/2. ~2–3% eval noise ≈
  our whole 0.010 gap — but that prices the BOARD's labels, not ours. TRAIN noise: no
  information-theoretic ceiling on ranking (Menon, ρ₊+ρ₋<1); GreenRFM (2603.06467, preprint) measured
  structured report-type noise costs only 0.03–0.09 AUC vs catastrophic random flips. Report-vs-pixels:
  measured and FN-dominant exactly on our soft channels — Krakowski cartilage report sens 24–42% (grade
  I 4.8%), Fritz lateral meniscus ~30% FN, Boutany reports matched surgery 35.5%. NULL: no report-vs-
  truth miss rates for effusion/synovitis/Baker's/contusion/fracture — 5 of 12 columns unmeasured.
  MRNet read-off: κ 0.508 task still yields AUC 0.937 — κ not a tight ceiling dial.
- D331 (M337): method with known asymmetric T = Natarajan NeurIPS 2013 (binary, unbiased) / Patrini
  CVPR 2017 forward correction. Multi-label composition follows from linearity but is UNPUBLISHED;
  soft-target correction defined mechanically, unpublished. KEY WARNING: BoMD NIH-Google pneumothorax —
  naive per-label F-correction 80.8 vs BCE 87.0 (LOST); report noise is correlated across columns
  (shared silence), diagonal T mis-specifies. Menon: FN-dominant noise preserves ranking PARAMETER-FREE
  (PU learning). Patrini: known-vs-estimated T buys ~10 pts on hard multi-class but ~0 on binary.
  BoMD NIHxPDC Table 4: ELR-family/re-labelling gains +1.4–4.0 from 0.84–0.86 baselines; nothing >0.90.
- M337 §2 state: audit half-adjudicated by main (202/402, running 37.1% affirmed).
- Research files: _label-noise-ceiling-and-report-vs-pixels-error.md, _loss-correction-with-measured-noise-matrix.md

- AUDIT LANDED (M340): B = 0.6934·0.0237 + 0.3066·0.2031 = **0.0787 → REFUTED as-is** (pre-registered
  bands held). Structure is the finding: unanimous cells 1.48% (bound 2.37%, LICENSED alone),
  discordant 15.6% (bound 20.3%) — 10.5× lift from a FREE selector (five-way disagreement), first
  measurement in its selector class. Point estimate 0.0582 (CONDITIONAL) recorded but bound is the
  read. Errors cluster: compartment confusion + entity substitution (Jain-shape, structured);
  r0130 = transposition written INTO the report (source-side, prompt-proof). Main ran the scorer
  himself (my per_column json line died on tuple keys — my bug).
- D333 (M338+M340+M339-ii): repair route real for STRUCTURED errors only — Jain 2104.00793
  (labeler F1 0.55→0.73 → AUROC +0.04, concentrated on repaired confusion); Khosravi PMC12920854
  (train-side tolerant; TEST-side schema moves AUC −0.05/−0.11); Wollek PMID 38295825 (report-source
  vs pixel-source label cost ~0.08 AUC at 99.7% extraction). GreenRFM repair never retrained —
  F1 0.943→0.985 unpropagated. Transposition noise: distinguished (AlleNoise nested-flip) but never
  priced vs iid flips — NULL. Shared report-silence factor: NULL (components exist, composition
  unpublished; audit found explicit negations not silences anyway). Selective repair = published
  OPTIMUM (FCF AAAI-24: over-cleaning adds noise, retain clean cells; DivideMix mask-cell option;
  Bernhardt 4× vs random; ActiveLab). Gradient check: Spearman(gold AUC, published miss) = **−0.50,
  n=7, p=0.25** (−0.33 with Effusion); extremes agree (Synovitis/LatOA vs ACL/MedMen), middle does
  not (MedOA, Effusion counter-examples) — report weakness explains extremes only.
- Research file: research/2026-09-11_extractor-repair-structured-noise-and-selective-repair.md

## PRIOR LATEST (12:25Z) — M331/M332 answered (D325); next free letter D326; main last seen M332 (09:55Z)
Listener died silently again (~11:17Z–14:17Z local gap): M331/M332 sat unread for ~40 min. Supervisor
relaunched (pid 54220). `.status` staleness check: `python inbox/agents.py --list` names STALE lanes.
- D325 (M332): Q1 broad-macro taxonomy ranked (coverage > in-domain SSL +0.011 > resolution plateau —
  our 336→392 null is ON the published plateau > ensembling > corpus/trunk wash > AUC-loss ~0.001–0.005
  > schedule/EMA NULL). Q2: NO published +0.01–0.03 broad gain from a ≥0.93 baseline exists; delta ∝
  (1−baseline) law measured 3× on MRNet (ELNet/TransMed/Belton); CheXpert steps shrink; Mahmood power
  law (2207.01725). Q3 concordance ruler: independence = conditional error independence, UNVERIFIABLE
  from labels alone (Albert & Dodd PMID 15180668); Sorin worked via cross-MODAL components — our 5
  same-report same-LLM-family tables sit nearer Funk 2015 (precision ≤18%, PMC4301805) than Sorin
  (<0.75%); residual error is the binding constraint (Northcutt 2103.14749: ~6% flips rankings);
  the tightening instrument = stratified audit of ~400 unanimous cells vs gold, (k+3)/400 bound.
  Sampling error at corpus scale is fine (±0.012 at N=4407). PPI (2301.09633) exists, PPI-for-AUC NULL.
- CORRECTION in D325: my D324 Hanley-McNeil CI (±0.19–0.24 at 27/31) was 2× too wide; true values
  ±0.117 @ AUC 0.80, ±0.085 @ 0.90, ±0.061 @ 0.95. Delegated computation I failed to re-derive.
  n=58 still can't resolve 0.05, but the floor is half what I quoted.
- Process defect recorded: D322 cleared a probe main had already spent (colhalf, scope:excluded rows) —
  check excluded board rows before pricing a "new" instrument.
- M332 repriced D323/D324: Synovitis & Lateral OA are ≤0.005 prizes; per-column axis closed by colhalf.
  ID fixes: Matsoukas = 2203.01825 (not 2204.02825); Azizi = 2101.05224.
- Backlog fully discharged: M310/M311/M312/M315/M317/M322 all answered earlier (D311/D313–D316).
- Research file: research/2026-09-11_broad-macro-gains-high-baseline-and-concordance-rulers.md

## PRIOR LATEST (10:05Z) — M327/M329/M330 answered (D322/D323/D324)
Note: listener delivers without harness notification — on resume ALWAYS `ls inbox/researcher/read/`
for new M-letters first, don't trust the absence of a notification (missed M323–M326 for ~1h that way).
- D322 (M327 Q1/Q2/Q4): per-column probe arithmetic CLEAN — sklearn midranks give exactly 0.5 for a
  constant column, column-local, mixed case == all-constant case (measured locally); macro-averaged
  per-column AUC confirmed by organiser statement + 2 public notebooks, so the flattening risk is
  closed. Rules: NO clause forbids probing — nearest are hand-labeling ban §4.b (not implicated),
  5/day limit (ordinary slot), catch-all §8.d (discretionary, no probing precedent). Rules live at
  comp/COMPETITION_RULES.txt (Kaggle web page is an SPA shell, unfetchable). Q4: 58 is the whole of
  it; CORRECTION to main's premise — the two community gold files are NOT byte-identical (gxpixel
  carries Report+lang cols) but ARE content-identical on the 12-col label matrix (58/58 UIDs, diff 0.0).
- D323 (M329): depth curve — only two on-regime measurements exist: Ilse ABMIL (1802.04712, attention
  ≫ max at bags 10–50) and MICCAI-2025 LN CT (9→18→27 slices: saturates ~18, degrades slightly at 27).
  NULL: no class-specific-vs-shared depth curve anywhere — ours would be first. Main's mechanism is
  verbatim in CLAM (max = single-instance gradient). Lateral OA: over-determined weakness — rarer (15%
  of patterns, Stoddart PMID 33253887), harder for radiologists+ CNNs (Fritz PMID 32170334, κ 0.741 vs
  0.876), worst channel in MOAKS itself (cartilage κ=0.36), under-read from reports (Krakowski, LTC AUC
  0.578), BMLs invisible without fat-supp (physics). Geometry hypothesis (0.1–0.9 band under-samples
  lateral): plausible via Bloecker 59% vs 50% meniscal coverage + convex lateral plateau, but
  UNMEASURED anywhere. Research file: research/2026-09-11_attention-pool-depth-saturation-and-lateral-oa-reads.md
- D324 (M330 + M327-Q3): montages LOSE to few orthogonal high-res slices (SpatialMed 2603.13800,
  −20.75 pts for medical VLM; UCSF-PDGM-VQA 2605.17140); text-in-tile indices untested (MedBLINK:
  marks seen not reasoned about); grade prompts UNDER-CALL (Chetla PMC12411675, sens 0.028–0.040 for
  mild); FP:FN levers = explanation prompts (+recall) / weak-learner prompts (−FP 78.5%), Guo MELBA
  2025:004; per-finding vs joint prompt = NULL; self-consistency measured (Liu PMC13060412).
  Validation floor: Hanley-McNeil at 27/31 → 95% CI ±0.19–0.24 AUC on full 58, ±0.27+ on split —
  split-the-gold can only certify coarse claims; named protocols = CheXpert split, partial
  identification (2312.04601), confident learning (1911.00068), concordance-adjudicated ruler (Sorin).
  M327-Q3: NULL for the exact design; family = Whitehill AUC/log-loss oracles (1506.01339,
  1707.01825), validity pricing = Dwork Science 2015 + Blum-Hardt Ladder. Research file:
  research/2026-09-11_vlm-montage-prompting-and-small-gold-validation.md

## PRIOR LATEST (09:10Z) — backlog cleared THROUGH M326
- D319 (M323-Q1, measured from local artefacts): fastMRI has NO laterality anywhere — knee.csv has 8
  columns none lateral; ISMRMRD headers (20 files, full tag enumeration) zero laterality tokens; and
  the DICOM batch has NO overlap with the 974 annotated k-space studies so it can't rescue it.
  Consequence flagged: main's coronal box-centre compartment assignment is a per-study coin flip
  without laterality (x-axis mirrors with knee side); only 208 both-cluster + 311 no-box meniscus
  studies are sound; only rescue = fibula-side detector from pixels. CORRECTION absorbed: fastMRI knee
  k-space is CORONAL PD/PD-FS, not sagittal (D318/M313 had it wrong; conclusions unchanged).
  M326-Q3: no more organiser gold than 58, closed.
- D320 (M323-Q2 + M326-Q1/Q2): threshold-merge estimator EXISTS — Raykar "Learning From Crowds" JMLR
  2010 (sens/spec per source; main's asymmetric-confidence encoding is a hand-set corner of its
  posterior); explicit per-source-cut model = Raykar NeurIPS-2011-workshop ordinal annotator model.
  GLAD wrong shape. Agreement-only ranking at 93.9% unanimity + shared generator = BELOW
  identifiability floor (Hui-Walter K≥3 independent; LLM cluster collapses to ~2 views); honest tool =
  partial identification (Maia Polo 2312.04601) whose bound width IS the answer → gold or the board.
  Eval-on-own-generator confound: no canonical name; Lampert TIP-2016 measured inflation; remedy =
  swap ruler MODALITY (gold-58 is that). Research file:
  research/2026-09-11_multi-source-label-merging-and-agreement-estimators.md.
- D321 (M324 + M325): view-as-samples supported exactly in our regime (Gori 2505.01109, 710
  experiments: instance-MIL matches/beats embedding-MIL once the encoder is foundation-class; MRNet
  itself is per-plane view-as-samples). Channels-vs-samples at fixed labels = empty cell, ours unrun.
  Consistency across plane+sequence views = NULL everywhere. TTA-adaptation on medical classification
  = near-NULL; BN line doesn't transfer to ViT (no BN layers); small-batch bias measured
  (2312.09486: insufficient class diversity per batch). Input×aggregator joint ablation = NULL, main's
  2x2 is the FIRST measured demonstration — flagged as publishable-shape. Dilution literature says
  small bags don't dilute; the max-pool suppression is argmax-suppression not classic dilution.
  Closest published model to this project found: Qiu et al. Nat Commun 2024 (PMC11368947), co-plane
  attention, 12 knee-abnormality labels. ID fixes: SAR=2302.12400; TransMIL no arXiv ID. Research
  file: research/2026-09-11_views-consistency-tta-and-mil-aggregation.md.
Overnight main's new measurements: 2x2 input×head (M325: extended scheme +0.0010 labelwise / −0.0059
max); label-table ranking on gold57 (M326: ours v2 is WORST at 0.2047 cell disagreement, FP:FN 3.4:1,
Contusion 36-vs-18 over-call); M323: owner ruled all corpora downloaded, blocker is labels, don't drop
data; per-cell weight machinery already in trainer (grades.py:36, trainer.py:634, __conf=0 idiom).

## PRIOR LATEST (07:15Z) — ruler regime REVERSED; D311 void; D313–D316 delivered
**LEDGER75 IS RETRACTED (2026-09-10, experiments/README.md:13641).** Its only out-of-sample test
(l75seat_v1, ref 56151868): midpoint predicted ~0.936, ledger75 0.949, board read 0.939. **Midpoint is
the default cut again; never quote a ledger75-fitted constant.** My D311 audit (calib_span reversal,
ema flip, corpus negatives evaporating) is VOID as a licence to re-open — main kept only the meta-point
(recut biased per family = signal about the default cut; Effusion mechanism the carrier, ledger75.py:8-14).
Main explicitly refused the offered pool recut.
Board record facts (M310): ruler monotone ≥Δ0.005 (0/71 inversions), blind below Δ0.002; ensemble
constant +0.0614 (n=23), solo +0.0650 (n=5); 0.955 needs ensemble 0.8936 offline ⇒ +0.0110 owed.
resnetlong_v1 (board 0.907): board resolved a −0.0067 offline delta as −0.008 — board sees deltas a
third of our noise bar.
Delivered this morning:
- D313 (M315): paired bootstrap on own holdouts (research/scripts/paired_boot.py; script sent as .txt).
  Paired SE 0.0024–0.0035 at n=909 → **paired floor ~0.005 (2sd), NOT 0.002**; pairing shrinks
  unpaired (2sd ~0.014) by 2.2–3.0×; standing 0.0062 floor was already ~paired-honest. Paired test
  licenses weights-level claims only, never recipe-level closures (seeds banned).
- D314 (M322): 0.000 board on the 79%→100% one-seat swap CANNOT distinguish β=0 from β=0.25 (even
  β=0.25 predicts ~0.001–0.002 board through 1/7 dilution — instrument null, not saturation proof).
  Literature: Zhang 2210.02189 (β 0.21–0.25 @N=100–2k CXR, steepest medical); Raghu 1902.07208 /
  Matsoukas 2203.01825 / Hernández 2102.01293 (pretraining flattens curve early); Figueroa c-term /
  Rolnick 1705.10694 (label noise = floor not slope). Verdict: β≈0 credible but unproven; only a
  bigger swing (×2 data) could resolve. Same-distribution data is weak on ANY published β.
- D315 (M319/M317bc/M320/M321): canonical born-again protocol is CLEAN on teacher-sees-eval (their
  dirty axis = best-epoch-on-test selection, admitted); contamination named+measured once, LLM side:
  Data Laundering arXiv:2412.15255 (BERT 25.76→74.75% on GPQA via distilled leak); "Students Parrot
  Their Teachers" NeurIPS23 = channel proof. Teacher-blind eval: NULL, ours is first. Iterated
  distillation saturates by gen-3, can invert (Mobahi = **arXiv:2002.05715**, NOT 2010.02162; Zhang &
  Sabuncu: 1 optimal-temperature gen beats 10-gen chain). Shared teacher correlates students, shrinks
  ensemble gain (Snapshot Distillation, in print). Main's calib×hold interaction SIGN PATTERN
  (harmful alone, harmless together) is UNPUBLISHED — published interactions are opposite sign
  (Müller 1906.02629 LS×KD harmful together; Lukasik 2003.02819: sign FLIPS under label noise).
- D316 (M316): literature's practice = report CV number for refit, unchecked (null as validated
  estimator); nested CV prices pipeline not instance; .632+ bootstrap (Efron-Tibshirani 1997) is the
  designed-for-this estimator, unused at deep-net scale; CV bias for refit is pessimistic in
  expectation and its size (~+0.003) is below any instrument's resolution (D313) — safe floor.
  Twin-ranks-refit-sibling: NULL, needs 2-3 more refit shipments to exist. Scheme: twins price the
  recipe, board prices the instance.
Overnight main's measurements: distill honest term +0.0056, contamination surcharge +0.0052,
pass-through ~6% (M319); calib×hold interaction +0.0077/+0.0092 at two corners, net zero at 12 slices
(M320/M321); teacher_gen2 +0.0121 five-for-five RETRACTED as residual-mimicry artefact (M318).
IN FLIGHT: nothing. Backlog CLEARED: M313–M322 all answered by D313–D318 (delivered 06:20–07:50Z).
- D317 (M314): licences — CONFIRMED NEGATIVE, no labelled knee corpus permits shipping weights.
  Only shippable MSK-MRI = TCIA Soft-tissue-Sarcoma (CC BY 3.0, 51 subj, no labels) + TCGA-SARC;
  M4Raw brain CC BY 4.0. SKM-TEA bans "derivative works" verbatim; OAI DUA research-only; K2S none;
  RIDER Knee GONE from TCIA. Survey at research/2026-09-11_shippable-corpus-licence-survey.md.
  Site axis = permanently offline-only for shipped members.
- D318 (M313-Q2): literature picks continued-SSL on fastMRI images over fastMRI+ supervised
  co-training. Cross-institution tax measured −0.07..−0.14 (Bien PMC6258509, Germann PubMed
  32168039, Lu 2409.04368 MRI worst modality −0.097). PU-learning caveat fits fastMRI+ structure.
  Organ-specific continuation distinction REAL (Azizi 2101.05224, RETFound, MoCo-CXR vs
  RadImageNet contested 2302.08272). Co-train-vs-pretrain order: NULL. Survey at
  research/2026-09-11_fastmri-transfer-mechanisms.md. Both research/ files UNCOMMITTED (main's call).

## Identity & rules (confirmed via HANDOVER_researcher_to_kimi_20260910.md, a prior instance)
Researcher lane, kneemri (RSNA knee Kaggle). Main = only correspondent, owns card/ledger/bd/git/slot.
Never train, never bd write, never git commit, repo writes are main's. Letters: `.part` then `mv`
into inbox/main/ as `<utc>-researcher-DNNN-<slug>.md`. **Scripts to main go as `.txt`, NEVER `.py`**
(a .py in inbox/main/ broke main's ruff format gate, M302 §5). Numbers carry STATISTIC@POPULATION@CUT.
CPU only: `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python` (-1 only; "" does NOT hide GPU). Bare python =
miniconda, not project. `grep -P` silent-fails — use -E. Never open adjudication_key.csv; do not grep
pack report text against raw/train.csv (M300 blind protocol). NO recommendations (M286): mechanism +
numbers only. Never arm another lane's box.
Numbering: mine D, main M. **Next free = D322** (main last seen M326; see LATEST section at top).

## Listener — UPGRADED 15:00Z: self-rearming supervisor (fixes the no-notification hole)
`scratchpad/researcher_mail_supervisor.sh` = bash loop: run listen_for_mail.py --box researcher
86000, on exit sleep 10, rerun. Exactly one live listener at any instant. Started via
Start-Process bash.exe with the script path. Do NOT also Start-Process the listener directly
(that doubles up). Log appended at logs/researcher_mail.log.
MAIN'S BOX: his listener is DEAD since ~13:00Z (pid 61080 gone, no heartbeat). D303+D304+D305
sit unread in inbox/main/. Main must re-arm his own box (lane rule: never arm another's) —
same supervisor pattern recommended; relay went via owner. Verified 15:02Z: my supervisor
running, one listener (pidfile 13244), heartbeat fresh.

## Ruler constants (from handover §3)
Midpoint default (scores.py:21 MIDPOINT=0.5); recut 0.75 (ledger75.py:44). Cut alone moves honest12 by
+0.026. Paired noise floor 0.0062 = 909-study 3-seed HOLDOUT constant — NOT for 58-study gold reads.
Hanley-McNeil 0.0053; BOARD_RESOLVABLE 0.0015. board−offline offset +0.0608 sd 0.0035 (D283).
Owner bar: ~1.5% absolute is a result, ~0.005 is noise; owner targets 0.945+/0.950 gold.
Pairs are NOT cells (D300/D301): always state which denominator a per-column number used.
Gold work scores vs train.csv radiologist rows only, never npz truth blocks.
Pre-reg rules lane enforces: no gold-58 per-column clause (macro only on gold; per-column on
foreign-896/L75mid-896 with n+); per-column claim vs per-column floor; symmetric-stat bars on
|stat−chance|; baseline must not be a deterministic function of target.

## Label machinery (handover §4)
core/data/labels/targets.py MEASURED_AGREEMENT reader levels: (0.08,0.041)(0.28,0.136)(0.68,0.378)
(0.82,0.688)(0.94,0.958). report_labels_v3.csv = CALIBRATED (0.0408/0.1356/0.3776/0.6879/0.9583);
report_labels_v2.csv = RAW levels. FINDING_OFFSETS sd 0.543 logits; UNCERTAIN_LEVEL 0.28;
ASSERTS_PRESENCE 0.5; ANNOTATED_WEIGHT 3.0; CONFIDENCE_FLOOR 0.25. Main's eval: kneemri/evaluation/
gold_estimators.py + oof_gold.py (MacroGold = board row flat 1/12; RankPooledGold = internal only).
Band-arm live on card since 12:39Z: experiments/arms/oof_band_folds.sh, 5 folds, 224/8, one token moved
(--labels report_labels_v6band50.csv), band re-spread symmetrically about rung; pre-reg on paired
gold@58 rankpooled: >=+0.020 proceed share-dial 0.75; <=-0.020 middle-rung route closes; between = null.

## Live thread (15:27Z)
D303 delivered 13:35Z; D304 delivered 14:26Z (queue check); D305 delivered 15:02Z (operational:
main's listener dead, D303/D304 unread, supervisor fix offered). Band arm landed NULL at gold58
(-0.0066, 1.06 paired sd) per M304 — ledger row written by main; middle-rung route not advanced.
mae_unitfix on card until ~19:30Z, reading is main's.
M304 (arrived, delivered 15:26Z, read 15:27Z): cross-tab the six supervision-side flags over
metrics237 (calibrated_targets, hold_annotations, eval_labels=v2, ema, distill_weight>0,
teacher_targets!=None): (1) 6x6 overlap matrix; (2) conditional deltas holding the largest
overlap flag fixed; (3) date spans True vs False groups; (4) which flags the champion's members
carry (comp/champion.json). "If all six set on champion, rest is academic, say so and stop."
Stand down on VisualCheXbert per M304. Do NOT watch band arm.
DONE 16:06Z: agent-21 mapping -> research/2026-09-10_unseen-candidates-mapped-onto-our-
measurements.md (11-row cross-ref, interpretation-file numbers) + D308 to main CORRECTING
D307 claim 1: joint multi-corpus pooling = merged3, built and null (-0.0024, ACL -0.0013 on
1,248 expert reads; board dose bends down; fastMRI variant licence-blocked). Supported: frozen
in-domain FM features (ortholarge +0.0202 four ways; open = recipe, ortho_fresh_lowlr unrun);
prioritised relabel RULER-side (blocker = gold-58 size: 15/876 + 5/502 flagged cells inside
ruler; "more gold, not a cleverer statistic"); curriculum-as-schedule (certainty_ramp +0.0077;
disagreed_ramp empty cell decides). Cheapest open: spectral_mae.py built, never run, 3-4h card.
Closed by repo: auto label cleaning (1.66% fixable vs +0.0053 ceiling), report-supervision
family (corpus 60% non-English; unread_dense +0.0020), Bien transfer. Next free letter D309.
IN FLIGHT: none. Next free letter D311.
M305 received 16:08Z (supervisor delivered), answered 16:20Z -> D310: joint pool = 5-of-12
labels at best (MRNet 1 direct + 1 coarse; Rijeka ~1+1; fastMRI+ ~5 coarse; OAI OA-only AND
licence-blocked); our concavity predicts +0.004-0.006 for ~3.4k foreign studies (0.63 doubling,
below 0.0062 floor, 3x under owner bar); site-axis sign NEGATIVE — finding 1 dead, close it;
Bien +0.087 = small-corpus-baseline in-domain-supervision price (n=1130 train, 183 val, ACL
only), not a pool increment on 4406. Main standing: card on mae_unitfix till ~20:30Z, VRAM 97%.
Main has read D303-D307; D308/D309/D310 unread in his box as of 16:20Z.
Mail health: BOTH lanes alive — main re-armed 16:08Z (after D305), my supervisor holding.
DONE 16:14Z: agent-22 round 2 + my own round 3 -> research/2026-09-10_round2-practitioner-
and-access-terrain.md + research/2026-09-10_round3-local-unrun-and-evaluation-economics.md +
D309 to main. Round-2 top: arXiv 2606.11606 frozen-probe small-signal (global pooling AT
CHANCE 0.500-0.524, patch-local recovers ~1.0, stratum gap +0.243) — lands on focal deficit +
decode toll; lumbar winners = hand-fixed labels + SHAPE alignment + coordinate pretraining
(geometry unpriced; our harmonisation is intensity-only); MIC-DKFZ seg-as-clf 7th/1147 vs
Demir negative; fastMRI licence conflict (paper no-commercial vs AWS MIT) — DUA read needed
before pooled weights; MRNet Kaggle mirror licence unknown; Kaggle knows 58/4407 gold publicly;
LLM label dataset public. Round-3 top: golden-set plateau 1-2k labels (EACL26 2512.08121;
gold-58 deep in steep part); Roelofs NeurIPS19 100+ Kaggle comps NO widespread test-set
overfitting + Mania model-similarity mitigation (board-reuse worry weakened); Gallas MRMC;
Ju 2022 disagreement-noise frame; physics-MRI-aug-for-classification SILENT; local unrun arms
(~76 name-matched, rough): fov_perstudy_read (readout, no card), diversity_curve (offline),
ortho_fresh_lowlr (probe control), multiscale (two-extent slot attn), sided_tissue (named
suspect), slice_count_whole_stack, latent_rate_ladder; report_supervision REFUTED in header.
DONE 15:47Z: agent-18 -> D306 delivered: M304 cross-tab verdict = ONE recipe generation, not
four levers. distill<->teacher share 13/13 (same arm, split direction empty); calibrated<->hold
invert sign inside partner-True stratum (-0.0024/-0.0031), +0.022 both ways partner-False;
eval +0.004-0.005 under hold (noise); ema +0.0050 n=6 / +0.0164 n=3. Era: no one-week confound
(True spans 13-24d, all start 08-24/25 = late-era generation bias). Champion stackseat_v1: all
six seats carry calibrated_targets+ema; stack_increments_ff alone carries all six; all six are
fold=-1 all-data NaN-auc fits (not in metrics237). Licensed: nothing without a paired flip run.
DONE: agent-19 -> four research/ files written+spot-verified (external-cohort one checked):
2026-09-10_cross-decode-toll-shrinkers-ranked-by-measured-effect.md / _external-cohort-ranking-
does-not-reliably-survive.md / _single-model-ceiling-interventions-medical-grading-ranked.md /
_crop-extent-and-physical-normalisation-literature.md. NOT committed (repo commits are main's).

## Board (STALE — 2026-09-09T19:53Z, do not quote without re-read; board_read is main's lane)
ours 0.944, leader 0.954, gap 0.010, rank 58/3421. Champion stackseat_v1.

## Traps register
- Listener: see Listener section (Start-Process only).
- temp cleaner DELETES /tmp/cluster_analysis data products (scripts survive, jsonl/json don't).
- Lexical sort misleads on letter numbers (D30 > D299) — use agents.py/registry not ls ordering.
- Package dir kneemri/evaluation/ not kneemri/eval. comp/colweights_v*.csv = OUR fusion weights.
- M302 filename defect: main's atomic write left .part inside final name — cosmetic, flagged D302 §6.
