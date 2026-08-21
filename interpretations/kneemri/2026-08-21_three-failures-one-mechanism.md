# Three failures, one mechanism, and the two axes nothing has touched

Three deliberate attempts to raise the score — a better label reader, a severity-aware loss, a better
backbone — each landed inside the noise floor or under it. This note gives the mechanism for each one
separately, then argues they share a single cause, then prices what is left.

The floor throughout is the measured holdout noise floor **0.0062**; the project bar is **1.5% absolute**;
the shipped champion is `basedense_v1`, **0.8690** holdout / **0.931** board.

---

## 1. Better backbone — refuted, and the refutation is a generalisation gap at matched fit

Three encoder swaps against `base_dense` (0.8512) / `band_dense` (0.8578):

| arm | encoder | solo | Δ solo | seat |
|---|---|---|---|---|
| `v3base_dense` | DINOv3 base, frozen+dense | 0.8455 | −0.0057 (inside the floor) | −0.0007 sixth, −0.0005 swap |
| `cnx_dense` | ConvNeXt trunk | 0.7927 | **−0.0585** | −0.0035 sixth, −0.0042 swap (≈ −0.0127 board) |
| DINOv2-large | wider ViT | — | +0.0021 | at 3.7× wall-clock |

**The mechanism is not underfitting.** `cnx_dense` was read at MATCHED TRAIN LOSS against `base_dense`:
0.8371 at loss 0.8456 versus 0.7927 at loss 0.8332. A **0.044 holdout gap at equal fit, with the losing
arm fitting slightly HARDER.** That removes "it needed more steps" and "it needed a different rate" as
explanations and leaves one: from the same fit on 3,498 training studies, that encoder generalises worse.
Eleven of twelve labels are negative and ten clear their paired half-widths, uniformly — focal −0.0585
equals regional −0.0585 to four decimals, so it is a worse model everywhere, not a redistribution.

Capacity is therefore not the short resource. Two corollaries fell out and both narrow earlier claims:

- **"DINOv3 is weak on this task" was an ADAPTATION finding, not an encoder finding.** `v3base_dense`
  0.8455 against its LoRA sibling `dinov3_base_lora` 0.8268 is **+0.0187** — LoRA-over-frozen costs about
  0.019 here. Every sentence in this repo saying DINOv3 is weak needs the scope "in the LoRA regime"
  written on it.
- **`base_dense`'s seat may have been bought by 768 WIDTH, not by its pretraining corpus** — it is the
  only wide member in a pool of 384-wide ones.

And the seat readings say the sharper thing: a second ViT of a different family does not buy a second
diversity seat. The next member has to be a different INDUCTIVE BIAS — which is what `cnx_dense` was, and
it lost by 0.058.

## 2. Loss by severity — mostly already shipped, and the new part is worth ~0.001–0.002

The design that motivated this axis — split the scalar target into an ordinal GRADE and a separate
CERTAINTY — **is already in the shipped pipeline**. `report_labels_v2.csv`'s unread `__verdict` column is
a perfect bijection with `value`:

    NO   sev 0  value 0.08  conf 0.85  22,298 cells
    UNK  sev 0  value 0.28  conf 0.05  14,445
    YES  sev 1  value 0.68  conf 0.95   6,461
    YES  sev 2  value 0.82  conf 0.95   6,427
    YES  sev 3  value 0.94  conf 0.95   3,241

Certainty is `__conf`, flat at 0.95 across all three YES levels, and `__conf` is **already consumed as
the loss weight**. The three YES levels are severity, not confidence. So the motivating collision — a
confident low grade against an uncertain high one — does not occur in this data.

The genuinely new piece, `--calibrated-targets`, has run twice: **+0.0038** at the old operating point and
**−0.0020** as `calib_span` on the recipe we actually ship, with every pre-registered clause inverting.
Stale operating point.

**But the axis is not worthless, and the second ruler is why.** Ordinal concordance — a ruler that costs
no card time — orders the seat correctly where the binary one inverted it:

| member in the seat | binary alone | binary in seat | ordinal labels | ordinal pairs | board |
|---|---|---|---|---|---|
| `lr1e-4_ws` | — | 0.8648 | 0.735 | 0.766 | 0.928 |
| `band_span` | **0.8565** (best alone) | 0.8661 | 0.736 | 0.767 | 0.927 |
| `depth_regions_narrow` | 0.8554 | 0.8653 | 0.743 | 0.769 | 0.928 |
| `calib_span` | 0.8545 (3rd alone) | **0.8670** | **0.759** | **0.791** | **0.929** |

`calib_span` carries real ordinal signal and does take the seat. The confound was tested:
`calibrated_targets`, the same flag on a different member, scores 0.739/0.776 mid-pack, so
by-construction does not explain it. It remains a lead (1 in 6 under random ordering), not a finding.

`severity_flat` — collapsing the three YES levels to one — cost **−0.0044**, so the gradation is live
signal that is already being used. `aprior_dense` (−0.0021) closes the other half: the head had ALREADY
recovered the plane-by-finding mapping from 4,407 study-level label vectors carrying no localisation, and
its loss tracked the control's to the fourth decimal.

**Verdict: the severity axis is worth roughly 0.001–0.002 — real, mechanism-sound, under the floor.** It
stays in by owner's call, correctly, as a free directional keep. It is not a lever.

## 3. LLM review of the labels — priced at +0.006, which IS the noise floor

An Opus subagent re-read the first 100 reports against 1,200 cells carrying the incumbent's **verbatim
rulebook** (recovered from `api_labeler.py`, which ships inside pilkwang's CC0 dataset). Result: **1,115
agreed (92.9%), 48 contested (4.0%), 37 uncertain (3.1%)**.

**The 4% is not the finding — the zero is.** MCL is the worst-scoring label and has **0 contested cells**.
ACL also 0. The contested cells concentrate in Fracture (19) and Contusion (9).

The error is one-way and it is a WEIGHT defect, not a value defect: NO→UNK 23, UNK→NO 13, UNK→YES 3,
NO→YES 3, YES→UNK 2, YES→NO 1. The reader fabricates DENIALS, never positives. Such a cell trains at
value 0.08 / weight **0.85** where the rulebook puts it at 0.28 / weight **0.05** — near-off. Two
rulebook ambiguities produce 36 of the 48.

**The arithmetic that decides it.** The metric is an unweighted mean of twelve AUCs. 58% of contested
cells are Fracture+Contusion, which are 17% of cells. A generous fix — Fracture 0.80→0.85, Contusion
+0.02 — gives **(0.05 + 0.02) / 12 = +0.006**. That is the noise floor, and a twentieth of the bar. And
the errors sit **3:1 on the label-bottlenecked half** of knee-dtk's split (6.0% versus 2.0%) — i.e. on
findings that already score well.

Language is refuted as the cause from two directions: non-English 4.9% versus English 2.7% in the pilot,
localised to one sentence shape; and `reader_agreement` puts English at 66.7% against non-English 64.2%,
**+0.025 [−0.076, +0.126]**.

**Two rulers, and they measure different things.** The pilot measures conformance to the RULEBOOK (92.9%)
— a re-read can fix that. `reader_agreement` measures conformance to the IMAGE across 58
radiologist-annotated studies: **~66% on positive calls, 140 wrong of 684, false positives 108 against 32
misses.** A re-read cannot fix that, and — the decisive point — the board is scored against
report-derived labels too, so the image-disagreement is the same noise on both sides and is not
recoverable by us at all.

One genuinely new observation the pilot did not have: **graded positive calls agree 63.2% versus plain
77.8%, −0.146 [−0.277, +0.005]** — a threshold defect on mild findings. Directional, filed, not planned
around.

---

## What the three share

All three added **information or capacity to a pipeline that is limited by neither.** The evidence that it
is limited by neither is not one arm, it is a sweep of them, each measured against a matched control:

| what was added | delta |
|---|---|
| more slices per study (20 vs 12) | −0.0001 |
| nine slots, 83,297 previously unseen slices | +0.0002 |
| every pooling rule, four operators on one recipe | spans 0.0019 total |
| slice ORDER (`depth_true`, order-aware aggregator) | −0.0015 |
| bigger / different / newer encoder | ≤ 0 at every try |
| better labels (ceiling, generously priced) | +0.006 |
| longer training + a regulariser that fired (`aug_long`) | −0.0076 |
| an EMA over weights (`weight_average`) | +0.0003 |
| head initialised with the localisation prior | −0.0021 |

Adding inputs does nothing. Adding capacity does nothing. Adding label quality is capped at the floor.
Adding training time does nothing **because holdout selection already absorbs the decline** — every arm
ships its best epoch, so a regulariser can only pay by raising the PEAK, and `aug_long` lowered it.

What binds instead is **error correlation among members over 3,498 training studies**:

- A per-label ORACLE handed the answers does not beat flat averaging over these members. That bounds
  *every* per-label weighting scheme writable over them; the measured attempt cost −0.004 to −0.005.
- Sixteen of seventeen mounted members are DINOv2. It is a monoculture.
- `deep24` is the **worst** member in the pool (0.8213) and greedy forward selection takes it anyway —
  because its errors are shaped differently, not because it is good.

Diversity buys seats and quality does not. That is a signature of a data-limited regime, not a
capacity-limited one. External corroboration points the same way: MRNet reaches ~0.97 on ACL with AlexNet
at 256 px — a **smaller** model at **lower** resolution than our 336 — against our 0.874
([Bien et al., PLOS Medicine 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/)).

---

## What is left: exactly two axes, and both are untouched in 107 runs

### A. External data

Competition rule §2.6a permits it explicitly: External Data must be *"publicly available and equally
accessible to use by all Participants of the Competition for purposes of the competition at no cost."*
**Nothing in 107 recorded runs has ever trained on external data.** The repo knows MRNet only as
slice-aggregation *literature*, never as a data source.

Candidates, with the licence read rather than assumed:

| dataset | size / labels | access | verdict against §2.6a |
|---|---|---|---|
| [MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/) (Stanford) | 1,370 knee exams, sag PD / cor T1+T2FS / ax PD-FS; labels abnormal / ACL tear / meniscal tear, extracted from clinical reports — the same derivation as ours | free, instant self-registration, Research Use Agreement | **best candidate.** Free and open to any participant, so "equally accessible at no cost" holds. **BUT** the RUA is non-commercial and forbids redistribution and link-sharing — and this competition pays $77,000 with a winner licence to the host. That conflict is a decision, not a measurement. |
| [fastMRI / fastMRI+](https://fastmri.med.nyu.edu/) | 22 knee pathology categories, 16,154 expert boxes + study-level labels | application form, institutional affiliation, approval | **fails §2.6a** — approval-gated, so not equally accessible. |
| OAI | KL osteoarthritis grades, maps to 3 of our 12 | signed institutional Data Use Certification | **fails §2.6a**, same reason, and OA-only. |

So the axis reduces to **one dataset with one licence question the owner must answer.**

The mechanism if it clears: not "+1,370 studies of extra label supervision" — only 2–3 of our 12 labels
map, and a generous per-label read of that gives about +0.006, under the bar. The mechanism that could
plausibly deliver 1.5% is **supervised pretraining of the encoder on knee pathology**, which attacks the
0.044-at-matched-loss generalisation gap directly. Every encoder we have tried was pretrained on natural
images (DINOv2/v3, ImageNet ConvNeXt) and the failure mode is *generalising badly from a good fit* —
exactly what a domain-matched initialisation is for. This is also the same goal the SSL MAE was chasing
and failed at three times, with the key difference that 1,370 labelled knee exams are far denser
supervision than a masked-autoencoder objective on unlabelled volumes.

### B. A true 3D encoder

`depth_true` closed slice order at the FEATURE level and stated its own exclusion: *a 3D encoder that
sees the volume before any per-slice feature exists is a different mechanism and untouched.* All 107 runs
are per-slice-features-then-pool. The aggregation stage is now saturated from three directions at once
(more slices −0.0001, more slots +0.0002, every combining rule within 0.0019), and both `attend_true` and
`depth_true` conclude the limit is upstream, in **what one slice's feature vector contains**. A trunk that
never produces a per-slice feature is the only thing that changes that.

Risk, stated up front: this is also a different inductive bias, and the last different inductive bias
(`cnx_dense`) lost by 0.058 at matched fit — on the same 3,498 studies. A 3D trunk has *more* parameters
per unit of data, so the same generalisation gap should be expected to be worse, not better, **unless it
is pretrained** — which points straight back at A.

---

## The plan, priced

The standing rule is wall-clock ÷ expected gain, said out loud before running.

**P1 — MRNet supervised pretraining (blocked on one owner decision).**
Cost: ~2 h download + preprocess into the existing cache format, then one 45-min arm at the shipped
recipe with the encoder initialised from the MRNet-pretrained weights instead of DINOv2. ≈ 3 h.
Plausible gain: 1.5–3% if the domain-gap diagnosis is right; 0 if the generalisation gap is about data
QUANTITY at fine-tune time rather than about the initialisation. Odds of clearing the bar: maybe 1 in 3 —
which is the best ratio on the board, because everything else on the board is priced at ≤ 0.006 with
higher confidence.
**Blocker: the RUA is non-commercial and this competition has a $77k purse and a winner licence. Owner
call, and it is a legal question rather than a technical one.** Everything else in P1 can be prepared
while that is decided — the loader, the cache writer and the pretraining arm are ours to write and cost
nothing to have ready.

**P2 — 3D trunk, cheap version first.**
Do not build a full video transformer. Inflate the existing 2D trunk's stem to 3D over a short depth
window and keep everything else identical — a single-variable arm against `band_dense` 0.8578. Cost one
45-min arm. Plausible gain: this is the instrument that tells us whether "the limit is in one slice's
feature vector" survives contact with a trunk that sees three. A null here retires the whole 3D axis for
about an hour; a positive licenses the expensive version.

**P3 — finish the queue already on the card.** `grade_ladder` (running, epoch 1 at 0.7853) then
`band_dense_llrd`. Neither is expected to clear the bar; both are instruments on axes already opened.

**Not doing, and why, so it does not get re-proposed:** report text as auxiliary supervision — already
run as `report_supervision`, −0.0053 at weight 0.3 and −0.0063 at weight 1.0, **monotone in the weight**,
which is the strongest form of kill. Per-label ensemble weighting — bounded above by an oracle that does
not beat the mean. More seeds or folds — banned by the owner, and correctly: it is the same thing tried
slightly differently. Longer schedules — holdout selection absorbs the decline.

---

## What would refute this diagnosis

Stated in advance so it is owed rather than argued afterwards:

1. If the MRNet-pretrained arm clears +0.015, the diagnosis "data-limited, not capacity-limited" is
   confirmed and the whole encoder chapter has to be re-read as "wrong pretraining corpus" rather than
   "capacity is not short".
2. If it lands flat AND the 3D stem lands flat, then the binding constraint is neither the encoder's
   initialisation nor its receptive field, and the only remaining reading is the label ceiling — which
   `reader_agreement` says we cannot cross, because the board's labels come from the same reports.
3. If any future board entry gains under 0.0015 offline and still moves a board tick, `BOARD_RESOLVABLE`
   is wrong and the ensemble ruler is not saturated after all.

The honest summary of the three failures is not that the work was wrong. Each of the three was a
well-argued idea, each was verified to have actually fired before its verdict was written, and each
returned a number. What they collectively bought is the knowledge that this pipeline is saturated on
every axis reachable from inside its own data — which is precisely the finding that makes external data
the next thing to run rather than the desperate thing to try.
