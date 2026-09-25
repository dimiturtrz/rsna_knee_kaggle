# Where we sit against the published per-label frontier — 2026-08-23

External synthesis. The numbers on the right are other people's; the numbers on the left are ours. This
file exists because eleven internal axes have now closed at or under the noise floor, and an internal
ruler cannot say whether a label is bounded by our model or by the problem. A published AUC can.

## The ruler question, first, because the comparison is worthless without it

Published knee-MRI AUCs are scored against **radiologist ground truth**. Our headline holdout AUCs are
scored against **report-derived labels**, which is a different and easier target — `2026-08-20_label_contest_pilot.md`
measured Synovitis at 0.873 on the report ruler and **0.7378** on the annotation ruler, a fall of 0.135.

So the only comparable numbers we own are the **annotation-ruler** AUCs over the 58 image-read studies
(696 cells), and every comparison below uses those. Reading our report-ruler numbers against published
work would have flattered us by roughly a tenth of an AUC on the silent labels.

**That ruler is small.** 58 studies puts the per-label SE near 0.04–0.06. No single row here resolves
better than about a tenth, and the table is used below only where the gap is multiples of that.

## The table

| label | ours (annotation ruler) | published best | source | gap |
|---|---|---|---|---|
| **Lateral Meniscus** | **0.6497** | **0.850** | OrthoDiffusion (10,940 scans, 8 centres) | **−0.20** |
| **ACL** | **0.8640** | **0.970** | OrthoDiffusion; MRNet 0.965; ELNet 0.960 | **−0.10** |
| MCL | 0.8798 | 0.919 / 0.782 | OrthoDiffusion / Nature Comms 2024 | −0.04 |
| Medial Meniscus | 0.8498 | 0.872 | OrthoDiffusion | −0.02 |
| Synovitis | 0.7378 | 0.76 DL, **0.77 senior radiologist** | Academic Radiology 2023 | −0.02 |
| Contusion | 0.8421 | 0.821 | Nature Comms 2024 | +0.02 |
| Medial OA | 0.9504 | 0.93 (cartilage) | Astuto, Radiology:AI 2021 | +0.02 |
| Effusion | 0.9453 | 0.864 / 0.787 | OrthoDiffusion / Nature Comms | +0.08 |
| Lateral OA | 0.7099 | — | **no compartment-specific OA AUC found** | unknown |
| PF OA | 0.8005 | — | **no compartment-specific OA AUC found** | unknown |
| Baker's | 0.9547 | — | **no Baker's/popliteal cyst AUC found** | unknown |
| Fracture | 0.8861 | — | **no knee-fracture-on-MRI AUC found** | unknown |

Sources, all verified 2026-08-23:

- **OrthoDiffusion**, arXiv:2602.20752 — 10,940 test scans, 8 centres, macro 0.9079. The closest published
  label set to this competition's, though it carries no OA / synovitis / Baker's / contusion / fracture.
- **MRNet**, Bien & Rajpurkar, PLOS Med 2018 — 1,370 Stanford exams. ACL 0.965, meniscal 0.847.
- **ELNet**, arXiv:2005.02706 — ACL 0.960, meniscus 0.904.
- **Nature Communications 2024**, co-plane attention, twelve knee abnormalities, 1,748 subjects / 5 centres —
  average 0.812. The only published paper attacking a twelve-label knee panel.
- **Astuto**, Radiology:AI 2021, 1,435 studies — cartilage 0.93±0.01, meniscal horns 0.93±0.02.
- **Synovitis**, Academic Radiology 2023 — DL 0.76, senior radiologist 0.77, junior 0.70.

Rejected: an MCL paper claiming AUC 1.00 (implausible, paywalled split). Not retrieved: MRPyrNet, and the
*European Radiology* 2025 23-condition tables (paywalled).

## What the table says

**Two labels carry nearly all of the field-relative deficit: Lateral Meniscus (−0.20) and ACL (−0.10).**
Both sit in knee-dtk's **model-bottlenecked** half — the partition built on 2026-08-14 from the same 58
annotations, which named ACL, MCL, Medial Meniscus, Lateral Meniscus, Lateral OA and PF OA as the findings
where the model, not the report, is the weak part. The external ruler and the internal partition agree,
and neither could have produced the other.

Priced on the competition metric, which is an unweighted mean of twelve:

    (0.20 + 0.10) / 12 = +0.025

That is the first quantity this project has produced that clears the owner's 0.02 bar, and it exceeds the
0.018 gap to first place (0.952). It is a **ceiling**, not a forecast: it assumes both labels reach the
best published number, on a different corpus, with different prevalence.

**Lateral Meniscus is the solid half of it.** A 0.20 gap is 3–4 SE on our own ruler. ACL's 0.10 is ~2 SE —
real but not comfortable, and weakened further by MRNet and ELNet being ACL-*specialist* models trained on
ACL-enriched corpora. If only Lateral Meniscus moves, the ceiling is +0.017, which sits just under the bar.

**Synovitis is retired, and only an external number could have retired it.** Published DL 0.76 against a
senior radiologist's 0.77 — the human ceiling on that finding is roughly where we already are (0.7378).
There is no headroom to chase because there is no headroom to be had. Internally this label looked like a
0.873 mid-pack performer with a supervision hole; externally it is finished.

**Three of our labels beat the published field** — Effusion by 0.08, Medial OA and Contusion by 0.02. Read
carefully: Effusion's comparators are from panels where effusion is a secondary target, so this is more
likely "their paper wasn't trying" than "our model is world-class".

**Four labels have no comparator at all.** Compartment-specific OA detection AUC from MRI: nothing found —
located OAI/MOST work reports Dice/IoU or PF-OA *incidence*, not detection AUC. Baker's cyst: nothing
(nearest, an SE-YOLOv5 cyst mAP 0.942). Knee fracture on MRI: nothing; every AUC located was radiograph-based.
Lateral OA at 0.7099 is our second-lowest number and it is **unpriceable** — it may be a large deficit or
it may be near the ceiling of a genuinely hard task, and no published work says which.

## What this does NOT say

- **It is not a measurement of our model on their data, nor theirs on ours.** Every gap here is confounded
  by corpus, prevalence, inclusion criteria and ground-truth protocol. In particular this competition grades
  ambiguous findings as NEGATIVE to favour specificity (organiser topic 733343), which no cited paper does.
- **It does not name a mechanism.** It says where the deficit is, not what causes it. Resolution, slice
  sampling, plane coverage, head structure and pretraining domain all remain candidates, and this table
  cannot rank them.
- **It does not license a submission.** Nothing here is a board reading.
- **The 58-study ruler is the weakest link and should be named as such in anything built on this.** Every
  gap above should be re-derived if the annotated set ever grows.

## The public-leaderboard context, gathered in the same sweep

Leader is **0.952** (2026-08-23 00:02); roughly twenty teams sit above our 0.934. The strongest **public**
notebook is 0.924 — `dreaddevelopment/knee-mri-twelve-findings-from-a-single-model` — and it reaches that
with **a single model**, no ensemble, on a CoAtNet (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`) with a 140 mm
physical crop, 42 slices at eval drawn from five (plane, fat-suppression) slots including Axial, and twelve
per-finding attention queries over slice features.

Its author's own board readings: *"CoAtNet alone scored 0.914 while every blend scored 0.914-0.915, so
ensembling is worth ~+0.001 there."* That is an independent team, on their own board draws, arriving at the
conclusion `2026-08-23_selection-overfit-explains-greedy7.md` reached offline. Two rulers, two teams, one
answer: **arrangement pays ~0.001 and contents are the variable.**

Its labelling claims are ones we have already banked: 3,155-study corpus 0.8923 → 4,349-study LLM-labelled
corpus 0.9054 (+0.0131), and LLM-vs-regex 0.8780 vs 0.8136 on the 58 gold studies. Our labeler **is** the
LLM path (`api_labeler.py`, from pilkwang's CC0 dataset), over 4,407 studies. Those gains are behind us,
not ahead.

## External-data status, since it gates knee-ombx

The host has ruled that sending report text to an external LLM for label extraction is **not** prohibited
private sharing (topic 733965, Po-Hao Chen, 2026-08-09). The separate question — whether registration-gated
knee-MRI datasets (MRNet, OAI, fastMRI+, SKM-TEA, KneeCoT) satisfy "publicly available and equally
accessible at no cost" — has been asked **three times** (733652, 734109, 735497) and **never answered**.
knee-ombx is therefore blocked on host silence, not on a rule.
