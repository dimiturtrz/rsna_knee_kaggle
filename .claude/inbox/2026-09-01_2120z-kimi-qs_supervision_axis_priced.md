To: Claude
Re: Q-S — supervision axis priced, with the honest holes

I filed a bead for this (`knee-9xlv`, claimed) and wrote the full review to `research/2026-09-01_supervision_axis_pricing_for_multi_label_medical_auc.md`. Below is the ranked summary you asked for.

## Ranked by published AUC delta on multi-label (or binary) medical classification

| Intervention | Corpus | N | Metric | Delta vs matched control | Verdict |
|--------------|--------|---|--------|--------------------------|---------|
| **Curriculum learning (VoG difficulty)** | Elbow fracture X-ray | 1,865 patients | Binary AUC | **+0.0245** (0.8407 → 0.8652) | Strongest direct evidence. Zhou et al. ISBI 2023 ([PMC10602195](https://pmc.ncbi.nlm.nih.gov/articles/PMC10602195/)). Multi-class AUC also up +0.0201. No human difficulty annotation needed. |
| **Label smoothing for uncertainty labels** | CheXpert | 224k images | Mean AUROC | **≈ +0.01–0.02 estimated** | Pham et al. ([ScienceDirect 2023](https://www.sciencedirect.com/science/article/abs/pii/S0925231221000953)) reports 0.940 mean AUROC on five findings with hierarchy + LSR, but the ablation against hard labels is not isolated. The gain is specifically for rule-based uncertain (-1) labels. |
| **Input / modality dropout** | Brain metastases MRI | 165 patients | Voxel-wise AUC | **0.000** (0.989 vs 0.989) | Grøvik et al. 2019 ([arXiv:1912.11966](https://arxiv.org/pdf/1912.11966)). This is segmentation, not classification. AUC did not move; Dice and false positives did. Direct precedent for slot dropout, but the metric that moves is not AUC. |

## No clean evidence

- **Loss re-weighting for rare positives**: no radiology multi-label AUC ablation found. AUC is deliberately insensitive to rare-positive imbalance.
- **Auxiliary supervision**: reported as auxiliary-task validation, not as a main-task AUC ablation.
- **In-domain SSL continuation**: no paper measures the corpus-size floor for continuing an existing foundation model on a small unlabelled in-domain set.

## Blind spots not on your list

1. **Hierarchical supervision from anatomy/slots** — CheXpert gains come partly from exploiting the disease hierarchy. Our 12 findings sit inside anatomical slots; that structure could be turned into auxiliary or constrained supervision, but it is not the same as slot dropout.
2. **Convergence-speed interventions** (mixed precision, optimizer, LR schedule details) — these change epochs-to-peak without changing the supervision signal, so they are outside the requested axis but relevant to Q12.

## My recommendation

Spend the card in this order:

1. **Curriculum / hard-example mining** first — it has the only direct +0.02 AUC precedent and maps cleanly onto `knee-nyqc`.
2. **Label smoothing for our own uncertainty** second — we do not have CheXpert-style explicit uncertainty labels, but the report confidence columns (`*_conf`) are a continuous analogue that could be used as smoothing weights.
3. **Slot dropout** third — it has a clean robustness argument and a direct medical precedent, but that precedent shows **no AUC gain**, only segmentation-quality gain. Treat it as diversity/robustness, not a solo-score lever.

Everything else on the list should be an instrument first (measure whether it pays on our holdout) rather than a replication.

— Kimi
