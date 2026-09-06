# 2026-09-06 — VisualCheXbert mechanism, Wollek-vs-Jain reconciliation, noisy-label pricing

Field synthesis for the label-route decision (mastermind 07:40Z order). Sources read in
full: arXiv 2102.11467 (VisualCheXbert), arXiv 2306.05997 (Wollek), Wei 2024 scoping
review (PMC11187424), Samala 2020 (PMC7981191).

## VisualCheXbert (arXiv 2102.11467) — the actual mechanism

- A DenseNet (Irvin et al. 2019 recipe) trained on REPORT labels (CheXpert rule labels;
  AUROC 0.875/0.883 on CheXpert test) outputs per-condition probabilities for every
  X-ray; those probabilities are the SUPERVISION for a BioBERT that maps report
  impression → image-proxy labels. Cross-fitting: DenseNet trained on CheXpert labels
  MIMIC images and vice versa.
- Output head: per-condition Youden-optimal threshold (BERT+Thresholding) or a logistic
  regression over all 14 BERT outputs per target (BERT+LogReg = "VisualCheXbert",
  the better one). So the image is used as a SOFT RELABELER, not a binary flipper;
  no "fraction of cells flipped per label" is reported — targets are probabilities.
- Per-label F1 gains over radiologists-reading-reports (high mapping): Enlarged
  Cardiomediastinum +0.52, No Finding +0.16, Lung Opacity +0.15, Cardiomegaly +0.15,
  Atelectasis +0.04 ns, Edema −0.04 ns, Support Devices +0.01 ns, Pleural Effusion
  −0.06 (significantly WORSE).
- Pattern: gains concentrate on hierarchy-affected parent labels; the hedged,
  threshold-dependent label (Pleural Effusion — "trace", "decreased", "residual"
  language in reports) gained NOTHING and lost slightly. Mapped onto us: our hedge-heavy
  findings (Effusion first among them) are exactly the CXR cell where image-grounding
  bought the least. The mechanism's published value is hierarchy correction, not
  hedge correction.
- Downstream price (Jain 2104.00793): text-only CheXbert re-label +0.00; VisualCheXbert
  image-grounded +0.04 mean weighted AUROC (0.83→0.87) on a 2019 DenseNet-121, CXR.

## Wollek 2306.05997 vs Jain 2021 — both real, different rulers and baselines

Wollek's 0.720/0.858/0.939 are downstream pneumothorax AUCs (DenseNet-121, 224px):
0.720 = CheXnet trained on ChestX-ray14; 0.858 = trained on rule-labeler labels;
0.934 = image+report radiologist labels; 0.939 = deep-labeler labels (BERT weak-supervised
on 66k rule-labeled German reports, fine-tuned on ~1k manual). So Wollek IS downstream,
and beats radiologists.

Reconciliation: (1) baseline weakness — Wollek's rule labeler is weak (negation F1 0.51,
uncertainty F1 0.51); Jain's CheXpert-rule baseline is already strong, so the upgrade has
no headroom. The downstream gain from a labeler upgrade is bounded by how broken the
current labeler is. (2) Confound — Wollek's DL labels beat image-informed radiologist
labels, which says the DL labeler also acts as a regularizer/denoiser at small n; that
effect is not "label correctness". (3) Task: single-label pneumothorax, internal set.

Implication for us: our extractor is an LLM re-read of v2, not a weak rule labeler —
the Wollek +0.081 regime does not apply; the Jain +0.00 text-only regime is the closer
prior unless the re-read specifically attacks our failure mode (hedge remap), which is
the image-grounded variant's row.

## Noisy-label correction literature — the honest map

- No knee-MRI or musculoskeletal downstream-AUC measurement of co-teaching /
  DivideMix-style correction exists. Wei 2024 (60-paper scoping review): MRI entries are
  brain/valve/spine; correction studies there compare clean-vs-noise, rarely publish a
  corrected-vs-baseline AUC delta. Nearest musculoskeletal: knee X-ray automated
  labeling (PMC10584746) — a labeling paper, not a correction delta.
- Quantified medical noise-impact results are "noise hurts": Samala 2020 (DBT): 10%
  corrupted labels materially increase generalization error; Jang: 2% random flips hurt
  CXR accuracy; review notes impact is largest at small sample size (our regime: 4.4k
  studies, 12 labels).
- General-vision correction numbers (CIFAR family): at 20% symmetric noise, co-teaching /
  DivideMix recover most of the drop (double-digit points); at 10-20% noise the
  recoverable margin shrinks to single digits. These are random-flip regimes — our noise
  is structured (hedge down-weighting), so the transfer is weak by construction.
- The one hedge-adjacent measured instance in-house: our v3 hedge remap, +0.0105 on
  gold-58 (already banked). The C3 "hedge-aware soft targets" strict form stays
  UNPRICED in the literature (searches: "hedge-aware soft targets radiology",
  "uncertainty label third class medical downstream AUC" — nothing with a downstream
  endpoint).
