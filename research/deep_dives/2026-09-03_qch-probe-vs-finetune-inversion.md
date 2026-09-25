# Q-CH — Does frozen linear-probe ranking predict fine-tuned ranking across trunks?

**Question (Claude, 2026-09-03):** When we compare trunks with a frozen linear probe, does the ranking stay the same after full fine-tuning? What is this called/measured in the literature? Does domain-continued pretraining change the relationship? What cheaper-than-full-fine-tune predictor best predicts the fine-tuned ranking, and what are the licence lines?

**TL;DR:** Rank inversion between frozen linear probing (LP) and full fine-tuning (FT) is a known, measured phenomenon. On ImageNet, ID LP and ID FT are correlated (Spearman r ≈ 0.90 for LP averaged against other ID metrics, FT is the most idiosyncratic protocol with average r ≈ 0.79), but full FT is a *weaker* predictor of out-of-domain FT than 10%-shot FT or even ID probing. Domain-continued pretraining has not been explicitly measured for LP-vs-FT rank stability in knee MRI; the closest radiology work only compares absolute LP/FT gains, not cross-trunk rank correlation. For a cheap predictor of final FT ranking, the best published evidence points to **short-schedule fine-tuning on ~10% of the data** (or a 100-step probe + meta-features à la TuneAhead) rather than a frozen LP. LogME gives a quick frozen-feature transferability score, but it is not a direct proxy for full-FT ranking.

---

## 1. What the phenomenon is called and how it is measured

The problem is usually discussed under two overlapping names:

- **Protocol correlation / rank correlation** — how well one evaluation protocol (LP, kNN, FT, few-shot FT) ranks a set of pre-trained models relative to another protocol on the same or a different dataset. The standard metric is **Spearman’s rank correlation coefficient** (sometimes Kendall’s τ or weighted Kendall’s τ).
- **Transferability estimation** — predicting which pre-trained model will transfer best *without* training it to convergence. LogME, LEEP, NCE, and H-Score fall in this family.

The closest direct measurement is **Marks et al. / Knott et al., “A Closer Look at Benchmarking Self-Supervised Pre-training with Image Classification” (2024/2025, IJCV)** [arXiv:2407.12210](https://arxiv.org/abs/2407.12210), code at [manuelknott/ssl_eval_protocols](https://github.com/manuelknott/ssl_eval_protocols).
They evaluate 26 SSL models on ImageNet-1k and 11 out-of-domain datasets and report:

- Among ID metrics, **linear probing** has the highest average correlation with other ID metrics (Spearman r ≈ 0.90), while **full fine-tuning** is the weakest (r ≈ 0.79).
- For **OOD fine-tuning ranking**, **10%-shot ID fine-tuning** is the strongest predictor.
- ID linear probing is still a strong OOD predictor (r ≈ 0.85 averaged over OOD metrics), but it is *not* the same as predicting full-FT ranking.
- Newell & Deng (referenced in the related-work section) previously found that “LP performance does not correlate with FT performance.”

So the answer to “does LP ranking predict FT ranking?” is **mostly, but not reliably enough to bet an arm on**: in-domain correlations are high, but the full-FT protocol is itself the most idiosyncratic, and cross-trunk inversions are expected when the domain shift or the trunk family changes.

### Mechanistic vocabulary

- **LP-FT** (Kumar et al., ICLR 2022) gives the conceptual frame: full fine-tuning can **distort pretrained features**, especially for out-of-distribution data, and the simple two-stage recipe — train a linear head on frozen features, then fine-tune the whole network from that initialization — recovers both ID and OOD performance. The paper reports ~1% ID gain and ~10% OOD gain over full FT alone. [arXiv:2202.10054](https://arxiv.org/abs/2202.10054), code at [AnanyaKumar/transfer_learning](https://github.com/AnanyaKumar/transfer_learning).
- **Evidence > Intuition** (Bassignana et al., EMNLP 2022) applies LogME to NLP encoders and finds that LogME correlates positively with final performance in 94% of setups, **but predictive power drops after full fine-tuning because frozen and fine-tuned representations misalign**. This is exactly the inversion mechanism we are worried about. [mainlp/logme-nlp](https://github.com/mainlp/logme-nlp).

---

## 2. Domain-continued pretraining and the LP-vs-FT relationship

We did not find a paper that explicitly reports **cross-trunk rank correlation between LP and FT in a domain-continued-pretraining setting** (natural-image trunk → medical continuation → knee task). The phenomenon is likely real but under-measured.

Relevant neighbouring results:

- **“Evaluating General Purpose Vision Foundation Models for Medical Image Analysis: An Experimental Study of DINOv2 on Radiology Benchmarks”** (Ouyang et al., 2023) shows that domain-adaptive continued pretraining of DINOv2 on radiology data improves both LP and FT results, but it reports absolute numbers, not rank correlations across continuation depths or trunks. [arXiv:2312.02366](https://arxiv.org/abs/2312.02366).
- **DIET-CP** (2025) studies lightweight continued pretraining on small medical datasets and finds that standard SSL recipes need redesign for tiny domains. [arXiv:2509.06990](https://arxiv.org/abs/2509.06990).
- **ULMFiT-style discriminative learning rates** (Howard & Ruder, 2018) are the classic way to prevent feature distortion when adapting a pretrained model to a new domain: lower layers get smaller learning rates than upper layers. This is directly relevant to rescuing a continued trunk if full FT starts distorting the domain-aligned features.

**Conjecture for our setting:** Continued pretraining may make features *more linearly separable* on the target domain (good LP) while also making them *more brittle to full FT* because the features are tuned to a narrower distribution. The ortholarge-vs-v3large inversion is therefore plausible and should be measured directly rather than assumed away.

---

## 3. Cheaper predictors of fine-tuned ranking

From strongest evidence to weakest:

| Predictor | Cost | Predicts FT ranking? | Evidence |
|-----------|------|----------------------|----------|
| **10%-shot fine-tuning** on the target task | ~5–10% of a full run | **Best OOD FT predictor** in Knott et al. | Knott et al. 2024/2025 |
| **Short-schedule probe + meta-features** (TuneAhead) | <5% of a full run | RMSE 1.47 pp, 95% within ±3 pp on Qwen2.5-7B | Long et al. 2026, [arXiv:2606.17660](https://arxiv.org/abs/2606.17660) |
| **Frozen linear / kNN probing** | Very cheap | Strong OOD predictor on average, but inversions occur | Knott et al. 2024/2025 |
| **LogME** | Very cheap | Good for model selection, not a direct FT-ranking proxy | You et al. ICML 2021 / JMLR 2022 |

- **LogME** (You et al., ICML 2021; extended JMLR 2022) estimates the Bayesian evidence of labels given frozen features. It ranks PTMs with weighted Kendall’s τ_w ≈ 0.5–0.8 on most vision tasks and is perfect on some NLP tasks. It is *not* the same as LP accuracy and does not claim to predict full-FT ranking. Code: [thuml/LogME](https://github.com/thuml/LogME).
- **TuneAhead** is the modern version of the idea for LLMs: a 100-step standardized probe plus static dataset descriptors predicts final fine-tuning performance within a few percentage points. The design is portable to vision: run a few hundred steps of FT on a small subset, extract training-curve shape features, and predict final score. No public vision code was found.

**Practical recommendation:** For trunk screening, do not trust a frozen LP alone. Run a **10%-study short fine-tune** (or a 1–2 epoch probe) on each candidate trunk and rank by that. It is more expensive than LP but far cheaper than a full 12-epoch run, and the literature shows it predicts final FT ranking better.

---

## 4. Licence lines

| Work / code | Licence | Notes |
|-------------|---------|-------|
| **LogME** code ([thuml/LogME](https://github.com/thuml/LogME)) | **MIT** (`LICENSE.txt`) | Copyright 2023 youkaichao |
| **Knott et al. SSL eval protocols** ([manuelknott/ssl_eval_protocols](https://github.com/manuelknott/ssl_eval_protocols)) | **MIT** | Copyright 2024 Manuel Knott |
| **LP-FT** code ([AnanyaKumar/transfer_learning](https://github.com/AnanyaKumar/transfer_learning)) | **No explicit licence** | No `LICENSE` file; repo is a research snapshot |
| **TuneAhead** (Long et al., 2026) | **No public code repository found** | Paper only ([arXiv:2606.17660](https://arxiv.org/abs/2606.17660)) |
| **Evidence > Intuition** LogME-NLP ([mainlp/logme-nlp](https://github.com/mainlp/logme-nlp)) | Not checked | Not central to the answer |

---

## 5. Actionable take-aways for the project

1. **Do not use frozen LP as the sole trunk filter.** The literature shows LP and FT rankings can invert, especially across backbone families and domain shifts.
2. **Add a 10%-study / 1–2 epoch FT probe to `arm_check` or the trunk evaluation pipeline.** Measure its Spearman correlation with the final 12-epoch score on our own data; this gives a project-specific predictor.
3. **If we continue pretraining more trunks in-domain, explicitly measure LP-vs-FT rank correlation.** The radiology literature does not give us this number; we have to produce it.
4. **Consider LP-FT as a training recipe for continued trunks** — it may recover OOD/generalization performance that naive full FT destroys.
5. **Licence hygiene:** LogME and Knott’s code are MIT and safe to reuse/adapt. LP-FT code has no licence, so we should treat it as read-only reference unless we contact the authors. TuneAhead has no public code yet.
