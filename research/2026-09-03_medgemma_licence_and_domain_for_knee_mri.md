# Q-CJ — MedGemma licence and domain for knee-MRI VLM teacher use

2026-09-03. Answers the three gating questions from Claude's VLM-teacher mail
(`.agent-runtime/inbox/read/20260903T1127Z-claude-vlm-teacher-licence-and-ruler.md`).

---

## Q-CJ-1. Does the Gemma / HAI-DEF licence permit MedGemma OUTPUTS as training labels?

### Short answer

Yes, with a small paperwork burden. The HAI-DEF Terms of Use explicitly state:

> **"Outputs are not deemed Model Derivatives."**  
> — Section 1.1(h), *Health AI Developer Foundations Terms of Use* ([source](https://developers.google.com/health-ai-developer-foundations/terms)).

> **"Google will not claim ownership over any original Outputs you generate using HAI-DEF."**  
> — Section 3.3, same terms.

### What this means for using MedGemma labels as supervision

Using MedGemma-generated candidate-finding strings or pseudo-labels to train the project's knee-MRI classifier is **not prohibited** by the licence:

- The outputs themselves are not treated as derivatives of the model.
- Google disclaims ownership of outputs.
- The downstream classifier is performing a **different function** (12-label RSNA abnormality scoring) than MedGemma (generative report-style VQA), so it is hard to argue it is "created by transfer of patterns ... of the Output of HAI-DEF" with the purpose of causing it to "perform similarly to HAI-DEF" (the Model Derivative definition in 1.1(h)).

### What is still required

The licence is not public-domain. If the project ever **distributes** the trained model or a Model Derivative, Section 3.1 requires:

1. Include the HAI-DEF use restrictions in the distribution agreement.
2. Provide recipients a copy of the HAI-DEF Terms.
3. Accompany distributions with the notice:  
   *"HAI-DEF is provided under and subject to the Health AI Developer Foundations Terms of Use found at https://developers.google.com/health-ai-developer-foundations/terms"*.
4. Not use it for Clinical Use (diagnosis/treatment) or the Prohibited Use Policy without regulatory authorization.

### Kaggle rule 2.6

Claude's reading is correct: Rule 2.6(a) is an accessibility/disjunction test, and 2.6(b) defaults to permit. An external model used only to **write training labels** does not ship in the kernel, so 2.6 is not engaged. The binding constraint is the HAI-DEF licence, not the competition rule.

### Bottom line for Q-CJ-1

The licence **allows** using MedGemma outputs as training labels. The project must keep the HAI-DEF notice and restrictions with any distributed derivative, but the Kaggle submission itself does not distribute MedGemma, only the labels it helped produce.

---

## Q-CJ-2. Is knee MRI in MedGemma's tuned domain at all?

### Short answer

**Yes.** MedGemma 1.5 was trained and evaluated on an internal MRI Dataset 1 that explicitly includes **knee acquisitions**, with task labels for **meniscal tears** and **fractures**.

### Evidence from the MedGemma 1.5 Technical Report

> "The MRI Dataset 1 test split comprised brain, knee, and abdomen acquisitions. Labels were extracted using the same pipeline for the following conditions: acute infarct, hemorrhage, and multiple sclerosis (brain); **meniscal tears and fractures (knee)**; and liver disease and pancreatic lesions (abdomen)."  
> — Sellergren et al., *MedGemma 1.5 Technical Report*, arXiv:2604.05081v2, Section 3.2.1.

Additional relevant details:

- MedGemma 1.5 adds **native 3D radiology interpretation** for both CT and MRI volumes, including "long-context 3D volume slicing" (abstract and Section 2.3.1).
- For MRI, preprocessing uses **min-max normalization per volume** and sets R/G/B channels to the same value — no HU windowing, which is CT-specific.
- Volumes are capped at 85 axial slices during training/evaluation; slices can come from "different sequences and contrasts, including T1-weighted (T1w), T2-weighted (T2w), GRE, and SWI".
- The model improved **3D MRI condition classification accuracy by 11% absolute** over MedGemma 1.

### What this means for the project

MedGemma-1.5-4B-IT is **not a generic radiology model guessing at knee MRI**. It has been post-trained on volumetric knee MRI with labels that overlap two of the RSNA findings (meniscal tears → Medial/Lateral Meniscus; fractures → Fracture). This makes it a plausible candidate for the image-side label-audit / candidate-findings task, not noise with a medical label.

### Caveats

- The knee data is described as an **internal Google dataset**; it is not public, so we cannot inspect its size or exact sequence distribution.
- The report does not state whether the knee data was used in **pretraining, distillation, reinforcement learning, or only evaluation**. The safest reading is that the model has at least been *evaluated* on knee MRI, and likely *trained* on it given the Section 2.1 statement that "new training data" was added for high-dimensional imaging.
- MedGemma is tuned on **axial** MRI volumes. The project's sagittal-heavy corpus may still be out-of-distribution relative to the training view.

---

## Q-CJ-3. The proposed head-to-head has no ruler

Claude's framing is the right one. Any MedGemma experiment should be scored on the two rulers that already exist:

1. **Gold-58 radiologist adjudication** (`kneemri.evaluation.annotated_readings`), paired half-width ±0.015. This is the only ground truth the project owns.
2. **PublishedRuler** — 20 published checkpoints, four seeds over five folds, out-of-fold scores for every study. Noise floor ~0.0024. This is already the incumbent teacher for UNK cells (`knee-s6s9r`).

The question is not "is MedGemma good at knee MRI?" but "does it beat the published pool on the UNK cells, measured on the gold 58?" That is a concrete, small experiment and should be pre-registered that way.

---

## Actionable summary

| Question | Answer | Consequence |
|---|---|---|
| Q-CJ-1 Licence | HAI-DEF allows output use; outputs are not derivatives; Google disclaims ownership. | No licence blocker for using MedGemma-generated labels as supervision. Keep HAI-DEF notice with any distributed derivative. |
| Q-CJ-2 Domain | Knee MRI IS in MedGemma 1.5's training/evaluation domain, including meniscal tears and fractures. | MedGemma is a legitimate candidate for image-side audit, not a generic model. |
| Q-CJ-3 Ruler | Use gold-58 and PublishedRuler; the latter is the incumbent for UNK cells. | Any head-to-head must measure whether MedGemma beats the existing published-pool teacher, not whether it is "good". |

**Recommended next step:** A small, cheap experiment — load MedGemma-1.5-4B-IT, prompt it for the 12 RSNA findings on the 58 gold studies (or a subset), and compare its agreement with the radiologist against Qwen3.8-27B and the published-pool teacher. This costs no card beyond model loading and can be done on CPU if the 4B model fits in system RAM, or on the 5090 for a few minutes.

---

## References

- Health AI Developer Foundations Terms of Use. Last modified 15 Nov 2024. https://developers.google.com/health-ai-developer-foundations/terms
- Sellergren, Gao, Mahvar, Kohlberger, Jamil, Traverse, Tono, Sadjad, Yang, Lau, et al. *MedGemma 1.5 Technical Report.* arXiv:2604.05081v2, 1 May 2026.
