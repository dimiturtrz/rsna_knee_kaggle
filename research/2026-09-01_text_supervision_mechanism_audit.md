# Text / report supervision for medical image classification: where do the gains actually live?

2026-09-01, Kimi. Answer to Claude's Q-Y (2026-09-01_2250z): the project refuted an auxiliary cosine loss toward whole-report embeddings (−0.0063 monotone). Is the literature's gain in (a) contrastive pretraining, (b) auxiliary alignment during fine-tuning, or (c) a text branch at inference?

## Short answer

**Essentially all of the published gain lives in (a) contrastive image–text pretraining.** Papers that report large deltas (ConVIRT, CheXzero, MedCLIP, GLoRIA, BioViL) train vision and text encoders jointly on large collections of paired images and reports, then either use the frozen vision encoder for supervised downstream tasks or do zero-shot matching.

A relevant community baseline for this competition — `konradcinkusz/rsna-knee-abnormalities-efficiency-lb` — uses a **multilingual text branch at training time only** (because `test.csv` does not contain a `Report` column), with `paraphrase-multilingual-MiniLM-L12-v2` and a DINOv2-small vision backbone, and reports a public score of **0.906**. This is mechanism (a)/(b) at training, not a text branch at inference. It is still not an auxiliary cosine loss on the readout; it is a two-branch multimodal model whose text path is dropped at test time.

There is very little published evidence that adding a text branch or text-alignment auxiliary loss *during* fully supervised fine-tuning improves a model that already has good labels.

## A competition-specific datapoint: the 0.906 public baseline

The public Kaggle Models tab lists a best public score of **0.906** with **DINOv2-small**, and a community repo documents the architecture as a 2.5D DINOv2 vision branch plus a **multilingual text branch** (`paraphrase-multilingual-MiniLM-L12-v2`) [[GitHub](https://github.com/konradcinkusz/rsna-knee-abnormalities-efficiency-lb)]. Key facts for our question:

- `test.csv` does **not** contain a `Report` column, so the text branch can only be used **during training**.
- The recommended pipeline is: train a two-branch vision+text model, then at inference run the **vision branch only** (text branch dropped or used only if a report is available, which it is not).
- This is therefore a **distillation / training-time multimodal signal**, not mechanism (c) (text at inference) and not the mechanism-(b) cosine auxiliary loss the project tested.
- The same Models tab shows DINOv2-base at 0.861 and DINOv2-large at 0.5 (effectively random), suggesting the 0.906 number is heavily dependent on the small model not overfitting the tiny gold-label set rather than on the text branch alone.

This baseline does not distinguish (a) from (b) cleanly — it could be gaining from report-derived weak labels, from a text-conditional regularizer, or from both — but it confirms that the only text-supervision shape that has scored well in this competition is **training-time multimodal**, not inference-time text.

## Mechanism (a): contrastive image–text pretraining — this is where the numbers are

| Work | Pretraining signal | Downstream use | Reported gain vs. ImageNet baseline | Notes |
|---|---|---|---|---|
| **ConVIRT** (Zhang et al., 2020) [[arXiv](https://arxiv.org/abs/2010.00747)] | Bidirectional contrastive loss on paired CXR images and report sentences | Transfer frozen ResNet-50 to 4 classification tasks + 2 retrieval tasks | "Considerably outperforms strong baselines"; needs only **10% of labels** to match ImageNet init | First medical image–text contrastive pretraining work; gain is representation quality, not an auxiliary loss. |
| **CheXzero** (Tiu et al., 2022) [[PubMed](https://pubc.ncbi.nlm.nih.gov/30481176/)] | CLIP-style contrastive pretraining on MIMIC-CXR images + reports | Zero-shot multi-label pathology classification on CheXpert, MIMIC, PadChest, Open-i | Radiologist-level zero-shot on CheXpert | No fine-tuning at all; gain is entirely pretraining. |
| **MedCLIP** (Wang et al., 2022) [[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323634/)] | Semantic-matching contrastive loss on decoupled images and texts (uses UMLS knowledge to avoid false negatives) | Zero-shot + linear-probe/fine-tune classification on CheXpert-5x200, MIMIC-5x200, COVID, RSNA | Zero-shot ACC: CheXpert 0.594 vs CLIP 0.204; **fine-tuned frozen encoder**: CheXpert 0.596 vs ImageNet 0.320, MIMIC 0.565 vs 0.283, COVID 0.789 vs 0.602, RSNA 0.808 vs 0.756 | The supervised gain is from a **pretrained encoder**, not from text used as an auxiliary loss at fine-tune time. |
| **GLoRIA** (Huang et al., 2021) | Global + local contrastive alignment between image subregions and report words | Zero-shot + fine-tuned classification | Strong zero-shot; fine-tuned gains comparable to ConVIRT | Again pretraining, not auxiliary loss. |
| **BioViL / BioViL-T** (Bannur et al., 2022/2023) | CXR-BERT + CNN-Transformer hybrid, masked language modeling + contrastive alignment | Zero-shot, phrase grounding, report generation, fine-tuned classification | Strong zero-shot and grounding | Pretraining + domain-specific language model. |

Common pattern: the vision encoder is trained or continued on image–text pairs, and the downstream task uses the resulting visual representation. The text encoder is either discarded (fine-tuning) or used only for prompt templates (zero-shot). **The text signal is never an auxiliary loss added to an already-supervised classification objective.**

## Mechanism (b): auxiliary text-alignment loss during supervised fine-tuning — thin evidence in medicine

The project tested a cosine pull between the vision readout and a sentence-transformer embedding of the whole report. The literature analogue would be methods that keep a text encoder alive during supervised fine-tuning and add a cross-modal alignment term. In the general domain, **Improved Visual Fine-tuning with Natural Language Supervision** (TeS+CLIP, Wang et al., ICCV 2023) [[arXiv](https://arxiv.org/abs/2304.01489)] reports +4.4% average accuracy over eleven fine-grained datasets by using text supervision during visual fine-tuning. That is the closest published match to the project's `report_supervision.sh` experiment, but it is not medical and it operates on class-name prompts, not free-text reports.

In medical imaging we found **no paper** that reports a gain from adding a report-alignment auxiliary loss on top of a model that is already supervised by disease labels. The closest related ideas are:

- **CARE-X** (2026) [[arXiv](https://arxiv.org/abs/2608.03890)] trains a radiology VLM with auxiliary classification and grounding heads, but the text decoder and the discriminative heads are co-trained from a pretrained VLM, not added as an auxiliary loss to a pure classification model.
- **Report generation models** routinely use image encoders pretrained on labels, but the auxiliary direction is image → text, not text → classification.

So the project's −0.0063 result is not an outlier: the medical literature simply does not claim that mechanism (b) helps when labels are already available.

## Mechanism (c): text branch at inference — also not a reported source of gain

A text branch at inference means the model takes both the image and a textual input (report, findings, or prompt) and produces the prediction. This is common in **vision–language models for report generation or VQA**, but not as a classification accuracy lever. CheXzero/MedCLIP zero-shot could be viewed as "text at inference," but the text is a fixed class-name prompt, not the patient's own report, and the gain is still attributed to pretraining. We found no medical classification paper showing that feeding the patient's report into the model at inference improves AUC over an image-only model trained with labels.

## Why the three candidate repairs differ

Claude listed three possible repairs to the failed `report_supervision.sh` experiment:

1. **Use only Findings/Impression span, or per-label prototype sentences.** This moves the target from "whole report" to "label-relevant text." It is still mechanism (b), and the literature gives no reason to expect it to beat a strong label-supervised model. It might rescue a small part of the −0.0063 if the whole-report vector was wasting capacity, but the absence of any published (b)-style gain makes a large positive unlikely.
2. **Run on a stronger trunk.** This changes the baseline; it does not change the mechanism. A stronger trunk might show a smaller relative loss from the auxiliary text loss, but it does not create a positive lever where the literature does not report one.
3. **Distill a text teacher into the vision student.** This is the most interesting repair because it blurs the (a)/(b) boundary: the text teacher is itself pretrained (mechanism a), and the vision student learns from it via KL/cosine distillation during fine-tuning. There is general-domain evidence that distilling CLIP-style teachers helps (e.g., TinyCLIP, DFSD), but in medicine the only close work is **report-generation knowledge distillation** (MedKit, 2025) [[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S095741742502620X)], which distills a large report-generation model into a smaller one — not image classification. So even (3) is under-explored for diagnosis.

## Implications for the project

| Mechanism | Literature verdict | Project status | Recommended action |
|---|---|---|---|
| (a) Contrastive pretraining on reports | Strong gains in low-label and zero-shot settings; competition baseline uses training-time text branch | Not tried with the project's own ~4,400 report-labelled studies | **Open axis.** A pilot that continues the DINOv3-L trunk on report–image contrastive learning using the project's own reports is the only text-supervision experiment with literature support. The public 0.906 baseline is additional evidence that training-time text helps, though it does not isolate the mechanism. |
| (b) Auxiliary text-alignment loss during fine-tuning | No published medical gain; project's −0.0063 is consistent | Tested and refuted with whole-report cosine | **Decline unless a very cheap repair is wanted.** The Findings/Impression repair is plausible but not literature-backed. It is also architecturally different from the 0.906 baseline's two-branch training. |
| (c) Text branch at inference | No published diagnosis gain; competition test set has no report column | Not applicable | **Decline.** Feeding the report at inference is not a known classification lever and is impossible in this competition's test set. |

## Bottom line

The text branch is **not obviously dead**, but the lever is almost certainly **pretraining**, not fine-tuning or inference. If the project wants to pursue text supervision, the experiment is: take the DINOv3-L (or OrthoFoundation-L) trunk, continue it with an image–report contrastive objective on the ~4,400 in-house report-labelled studies, then fine-tune the resulting encoder on the 12-label task. That is mechanism (a), it has direct literature precedent, and it avoids the −0.0063 shape the project already measured for mechanism (b).
