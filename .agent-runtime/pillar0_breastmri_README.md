---
license: ecl-2.0
language:
- en
extra_gated_fields:
  Name: text
  Organization: text
---
# Model Card for Pillar0-BreastMRI

## Model Details

### Model Description

**Pillar-0 (Breast MRI)** is a general-purpose radiology foundation model designed for high-resolution volumetric MRI understanding. It is part of the Pillar-0 family of models, which leverages the native 3D structure of medical imaging rather than processing volumes as independent 2D slices.

This specific checkpoint utilizes the **Atlas** vision backbone efficiently scaled for 3D inputs and is pretrained via asymmetric contrastive learning against the **Qwen3-Embedding-8B** Large Language Model (LLM). The model was trained on a dataset of breast MRI exams and their corresponding radiology reports to capture rich clinical context. It achieves state-of-the-art performance on internal benchmarks, significantly outperforming general-purpose medical foundation models like MedGemma, MedImageInsight, and Lingshu.

- **Model type:** Vision-Language Foundation Model .
- **Architecture:** Atlas Vision Encoder (Backbone) aligned with Qwen3-Embedding-8B (Text Encoder).
- **Language(s) (NLP):** English (Radiology Reports).

### Model Sources

- **Repository:** https://huggingface.co/collections/YalaLab/pillar-0
- **Code:** https://github.com/YalaLab/pillar-pretrain
- **Paper:** Pillar-0: A New Frontier for Radiology Foundation Models

## Uses

The model is designed to generate high-fidelity volumetric embeddings for Breast MRI exams. It can be used for:
* **Feature Extraction:** Extracting rich 3D representations from breast MRI volumes for analysis.
* **Zero-shot/Few-shot Analysis:** Leveraging the aligned text-image space for retrieval or classification tasks (evaluated primarily via linear probing in the paper).

## How to Get Started with the Model

Detailed instructions for loading the weights and preprocessing volumes using the RATE-Evals (Radiology Vision Engine) framework are available in the [GitHub Repository](https://github.com/YalaLab/rate-eval).

## Evaluation

### Testing Data, Factors & Metrics

#### Metrics
* **RATE (Radiology Text Engine):** Evaluation performed on **35 clinically grounded findings** (e.g., "Are any masses present?", "Is there skin thickening?") extracted from reports by board-certified radiologists.
* **Protocol:** Linear probing on frozen embeddings (RATE-Evals).
* **Primary Metric:** Area Under the Receiver Operating Characteristic curve (AUROC).

### Results

Pillar-0 (Breast MRI) establishes a new performance frontier compared to existing medical foundation models.

| Model | Mean AUROC (Breast MRI) | Win Rate vs Pillar-0 |
| :--- | :--- | :--- |
| **Pillar-0 (Ours)** | **82.9** | **-** |
| MedGemma (Google) | 67.1 | 2.9% |
| MedImageInsight (Microsoft) | 64.4 | 5.7% |
| Lingshu (Alibaba) | 57.0 | 0.0% |
| Merlin (Stanford) | N/A | N/A |


#### Summary
Pillar-0 outperformed all baselines, achieving the highest score in **91.4%** of the evaluated breast MRI tasks.

## Citation

Please cite Pillar-0 if you find this work helpful.

```bibtex
@article{pillar0,
  title   = {Pillar-0: A New Frontier for Radiology Foundation Models},
  author  = {Agrawal, Kumar Krishna and Liu, Longchao and Lian, Long and Nercessian, Michael and Harguindeguy, Natalia and Wu, Yufu and Mikhael, Peter and Lin, Gigin and Sequist, Lecia V. and Fintelmann, Florian and Darrell, Trevor and Bai, Yutong and Chung, Maggie and Yala, Adam},
  year    = {2025}
}