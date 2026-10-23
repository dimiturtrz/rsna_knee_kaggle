# Papers read for this project

The PDFs are copyrighted by their authors and publishers, so they are not in git. Research docs cite a
paper by its local filename; this table maps each filename to the paper so it can be fetched again.

| Local file | Paper | Source |
|---|---|---|
| `.agent-runtime/brainfound.pdf` | Towards Generalisable Foundation Models for Brain MRI | arXiv:2510.23415 |
| `.agent-runtime/diet.pdf` | DIET-CP: Lightweight and Data Efficient Self Supervised Continued Pretraining | arXiv:2509.06990 |
| `.agent-runtime/dino.pdf` | Emerging Properties in Self-Supervised Vision Transformers (Caron et al.) | arXiv:2104.14294 |
| `.agent-runtime/dino3dreg.pdf` | Effective Feature Learning for 3D Medical Registration via Domain-Specialized DINO Pretraining | arXiv:2603.14086 |
| `.agent-runtime/dinocell.pdf` | Self-supervised Pretraining of Cell Segmentation Models | arXiv:2604.10609 |
| `.agent-runtime/explora.pdf` | ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts | arXiv:2406.10973 |
| `.agent-runtime/frees.pdf` | Towards Optimal Convolutional Transfer Learning Architectures for Breast Lesion Classification and ACL Tear Detection | arXiv:2508.17567 |
| `.agent-runtime/juodelyte.pdf` | Juodelyte et al., TMLR 09/2023 | arXiv:2302.08272 |
| `.agent-runtime/khan23a.pdf` | Khan et al., ML4H 2023 | PMLR 225:217-231 |
| `.agent-runtime/krones.pdf` | From pre-training to downstream performance: Does domain-specific pre-training make sense? | arXiv:2605.08819 |
| `.agent-runtime/lee.pdf` | Knowledge Transfer Scaling Laws for 3D Medical Imaging | arXiv:2605.06859 |
| `.agent-runtime/meta_overfitting.pdf` | A Meta-Analysis of Overfitting in Machine Learning (Roelofs et al.) | NeurIPS 2019 |
| `.agent-runtime/mmdinov2.pdf` | MM-DINOv2: Adapting Foundation Models for Multi-Modal Medical Image Analysis | arXiv:2509.06617 |
| `.agent-runtime/orthodiffusion_paper.pdf` | OrthoDiffusion: A Generalizable Multi-Task Diffusion Foundation Model for Musculoskeletal MRI Interpretation | arXiv:2602.20752 |
| `.agent-runtime/orthofoundation.pdf` | A multimodal vision foundation model for generalizable knee pathology | — |
| `.agent-runtime/pillar0_paper.pdf` | Pillar-0: A New Frontier for Radiology Foundation Models | arXiv:2511.17803 |
| `.agent-runtime/recht_imagenet.pdf` | Do ImageNet Classifiers Generalize to ImageNet? (Recht et al.) | ICML 2019 |
| `.agent-runtime/zhang2024.pdf`, `zhang2403.pdf` | Self-Supervised Learning for Medical Image Data with Anatomy-Oriented Imaging Planes | arXiv:2403.16499 |
| `mahmood.pdf` | How Much More Data Do I Need? Estimating Requirements for Downstream Tasks | arXiv:2207.01725 |
| `zhang.pdf` | A Generalizable Artificial Intelligence Model for COVID-19 Classification Task Using Chest X-ray Radiographs | — |

## Third-party material fetched rather than tracked

| What | Fetch |
|---|---|
| Public notebooks `hyakuman937/rsna-knee-0937-meniscus-resid-repro`, `renta937/rsna-knee-0-937-weak-label-dinov2-meniscus-resid` (read in `research/deep_dives/`) | `kaggle kernels pull <owner>/<slug> -p research/external/pubkernels/<owner>` |
| fastMRI+ knee bounding-box annotations (was `comp/external/fastmri_knee.csv`) | `knee.csv` from https://github.com/microsoft/fastmri-plus (`Annotations/`) |
| Competition rules (was `comp/COMPETITION_RULES.txt`, a verbatim copy) | https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules |

Git history from before 2026-10-23 still holds these files.
