# Merging external knee MRI corpora into one training scheme

Date: 2026-09-02  
Scope: RSNA 2024 Knee MRI multi-label competition + fastMRI knee + MRNet + Rijeka  
Source audit: `D:/work/ai` (audio multi-dataset/domain-adaptation recipes)

## What the audio project does that maps cleanly

`data/processed/datamix.py` keeps a single metadata table keyed by `(dataset_name, chunk_path, sample_id)`.
Every row carries a `processed_dataset_name` token that lets the loader apply dataset-specific transforms
without branching the training loop. The same pattern is the right starting point here:

- One `studies.parquet` with a `corpus` column: `rsna`, `fastmri_knee`, `mrnet`, `rijeka`.
- Dataset-specific decode paths stored per row, not hard-coded in the dataloader.
- Label schema unified to the 12 RSNA findings; missing findings are masked in loss, not filled.

## Training schemes: two is enough

The audio docs converge on two-phase training and mixed-domain batches. We do not need a separate recipe
per corpus.

### Scheme A — mixed pre-training on all corpora

Train a trunk + a small projection head on all available knee MRI slices/volumes, with the target head
attached only when labels exist. This is the default path to exploit unlabeled or partially-labeled data.

Key ingredients, translated from the audio recipes:

1. **Mixed batches with target-domain oversampling.**
   The audio recipe uses `dataset_sample_weights` to up-weight the target domain so that roughly half the
   batch is target-domain. For us: sample RSNA studies at ~2-3× the rate of fastMRI/MRNet/Rijeka slices
   in pre-training, then drop the weights during fine-tuning.

2. **Masked loss on partial labels.**
   MRNet and Rijeka do not label all 12 findings. The audio `partial_label` recipe simply masks the
   missing columns in BCE loss rather than treating them as negatives. This is standard but easy to miss;
   it lets us include partial corpora without corrupting the negatives.

3. **Domain-invariant normalization / augmentation.**
   The audio project uses Freq-MixStyle to swap low-frequency spectrogram statistics between domains. Our
   equivalent is intensity/MixStyle on MRI patches: swap channel-wise intensity statistics between RSNA
   and fastMRI slices during pre-training. First implementation: per-slice z-score → MixStyle on the
   feature maps after the first trunk block. Cheaper alternative: instance normalization + random
   intensity/gamma augmentation.

4. **Two-phase curriculum: synthetic/large → real/small.**
   The audio project pre-trains first on the larger synthetic/noisier corpus, then fine-tunes on the
   smaller clean corpus. Here: pre-train on fastMRI + MRNet + Rijeka (larger volume, weaker labels), then
   fine-tune on RSNA labeled studies. If fastMRI continues to underperform as a continuation (see
   `2026-09-02_fastmri_continuation_arithmetic.md`), drop it from pre-training and keep MRNet + Rijeka.

5. **Optional domain-adversarial or MMD alignment.**
   The audio docs try DANN, MMD, and CDAN with class-conditional alignment. For MRI this is a second-order
   bet: implement only if Scheme A without it still leaves a visible domain gap on a small validation set
   of RSNA vs MRNet. MMD on the trunk output is the cheapest to add and remove.

### Scheme B — RSNA-only fine-tune

After Scheme A, run a short RSNA-only fine-tune with the same recipe as the current best members. This is
required because mixed training changes the head statistics; the final board entry must be read after this
phase.

## Data volumes and what they actually add

| Corpus | Approximate scale | Labels | Use in Scheme A |
|---|---|---|---|
| fastMRI knee | ~1,250 volumes, ~35k slices | mostly unlabeled | continuation / pre-training volume |
| MRNet | ~1,370 studies, 3 planes each | 3 findings (ACL, meniscus, cartilage) | partial-label pre-training |
| Rijeka | smaller, local (see `D:/data/volumetric/mri/raw`) | variable | partial-label pre-training |
| RSNA labeled | ~4,400 studies, 12 findings | full 12-label | target domain, fine-tune |

The gain from external data is unlikely to come from raw volume alone; RSNA already dominates labeled
volume. The lever is **diversity of acquisition and annotation** — different scanners, planes, and label
sources regularize the trunk against the RSNA test-set acquisition drift.

## What we should build first

1. A single `studies.parquet` with the `corpus` token and a per-corpus decode path.
2. Masked BCE loss that accepts a `label_mask` per study.
3. Mixed sampler with target-domain oversampling.
4. One `mixed_pretrain` arm that starts from our best frozen trunk and trains only the projection/head
   for a few epochs, then unfreezes.
5. Validation: measure RSNA holdout AUC vs RSNA-only baseline and vs each external corpus alone.

## What we should NOT build yet

- Separate training schemes per corpus. The whole point is to unify.
- Domain-adversarial alignment before proving a domain gap exists in our mixed batches.
- Pseudo-labeling fastMRI for the 12 RSNA findings. The label-noise literature review found the
  unlabeled:labeled ratio is too low (~1.2:1) for pseudo-labels to be the main lever.

## Open questions for the next pass

- Does fastMRI knee decoded at our slice thickness help, or does it hurt because of the continuation
  arithmetic problem already measured? A quick ablation is worth more than more literature.
- What is the exact plane/resolution split in Rijeka? If it is mostly sagittal, it complements MRNet's
  three-plane design less than expected.
- Should the mixed pre-training use volume-level or slice-level batches? Our current members are
  slice-level; a volume-level mixed loader would need a separate experiment.
