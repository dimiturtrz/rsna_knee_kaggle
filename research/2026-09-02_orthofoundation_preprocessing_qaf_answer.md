# OrthoFoundation-L preprocessing — answer to Claude's Q-AF

Source: direct inspection of the released repository `ytrsk/OrthoFoundation` and the local checkpoint `D:/data/volumetric/mri/reference/orthofoundation/OrthoFoudation-L.pth`.

## 1. Input normalization

**Mean/std: ImageNet LVD-1689M constants, unchanged.**

- `datasets/augmentation.py` sets:
  ```python
  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
  ```
- `datasets/knee_multidata.py` repeats the same constants and applies them in `self.trans_img`.
- There is **no per-volume or per-slice intensity normalization** in the released pretraining code; the model consumes RGB PNG slices normalized with the standard DINOv3 constants.

Citation: [OrthoFoundation `datasets/augmentation.py`](https://github.com/ytrsk/OrthoFoundation/blob/main/datasets/augmentation.py), [OrthoFoundation `datasets/knee_multidata.py`](https://github.com/ytrsk/OrthoFoundation/blob/main/datasets/knee_multidata.py).

## 2. Input resolution and patch grid

**Continued pretraining resolution: 224 × 224, patch 16 (14 × 14 grid).**

- `knee_multidata.py` defaults to `self.image_size = kwargs.get("size", 224)`.
- `DataAugmentationDINO` in `augmentation.py` uses `transforms.RandomResizedCrop(224)` for both global crops.
- The DINOv3-L backbone is loaded with `backbone: vitl16` in `exp/pretrain/dinov3/full/config.yaml`; DINOv3 ViT-L/16 defaults to `img_size=224` and `patch_size=16`.
- The local checkpoint contains `backbone.patch_embed.proj.weight` of shape `[1024, 3, 16, 16]`, confirming patch 16 and 1024-dim ViT-L.

The paper README does not contradict this; the released checkpoint is therefore a 224 px continuation. If your probe loads at 336 px, the position embeddings are being interpolated away from the continuation resolution.

## 3. Register / storage tokens and student vs teacher

**4 register tokens survive unchanged.**

- The checkpoint key `backbone.storage_tokens` has shape `[1, 4, 1024]`.
- DINOv3's architecture exposes these as `x_storage_tokens`; the MODEL_CARD explicitly lists 4 register tokens for ViT-L. This is the same register count as DINOv2-with-registers.

**Released checkpoint is the student backbone (inferred).**

- The training config defines both `teacher_model` and `student_model` with tag `dinov3_vit`.
- `process/processes.py` saves a backbone-only checkpoint for every training `model_process` as:
  ```python
  torch.save(self.model.module.backbone.state_dict(), path.replace(self.model_name, self.model_name+"backbone"))
  ```
- Both processes share `model_name = "dinov3_vit"`, so both would write the same filename `dinov3_vitbackbone_epoch_*.pth`. The config lists `teacher_model` first and `student_model` second; the student save overwrites the teacher save.
- The released `OrthoFoudation-L.pth` contains only `backbone.*` keys with no teacher/student prefix, consistent with that backbone-only save path.
- Conclusion: treat the released weights as the **student backbone**.

## Bottom line for `ortho_probe`

Feed it:
- **224 px input** (or accept that 336 px is an interpolation experiment)
- **mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)**
- **4 register/storage tokens** enabled
- weights are the **student** of the DINO student-teacher continuation

If the probe still underperforms against DINOv3-L under these settings, the gap is real and not a preprocessing mismatch.
