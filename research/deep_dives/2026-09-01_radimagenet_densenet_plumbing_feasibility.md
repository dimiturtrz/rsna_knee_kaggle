# RadImageNet DenseNet-121 plumbing feasibility

Context: Claude asked whether the RadImageNet DenseNet-121 plumbing job is payable, i.e. whether the weights can be mirrored to Kaggle, what the checkpoint actually contains, and whether Nilay et al. attribute the ensemble gain to the RadImageNet branch.

## Q-L1 — licence and mirror permission

The RadImageNet GitHub repository ships under the **MIT licence** ([BMEII-AI/RadImageNet LICENSE](https://github.com/BMEII-AI/RadImageNet/blob/main/LICENSE)). The MIT licence permits redistribution, including as a Kaggle dataset, provided the licence and copyright notice are included. The dataset itself is request-gated, but the released pretrained PyTorch weights are the MIT-licensed asset. A private or public Kaggle mirror of the DenseNet-121 `.pt` file is therefore permitted in principle.

One practical caveat: the PyTorch weights are distributed through a Google Drive link in the repository README, not through a stable API or HuggingFace hub. A mirror would need to carry the weights file plus the MIT licence, and the Kaggle dataset description should state the source and licence to satisfy Kaggle rule 2.6a.

## Q-L2 — what is inside the PyTorch checkpoint

I downloaded `RadImageNet_pytorch/DenseNet121.pt` (the official Google Drive release) and inspected its state dict.

- **Architecture match.** The checkpoint is a plain `torchvision.models.densenet121` feature extractor wrapped as `nn.Sequential(*list(densenet121().children())[:-1])`. All keys are prefixed `backbone.0.*` and map one-to-one onto the `features` module of `torchvision.models.densenet121(weights=None)`. There are 725 entries, all feature-block weights and BatchNorm buffers.
- **Classifier head.** There is **no classifier head** in the `.pt` file. The released weights stop at the final feature norm (`backbone.0.norm5`). The original RadImageNet pretraining task had 165 pathologic labels, but the public PyTorch release does not include the 165-way classifier.
- **Input preprocessing.** This is the weak point. The repository contains two inconsistent recipes:
  1. TensorFlow training code: `rescale=1./255` followed by Keras `preprocess_input` in `caffe` mode — i.e. BGR channel order, subtract ImageNet mean [103.939, 116.779, 123.68] with no division by std.
  2. PyTorch demo in the repository: `image = (image-127.5)*2 / 255` using OpenCV (`cv2.imread` returns BGR), which scales to [-1, 1] with BGR order and no per-channel mean/std subtraction.

Nilay et al. (arXiv 2603.14621v1), who use the same RadImageNet DenseNet-121 weights, report normalizing with standard ImageNet statistics: mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225], applied after resizing to 256 and center-cropping to 224. They presumably handle RGB/BGR ordering by loading with a PIL-based pipeline.

Bottom line: there is no single documented preprocessing that is obviously authoritative. The weights themselves were trained in TensorFlow, so the TensorFlow recipe is the most faithful to pretraining. The safest practical path is to treat the PyTorch release as a feature extractor and test both (a) TensorFlow-style BGR mean subtraction on [0,1] and (b) ImageNet-style RGB normalization, because Nilay et al. obtained a reported result with the latter. The wrong normalization will silently degrade transfer, exactly as the project saw with RAD-DINO.

## Q-L3 — DenseNet branch attribution in Nilay et al.

Nilay et al., *A Heterogeneous Ensemble for Multi-Center COVID-19 Classification from Chest CT Scans* (arXiv 2603.14621v1, 2026), reports:

- Best single model: DINOv2, F1 = 0.8969.
- Final nine-model ensemble: F1 = 0.9280.
- Gain over best single: +0.031.

There is **no per-branch ablation** that removes the RadImageNet DenseNet-121 branch. The paper compares the full ensemble against individual models and against a ResNet-50 baseline (Figure 2 / Table 2 shows Baseline 0.700 vs Ensemble 0.928), but it does not report "ensemble without DenseNet" or isolate the DenseNet contribution. The +0.031 is a headline ensemble gain, not an attributable DenseNet gain. This is exactly the kind of unattributed number the project has previously quoted as a per-branch precedent; it should be treated as a total, not a DenseNet-specific one.

## Recommendation

The plumbing is payable, but only if the preprocessing uncertainty is resolved before the card run. The licence is clean, the checkpoint is a plain torchvision DenseNet-121 feature extractor with no classifier to remap, and the weights can be mirrored. The main risk is the same as RAD-DINO: feeding the trunk a normalization it was not trained on. If the project proceeds, I would pre-register a preprocessing choice and verify it with a small sanity forward pass before starting the full arm.
