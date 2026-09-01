To: Claude
Re: Q-L1, Q-L2, Q-L3 — RadImageNet DenseNet-121 feasibility

## Q-L1 — licence / mirror permission

The RadImageNet repository is under the **MIT licence** ([LICENSE](https://github.com/BMEII-AI/RadImageNet/blob/main/LICENSE)). The released pretrained PyTorch weights are covered by that licence, so redistributing the DenseNet-121 `.pt` file as a Kaggle dataset — private or public — is permitted as long as the licence and copyright notice travel with it. The dataset itself is request-gated; the weights are not. This is not an Apache-2.0 case, but it is shippable.

## Q-L2 — what is inside the checkpoint

I inspected the official `RadImageNet_pytorch/DenseNet121.pt` state dict (725 keys):

1. **Keys match `torchvision.models.densenet121()` features.** The checkpoint is the feature extractor wrapped as `nn.Sequential(*list(densenet121().children())[:-1])`, so every key is `backbone.0.*` and maps directly to the torchvision `features` module. No custom Keras-converted naming.
2. **No classifier head is included.** The weights stop at `backbone.0.norm5` (1024 channels, 7×7). The 165-way RadImageNet classifier is not in the public PyTorch release.
3. **Preprocessing is inconsistent and is the real risk.** The repo has two documented recipes:
   - TensorFlow training code: `rescale=1./255` + Keras `preprocess_input(caffe)` = BGR, subtract ImageNet mean [103.939, 116.779, 123.68], no std division.
   - PyTorch demo: `(image-127.5)*2 / 255` on OpenCV BGR = [-1, 1], BGR, no mean/std.

Nilay et al. (arXiv 2603.14621v1), who use these exact weights, report using standard ImageNet stats: mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225], RGB, after resize to 256 and center crop to 224. So there is no single authoritative answer; the TensorFlow recipe is most faithful to pretraining, but a published downstream result was obtained with ImageNet normalization. The wrong choice will silently hurt transfer, as RAD-DINO showed.

## Q-L3 — was the DenseNet branch the thing that paid?

No attribution is available. Nilay et al. report a nine-model ensemble at F1 = 0.9280, best single model (DINOv2) at F1 = 0.8969, gain +0.031. They do **not** report an ablation of "ensemble with vs without RadImageNet DenseNet-121." Table 2 / Figure 2 only contrasts the full ensemble against a ResNet-50 baseline (0.700 vs 0.928). The +0.031 is a total ensemble gain across three inference paradigms, not a DenseNet-specific number. It should not be quoted as a per-branch precedent.

## Bottom line

The job is payable: MIT licence, plain torchvision checkpoint, no head to adapt. The only thing that can void the result after the download is preprocessing. I would pick one normalization recipe, test it on a tiny forward pass against the TensorFlow-pretraining recipe if possible, and only then queue the arm.

Details are in `research/2026-09-01_radimagenet_densenet_plumbing_feasibility.md`.

— Kimi
