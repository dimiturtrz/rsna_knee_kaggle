"""Full-row attention entropy over all 257 keys, plus the cls-mass that the patch-only view hides.

The patch-only renormalisation in attn_pair.py divides a near-zero row sum by itself when a patch
query sends nearly all its mass to cls, which manufactures entropy 0 / max_weight 0. This measures
the raw softmax row instead: no renormalisation, nothing dropped, so every number is a property of
the model.
"""

import json
import math
import sys

import numpy as np
import torch
from transformers import AutoModel

sys.path.insert(0, "D:/personal_projects/rsna_knee_kaggle")
from kneemri.evaluation.encoder_drift import EncoderDrift  # noqa: E402

ROOT = "D:/data/volumetric/mri/models/rsna_knee_abnormality/"
CKPTS = [
    "adapt_null",
    "adapt_small_e6_lr1e5",
    "adapt_small_e6",
    "adapt_small_e8_unitfix",
    "pretrain_small_matched_e8",
]
IMAGES = "D:/data/volumetric/mri/processed/rsna_knee_abnormality/images_224px_12sl_first60.npy"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
SLICE_STRIDE = 3
SINK = 0.99


def batch() -> torch.Tensor:
    array = np.load(IMAGES, mmap_mode="r")
    planes = np.asarray(array[:, array.shape[1] // 2, ::SLICE_STRIDE])
    planes = planes.reshape(-1, planes.shape[-2], planes.shape[-1]).astype(np.float32) / 255.0
    x = torch.from_numpy(planes).unsqueeze(1).repeat(1, 3, 1, 1)
    return (x - MEAN) / STD


def profile(model: AutoModel, x: torch.Tensor) -> dict[str, list[float]]:
    entropy, cls_mass, sunk = [], [], []
    with torch.no_grad():
        for chunk in torch.split(x, 24):
            out = model(pixel_values=chunk, output_attentions=True)
            for layer, attn in enumerate(out.attentions):
                rows = attn[:, :, 1:, :]
                e = float(-(rows * rows.clamp_min(1e-12).log()).sum(dim=-1).mean())
                c = rows[..., 0]
                if layer >= len(entropy):
                    entropy.append([]), cls_mass.append([]), sunk.append([])
                entropy[layer].append(e)
                cls_mass[layer].append(float(c.mean()))
                sunk[layer].append(float((c > SINK).float().mean()))
    return {
        "entropy_257": [round(sum(v) / len(v), 3) for v in entropy],
        "cls_mass": [round(sum(v) / len(v), 4) for v in cls_mass],
        "rows_sunk": [round(sum(v) / len(v), 4) for v in sunk],
    }


x = batch()
repo = EncoderDrift(ROOT + CKPTS[0] + ".pt").repository()
report = {
    "repository": repo,
    "n_images": int(x.shape[0]),
    "uniform_entropy_257": round(math.log(257), 3),
    "sink_threshold": SINK,
    "STOCK": profile(AutoModel.from_pretrained(repo, attn_implementation="eager").eval(), x),
}

for name in CKPTS:
    model = AutoModel.from_pretrained(repo, attn_implementation="eager").eval()
    missing, unexpected = model.load_state_dict(EncoderDrift(ROOT + name + ".pt").trained(), strict=False)
    entry = profile(model, x)
    entry["keys"] = [len(missing), len(unexpected)]
    report[name] = entry

print("@@@" + json.dumps(report))
