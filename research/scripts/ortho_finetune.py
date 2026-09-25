"""OrthoDiffusion Stage II: fine-tune the three plane denoisers' encoder halves end to end (knee-mxz08).

Their recipe (`external/OrthoDiffusion/finetune_classifier.py`): read the noised volume through the
input + middle blocks, GAP the chosen middle block, linear head; trunk params at lr x 0.05, Adam,
cosine, BCE, 5 epochs. Here all six slots feed ONE head (late fusion folded into the linear layer), a
fresh noise draw per step is the augmentation, and eval uses the fixed NOISE_SEED draw.
"""

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from core.data.corpus.cache_flags import CacheFlags
from core.data.corpus.folds import ReportFolds
from core.data.corpus.slots import SLOTS
from core.data.labels.ruler import DEFAULT_LABELS
from core.data.labels.targets import Targets
from core.findings import TARGETS
from core import cli
from kneemri.models.ortho_diffusion import MIDDLE_WIDTH, NOISE_SEED, PLANE_READS, OrthoDiffusionTrunk
from kneemri.models.ortho_weights import DenoiserWeights
from kneemri.runs.holdout_record import HoldoutRecord

logger = logging.getLogger(__name__)
PLANES = sorted({slot.plane for slot in SLOTS})


class OrthoClassifier(nn.Module):
    def __init__(self, trunk: OrthoDiffusionTrunk) -> None:
        super().__init__()
        self.trunk = trunk
        self.head = nn.Linear(len(SLOTS) * MIDDLE_WIDTH, len(TARGETS))

    def forward(self, pixels: torch.Tensor, mask: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        batch = len(pixels)
        features = torch.zeros(batch, len(SLOTS), MIDDLE_WIDTH, device=pixels.device)
        for plane in PLANES:
            index = [i for i, slot in enumerate(SLOTS) if slot.plane == plane]
            read = PLANE_READS[plane]
            volume = self.trunk.prepared(pixels[:, index].flatten(0, 1))
            noisy = self.trunk.schedule.noised(volume, read.timestep, generator)
            steps = torch.full((len(noisy),), read.timestep, dtype=torch.long, device=pixels.device)
            _, info = self.trunk.denoisers[plane](noisy, steps, ret_mid=True)
            pooled = self.trunk.pooled(info["mid"][read.block].float()).view(batch, len(index), -1)
            features[:, index] = pooled
        return self.head((features * mask.unsqueeze(-1)).flatten(1))


def unfrozen(trunk: OrthoDiffusionTrunk) -> list[nn.Parameter]:
    chosen = []
    for name, parameter in trunk.denoisers.named_parameters():
        wanted = "input" in name or "middle" in name
        parameter.requires_grad_(wanted)
        if wanted:
            chosen.append(parameter)
    return chosen


def predict(
    model: OrthoClassifier,
    images: np.ndarray,
    mask: np.ndarray,
    rows: np.ndarray,
    batch: int,
    device: torch.device,
    *,
    precise: bool,
) -> np.ndarray:
    model.eval()
    out = []
    generator = torch.Generator(device=device).manual_seed(NOISE_SEED)
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            chunk = np.sort(rows[start : start + batch])
            pixels = torch.from_numpy(np.ascontiguousarray(images[chunk])).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precise is False):
                logits = model(pixels, torch.from_numpy(mask[chunk]).to(device), generator)
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def auc(scores: np.ndarray, truth: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    values = [
        roc_auc_score(truth[:, k] > 0.5, scores[:, k])
        for k in range(truth.shape[1])
        if 0 < (truth[:, k] > 0.5).sum() < len(truth)
    ]
    return float(np.mean(values))


def main() -> None:
    command = cli.Command(__doc__, logger)
    CacheFlags.add_to(command.parser)
    command.parser.add_argument("--epochs", type=int, default=6)
    command.parser.add_argument("--batch", type=int, default=4)
    command.parser.add_argument("--lr", type=float, default=5e-4)
    command.parser.add_argument("--trunk-scale", type=float, default=0.05)
    command.parser.add_argument("--limit", type=int, default=0, help="smoke: studies per split")
    command.parser.add_argument("--fp32", action="store_true")
    command.parser.add_argument("--vram-gb", type=float, default=22.0, help="hard cap: the card is shared")
    command.parser.add_argument("--out", type=Path, required=True)
    arguments: argparse.Namespace = command.parsed()

    rooted = command.rooted(arguments)
    corpus, identifiers = CacheFlags.cache(arguments, rooted.processed()).load()
    folds = ReportFolds()
    training, holdout = folds.holdout(folds.assign(str(rooted.raw() / "train.csv"), identifiers), 0)
    training, holdout = np.sort(training), np.sort(holdout)
    if arguments.limit:
        training, holdout = training[: arguments.limit], holdout[: arguments.limit // 4]
    graded = Targets.load(
        rooted.processed() / "labels" / DEFAULT_LABELS, rooted.raw() / "train.csv", identifiers
    ).scores
    graded = np.asarray(graded, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            arguments.vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory
        )
    trunk = DenoiserWeights(rooted.kind("external") / "orthodiffusion").trunk(device)
    model = OrthoClassifier(trunk).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": unfrozen(trunk), "lr": arguments.lr * arguments.trunk_scale},
            {"params": model.head.parameters(), "lr": arguments.lr},
        ]
    )
    steps_per_epoch = math.ceil(len(training) / arguments.batch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, arguments.epochs * steps_per_epoch)
    rng = np.random.default_rng(2026)
    best = -1.0
    for epoch in range(arguments.epochs):
        model.train()
        started, total = time.time(), 0.0
        shuffled = rng.permutation(training)
        for step, start in enumerate(range(0, len(training), arguments.batch)):
            chunk = np.sort(shuffled[start : start + arguments.batch])
            pixels = torch.from_numpy(np.ascontiguousarray(corpus.images[chunk])).to(device)
            target = torch.from_numpy(graded[chunk]).to(device)
            known = ~torch.isnan(target)
            # Their blocks re-run forward inside backward (gradient checkpointing), so autocast has to span it.
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not arguments.fp32):
                logits = model(pixels, torch.from_numpy(corpus.mask[chunk]).to(device), None).float()
                loss = functional.binary_cross_entropy_with_logits(logits[known], target[known])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
            broken = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ]
            if broken:
                logger.info(
                    f"  non-finite grads in {len(broken)}: {broken[:3]} rows {chunk.tolist()} target {target.tolist()}"
                )
            optimizer.step()
            scheduler.step()
            total += float(loss)
            if step % 50 == 0 or not math.isfinite(float(loss)):
                reserved = torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else 0.0
                logger.info(
                    f"  step {step}/{steps_per_epoch} loss {float(loss):.4f} {reserved:.1f} GiB known {int(known.sum())} logit_nan {int(torch.isnan(logits).sum())}"
                )
        scores = predict(
            model, corpus.images, corpus.mask, holdout, arguments.batch * 2, device, precise=arguments.fp32
        )
        truth = np.nan_to_num(graded[holdout])
        value = auc(scores, truth)
        logger.info(
            f"epoch {epoch}: loss {total / steps_per_epoch:.4f}  holdout mean AUC {value:.4f}  ({time.time() - started:.0f}s)"
        )
        if value > best:
            best = value
            HoldoutRecord(arguments.out.with_suffix(".holdout.npz")).write(
                HoldoutRecord.named(identifiers, holdout), scores, graded[holdout], epoch
            )
            torch.save({"head": model.head.state_dict(), "trunk": trunk.state_dict(), "epoch": epoch}, arguments.out)
    logger.info(f"ORTHO FINETUNE DONE best {best:.4f}")


if __name__ == "__main__":
    main()
