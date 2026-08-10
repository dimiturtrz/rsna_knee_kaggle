"""Fit one member: `python -m kneemri.train --fold 0 --epochs 8`.

Writes a checkpoint in the same shape the published package uses — weights under `model` beside the
config, the fold scores and a fingerprint — so our own members and theirs load through one code path.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from core.paths import DataRoot
from kneemri.folds import ReportFolds
from kneemri.model import TARGETS, StudyModel
from kneemri.slot_reader import CROP_MM, IMAGE_SIZE, SLICE_BAND
from kneemri.slots import SLOTS
from kneemri.study_cache import StudyCache
from kneemri.targets import Targets
from kneemri.trainer import UNFREEZE_LAST, Trainer

DINOV2 = "facebook/dinov2-small"
DIM = 384
GROUP = 3

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--slices", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--one-cycle", action="store_true", help="attach a OneCycleLR schedule")
    parser.add_argument("--labels", type=Path, required=True, help="report-derived label table")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    torch.manual_seed(arguments.seed)
    generator = np.random.default_rng(arguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = DataRoot.from_config(arguments.paths)
    cache = StudyCache(root.processed(), slices=arguments.slices)
    corpus, identifiers = cache.load()
    targets = Targets.load(arguments.labels, root.raw() / "train.csv", identifiers)
    assignments = ReportFolds(arguments.folds, arguments.seed).assign(str(root.raw() / "train.csv"), identifiers)
    training, holdout = ReportFolds(arguments.folds, arguments.seed).holdout(assignments, arguments.fold)
    logger.info(f"{len(identifiers)} studies: {len(training)} train / {len(holdout)} holdout (fold {arguments.fold})")
    logger.info("target coverage:", {k: round(v, 3) for k, v in targets.coverage().items()})

    model = StudyModel(AutoModel.from_pretrained(DINOV2), DIM, len(SLOTS))
    trainer = Trainer(model, device)
    if arguments.one_cycle:
        # ceil, not floor: our batcher KEEPS the final partial batch, so a floor undercounts by one
        # step per epoch and OneCycleLR raises on the very last step of the run — after every epoch has
        # been logged and before the checkpoint is written, which is the most expensive place to fail.
        steps_per_epoch = -(-len(training) // arguments.batch)
        trainer.one_cycle(arguments.epochs * max(steps_per_epoch, 1))

    best, best_state = -1.0, None
    for epoch in range(arguments.epochs):
        started = time.time()
        mean_loss = trainer.fit_epoch(corpus, targets, training, arguments.batch, generator)
        predictions = trainer.predict(corpus, holdout, arguments.batch, generator)
        per_label = trainer.evaluate(predictions, targets.scores[holdout])
        mean = float(np.nanmean(list(per_label.values())))
        logger.info(f"epoch {epoch}: loss {mean_loss:.4f}  holdout mean AUC {mean:.4f}  ({time.time() - started:.0f}s)")
        logger.info("    %s", {name: round(value, 3) for name, value in per_label.items()})
        if mean > best:
            best = mean
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    destination = arguments.out or (root.processed() / f"member_fold{arguments.fold}.pt")
    model.load_state_dict(best_state)  # pyrefly: ignore[bad-argument-type]
    torch.save(
        {
            "model": best_state,
            "holdout": best,
            "annot": float("nan"),
            "epochs_done": arguments.epochs,
            "targets": list(TARGETS),
            "slots": [slot.name for slot in SLOTS],
            "config": {
                "img": IMAGE_SIZE,
                "slices": arguments.slices,
                "group": GROUP,
                "crop_mm": CROP_MM,
                "band": list(SLICE_BAND),
                "slots": [slot.name for slot in SLOTS],
                "rules": {"order": "normal", "lat": "centre", "slot_fallback": False, "decode_fill": "nearest"},
                "variant": "small",
                "pool": "cls_mean",
                "unfreeze_last": UNFREEZE_LAST,
            },
            "fingerprint": model.to(device).fingerprint(IMAGE_SIZE, GROUP, device).tolist(),
            "fingerprint_config": {"img": IMAGE_SIZE, "n_slot": len(SLOTS), "group": GROUP, "seed": 2026},
        },
        destination,
    )
    logger.info(f"best holdout mean AUC {best:.4f} -> {destination}")
    logger.info(json.dumps({"fold": arguments.fold, "holdout": best}))


if __name__ == "__main__":
    main()
