"""Fit one member: `python -m kneemri.train --fold 0 --epochs 8`.

Writes a checkpoint in the same shape the published package uses — weights under `model` beside the
config, the fold scores and a fingerprint — so our own members and theirs load through one code path.
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from core.paths import DataRoot
from kneemri.backbone import ENCODERS, SMALL
from kneemri.folds import ReportFolds
from kneemri.model import TARGETS, StudyDesign, StudyModel
from kneemri.scores import LabelScores
from kneemri.slot_reader import CROP_MM, IMAGE_SIZE, SLICE_BAND
from kneemri.slots import SLOTS
from kneemri.study_cache import StudyCache
from kneemri.targets import CONFIDENCE_FLOOR, Targets
from kneemri.trainer import UNFREEZE_LAST, AugmentConfig, Trainer

GROUP = 3

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingOptions:
    """Everything one run varies, parsed once.

    The parser lived in `main`, where it grew a statement per flag and pushed the function over the
    complexity gate twice in one session. A flag is configuration, not control flow, so it belongs in a
    type — and a type can then be constructed in a test without a command line.
    """

    fold: int = 0
    folds: int = 5
    epochs: int = 8
    batch: int = 12
    slices: int = 12
    seed: int = 2026
    split_seed: int = 2026
    one_cycle: bool = False
    intensity_aug: bool = False
    weight_floor: bool = False
    normalise_by_count: bool = False
    slice_pooling: str = "channels"
    anatomical_prior: bool = False
    variant: str = SMALL
    labels: Path = Path("labels.csv")
    eval_labels: Path | None = None
    out: Path | None = None
    paths: Path = Path("paths.yaml")

    @classmethod
    def parse(cls) -> "TrainingOptions":
        """Read the command line into an options object."""
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--fold", type=int, default=0)
        parser.add_argument("--folds", type=int, default=5)
        parser.add_argument("--epochs", type=int, default=8)
        parser.add_argument("--batch", type=int, default=12)
        parser.add_argument("--slices", type=int, default=12)
        parser.add_argument("--seed", type=int, default=2026, help="model init and data order ONLY")
        parser.add_argument(
            "--split-seed",
            type=int,
            default=2026,
            help="fold assignment. SEPARATE from --seed on purpose: deriving the split from the "
            "training seed gives every seed a different holdout, so two runs are not comparable and "
            "cross-scoring them leaks training data.",
        )
        parser.add_argument("--one-cycle", action="store_true", help="attach a OneCycleLR schedule")
        parser.add_argument("--intensity-aug", action="store_true", help="gamma/blur/contrast/noise too")
        parser.add_argument("--weight-floor", action="store_true", help="an unmentioned finding pulls at 0.25")
        parser.add_argument("--normalise-by-count", action="store_true", help="loss .mean() not /weight-sum")
        parser.add_argument(
            "--slice-pooling",
            choices=("channels", "max"),
            default="channels",
            help="how a group of slices becomes one feature. 'channels' stacks three slices as RGB, "
            "which couples a group's physical thickness to the slice count; 'max' encodes each slice "
            "on its own and takes the elementwise max, which does not, and is what the .903 reference "
            "does. 'max' costs three encoder passes per group.",
        )
        parser.add_argument("--labels", type=Path, required=True, help="label table to TRAIN on")
        parser.add_argument(
            "--eval-labels",
            type=Path,
            default=None,
            help="table to SCORE against (default: --labels). Hold this fixed across arms when the "
            "training labels are the variable, or the arms are measured with different rulers.",
        )
        parser.add_argument(
            "--anatomical-prior",
            action="store_true",
            help="start each diagnosis's slot attention from how the finding is actually read",
        )
        parser.add_argument(
            "--variant",
            choices=tuple(ENCODERS),
            default=SMALL,
            help="DINOv2 size. base is 86M parameters and 768-wide against small's 22M and 384.",
        )
        parser.add_argument("--out", type=Path, default=None)
        parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
        return cls(**vars(parser.parse_args()))


def main() -> None:
    arguments = TrainingOptions.parse()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    torch.manual_seed(arguments.seed)
    generator = np.random.default_rng(arguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = DataRoot.from_config(arguments.paths)
    cache = StudyCache(root.processed(), slices=arguments.slices)
    corpus, identifiers = cache.load()
    targets, evaluation = Targets.pair(
        arguments.labels,
        arguments.eval_labels,
        root.raw() / "train.csv",
        identifiers,
        CONFIDENCE_FLOOR if arguments.weight_floor else 0.0,
    )
    # The split is a property of the CORPUS, not of this run: it must not move when the model seed
    # does. Deriving it from --seed gave seed 7 a fold-0 holdout sharing only 192 of 909 studies with
    # seed 2026's, so their holdout numbers were never comparable and scoring one on the other's split
    # was scoring a model on 717 studies it had trained on.
    folds = ReportFolds(arguments.folds, arguments.split_seed)
    training, holdout = folds.holdout(folds.assign(str(root.raw() / "train.csv"), identifiers), arguments.fold)
    logger.info(f"{len(identifiers)} studies: {len(training)} train / {len(holdout)} holdout (fold {arguments.fold})")
    logger.info("target coverage:", {k: round(v, 3) for k, v in targets.coverage().items()})

    design = StudyDesign(slice_pooling=arguments.slice_pooling, anatomical_prior=arguments.anatomical_prior)
    encoder = ENCODERS[arguments.variant]
    model = StudyModel(AutoModel.from_pretrained(encoder.repository), encoder.dim, len(SLOTS), design)
    scores = LabelScores()
    trainer = Trainer(
        model,
        device,
        augment_config=AugmentConfig() if arguments.intensity_aug else AugmentConfig.geometric_only(),
        normalise_by_count=arguments.normalise_by_count,
    )
    if arguments.one_cycle:
        # ceil, not floor: our batcher KEEPS the final partial batch, so a floor undercounts by one
        # step per epoch and OneCycleLR raises on the very last step of the run — after every epoch has
        # been logged and before the checkpoint is written, which is the most expensive place to fail.
        steps_per_epoch = -(-len(training) // arguments.batch)
        trainer.one_cycle(arguments.epochs * max(steps_per_epoch, 1))

    destination = arguments.out or (root.processed() / f"member_fold{arguments.fold}.pt")
    best, best_state = -1.0, None
    for epoch in range(arguments.epochs):
        started = time.time()
        mean_loss = trainer.fit_epoch(corpus, targets, training, arguments.batch, generator)
        predictions = trainer.predict(corpus, holdout, arguments.batch, generator)
        per_label = scores.per_label(predictions, evaluation.scores[holdout])
        mean = float(np.nanmean(list(per_label.values())))
        logger.info(f"epoch {epoch}: loss {mean_loss:.4f}  holdout mean AUC {mean:.4f}  ({time.time() - started:.0f}s)")
        logger.info("    %s", {name: round(value, 3) for name, value in per_label.items()})
        if mean > best:
            best = mean
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            trainer.save_partial(destination, best_state, best, epoch)

    model.load_state_dict(best_state)  # pyrefly: ignore[bad-argument-type]
    torch.save(
        {
            "model": best_state,
            "holdout": best,
            "annot": float("nan"),
            "epochs_done": arguments.epochs,
            # The recipe, not just the image geometry. Without this a checkpoint cannot say which arm
            # produced it: two seed runs were left on disk whose one-cycle/floor/label flags are now
            # unrecoverable, so they cannot be used as a noise floor for anything.
            # str(), not the dataclass as-is: every loader here reads with weights_only=True, which
            # refuses to unpickle a Path, so saving the options raw would make each new checkpoint
            # unloadable by our own code.
            "options": {name: str(value) for name, value in asdict(arguments).items()},
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
                "variant": encoder.variant,
                "pool": "cls_mean",
                "slice_pooling": arguments.slice_pooling,
                "anatomical_prior": arguments.anatomical_prior,
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
