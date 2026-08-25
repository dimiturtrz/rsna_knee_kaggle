"""Fit one member: `python -m kneemri.training.train --fold 0 --epochs 8`.

Writes a checkpoint in the same shape the published package uses — weights under `model` beside the
config, the fold scores and a fingerprint — so our own members and theirs load through one code path.

THE FLAGS live in `options.py` and WHAT THEY BUILD lives in `run_setup.py`. This module is the run
itself: read the corpus, build the model, check the switches landed, fit, save.
"""

import json
import logging

import torch

from core.data.corpus.folds import ReportFolds
from core.data.corpus.slots import SCHEMES
from core.data.labels.targets import Targets
from core.paths import DataRoot
from kneemri.models.backbone import ENCODERS
from kneemri.models.model import StudyDesign
from kneemri.runs.member_record import MemberRecord
from kneemri.training.fitting import Fitting, Fold, RunPlan
from kneemri.training.landed import Landed
from kneemri.training.options import TrainingOptions
from kneemri.training.run_setup import RunSetup

logger = logging.getLogger(__name__)


def main() -> None:
    arguments = TrainingOptions.parse()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # The MODEL's initialisation. The data order is seeded from the same value inside `Fitting`, which
    # is where the only consumer of it lives.
    torch.manual_seed(arguments.seed)

    # THE CACHE BEFORE THE CARD. A reading nothing has decoded is a `FileNotFoundError` on a memory map,
    # and taking the device first meant an arm held the GPU to discover it — the queue slot is spent
    # either way, but the card need not be. Mapping is cheap: no pixels are read here.
    root = DataRoot.from_config(arguments.paths)
    cache = arguments.cache(root.processed())
    corpus, identifiers = cache.load()
    # The slice count as well as the flag: the reserved peak is a function of it, so an arm that names
    # no --vram-gb still gets a cap fitted to what it will actually hold (knee-ylp0). The SIZE too, or
    # the cap is a 336 px number handed to a run that encodes a bigger grid — the same `train_size or
    # image_size` the record and the members already read, so the cap sees the size the step does.
    # And the ENCODER, or the cap is a `small`-fitted number handed to a wider one: that is what killed
    # v3base_band on 2026-08-20, capped at 12.68 GB with 17.51 free. The encoder answers with ONE
    # number rather than a patch and a width, because a trunk whose blocks do not all see the same grid
    # cannot be described by that pair — which is what capped cnx_dense at 2.84 GB with 28 free.
    # And how many blocks SUPERVISION OPENED, or the cap is a six-block number handed to a run that
    # opened twelve: that is what refused encoder_adapt_thawed on 2026-08-21 at 9.56 GB with 9.55 free.
    # All of it assembled in `RunSetup.budget`, because the QUEUE needs the same answer before it will
    # start an arm and was using a constant for it.
    setup = RunSetup(arguments)
    device = setup.budget().select()
    targets, evaluation = Targets.pair(
        arguments.labels,
        arguments.eval_labels,
        root.raw() / "train.csv",
        identifiers,
        arguments.supervision,
    )
    # The split is a property of the CORPUS, not of this run: it must not move when the model seed
    # does. Deriving it from --seed gave seed 7 a fold-0 holdout sharing only 192 of 909 studies with
    # seed 2026's, so their holdout numbers were never comparable and scoring one on the other's split
    # was scoring a model on 717 studies it had trained on.
    folds = ReportFolds(arguments.folds, arguments.split_seed)
    training, holdout = folds.holdout(folds.assign(str(root.raw() / "train.csv"), identifiers), arguments.fold)
    positions = arguments.positions
    training, holdout = positions.sampled(training, holdout)
    training = positions.thinned(training)
    for caveat in positions.caveats(training):
        logger.warning(caveat)
    # ONE call, and %s rather than a second positional: logging treats extra positionals as %-format
    # arguments, so `logger.info("target coverage:", {...})` printed the label and silently dropped the
    # dict. Every run so far has logged the words "target coverage:" and none of the coverage.
    logger.info(
        "%d studies: %d train / %d holdout (fold %d) | target coverage: %s",
        len(identifiers),
        len(training),
        len(holdout),
        arguments.fold,
        {k: round(v, 3) for k, v in targets.coverage().items()},
    )

    design = StudyDesign(
        pool=arguments.pool,
        slice_pooling=arguments.slice_pooling,
        slice_regions=arguments.slice_regions,
        through_plane=arguments.through_plane,
        slot_pooling=arguments.slot_pooling,
        anatomical_prior=arguments.anatomical_prior,
        layer=arguments.encoder_layer,
    )
    encoder = ENCODERS[arguments.variant]
    # Before the weights are fetched and the cache is mapped: a read size the patch grid cannot tile is
    # a configuration mistake, and the failure without this is either a crash deep in the encoder or a
    # silent interpolation of the position embeddings onto a grid the weights never saw.
    encoder.require_tiling(arguments.train_size or arguments.image_size)
    model = setup.model(encoder, len(SCHEMES[arguments.scheme]), design, device)
    trainer = setup.trainer(model, device, identifiers, training)
    # BEFORE THE CARD IS SPENT, and this is the last moment it can be asked: from here on the only
    # evidence that a switch was wanted is the config, and the config records the ASK. `low_rank` once
    # matched an encoder's projections by name and wrapped nothing, so an arm trained a plain model
    # under a LoRA recipe and cost a queue slot to discover. Two seconds of attribute reads.
    unlanded = Landed(model, trainer, targets).missing(setup.requested(device))
    if unlanded:
        message = "the run asked for switches the built model does not show: " + "; ".join(unlanded)
        raise ValueError(message)

    destination = arguments.out or (root.processed() / f"member_fold{arguments.fold}.pt")
    record = MemberRecord(arguments, encoder, model, device, [slot.name for slot in SCHEMES[arguments.scheme]])
    plan = RunPlan(
        destination=destination,
        epochs=arguments.epochs,
        batch=arguments.batch,
        patience=arguments.patience,
        seed=arguments.seed,
    )
    kept = Fitting(trainer, record, plan).run(
        Fold(corpus=corpus, targets=targets, evaluation=evaluation, training=training, holdout=holdout)
    )

    model.load_state_dict(kept.state)
    # The SAME assembler the periodic save uses, so a run killed at its best epoch and a run allowed to
    # finish produce checkpoints of identical shape — and the EPOCH is the kept one in both, where the
    # final save used to record the last epoch index and so contradicted its own partial under patience.
    torch.save(record.payload(kept.state, kept.holdout, kept.epoch, kept.witness), destination)
    logger.info(
        "best holdout mean AUC %.4f -> %s | %s",
        kept.holdout,
        destination,
        json.dumps({"fold": arguments.fold, "holdout": kept.holdout}),
    )


if __name__ == "__main__":
    main()
