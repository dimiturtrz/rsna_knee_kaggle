"""Serve mattiaangeli's five released fold checkpoints on OUR studies, through THEIR decode.

`python -m kneemri.inference.foreign_folds`.

WHAT THIS ANSWERS: is their published family better than ours, and does it add anything to our seating —
priced on studies each fold never trained on. Nothing here reads a board number: the split
(`kneemri.evaluation.foreign_split`) says which of our studies fold *f* held out, and only those may
carry fold *f*'s AUC.

NOTHING OF THEIRS IS RETYPED. The decode, the model classes and the serving loop are CUT from their
kernel by `foreign_source.ForeignSource` at the line range of the ONE program in that file that serves
the released checkpoints — `SERVING`, below. Their file holds three programs sharing a namespace, and
the training program's `SLICE_BAND` and the second family's `CROP_MM` are both wrong for these weights.
A port that resolved names by exec-then-look-up would feed their model an input their model has never
seen, and the low score would read as "their model is not better than ours".

WHAT IS OURS, AND WHY. Three things their kernel got from its Kaggle mount are injected instead of
lifted: `SERIES_ROOT` (our DICOM root), `models` and `CFG` (filled after the checkpoints load). The
checkpoint construction is a top-level `for` in their file, so it cannot be lifted as a symbol; `nets`
transcribes it and is the only transcription in this module, kept to their twelve lines including the
refusal their two asserts encode. Their `ProcessPoolExecutor` becomes a thread pool, because a lifted module is not
importable in a child process — same calls, same values, no compute restructured.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.findings import STUDY
from kneemri.inference.foreign_source import ForeignModule, ForeignSource

logger = logging.getLogger(__name__)

SERVING = (1600, 2320)
"""The line range of the program the released folds are served under, in their ensembled kernel."""

HEADER = """\
import gc, math, os, time, warnings
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')
cv2.setNumThreads(1)
"""

SYMBOLS = (
    "CROP_MM",
    "SIZE",
    "SLICE_BAND",
    "N_SLICE",
    "INTENSITY",
    "SLOTS",
    "N_SLOT",
    "LABELS",
    "DEV",
    "ordered_files",
    "series_side",
    "read_crop",
    "window",
    "render",
    "build_study",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "N_PLANE",
    "N_CONTRAST",
    "_PLANE_OF",
    "_CONTRAST_OF",
    "N_SLOT_TYPES",
    "MASK_IDX",
    "segment_softmax",
    "_pad_kv",
    "_seg_mean_max",
    "_GatedDelta",
    "_GatedDepthBlock",
    "SlotDepthMixer",
    "DepthCompress",
    "ViTSlotToken",
    "ClsAddPool",
    "MeanMaxPool",
    "LabelAttentionPool",
    "TokenXAttnPool",
    "TokenResidualPool",
    "CodexResidualPool",
    "Readout",
    "Net",
    "AMP_PREF",
    "amp_for",
    "AMP_DT",
    "AMP_ON",
    "CHUNK",
    "MICRO",
    "_norm_",
    "_micro",
    "predict",
)
"""Every symbol their serving program needs, in no particular order — the lift emits them in file order."""

WORKERS = 8
WORKERS_REASON = "their ProcessPoolExecutor, as threads; a lifted module cannot be imported in a child"


@dataclass(frozen=True)
class ForeignFolds:
    """Their released folds, built from their own source and served on our studies."""

    kernel: Path
    checkpoints: Path
    series_root: Path
    symbols: tuple[str, ...] = SYMBOLS
    scope: tuple[int, int] = SERVING
    """The line range of the program being served; a port states which of their three programs it is."""

    def module(self) -> ModuleType:
        """Their serving program as an importable module, with the Kaggle-mount globals bound."""
        lifted = ForeignSource(self.kernel, header=HEADER, scope=self.scope).lift(list(self.symbols)).build()
        ForeignModule.inject(lifted, SERIES_ROOT=self.series_root)
        return lifted

    def nets(self, module: ModuleType) -> list[torch.nn.Module]:  # devtools-ignore: test-mirror
        """Their twelve-line construction, transcribed once, and bound as the `models` their loop reads.

        Transcribed rather than lifted because it is a top-level `for` in their file, not a symbol. Their
        two asserts are kept as one refusal (`assert` is a banned statement here): a missing key outside
        `enc.`, or any unexpected key, means the checkpoint and the lifted classes have drifted apart —
        which must stop the port rather than quietly degrade its score.
        """
        built: list[torch.nn.Module] = []
        cfg: dict[str, object] = {}
        for path in sorted(self.checkpoints.glob("*_f*.pt")):
            state = torch.load(path, map_location="cpu", weights_only=False)
            cfg = dict(state["cfg"])
            backbone = str(cfg["backbone"])
            stem = cfg.get("stem", "native")
            channels = 3 if stem == "compress" else cfg.get("n_slice", 16)
            sizing: dict[str, object] = {"img_size": cfg["img"]} if "vit_" in backbone else {}
            encoder = module.timm.create_model(backbone, pretrained=False, num_classes=0, in_chans=channels, **sizing)
            if cfg["cond"] == "token":
                encoder = module.ViTSlotToken(encoder, module.N_SLOT_TYPES)
            net = module.Net(
                encoder, cfg["cond"], cfg.get("n_meta", 0), cfg["pool"], stem=stem, n_slice=cfg.get("n_slice", 16)
            )
            missing, unexpected = net.load_state_dict(state["state_dict"], strict=False)
            if drifted := [key for key in missing if not key.startswith("enc.")] + list(unexpected):
                message = f"{path.name} and the lifted classes have drifted apart: {drifted[:5]}"
                raise ValueError(message)
            built.append(net.to(module.DEV).eval())
            logger.info("loaded %s fold %s %s pool=%s", path.name, state["fold"], backbone, cfg["pool"])
        ForeignModule.inject(module, models=built, CFG=cfg)
        return built

    @staticmethod
    def records(series: pd.DataFrame, studies: list[str]) -> dict[str, list[dict[str, object]]]:
        """Their `by`: each study's series rows as dicts, which is what their `build_study` reads."""
        wanted = series[series[STUDY].isin(set(studies))]
        return {
            str(study): [{str(key): value for key, value in row.items()} for row in group.to_dict("records")]
            for study, group in wanted.groupby(STUDY)
        }

    def serve(  # devtools-ignore: test-mirror
        self, module: ModuleType, studies: list[str], by: dict[str, list[dict[str, object]]]
    ) -> Float[NDArray[np.float32], "folds studies labels"]:
        """Their chunked loop, per fold, kept SEPARATE — combining folds is the caller's decision, not this one."""
        out = np.full((len(module.models), len(studies), len(module.LABELS)), np.nan, np.float32)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for start in range(0, len(studies), module.CHUNK):
                block = studies[start : start + module.CHUNK]
                images = np.zeros((len(block), module.N_SLOT, module.N_SLICE, module.SIZE, module.SIZE), np.uint8)
                masks = np.zeros((len(block), module.N_SLOT), np.uint8)
                for index, array, mask in pool.map(
                    module.build_study, [(i, s, by.get(s, [])) for i, s in enumerate(block)]
                ):
                    images[index], masks[index] = (array, mask)
                out[:, start : start + len(block)] = module.predict(images, masks)
                logger.info("%d/%d studies", start + len(block), len(studies))
        return out


def main() -> None:
    """Serve their folds over a study list and write the per-fold probabilities, unpooled."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--kernel", type=Path, required=True)
    command.parser.add_argument("--checkpoints", type=Path, required=True)
    command.parser.add_argument("--series-root", type=Path, required=True)
    command.parser.add_argument("--series", type=Path, required=True)
    command.parser.add_argument("--studies", type=Path, required=True)
    command.parser.add_argument("--out", type=Path, required=True)
    arguments = command.parsed()

    studies = [str(one) for one in np.load(arguments.studies, allow_pickle=True)["studies"]]
    folds = ForeignFolds(arguments.kernel, arguments.checkpoints, arguments.series_root)
    module = folds.module()
    folds.nets(module)
    predictions = folds.serve(module, studies, folds.records(pd.read_csv(arguments.series), studies))
    np.savez(arguments.out, predictions=predictions, studies=np.array(studies), labels=np.array(module.LABELS))
    command.report([f"wrote {arguments.out} {predictions.shape} (folds, studies, labels), unpooled"])


if __name__ == "__main__":
    main()
