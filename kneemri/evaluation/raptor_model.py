"""The raptor model, reimplemented as OUR code so a foreign checkpoint can be SERVED, not only measured.

`python -m kneemri.evaluation.raptor_model --studies 6` — score the same studies through their module
and through this one, and assert the two probability vectors are equal element for element.

WHAT IS ACTUALLY FOREIGN HERE. Nothing but the weights. The architecture is a timm backbone plus a
forty-line attention-MIL head — a `LayerNorm`, a two-layer gate softmaxed over the window axis, and a
per-finding linear read-out. Their notebook publishes it in full. Written out here it carries the same
parameter NAMES, which is the whole trick: `load_state_dict(..., strict=True)` then either accepts
their checkpoint or refuses it, with no key remapping to get subtly wrong.

WHY IT HAD TO BE WRITTEN. `raptor_feed` made the decode ours; this makes the forward pass ours. Between
them, `external/raptor` — gitignored third-party code with no LICENSE file — leaves the serving path
entirely, and a kit built from `comp/build_kit.py` ships the member with no vendored code in it. Their
weights stay what they always were: a public Kaggle dataset, mounted.

FAITHFUL MEANS EQUAL, NOT CLOSE — AND THE ORACLE HERE IS END-TO-END. `raptor_feed`'s oracle stops at the
`uint8` volume. This one runs the rest of the chain: windows, backbone, head, sigmoid. So the two
oracles together cover every step from a DICOM directory to twelve probabilities, and the windowing
that sits between them — the one stage neither module's own test could reach — is checked by the pair.

Equality is exact on CPU in fp32 and NOT claimed under autocast: their `infer_probs` runs fp16 on CUDA,
which is a serving choice, not part of the specification. `--device cpu` is what proves the port.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, override

import numpy as np
import timm
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor, nn

from core import cli
from core.findings import TARGETS
from kneemri.evaluation import foreign_member, raptor_feed

logger = logging.getLogger(__name__)

FINDINGS = 12
"""Outputs of the head, one per finding — the same twelve `core.findings.TARGETS` names, in their order."""

ATTENTION_WIDTH = 256
"""Hidden width of the gate that pools windows into a study. Fixed by the checkpoint's shapes."""

DROPOUT = 0.2
"""Kept only because it sits between the gate's two layers and shifts every parameter index behind it."""

INIT_STD = 0.02
"""Init of the read-out, irrelevant to serving and kept so a randomly-initialised member matches theirs."""

HYBRID_PREFIXES = ("maxvit", "maxxvit", "coatnet", "coat_", "convnext")
"""Conv-attention hybrids: no CLS token and no interpolatable position embedding, so they average-pool.
The `vit` inside `maxvit` and `coatnet` must NOT route them down the transformer path."""

TRANSFORMER_MARKERS = ("vit", "deit", "dinov2", "eva", "beit")
"""Architectures that really do carry a CLS token, checked only after the hybrids are taken out."""

DEFAULT_ARCHITECTURE = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
"""The arm their published kit ships. A checkpoint carrying its own `arch` overrides this."""

DEFAULT_RESOLUTION = 384

CUDA = "cuda"


@dataclass(frozen=True)
class Architecture:
    """A backbone named by a checkpoint, and the input resolution it was fitted at."""

    name: str
    resolution: int
    options: raptor_feed.TimmOptions = field(default_factory=raptor_feed.TimmOptions)
    """Extra `timm.create_model` options the checkpoint was built with — its `Layout.backbone`."""

    @property
    def transformer(self) -> bool:
        """Whether this one pools a CLS token rather than averaging a feature map."""
        hybrid = self.name.startswith(HYBRID_PREFIXES)
        return not hybrid and any(marker in self.name for marker in TRANSFORMER_MARKERS)

    def built(self) -> nn.Module:
        """Construct the feature extractor with the classifier removed, weights left uninitialised."""
        if self.transformer:
            return timm.create_model(
                self.name,
                pretrained=False,
                num_classes=0,
                in_chans=3,
                global_pool="token",
                dynamic_img_size=True,
                **self.options,
            )
        return timm.create_model(
            self.name, pretrained=False, num_classes=0, in_chans=3, global_pool="avg", **self.options
        )

    @staticmethod
    def width(backbone: nn.Module) -> int:
        """The backbone's feature width, read through `nn.Module.__getattr__` and narrowed to an `int`."""
        return int(cast("int", backbone.num_features))


class RaptorClassifier(nn.Module):
    """A backbone read over every window, pooled by a per-finding attention gate.

    The attribute names are theirs on purpose — `backbone`, `norm`, `att`, `clsW`, `clsb` — because the
    state-dict keys are the contract with a checkpoint we did not train.
    """

    def __init__(self, backbone: nn.Module, features: int, findings: int = FINDINGS) -> None:
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(features)
        self.att = nn.Sequential(
            nn.Linear(features, ATTENTION_WIDTH),
            nn.Tanh(),
            nn.Dropout(DROPOUT),
            nn.Linear(ATTENTION_WIDTH, findings),
        )
        self.clsW = nn.Parameter(torch.zeros(findings, features))
        self.clsb = nn.Parameter(torch.zeros(findings))
        nn.init.trunc_normal_(self.clsW, std=INIT_STD)
        self.n = findings

    def encode(
        self, windows: Float[Tensor, "batch windows channels size size"]
    ) -> Float[Tensor, "batch windows features"]:
        """One feature vector per window, the batch and window axes flattened for the backbone."""
        batch, count = windows.shape[:2]
        flat = self.backbone(windows.flatten(0, 1))
        return flat.view(batch, count, -1)

    def pooled(self, features: Float[Tensor, "batch windows features"]) -> Float[Tensor, "batch findings features"]:
        """Pool the windows once PER FINDING — the gate is softmaxed over windows, not over findings.

        Split out of `head` rather than duplicated, because this is what a DISTILLATION target is read
        from and a second spelling of an attention pool is a second thing that can drift from the
        checkpoint we did not train. `head` is unchanged in behaviour, which `main`'s equality oracle is
        the witness for.
        """
        normed = self.norm(features)
        attention = torch.softmax(self.att(normed), dim=1)
        return torch.einsum("bkn,bkf->bnf", attention, normed)

    def head(self, features: Float[Tensor, "batch windows features"]) -> Float[Tensor, "batch findings"]:
        """Twelve logits from the per-finding pooled features — one learned direction each."""
        pooled = self.pooled(features)
        return (pooled * self.clsW).sum(-1) + self.clsb

    def embedding(self, windows: Float[Tensor, "batch windows channels size size"]) -> Float[Tensor, "batch features"]:
        """What this member READ from each study, before its head split that reading twelve ways.

        The mean over the FINDING axis, so the target a student is pulled onto is the study's
        representation rather than the twelve directions this particular head happens to project it
        along. Distilling the latter teaches the teacher's head, which is the level `swin_distill_fit`
        measured at -0.0077 against a cold control.
        """
        return self.pooled(self.encode(windows)).mean(dim=1)

    @override
    def forward(self, windows: Float[Tensor, "batch windows channels size size"]) -> Float[Tensor, "batch findings"]:
        """Logits for a batch of studies, each given as its own stack of windows."""
        return self.head(self.encode(windows))


@dataclass(frozen=True)
class RaptorMember:
    """A loaded checkpoint, ready to score decoded studies."""

    model: RaptorClassifier
    architecture: Architecture
    labels: tuple[str, ...] = TARGETS
    """The findings in the order the HEAD emits them, read from the checkpoint's own `lab` key.

    Carried rather than assumed: the two orders agree today, and a submission whose columns were
    assigned positionally would keep scoring — around chance — the day an arm's did not."""
    layout: raptor_feed.Layout = raptor_feed.RAPTOR
    """The decode and windowing this checkpoint was fitted under; the feed that decodes for it must match."""

    @classmethod
    def loaded(
        cls,
        checkpoint: Path,
        device: str,
        architecture: str = DEFAULT_ARCHITECTURE,
        resolution: int = DEFAULT_RESOLUTION,
        layout: raptor_feed.Layout = raptor_feed.RAPTOR,
    ) -> "RaptorMember":
        """Read a checkpoint, build the architecture IT names, and load its weights strictly.

        `weights_only=True`: these are foreign files, and every one on disk loads under it.
        """
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        fitted = int(state.get("res", resolution))
        named = Architecture(str(state.get("arch", architecture)), layout.resolution(fitted), layout.backbone)
        backbone = named.built()
        model = RaptorClassifier(backbone, features=Architecture.width(backbone))
        model.load_state_dict(state["model"], strict=True)
        model.eval().to(device)
        labels = tuple(str(name) for name in state.get("lab", TARGETS))
        return cls(model, named, labels, layout)

    @torch.no_grad()
    def probs(
        self, windows: Float[Tensor, "windows channels size size"], device: str
    ) -> Float[NDArray[np.float32], " findings"]:
        """Twelve probabilities for one study, from the windows `raptor_feed` stacked for it."""
        half = self.layout.half and torch.device(device).type == CUDA
        with torch.autocast(CUDA, dtype=torch.float16, enabled=half):
            logits = self.model(windows.unsqueeze(0).to(device))
        return torch.sigmoid(logits.float())[0].cpu().numpy()

    def viewed(
        self,
        volume: Float[NDArray[np.uint8], "depth height width"],
        mask: Float[NDArray[np.uint8], " depth"],
    ) -> list[Float[Tensor, "windows channels size size"]]:
        """Every view this member's layout averages, windowed the way it was fitted."""
        windows = raptor_feed.Windows.of(self.layout)
        return windows.views(volume, mask, self.architecture.resolution, self.layout.flips)

    def scored(
        self,
        volume: Float[NDArray[np.uint8], "depth height width"],
        mask: Float[NDArray[np.uint8], " depth"],
        device: str,
    ) -> Float[NDArray[np.float32], " findings"]:
        """The whole serving step for one decoded study: window it, then read the member, views averaged."""
        read = [self.probs(view, device) for view in self.viewed(volume, mask)]
        return np.asarray(np.mean(read, axis=0), dtype=np.float32)

    @torch.no_grad()
    def embedded(
        self, windows: Float[Tensor, "windows channels size size"], device: str
    ) -> Float[NDArray[np.float32], " features"]:
        """One study's pooled representation, from the windows `raptor_feed` stacked for it."""
        return self.model.embedding(windows.unsqueeze(0).to(device)).float()[0].cpu().numpy()

    def represented(
        self,
        volume: Float[NDArray[np.uint8], "depth height width"],
        mask: Float[NDArray[np.uint8], " depth"],
        device: str,
    ) -> Float[NDArray[np.float32], " features"]:
        """The whole teaching step for one decoded study: window it, then read what the member READ.

        `scored`'s sibling and windowed identically, which is the point — a representation taken under a
        different windowing would be a target the member never actually produces at serving time.

        NO `TARGETS` REORDER, unlike `ordered`, and that is not an omission: the finding axis has already
        been averaged away, so there is no column order left for a label list to disagree about.
        """
        windows = raptor_feed.Windows.of(self.layout).stacked(volume, mask, self.architecture.resolution)
        return self.embedded(windows, device)

    def width(self) -> int:
        """How wide this member's reading is — the teacher side of a student's projection."""
        return Architecture.width(self.model.backbone)

    def ordered(
        self,
        volume: Float[NDArray[np.uint8], "depth height width"],
        mask: Float[NDArray[np.uint8], " depth"],
        device: str,
    ) -> Float[NDArray[np.float32], " findings"]:
        """One study scored and put in `TARGETS` order — the order a submission's columns are in."""
        scored = self.scored(volume, mask, device)
        return scored[[self.labels.index(target) for target in TARGETS]]


def main() -> None:
    """Score the same studies both ways and refuse anything short of equality."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--studies", type=int, default=6, help="how many studies to compare")
    command.parser.add_argument("--device", default="cpu", help="equality is claimed on cpu/fp32 only")
    command.parser.add_argument("--checkpoint", type=Path, required=True, help="a checkpoint of theirs")
    arguments = command.parsed()
    raw = command.rooted(arguments).raw()

    theirs = foreign_member.Decode.checked(
        foreign_member.Decode(foreign_member.EXTERNAL / foreign_member.SCRIPT).loaded()
    )
    arm = theirs.ARMS[0]
    their_model, their_resolution = theirs.load_model(
        str(arguments.checkpoint), arm["arch"], arm["res"], arguments.device
    )
    member = RaptorMember.loaded(arguments.checkpoint, arguments.device)
    if member.architecture.resolution != their_resolution:
        message = f"resolution disagrees: ours {member.architecture.resolution}, theirs {their_resolution}"
        raise ValueError(message)

    series_root = raw / "train_series"
    records = raptor_feed.RaptorFeed.records(raw / "train_series.csv")
    feed = raptor_feed.RaptorFeed(series_root)

    for study in list(records)[: arguments.studies]:
        volume, mask = feed.study(study, records)
        mine = member.scored(volume, mask, arguments.device)
        their_windows = theirs.eval_windows(volume, mask, theirs.K_EVAL, their_resolution)
        theirs_probs = theirs.infer_probs(their_model, their_windows, arguments.device)
        if not np.array_equal(mine, theirs_probs):
            worst = float(np.abs(mine - theirs_probs).max())
            message = f"{study}: our probabilities differ from theirs by up to {worst:.3e}"
            raise ValueError(message)
        logger.info("%s: identical", study)
    logger.info("%d studies score identically", arguments.studies)


if __name__ == "__main__":
    main()
