"""Embed each study's radiology report once, so the image model can be supervised by all of it.

Our supervision is twelve graded numbers per study. The report they were extracted from has a median
of **129 words** and a 90th percentile of 286, in twelve languages — the anatomy named, the severity
wording, the incidental findings, the negations. Everything the twelve columns have no slot for is
discarded before training begins, and the model this project measured today is
supervision-limited rather than capacity-limited.

So the report itself becomes a target: embed it once with a multilingual text encoder, and let an
auxiliary head predict that vector from the study representation. This is the ConVIRT / GLoRIA /
BioViL setting, and it fits this competition exactly — `test.csv` has no `Report` column, so text
supervises at training time and inference stays image-only.

The multilingual corpus is an ASSET here rather than a liability: a multilingual encoder maps
*"Rotura de menisco interno"* and *"medial meniscal tear"* to nearby vectors, where a rule-based label
reader has to be taught both.

This runs OFFLINE and its output is a file. The text encoder never enters the Kaggle kit.

`python -m kneemri.embeddings.report_embeddings --out <processed>/labels/report_embeddings.npz`
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float

from core import cli
from kneemri.device import VisibleCards

ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
"""384-wide, 50+ languages, and small enough to embed 4,407 reports on a CPU while the GPU trains."""

MAX_TOKENS = 512
"""The encoder's window. It covers the 90th-percentile report; the tail is truncated, not dropped."""

BATCH = 32

logger = logging.getLogger(__name__)


class ReportEmbeddings:
    """Turns reports into one L2-normalised vector each, and reduces them to a trainable width."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def encode(self, reports: list[str], tokenizer: object, model: torch.nn.Module) -> Float[np.ndarray, "n d"]:
        """Mean-pool the token states over the attention mask, then L2-normalise.

        Mean pooling over the MASK rather than over the padded length: padding tokens would otherwise
        drag every short report toward the same vector, which is precisely the reports whose findings
        are shortest and most decisive.
        """
        model.eval().to(self.device)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(reports), BATCH):
                batch = reports[start : start + BATCH]
                encoded = tokenizer(  # pyrefly: ignore[not-callable]
                    batch, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                states = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                chunks.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
        return np.concatenate(chunks).astype(np.float32)

    def reduce(self, vectors: Float[np.ndarray, "n d"], dims: int) -> Float[np.ndarray, "n k"]:
        """The leading principal components, scaled as a whole to an average unit norm.

        Reduced because the auxiliary head has to predict this from a study representation fitted on
        4,407 examples: a 384-wide target over that many studies is mostly directions no image could
        determine. The components are computed on OUR reports, so they describe the axes this corpus
        actually varies along rather than the encoder's general ones.

        Scaled by ONE global factor rather than per study. Re-normalising each row to unit length
        throws away how far a report sits from the corpus mean, and does it worst where the reduction
        is sharpest: at a single component every study collapses to +1 or -1, so the target keeps the
        sign of the leading axis and discards its magnitude entirely. A global factor puts the target
        on a sane scale for a regression head while leaving the relative geometry between studies
        exactly as the reports arranged it.
        """
        centred = vectors - vectors.mean(axis=0, keepdims=True)
        _, _, components = np.linalg.svd(centred, full_matrices=False)
        projected = centred @ components[:dims].T
        scale = float(np.linalg.norm(projected, axis=1).mean())
        return (projected / max(scale, 1e-9)).astype(np.float32)


def main() -> None:
    """Embed every training report and write the vectors beside their study identifiers."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--dims", type=int, default=64, help="principal components kept as the target")
    command.parser.add_argument("--encoder", type=str, default=ENCODER)
    command.parser.add_argument("--out", type=Path, default=None)
    arguments = command.parsed()

    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415 - a 90 MB import only this CLI needs

    root = command.rooted(arguments)
    table = pd.read_csv(root.raw() / "train.csv")
    identifiers = table["StudyInstanceUID"].astype(str).tolist()
    reports = table["Report"].astype(str).tolist()
    logger.info("embedding %d reports with %s", len(reports), arguments.encoder)

    device = VisibleCards().device()
    embeddings = ReportEmbeddings(device)
    vectors = embeddings.encode(
        reports, AutoTokenizer.from_pretrained(arguments.encoder), AutoModel.from_pretrained(arguments.encoder)
    )
    reduced = embeddings.reduce(vectors, arguments.dims)

    destination = arguments.out or (root.processed() / "labels" / "report_embeddings.npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, identifiers=np.array(identifiers), vectors=reduced)
    logger.info("wrote %s %s", destination, reduced.shape)


if __name__ == "__main__":
    main()
