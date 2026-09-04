"""The encoder VOCABULARY: which trunk a `--variant` string names, and which family it belongs to.

A leaf module on purpose. The registry rows live in two files now (`backbone.py` for DINOv2,
`foreign_encoders.py` for everything else) and both have to spell the same names, so the names cannot
live in either one without making the other import it. `backbone` re-exports every name here for the
forty-odd modules that address them as `kneemri.models.backbone.<name>`.
"""

SMALL = "small"
BASE = "base"
LARGE = "large"
V3_SMALL = "v3small"
V3_SMALL_PLUS = "v3splus"
V3_BASE = "v3base"
V3_LARGE = "v3large"
ORTHO_LARGE = "ortholarge"
CONVNEXT_BASE = "cnxbase"
VJEPA2_LARGE = "vjepa2l"
RAD_DINO = "raddino"
SWIN_BASE = "swinbase"
VIT_IN21K = "vitin21k"
RADIMAGENET_DENSENET = "radnet121"
RADIODINO_BASE = "radiodinob16"
RADIODINO_SMALL = "radiodinos16"
RADIODINO_SMALL_P8 = "radiodinos8"
MEDSAM_BASE = "medsam"
COATNET_2 = "coatnet2"

DINOV2 = "dinov2"
DINOV3 = "dinov3"
CONVNEXT = "convnext"
VJEPA2 = "vjepa2"
RADDINO = "rad-dino"
SWIN = "swin"
VIT = "vit"
RADIMAGENET = "radimagenet"
RADIODINO = "radiodino"
MEDSAM = "medsam"
COATNET = "coatnet"
