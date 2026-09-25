"""The registry rows for trunks this project did not pretrain on: every encoder that is not DINOv2.

Split out of `backbone.py` when that file passed the god-file ceiling a second time. The line is the one
the ledger already draws — the DINO family is what the pipeline was built around and what every baseline
is measured against, and everything here is a FOREIGN trunk mounted to answer a diversity question. The
table is declaration-only, which is the case `CLAUDE.md` exempts from "never bucket a module by kind":
there is no behaviour to split, only rows, and `backbone.ENCODERS` merges them back into one namespace so
no caller learns that the split happened.
"""

from kneemri.models.encoder import (
    HALF,
    IMAGENET,
    UNIT,
    Encoder,
    Normalization,
    Stage,
)
from kneemri.models.sam_trunk import MEDSAM_VIT_B
from kneemri.models.snapshot_trunk import ORTHOFOUNDATION_L
from kneemri.models.state_dict_trunk import RADIMAGENET_DENSENET121
from kneemri.models.timm_trunk import MRICORE_VITB, TimmHubWeights, TimmMapWeights
from kneemri.models.trunk_tokens import TrunkAsTokens
from kneemri.models.variants import (
    COATNET,
    COATNET_2,
    CONVNEXT,
    CONVNEXT_BASE,
    DINOV3,
    MEDSAM,
    MEDSAM_BASE,
    MRICORE,
    MRICORE_BASE,
    ORTHO_LARGE,
    RAD_DINO,
    RADDINO,
    RADIMAGENET,
    RADIMAGENET_DENSENET,
    RADIODINO,
    RADIODINO_BASE,
    RADIODINO_SMALL,
    RADIODINO_SMALL_P8,
    RESNET,
    RESNET_50D,
    SWIN,
    SWIN_BASE,
    VIT,
    VIT_IN21K,
    VJEPA2,
    VJEPA2_LARGE,
)

FOREIGN_ENCODERS: dict[str, Encoder] = {
    # THE FIRST TRUNK PRETRAINED ON THIS ANATOMY, added 2026-09-02. Every encoder above is natural
    # images (the DINOs, ConvNeXt, Swin, ViT), mixed radiology (RadImageNet, RAD-DINO) or video
    # (V-JEPA2); none has seen a knee. `radnet_family` closed EMPTY at 0.6716 against a 0.75 floor, and
    # the reading that survives it is that a mixed-radiology corpus is not an in-domain corpus.
    #
    # `knee-y59c` MEASURED SOLO STRENGTH ANTI-CORRELATING WITH DISTANCE FROM DINOv2 across seven trunks,
    # and this entry is the one point that separates the two readings of that curve. It is DINOv3-L
    # weights — near DINOv2 on the family axis, where the curve predicts a strong solo — that have then
    # been moved by 894k knee-MRI slices. If distance from DINOv2 is what the curve is really about, this
    # reads strong; if the curve was a proxy for natural-image pretraining quality, continuation costs it.
    #
    # WHY IT IS THE CHEAPEST TEST OF THE EXTERNAL-DATA LANE. Ilse et al. 2025 price DINO-family
    # continuation as needing ~100k in-domain images to beat its own open weights, and the unlabelled
    # knee substrate on this disk is ~72k slices — below the threshold, so running the continuation here
    # is priced to fail before it starts. This release IS that continuation at 1.25M images, done by
    # somebody else and licensed MIT. The lane is decided by loading it, not by pretraining.
    #
    # MOUNTED 2026-09-03 as `dimiturnt/dinov3-orthofoundation-hf`. The slug carries the FAMILY because
    # `locates` demands both — `family in path` AND `leaf in segments` — and the first upload of this
    # encoder was flat as `orthofoundation-hf`, which holds no "dinov3" and so could not find itself.
    # `test_every_mount_is_locatable` caught it; nothing else would have. Its docstring already priced
    # the failure: `find_encoder` raises, `kernel_run` drops that member, and the entry SCORES without
    # it. Nesting under the existing `dinov3-hf` would also satisfy the rule, but re-versioning the
    # dataset the champion kernels mount to tidy a name risks a working entry, so this got its own slug
    # and the flat upload is an orphan for deletion. Two licences ship inside it because two apply and
    # both require it: the
    # release's own MIT text (verified from the file, not from prose), and Meta's DINOv3 Agreement, whose
    # section 1(b)(i) grants redistribution of DERIVATIVE works only under that same Agreement and only
    # with a copy provided. This is what separates the upload from the RadImageNet one: there the grant
    # did not exist, here it does, and the condition is met by shipping the text.
    ORTHO_LARGE: Encoder(
        ORTHO_LARGE,
        "",
        1024,
        patch=16,
        family=DINOV3,
        leaf="dinov3-orthofoundation-hf",
        mount="dimiturnt/dinov3-orthofoundation-hf",
        params_m=303.1,
        depth=24,
        builder=ORTHOFOUNDATION_L.build,
        normalization_evidence=(
            "The release publishes no preprocessing document — it is a bare training checkpoint, and the "
            "converted snapshot carries only the architecture. Its own paper states DINOv3 preprocessing "
            "unchanged, and the weights it continues from declare ImageNet constants, so ImageNet is what "
            "this checkpoint was continued under rather than a default falling through."
        ),
    ),
    # THE FIRST NON-TRANSFORMER, added 2026-08-21 because the transformer axis closed. `base_dense`
    # (DINOv2-base among seven DINOv2-small members) took the board from 0.930 to 0.931; `v3base_dense`
    # (DINOv3-base, same recipe, a whole different pretraining corpus and teacher) was worth -0.0007 as
    # a sixth seat. A second ViT family does not buy a second diversity seat, so the next member has to
    # differ in INDUCTIVE BIAS rather than in weights: local convolution and a stagewise hierarchy make
    # different mistakes than global attention over a flat grid, and that is the quantity an ensemble
    # is paid for.
    #
    # ConvNeXtV2-base is chosen for capacity parity with the member that worked: 87.7M parameters
    # against DINOv2-base's 86M, so the comparison is about the bias and not about size. FCMAE
    # pretraining keeps it self-supervised like the DINOs, then IN-22k finetuning — which is itself a
    # difference worth having, since every incumbent is SSL-only.
    #
    # STRIDE 32, NOT PATCH 32, and `patch` is the right field for it anyway: four stages halving twice
    # over, so the read must be a multiple of 32 for the grid to come out whole. 336 does NOT tile;
    # 352 gives an 11x11 grid of 121 tokens against a ViT's 576 at 336, which is why a convolutional
    # trunk of the same parameter count is cheaper per slice at a higher resolution.
    #
    # NO MOUNT, so this trains here and cannot be submitted yet. The weights are Apache-2.0 and mirror
    # cleanly the way `dinov3-hf` did; that is a dataset upload, not a code change, and it is deferred
    # until a member proves it is worth a seat.
    CONVNEXT_BASE: Encoder(
        CONVNEXT_BASE,
        "facebook/convnextv2-base-22k-224",
        1024,
        patch=32,
        family=CONVNEXT,
        leaf="convnextv2-base",
        params_m=87.7,
        adapter=TrunkAsTokens.from_map,
        # `depths` and `hidden_sizes` off ConvNeXtV2-base's own config, at the strides the four stages
        # run at. Stated rather than read from the loaded config because the cap has to be derived
        # BEFORE any weights are fetched — the whole point of it is to refuse a run that will not fit.
        stages=(Stage(3, 4, 128), Stage(3, 8, 256), Stage(27, 16, 512), Stage(3, 32, 1024)),
    ),
    # THE FIRST TRUNK THAT WAS NOT PRETRAINED WITHOUT LABELS, added 2026-08-30. Every encoder above is
    # self-supervised — DINOv2 and DINOv3 by distillation, ConvNeXtV2 by masked autoencoding, V-JEPA2 by
    # latent prediction — so "a different trunk" has so far always meant a different way of having no
    # labels. Swin-base is IN-22k SUPERVISED: 21,841 human-assigned classes, an objective that rewards
    # exactly the discriminative structure our head is then asked to find, and the one axis of pretraining
    # this project has never once varied. That is the diversity claim, and it is about the OBJECTIVE
    # rather than about the architecture, which is why it is worth a run after ConvNeXtV2's read PARTIAL.
    #
    # THE ARCHITECTURE DIFFERS TOO, and differently from ConvNeXt's. Shifted-window attention is neither
    # global attention over a flat grid (DINOv2) nor pure local convolution (ConvNeXtV2): attention within
    # a 12x12 window, with the windows displaced every other block so information crosses their seams.
    # `knee-y59c` measured distance-from-DINOv2 anti-correlating with solo strength across seven trunks,
    # so this sits deliberately in the MIDDLE of that axis rather than at the far end where `mae_matched`
    # (0.6735) and `focus_mcl` (0.5332) died.
    #
    # 384 IS THE ONLY LEGAL READ, and `require_tiling` is NOT sufficient to enforce it. The window size is
    # baked into the pretrained relative-position tables at 12, and a 12x12 window tiles a 384 px read at
    # stride 32 exactly. 352 tiles the stride and NOT the window: transformers will pad and interpolate
    # rather than refuse, handing back a plausible answer computed on a geometry the tables never saw.
    # The arm therefore pins --train-size 384 and this comment is the reason it may not be swept.
    #
    # MIRRORED 2026-08-30, so this trunk is shippable: `dimiturnt/swin-base-384` exists and `mount` below
    # names it. Apache-2.0, so mirroring was permitted outright — no licence question of the kind DINOv3
    # needed. This header read "NO MOUNT YET ... deferred until a member earns the seat" for hours after
    # the upload landed, while the entry's own `mount` and `leaf` comments described the mirror as built
    # and verified. A stale header here is not cosmetic: it is the line a reader checks before deciding
    # whether a finished arm can be submitted at all.
    SWIN_BASE: Encoder(
        SWIN_BASE,
        "microsoft/swin-base-patch4-window12-384-in22k",
        1024,
        patch=32,
        family=SWIN,
        # THE DATASET'S SLUG, not a prettier name for the trunk. `leaf` has exactly one consumer,
        # `locates`, which requires it as an exact path SEGMENT of the mounted directory — and a flat
        # mount's directory IS the slug, `/kaggle/input/swin-base-384`. Spelled `swin-base` this entry
        # matched no mount, so `find_encoder` raised and `kernel_run` dropped the member from the entry
        # with the other members still loading: a submission that scores, without the thing it was for.
        leaf="swin-base-384",
        params_m=86.9,
        adapter=TrunkAsTokens.from_sequence,
        # FLAT, like `dimiturnt/rad-dino` and unlike `dimiturnt/dinov3-hf/<leaf>`: the leaf shape exists
        # because DINOv3 ships four variants from one dataset, and there is one Swin. Written with
        # `SwinModel.save_pretrained`, so the mirror holds exactly what `load` reads back — verified by
        # loading it from disk through this entry, which is what a kernel does with a mounted dataset.
        # Apache-2.0 and ungated (read from the HF API, not the model card), so §2.6.a is met by the
        # upstream repository and the mirror exists only because a kernel has no internet.
        mount="dimiturnt/swin-base-384",
        # `depths` and `embed_dim` off Swin-base's own config, at the strides its four stages run at —
        # stated ahead of any fetch for the same reason ConvNeXtV2's are: the cap refuses a run that
        # will not fit, so it cannot wait for the weights to say.
        stages=(Stage(2, 4, 128), Stage(2, 8, 256), Stage(18, 16, 512), Stage(2, 32, 1024)),
    ),
    # THE FIRST TRUNK WHOSE TOP BLOCKS `--unfreeze-last` ACTUALLY REACHES, among the hierarchical ones.
    # That is the reason it is here rather than a fourth restatement of "another non-DINO family". Swin
    # and ConvNeXtV2 both hold their deepest block run in stage 2 (18 of 24, 27 of 36), so a suffix of
    # six opened stage-2 blocks while the widest stage stayed frozen; `swin_wholetop` then measured what
    # opening it is worth and came back -0.0057, which is what closed both families on a priced argument.
    # CoAtNet's depths are 2/6/14/2 and its stages are NOT one block type: the first two are MBConv, the
    # last two are transformer. `EncoderSurgery.blocks` groups by element TYPE, so the run it counts is
    # the 16 transformer blocks across stages 2 AND 3, and `--unfreeze-last 6` opens the last four of
    # stage 2 plus BOTH of stage 3. The convolutional stem stays frozen, which is the part a 4,400-study
    # corpus has no business refitting anyway.
    #
    # THE OBJECTIVE IS SUPERVISED AND THE ARCHITECTURE IS A HYBRID. IN-12k supervised then IN-1k finetuned,
    # so it shares Swin's pretraining axis; what differs is the mixing rule — depthwise convolution with
    # squeeze-excite for the first two stages, then GLOBAL relative attention over the whole 12x12 grid for
    # the last two. Neither ConvNeXtV2's purely local convolution nor Swin's windowed attention: the one
    # trunk in this table where a low-level convolutional prior feeds a globally-attending top.
    #
    # 384 IS THE ONLY LEGAL READ, and unlike Swin's case it is not the window that pins it. `rmlp` is a
    # relative-position MLP whose table is built for the stage-3 grid at construction, so 448 raises
    # `The size of tensor a (784) must match the size of tensor b (576)` on the first forward — measured
    # here, not inferred. 12x12 = 144 final tokens, FEWER than ConvNeXtV2's 196 at 448. That is not the
    # handicap it looks like: `native224` reads 256 tokens for 0.8502 against the 336 corner's 0.851, so
    # this pipeline loses almost nothing per token dropped, and the 0.03 non-DINO deficit is a fact about
    # the trunks rather than about their resolution.
    #
    # NO MOUNT, so this trains here and cannot be submitted yet — Apache-2.0, mirrors cleanly, and that
    # upload is deferred until a member earns the seat.
    COATNET_2: Encoder(
        COATNET_2,
        "timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
        1024,
        patch=32,
        family=COATNET,
        leaf="coatnet-2-384",
        params_m=72.9,
        normalization=HALF,
        normalization_evidence=(
            "timm ships no preprocessor_config.json; its config.json carries `pretrained_cfg` and that "
            "document states mean and std (0.5, 0.5, 0.5) for this tag, read from the file rather than "
            "from prose. NOT a default: the entry ran one epoch under the IMAGENET constants before "
            "`test_every_encoder_states_its_own_constants` refused the silent declaration."
        ),
        # Both wrappers, applied inside the builder rather than by `adapter`: timm answers
        # `forward_features` and emits a map, so the CALL and the LAYOUT are separately wrong.
        builder=TimmMapWeights(
            "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
            1024,
            repository="timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
            classes=1000,
        ).build,
        # `depths` and `embed_dim` off coatnet_2_rw's own config, at the strides its four stages run at —
        # stated ahead of any fetch because the VRAM cap has to refuse an impossible run before weights.
        stages=(Stage(2, 4, 128), Stage(6, 8, 256), Stage(14, 16, 512), Stage(2, 32, 1024)),
    ),
    # THE PLAIN CNN, added 2026-09-07 — the trunk every one of this project's baselines skipped past.
    # Twelve entries above are transformers or hybrids; the one pure convolutional line here is
    # `radnet121`, and it closed EMPTY at 0.6716 against a 0.75 floor. That number was read as a verdict
    # on a mixed-radiology CORPUS, and it cannot also be read as a verdict on convolution: no ImageNet
    # CNN has ever been fitted here, so the CNN lane has never actually been priced.
    #
    # This row exists to price it, at the recipe the foreign 2x2 is replicating rather than at ours — 224
    # px, 2.5D three-slice RGB, drop-path 0.1 — because the cell asks whether a CNN can reach the band at
    # all, and answering with our own dense-window recipe would confound the trunk with the recipe.
    #
    # THE MOUNT WAS ARRANGED ON 2026-09-09, ONCE THE LANE WAS PRICED AND NOT BEFORE. This row read "NO
    # MOUNT, deliberately" until `resnet_224_short` cleared its own pre-registered 0.80 plumbing gate on
    # the 909 (epoch 3, 0.8254), which is the condition that sentence named: an upload spent on a trunk
    # with no measured seat is the thing it refused, and the seat now has a number behind it.
    #
    # THE SLUG IS THE LEAF, EXACTLY, AND THAT IS NOT A STYLE CHOICE. `locates` reads `leaf in segments`
    # — segment EQUALITY, not a substring — so a mount whose slug merely CONTAINS the leaf does not
    # resolve at its own root. The first upload went out as `kneemri-resnet-timm` holding a
    # `resnet50d-224/` directory, which reads correctly to a human and fails `str.split`: exactly the
    # swin defect this file's mount test was written for. `test_every_mount_is_locatable` caught it
    # before a kernel did, which is the whole point of that test — `find_encoder` raises, `kernel_run`
    # catches per member and DROPS it, and the entry scores without the member it was built for. Every
    # other mount here obeys the same rule (`swin-base-384`, `vjepa2-hf`, `rad-dino`). It is its OWN
    # dataset rather than a directory added to an existing one: re-versioning a dataset that champion
    # kernels already mount is a risk taken for no gain.
    #
    # REDISTRIBUTION IS LICENSED. `timm/resnet50d.ra4_e3600_r224_in1k` is Apache-2.0, read from the Hugging
    # Face model API on 2026-09-09 (`cardData.license`, plus the `license:apache-2.0` tag) rather than from
    # prose, and the licence text ships beside the weights as section 4(a) of that licence requires.
    RESNET_50D: Encoder(
        RESNET_50D,
        "timm/resnet50d.ra4_e3600_r224_in1k",
        2048,
        patch=32,
        family=RESNET,
        leaf="resnet50d-224",
        mount="dimiturnt/resnet50d-224",
        params_m=25.6,
        normalization=HALF,
        normalization_evidence=(
            "timm ships no preprocessor_config.json; its config.json carries `pretrained_cfg` and that "
            "document states mean and std (0.5, 0.5, 0.5) for THIS tag, fetched from the hub on "
            "2026-09-07 and read off the file. The reading is tag-specific and the sibling tags "
            "disagree: `resnet50d.ra2_in1k` and `resnet50d.a1_in1k` both state the ImageNet constants "
            "on the same architecture, so a family-level default would be wrong here by (0.485, 0.456, "
            "0.406) against (0.5, 0.5, 0.5)."
        ),
        # A map-emitting trunk called through `forward_features`, so both wrappers, as CoAtNet needs.
        builder=TimmMapWeights(
            "resnet50d.ra4_e3600_r224_in1k",
            2048,
            repository="timm/resnet50d.ra4_e3600_r224_in1k",
            classes=1000,
            drop_path=0.1,
        ).build,
        # ResNet-50's own (3, 4, 6, 3) bottleneck blocks, at the four strides and widths they run at.
        stages=(Stage(3, 4, 256), Stage(4, 8, 512), Stage(6, 16, 1024), Stage(3, 32, 2048)),
    ),
    # THE FIRST TRUNK PRETRAINED ON MEDICAL IMAGES WITH LABELS, added 2026-09-01. Laid out on the two
    # axes this project has actually varied, every encoder above falls in one of two cells:
    #
    #                     self-supervised            supervised
    #   natural images    DINOv2, DINOv3,            Swin-base IN-22k, ViT IN-21k
    #                     ConvNeXtV2 (FCMAE), V-JEPA2
    #   medical images    RAD-DINO                   -- EMPTY --
    #
    # That empty cell is the whole reason for this entry. `knee-nj4l` says the champion is an
    # architecture monoculture and that this is the mechanical cause of the 0.937-0.941 plateau; seven
    # independent board flats (reweight, add, swap, count, precision, pooling lens, distance-with-quality)
    # closed the SEATING axis, which leaves only what a member IS and what it is TAUGHT. RadImageNet is
    # both at once: a different architecture AND the one pretraining objective nothing here has seen.
    #
    # 1.35M radiology images across 165 supervised classes, DenseNet-121. It is not knee MRI — the corpus
    # is CT, MRI and ultrasound across abdomen, thorax, knee, shoulder and brain — so the claim is domain
    # proximity, not task transfer.
    #
    # WHAT THE PRECEDENT DOES AND DOES NOT SAY. Nilay et al. report a nine-model ensemble beating its best
    # solo member by +0.031 F1 with these exact weights in the pool. They do NOT report an ablation with
    # and without the DenseNet branch, so that number is a TOTAL ensemble gain across three inference
    # paradigms and is not a per-branch precedent. It is recorded here as the reason to look, and the
    # argument for actually spending a card is the empty cell above.
    #
    # SEVEN MILLION PARAMETERS against 86-88M for every incumbent, which is the honest objection: this is
    # a twelfth the capacity of the members it would sit beside, and a weak solo is the likeliest outcome.
    # It is queued anyway because the ensemble is not paid for member strength — `knee-y59c` measured
    # distance-from-DINOv2 ANTI-correlating with solo strength across seven trunks, and every seat this
    # project has bought was bought with a mechanism nothing else in the pool had.
    #
    # HALF, NOT IMAGENET, and this was measured rather than assumed. The release documents two mutually
    # exclusive preprocessing recipes (TensorFlow: rescale 1/255 then Keras caffe mean subtraction;
    # PyTorch demo: (x-127.5)*2/255 on [0,255], i.e. [-1,1]), and a published downstream user of these
    # weights (Nilay et al.) reports a third, plain ImageNet. `norm0` sits in the FROZEN prefix under
    # every recipe this project runs, so its running statistics cannot adapt to a wrong input scale —
    # which makes the recipe decidable: feed each candidate and keep the one whose post-`norm0` output is
    # closest to the unit distribution those statistics encode. Both caffe forms diverge by five orders of
    # magnitude (post-`norm0` std 181 and 249, final std 1.2e4 and 5.5e4); of the three [0,1]-scale forms,
    # HALF is nearest unit variance (1.93, against ImageNet's 3.59) AND is the recipe the PyTorch release
    # itself documents. Two independent lines agreeing is why this is declared rather than swept.
    #
    # BGR VERSUS RGB DOES NOT ARISE, which removes half the published ambiguity by construction:
    # `slice_read.py:126` expands one greyscale channel to three identical ones, so any channel
    # permutation is the identity here.
    #
    # NOT SHIPPABLE, AND THE MOUNT WAS WITHDRAWN 2026-09-02. A mirror `dimiturnt/radimagenet-densenet121`
    # was created earlier that day on a reading of MIT from the `Lab-Rasool/RadImageNet` re-host, and the
    # mount was declared here. That reading does not hold: the RadImageNet weights are request-gated at
    # source and publish no licence of their own, so a re-host cannot confer one and our public mirror was
    # redistribution we had no grant for. The mirror is being deleted and `mount` stays absent, which is
    # what makes this trunk unshippable rather than merely unbuilt — `mount_check` refuses it by design.
    # The MEASUREMENTS stand and are not withdrawn: a ridge probe over this trunk's cached features reads
    # 0.8197 against the incumbent `v3large`'s 0.8046, and a trained head over the same caches keeps that
    # ordering (0.7800/0.7841 against 0.7726, knee-j0h5). They were taken on locally-held weights and they
    # are why the trunk FAMILY is worth pursuing — through a trunk whose licence permits redistribution.
    # OrthoDiffusion (knee-6hfi) is that route.
    RADIMAGENET_DENSENET: Encoder(
        RADIMAGENET_DENSENET,
        "Lab-Rasool/RadImageNet",
        1024,
        patch=32,
        family=RADIMAGENET,
        leaf="radimagenet-densenet121",
        params_m=7.0,
        normalization=HALF,
        normalization_evidence=(
            "The release is a bare OrderedDict and publishes no preprocessing document, and its own "
            "documentation states two mutually exclusive recipes while a published downstream user "
            "reports a third. Settled by measurement instead: norm0 sits inside the frozen prefix at "
            "--unfreeze-last 6, so its running statistics cannot adapt to a wrong input scale, which "
            "makes the right recipe the one whose post-norm0 output is nearest the unit distribution "
            "those statistics encode. On uniform-random 4x3x224x224 the post-norm0 std reads 3.585 "
            "(imagenet), 1.929 (half), 2.484 (raw01), 181.4 and 248.5 (the two caffe forms, which "
            "diverge by five orders of magnitude downstream). HALF wins, and is also what the PyTorch "
            "release documents."
        ),
        adapter=TrunkAsTokens.from_map,
        builder=RADIMAGENET_DENSENET121.build,
        # Four dense blocks of 6/12/24/16 layers at strides 4/8/16/32, widths taken at each block's
        # OUTPUT (256, 512, 1024, 1024) — a transition halves the width again before the next block, so
        # these are the peaks rather than the plateaus, which is the conservative side for a cap whose
        # job is refusing a run that will not fit.
        stages=(Stage(6, 4, 256), Stage(12, 8, 512), Stage(24, 16, 1024), Stage(16, 32, 1024)),
    ),
    # THE OBJECTIVE AXIS WITH THE ARCHITECTURE HELD FIXED, added 2026-08-30. `swinbase` above is this
    # registry's first supervised trunk and it varies TWO things at once against every incumbent: the
    # pretraining objective AND the architecture. Whatever it reads, it cannot say which half carried it.
    # This entry can. It is an exact architectural twin of `v3base` — ViT-B/16, twelve blocks, 768 wide,
    # global attention over the same patch grid — and differs from it in the objective ALONE: AugReg
    # supervised IN-21k against DINOv3 self-distillation on LVD-1689M. `v3base_dense` already ran the
    # dense recipe on the twin and read -0.0007 as a sixth seat, so the control is on disk.
    #
    # `knee-y59c` MEASURED SOLO STRENGTH ANTI-CORRELATING WITH DISTANCE FROM DINOv2 across seven trunks,
    # and an architectural twin sits at ~0 on that axis — which is why this is expected to produce a real
    # number rather than another `mae_matched` (0.6735) or `focus_mcl` (0.5332). A strong solo with a
    # worthless seat would itself be the finding: it would say the objective moves the trunk's COMPETENCE
    # and not its OPINION, which is the quantity `knee-m69u` asks about and the one Swin confounds.
    #
    # NOT IMAGENET, and this one is not a corpus story. See `HALF`.
    #
    # 224 IS THE PRETRAINED READ and 16 tiles it at 14x14. Larger reads interpolate the position
    # embeddings, which every ViT here already does — so unlike Swin's window tables this may be swept.
    #
    # NO MOUNT YET. Apache-2.0 and ungated, so mirroring is permitted outright and is the same dataset
    # upload `dimiturnt/swin-base-384` is; deferred until a member earns the seat.
    VIT_IN21K: Encoder(
        VIT_IN21K,
        "google/vit-base-patch16-224-in21k",
        768,
        patch=16,
        family=VIT,
        normalization=HALF,
        leaf="vit-base-in21k",
        params_m=86.4,
    ),
    # THE FIRST TRUNK THAT WAS NOT PRETRAINED ON FLAT IMAGES, added 2026-08-21 after `through_dense`
    # answered the cheap version of this question and answered it flat. That arm put slices i-1, i, i+1
    # into the three input channels and read -0.0020 against its control, and the mechanism its null
    # leaves standing is precise: `Conv2d(3, D, k=14, s=14)` over three slices is IDENTICAL to
    # `Conv3d(1, D, k=(3,14,14), s=(1,14,14))`, so a 2D stem at window 3 collapses the window to ONE
    # linear projection per output dimension and emits ONE depth position. What it cannot do at any
    # window size is emit SEVERAL. That is the residue, and a tubelet trunk is the only thing that buys
    # it — which makes this the expensive version of a question already asked cheaply, not a new guess.
    #
    # PICKED ON LICENCE, NOT ON BENCHMARKS. The HF API says MCG-NJU/videomae-base and
    # facebook/timesformer-base-finetuned-k400 are both cc-by-nc-4.0, while this is MIT. A
    # non-commercial clause plausibly propagates to weights trained from it, which is a competition
    # ELIGIBILITY question rather than a modelling one, and it is cheaper to answer with one API call
    # than to discover after a member scores.
    #
    # THE RISK, STATED RATHER THAN DISCOVERED. It is pretrained on natural VIDEO, so this is natural-
    # domain pretraining again — and our own cautionary instance is on the ledger, where DINOv3, a
    # stronger natural-image encoder, scored ~0.04 WORSE than DINOv2 with the deficit in focal findings.
    # Worse, video is genuinely TEMPORAL and MRI depth is SPATIAL; nothing measured anywhere here says a
    # trunk pretrained on motion transfers to a spatial third axis. That is exactly why the first read
    # is a frozen embedding pass plus a head fit rather than an arm.
    #
    # NO MOUNT, so this trains here and cannot be submitted until the weights are mirrored — a dataset
    # upload, deferred until the probe says it is worth one.
    VJEPA2_LARGE: Encoder(
        VJEPA2_LARGE,
        "facebook/vjepa2-vitl-fpc64-256",
        1024,
        patch=16,
        family=VJEPA2,
        leaf="vjepa2-hf",
        # V-JEPA2 has no official Kaggle mirror, so the snapshot is published as a private dataset the
        # same way DINOv3 is. Without this the variant trains locally and cannot ship:
        # `package_members.shippable` refuses a member whose encoder no dataset mirrors, which is the
        # half `knee-rpkw`'s recorded shippability clearance did not check — it verified the kit carried
        # the CODE and concluded the member could fire, while a kernel has no internet and reaches
        # weights only through a mount.
        #
        # FLAT, NAMED FOR ITS OWN LEAF — `dimiturnt/vjepa2-hf`, not a `vjepa2-vitl/` directory under it.
        # This line said the latter from the mount until 2026-08-30 and the member could not load on
        # Kaggle for a single day of that. The staging tree on disk is correct
        # (`kaggle_upload/vjepa2-hf/vjepa2-vitl/`) and the upload flattened it: `--dir-mode zip` zips a
        # subdirectory and Kaggle unpacks it into the ROOT, so the published dataset holds `config.json`
        # at top level and the segment `locates` matches on does not exist. That is the SAME trap
        # `rad-dino` hit and documented one day earlier, and V-JEPA2 was uploaded into it anyway, so the
        # lesson is now spent in the only place that enforces it: the declared mount, which needs no
        # directory when the dataset is named for the encoder.
        #
        # WHAT IT COST, because a mount bug that merely fails loudly costs nothing. `KernelRun` DROPS an
        # unloadable member and prints `ENTRY IDENTITY CHANGED`, so `farseat_v1` shipped six members
        # under a seven-member name, scored dark6's exact 0.937, and that flat read was recorded as
        # closing the cross-family ensemble-diversity axis. It closed nothing; the seventh seat was never
        # in the entry. See knee-nbk3.
        mount="dimiturnt/vjepa2-hf",
        depth=24,
        # The only tubelet trunk in the registry: `fpc64` tokenises TWO frames per token, so the same
        # window costs half the per-slice activations a 2D ViT-L would. Without this the cap OVERSTATES
        # it by 2x and parks the run in `await_headroom` on memory it would never allocate (knee-2nxh).
        tubelet=2,
        # Counted off the loaded weights once, here, rather than guessed from "ViT-L": 325.97M.
        params_m=326.0,
    ),
    # THE FIRST TRUNK NOT PRETRAINED ON NATURAL IMAGES, added 2026-08-23. Every encoder above it —
    # DINOv2, DINOv3, ConvNeXtV2, V-JEPA2 — was pretrained on photographs or on video of the world, and
    # the failure mode they share is stated on the ledger: `base_dense` reaches 0.8371 holdout at train
    # loss 0.8456 while `cnx_dense` reaches 0.7927 at a BETTER train loss of 0.8332. A trunk that fits
    # our pixels harder generalises 0.044 worse on them. That gap is what a domain-matched
    # initialisation is for, and it is the only mechanism left priced above the 1.5% bar (knee-ombx).
    #
    # AN EXACT CONTROL EXISTS, which is the whole reason this one is worth an arm. RAD-DINO is
    # FINETUNED FROM `facebook/dinov2-base` — same ViT-B/14, same 768 width, same 12 blocks, no
    # registers — so it is a drop-in for BASE through the identical `AutoModel` path, with no
    # conversion, no wrapper and no new dependency. `base_dense_s2026` is that encoder at this exact
    # recipe (0.8512). Swapping only the initialisation weights therefore reads the domain question
    # with ONE variable, which no previous encoder arm here has managed: every other family change
    # moved architecture and pretraining together.
    #
    # THE DOMAIN MATCH IS PARTIAL AND IS NOT OVERSOLD. 882,775 CHEST RADIOGRAPHS (MIMIC-CXR, CheXpert,
    # NIH-CXR, PadChest, BRAX) — grayscale clinical anatomy with radiological tissue contrast, and
    # neither the acquisition physics nor the body part of a knee MRI. So it prices below the class:
    # +0.005-0.015 plausible against the mechanism's 1.5-3%.
    #
    # IT ALSO PRICES RadImageNet, which is the better-matched and far dearer version of the same
    # question — 1.35M radiology images across 11 anatomies, its own published evaluation including ACL
    # and meniscus tear detection on MR, and Keras `.h5` weights with no PyTorch port, so 2-4 h of
    # conversion carrying a silent-mismatch failure mode that would read exactly like a null. If
    # radiology initialisation moves nothing against an EXACT control here, that conversion is not
    # worth buying; this is the cheap half of a cost-tiered pair.
    #
    # LICENCE MIT, checked before the arm rather than after, the same way V-JEPA2's was. MOUNTED on
    # 2026-08-23, mid-arm and BEFORE the number: deferring the upload "until a seat is earned" had the
    # ordering backwards, because an unmirrored encoder makes a WINNING member offline-only, so the
    # deferred work was not the reward for a good number but the thing standing between one and a slot.
    #
    # A DATASET OF ITS OWN, FLAT, rather than a leaf under a shared one like `dinov3-hf`. The first
    # attempt copied that layout and lost it: `kaggle datasets create --dir-mode zip` zips a subdirectory
    # and Kaggle unpacks it into the ROOT, so `dimiturnt/raddino-hf` holds `config.json` at top level and
    # the `rad-dino/` segment `locates` matches on does not exist. That was caught by listing `dinov3-hf`
    # — which DOES report `vits16/...` prefixes — rather than by a kernel, which is where it was heading.
    # Naming the dataset for the encoder makes the mount path its own leaf and needs no directory at all.
    #
    # THE ONE ENCODER HERE THAT IS NOT NORMALISED LIKE THE OTHERS, and until 2026-08-25 it was fed the
    # ImageNet constants anyway, because the model registered them as buffers for every trunk alike. Its
    # own `preprocessor_config.json` says 0.5307 / 0.2583 — greyscale, one number repeated across the
    # three channels — so the trunk was seeing a 0.20-std offset with a 12.8% scale error. That makes
    # `raddino_dense`'s 0.8137 against `base_dense_s2026`'s 0.8512 a reading of the mismatch as much as
    # of the domain, on top of the run being TRUNCATED at epoch 6 of 12 (knee-8wm6). The exact-control
    # argument above is undamaged and unmeasured: swapping only the initialisation is still the clean
    # single-variable read, and it has not actually been taken yet.
    RAD_DINO: Encoder(
        RAD_DINO,
        "microsoft/rad-dino",
        768,
        mount="dimiturnt/rad-dino",
        family=RADDINO,
        leaf="rad-dino",
        params_m=86.6,
        normalization=Normalization((0.5307, 0.5307, 0.5307), (0.2583, 0.2583, 0.2583)),
    ),
    # THE CORPUS AXIS WITH EVERYTHING ELSE HELD, added 2026-09-03. This is the one comparison the trunk
    # table has never been able to make cleanly. RadioDINO is ViT-B/16, twelve blocks, 768 wide, trained
    # with DINO — the same architecture and the same objective as `v3base` and as `vitin21k`'s twin — and
    # it differs from them in the PRETRAINING CORPUS alone: 1.35M RadImageNet images against LVD-1689M
    # natural ones. Every previous family change here moved two or three variables at once, which is why
    # `knee-m69u` cannot be answered from the existing arms.
    #
    # IT IS NOT A REPEAT OF `radnet_family`, and the distinction is the whole reason it is worth a card.
    # That arm read 0.6716, converged, and closed the medical-SUPERVISED cell of the 2x2 above. These
    # weights are SELF-SUPERVISED on the same corpus, which is a different cell — so the two together
    # separate "RadImageNet is the wrong corpus" from "165-way supervision is the wrong objective", and
    # a flat here would say something the DenseNet result cannot say on its own.
    #
    # WHAT THE CURVE PREDICTS. `knee-y59c` measured distance-from-DINOv2 anti-correlating with solo
    # strength across seven trunks. An architectural and objective twin sits near zero on that axis while
    # its corpus sits far, so this entry is a direct test of WHICH of those two distances the curve was
    # actually measuring. That is an instrument reading, not a seat bid, and it should be priced as one.
    #
    # NO PUBLISHED KNEE NUMBER exists for these weights — the release benchmarks MedMNIST, BUSI and a
    # RadImageNet kNN. So there is no precedent to inherit and no number to replicate, only the mechanism.
    #
    # IMAGENET, DECLARED RATHER THAN MEASURED, and the two agree: the shipped `pretrained_cfg` names the
    # ImageNet mean and std outright, unlike RadImageNet's bare OrderedDict which documented two mutually
    # exclusive recipes and forced the post-`norm0` measurement recorded above.
    #
    # 224 IS THE PRETRAINED READ and 16 tiles it at 14x14, the same geometry as `vitin21k`. Larger reads
    # interpolate position embeddings, which every ViT here already does, so this may be swept — Swin's
    # window tables are the only geometry in this table that may not.
    #
    # NO MOUNT YET. CC-BY-4.0 and ungated, so mirroring is permitted with attribution and is the same
    # dataset upload `dimiturnt/swin-base-384` was; deferred until a member earns the seat. That licence
    # is also why this trunk is reachable at all where RadImageNet's own weights are not.
    RADIODINO_BASE: Encoder(
        RADIODINO_BASE,
        "Snarcy/RadioDino-b16",
        768,
        patch=16,
        family=RADIODINO,
        normalization=IMAGENET,
        normalization_evidence=(
            "The repository holds only `config.json` and the weights — a bare `ViTModel` config with no "
            "preprocessing document — so the recipe comes from the model card's own usage snippet, which "
            "resizes to 224 and calls `transforms.Normalize(mean=[0.485, 0.456, 0.406], "
            "std=[0.229, 0.224, 0.225])`. That is IMAGENET exactly. It is also what the DINO framework "
            "this trunk was pretrained under uses by default, so the card and the pretraining pipeline "
            "agree rather than the card being the only witness."
        ),
        leaf="radiodino-b16",
        params_m=86.6,
        # ITS `config.json` LIES ABOUT ITS KEYS, which is why this entry has a builder and the other
        # ViTs do not. The file names `ViTModel`, so `AutoModel.from_pretrained` builds the right
        # architecture — and the weights beside it are keyed the way timm keys them, so it matches not
        # one of them and returns a correctly-shaped ViT-B/16 at its random initialisation. That ran:
        # `radiodino_corpus` trained three epochs of a from-scratch trunk against a pre-registration
        # about medical pretraining. `refuse_unloaded` now refuses it; `TimmHubWeights` loads it.
        builder=TimmHubWeights("vit_base_patch16_224", 768, repository="Snarcy/RadioDino-b16").build,
    ),
    # THE PATCH-SIZE AXIS, added 2026-09-04, and the two rows below exist ONLY as a pair. Dehghan et al.
    # (arXiv:2602.18614) measured smaller patches beating 14/28 across twelve medical datasets — up to
    # +12.78% balanced accuracy in 2D — which is a real mechanism for a task whose findings are thin
    # localised structures read off a slice. But `s8` against `b16` alone would move patch size and trunk
    # width together, and this registry already has enough family changes that varied two things at once.
    # `s16` is the control that holds width at 384 so the pair reads patch size and nothing else.
    #
    # SAME PACKAGING FAULT AS `b16`. Both `config.json` files name `ViTModel` over timm-keyed weights, so
    # both load through `TimmHubWeights` and neither may use `from_hub`. Verified against the actual key
    # prefixes (`blocks`, `cls_token`, `patch_embed`, `pos_embed`), not inferred from the sibling entry.
    #
    # `s8` IS FOUR TIMES THE TOKENS at a fixed read — 28x28 against 14x14 at 224 — so its cost is a
    # sequence-length cost, not a parameter one, and an arm on it must be priced at that rather than at
    # its 21M parameters.
    RADIODINO_SMALL: Encoder(
        RADIODINO_SMALL,
        "Snarcy/RadioDino-s16",
        384,
        patch=16,
        family=RADIODINO,
        normalization=IMAGENET,
        normalization_evidence=(
            "Same release, same card, same usage snippet as `radiodinob16`: resize to 224 then "
            "`transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`, which is "
            "IMAGENET exactly and is also the DINO framework's own default."
        ),
        leaf="radiodino-s16",
        params_m=21.7,
        builder=TimmHubWeights("vit_small_patch16_224", 384, repository="Snarcy/RadioDino-s16").build,
    ),
    RADIODINO_SMALL_P8: Encoder(
        RADIODINO_SMALL_P8,
        "Snarcy/RadioDino-s8",
        384,
        patch=8,
        family=RADIODINO,
        normalization=IMAGENET,
        normalization_evidence=(
            "Same release, same card, same usage snippet as `radiodinob16`: resize to 224 then "
            "`transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`, which is "
            "IMAGENET exactly and is also the DINO framework's own default."
        ),
        leaf="radiodino-s8",
        params_m=21.4,
        builder=TimmHubWeights("vit_small_patch8_224", 384, repository="Snarcy/RadioDino-s8").build,
    ),
    # THE FIRST TRUNK PRETRAINED ON A DENSE OBJECTIVE, added 2026-09-04. Every encoder above was fitted
    # either without labels (DINOv2/v3, ConvNeXtV2, V-JEPA2, RadioDino) or on one label per image (Swin,
    # ViT-IN21k, RadImageNet). MedSAM was trained to SEGMENT — to say what each pixel belongs to — on
    # 1M+ medical images spanning modalities, musculoskeletal MRI among them. So it varies TWO axes this
    # registry has never varied together: the objective's spatial granularity, and a corpus that actually
    # contains the anatomy we read. That is the diversity claim, and it is about the objective rather
    # than the architecture: the trunk underneath is a plain ViT-B, the same shape as `v3base`.
    #
    # 336 IS THE ONLY LEGAL READ, for a different reason than `swinbase`'s. Swin's window tables are baked
    # at one size; SAM's POSITION EMBEDDINGS are, as a learned (1, 64, 64, 768) parameter sized for the
    # 1024 px read it was pretrained at. `SamVisionWeights.build` resamples that table to 21x21 ONCE, and
    # training then fits the resampled table — so the checkpoint is a function of this number and a run at
    # any other size would interpolate a second time off an already-interpolated table. The pin lives in
    # `MEDSAM_VIT_B.read_size`, which is where a reader looking for it will be, and `require_tiling` is
    # not sufficient to enforce it: 384 and 224 both tile patch 16 cleanly and both would be wrong.
    #
    # 768 IS THE TRUNK, AND THE NECK IS NOT READ. This entry declared 256 until 2026-09-04, because
    # SAM's encoder runs 12 blocks at 768 wide and then projects to 256 for the mask decoder — a decoder
    # this project never builds. `TokensInsteadOfNeck` replaces that projection at build, so `dim` and
    # `stages` now agree at 768 and the head receives what the blocks produced.
    #
    # DECIDED ON A MEASUREMENT, not on the architecture argument alone (knee-h8luo). The published case
    # against the neck is strong — it linear-probes at 11.1% ImageNet top-1 against DINOv2's 77.4%
    # (SAMantics, arXiv:2411.15288) and the one medical model stating a read-out point drops it
    # (FODA-PG, arXiv:2409.03947) — but MedSAM fine-tuned every image-encoder parameter and published no
    # layer-wise breakdown, so a neck-concentrated adaptation would have made this change discard the
    # medical part of a medical checkpoint. `kneemri.models.sam_drift` (retired, last at dd9a4566) measured it against
    # `facebook/sam-vit-base`: all twelve blocks moved 0.85-1.72 of their own initial norm and the neck
    # moved 0.68, the least of the fifteen regions bar the position table. The adaptation is in the
    # trunk.
    #
    # NO MOUNT YET. It can be trained here and NOT submitted. The upstream licence permits a mirror —
    # apache-2.0 and ungated, read from the HF API rather than the model card — so this is a job to do
    # when a member earns the seat, not a licence question of the kind RadImageNet's weights are.
    MEDSAM_BASE: Encoder(
        MEDSAM_BASE,
        MEDSAM_VIT_B.repository,
        768,
        patch=16,
        family=MEDSAM,
        # NOT the registry default. MedSAM's `preprocessor_config.json` declares mean [0, 0, 0] and std
        # [1, 1, 1] — rescale to [0, 1] and nothing further — and its upstream inference code agrees. The
        # default would have shifted every pixel by 0.485 and scaled it by 0.229 before a trunk that never
        # saw that distribution, with no shape anywhere to disagree.
        normalization=UNIT,
        leaf="medsam-vit-b",
        params_m=89.7,
        stages=(Stage(12, 16, 768),),
        adapter=TrunkAsTokens.from_map,
        builder=MEDSAM_VIT_B.build,
    ),
    # THE LARGEST IN-DOMAIN PRETRAINING CORPUS THIS PROJECT CAN REACH, added 2026-09-05. Six point one
    # million MRI slices across eighteen body locations, the knee among them, under the DINOv2 objective
    # — against `ortholarge`'s 1.25M knee images and `radnet121`'s 1.35M mixed radiographs. It is the
    # third and largest point on the corpus axis `knee-m69u` asks about, and the only one whose corpus is
    # MRI rather than radiographs: every previous medical trunk here was trained on projection imaging,
    # which shares neither the contrast mechanism nor the volumetric read.
    #
    # IT IS ViT-B/16 UNDER DINOv2, so `vitin21k` and `radiodinob16` are both architectural twins and the
    # corpus is again the single moving variable — the comparison `radnet_family` could not make, because
    # that trunk varied architecture, objective and corpus together and closed at 0.6716.
    #
    # WHAT THE CURVE PREDICTS, and why this is worth a card rather than a probe alone. `knee-y59c`
    # measured solo strength anti-correlating with distance from DINOv2 across seven trunks, and
    # `ortholarge` has since split that curve's two readings: it led every trunk FROZEN (+0.0203 on
    # `feature_probe`) and gave the lead back FINE-TUNED. So the open question is not whether in-domain
    # pretraining shows up — it is whether it survives a fine-tune, and `ortho_slow_trunk` is the arm
    # testing that on the rate. This trunk is the same question asked of a corpus twenty times commoner
    # and an objective identical to the one our own baselines were pretrained under.
    #
    # UNIT, DECLARED BY THE RELEASE AND NOT BY THIS REGISTRY'S DEFAULT. Their inference code feeds a
    # `FloatTensor` in [0, 1] with no mean or std applied, which is MedSAM's recipe above and not
    # `vitin21k`'s HALF. The default would have shifted every pixel by 0.485 before a trunk that never
    # saw that distribution — the failure `raddino_dense` already paid for once.
    #
    # 224 IS THE PRETRAINED READ and 16 tiles it at 14x14, the same geometry as `vitin21k`. Larger reads
    # interpolate the position table per forward, which `dynamic_img_size` does inside `TimmChunkedWeights`.
    #
    # NO MOUNT YET. Apache-2.0 and ungated, so mirroring is permitted outright; the release is on Google
    # Drive alone, so the Kaggle dataset is the only way a kernel can ever see it. Deferred until a
    # member earns the seat, which is the ordering `raddino` says to invert once one does.
    MRICORE_BASE: Encoder(
        MRICORE_BASE,
        # No hub id: the weights are published on Google Drive and reach this machine through the data
        # root's `reference/` kind, which is why the row carries a builder.
        "",
        768,
        patch=16,
        family=MRICORE,
        normalization=UNIT,
        normalization_evidence=(
            "The release ships no preprocessing document — the checkpoint is a bare `teacher` state "
            "dict — so the recipe comes from their own inference code, which builds the input as a "
            "`FloatTensor` in [0, 1] and applies no mean or std. That is UNIT exactly, and it is also "
            "what the SAM-derived pipeline the release wraps expects."
        ),
        leaf="mricore-vitb",
        params_m=85.8,
        stages=(Stage(12, 16, 768),),
        builder=MRICORE_VITB.build,
    ),
}
