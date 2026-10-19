# ---------------------------------------------------------------
# SegMAN-T + BDFR + PLLA-v3 + HBPH
# COCO-Stuff-164K fair-comparison / performance-priority config
#
# Fairness constraints relative to the original SegMAN-T COCO setup:
#   - same COCO-Stuff-164K dataset
#   - same 512x512 data pipeline inherited from the official dataset config
#   - same 160K training budget
#   - same AdamW base LR = 6e-5
#   - same effective total batch size target = 16
#   - same single-scale whole-image validation protocol
#
# Performance optimization is restricted to the three proposed modules and
# their training schedules, so any gain remains directly comparable.
# ---------------------------------------------------------------

_base_ = [
    '../../_base_/models/segman.py',
    '../../_base_/datasets/coco-stuff164k.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

# Keep consistent with the original SegMAN COCO configuration.
find_unused_parameters = True


model = dict(
    type='EncoderDecoder',
    pretrained=None,

    backbone=dict(
        type='SegMANEncoder_t',
        pretrained='/root/autodl-tmp/pretrained/SegMAN_Encoder_t.pth.tar',
        style='pytorch'),

    decode_head=dict(
        type='SegMANDecoder',
        in_channels=[32, 64, 144, 192],
        in_index=[0, 1, 2, 3],
        channels=128,
        feat_proj_dim=192,
        dropout_ratio=0.1,

        # COCO-Stuff-164K has 171 semantic classes.
        num_classes=171,

        norm_cfg=norm_cfg,
        align_corners=False,


        # =======================================================
        # A. BDFR
        # Boundary-conditioned dual-frequency feature refinement
        # =======================================================
        #
        # ADE20K verified setting used 0.25.
        # COCO is more class-imbalanced and contains many small regions,
        # so reduce the auxiliary boundary pressure slightly while still
        # keeping BDFR strong enough to refine thin/small structures.
        #
        use_bdfr=True,
        bdfr_channels=32,
        bdfr_loss_weight=0.20,
        bdfr_dice_weight=1.0,
        bdfr_boundary_width=3,
        bdfr_auto_pos_weight=True,
        bdfr_pos_weight_max=10.0,

        # Keep the same warmup length as the successful ADE20K recipe.
        # This allows BDFR to come online early without dominating the
        # first few thousand iterations.
        bdfr_loss_warmup_iters=4000,

        bdfr_residual_gate_init=0.50,
        bdfr_residual_gate_scale=1.00,


        # =======================================================
        # B. PLLA-v3
        # Boundary-aware rank-gated low-rank logit adapter
        # =======================================================
        #
        # Keep the ADE20K-verified structure unchanged.
        # PLLA directly modifies inference logits, so preserving the
        # proven rank / scale setting is safer than increasing capacity
        # simply because COCO has more classes.
        #
        use_plla_adapter=True,
        plla_rank=32,
        plla_hidden_channels=64,
        plla_scale_init=0.20,
        plla_max_scale=0.40,
        plla_gate_init_bias=-1.0,
        plla_detach_prob=True,

        # Old ablation branches remain disabled.
        use_ucp_offset=False,
        use_dynamic_class_offset=False,


        # =======================================================
        # C. HBPH
        # Training-only hyperbolic auxiliary CE branch
        # =======================================================
        #
        # COCO's larger / more long-tailed class space is a reasonable
        # place for HBPH to help representation learning, but the actual
        # inference classifier should dominate near convergence.
        #
        use_hbph_aux=True,

        # Slightly weaker than the successful ADE20K 0.25 setting.
        hbph_aux_loss_weight=0.20,

        # Decay to a small non-zero regularization signal late in training.
        hbph_aux_min_loss_weight=0.05,

        # Keep the successful ADE20K warmup duration.
        hbph_aux_warmup_iters=8000,

        # Match the original 160K budget:
        # full HBPH influence through the main learning stage,
        # then progressively reduce train/test objective mismatch
        # during the final 40K iterations.
        hbph_aux_decay_start=120000,
        hbph_aux_decay_end=160000,

        # HBPH must backpropagate into the shared decoder feature.
        hbph_aux_detach_feature=False,

        hbph_adapter_bottleneck=64,
        hbph_init_input_scale=0.10,
        hbph_min_input_scale=0.05,
        hbph_max_input_scale=1.00,
        hbph_K=-1.0,
        hbph_dropout=0.0,

        # Initialize hyperbolic class directions/scales from conv_seg.
        hbph_init_from_classifier=True,


        # Main segmentation objective remains identical in form to SegMAN.
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),

    train_cfg=dict(),

    # Keep fair single-scale whole-image evaluation.
    test_cfg=dict(mode='whole')
)


# ===============================================================
# Optimizer
# ===============================================================
#
# Keep the original SegMAN base optimizer / LR.
# ImageNet-pretrained encoder uses base LR = 6e-5.
# Decoder and newly introduced modules use the verified larger LR.
# ===============================================================

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,

    paramwise_cfg=dict(custom_keys={
        'pos_block': dict(decay_mult=0.0),
        'norm': dict(decay_mult=0.0),

        # Original SegMAN decode head.
        'head': dict(lr_mult=10.0),

        # BDFR.
        'bdfr': dict(lr_mult=10.0),

        # PLLA-v3.
        # Class embedding is a class-specific low-rank direction matrix;
        # do not shrink it with weight decay.
        'plla_adapter.class_embed': dict(
            lr_mult=10.0,
            decay_mult=0.0),

        # The global PLLA residual amplitude is sensitive; learn it with
        # half the branch LR and without weight decay.
        'plla_adapter.scale_raw': dict(
            lr_mult=5.0,
            decay_mult=0.0),

        'plla_adapter': dict(lr_mult=10.0),

        # Keep HBPH on the ADE20K-verified branch LR.
        'hbmlr': dict(lr_mult=10.0),
    })
)


# ===============================================================
# LR schedule
# ===============================================================
#
# Keep the successful SegMAN/ADE recipe unchanged.
# schedule_160k_adamw.py supplies the 160K IterBasedRunner budget.
# ===============================================================

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)


# ===============================================================
# Data loader
# ===============================================================
#
# Adjust samples_per_gpu according to GPU count so that:
#
#       total effective batch size = 16
#
# Example:
#   4 GPUs -> 4 samples/GPU
#   2 GPUs -> 8 samples/GPU
#
# ===============================================================

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)


# ===============================================================
# Evaluation / checkpoint
# ===============================================================
#
# Evaluation is more frequent than the previous ADE config so the best
# validation checkpoint is less likely to be missed. This does NOT alter
# the 160K optimization budget or the model training objective.
# ===============================================================

evaluation = dict(
    interval=2000,
    metric='mIoU',
    save_best='mIoU')

checkpoint_config = dict(
    by_epoch=False,
    interval=4000,
    max_keep_ckpts=3)
