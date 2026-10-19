_base_ = [
    '../../_base_/models/segman.py',
    '../../_base_/datasets/cityscapes_1024x1024_repeat.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

# 官方 Cityscapes 配置中有；多卡训练建议保留
find_unused_parameters = True

model = dict(
    type='EncoderDecoder',
    pretrained=None,

    backbone=dict(
        type='SegMANEncoder_t',
        pretrained='/root/autodl-tmp/pretrained/SegMAN_Encoder_t.pth.tar',
        style='pytorch'
    ),

    decode_head=dict(
        type='SegMANDecoder',
        in_channels=[32, 64, 144, 192],
        in_index=[0, 1, 2, 3],
        channels=128,
        feat_proj_dim=192,
        dropout_ratio=0.1,

        num_classes=19,

        norm_cfg=norm_cfg,
        align_corners=False,

        # ---------------- BDFR ----------------
        use_bdfr=True,
        bdfr_channels=32,

        bdfr_loss_weight=0.04,

        bdfr_dice_weight=1.0,
        bdfr_boundary_width=3,
        bdfr_auto_pos_weight=True,
        bdfr_pos_weight_max=10.0,
        bdfr_loss_warmup_iters=4000,

        bdfr_residual_gate_init=0.50,
        bdfr_residual_gate_scale=1.00,


        # ---------------- PLLA ----------------
        use_plla_adapter=True,

        plla_rank=16,

        plla_hidden_channels=64,

        plla_scale_init=0.12,
        plla_max_scale=0.30,

        plla_gate_init_bias=-1.25,
        plla_detach_prob=True,

        use_ucp_offset=False,
        use_dynamic_class_offset=False,


        # ---------------- HBPH ----------------
        use_hbph_aux=True,

        hbph_aux_loss_weight=0.10,
        hbph_aux_min_loss_weight=0.02,

        hbph_aux_warmup_iters=8000,

        hbph_aux_decay_start=100000,
        hbph_aux_decay_end=160000,

        hbph_aux_detach_feature=False,

        hbph_adapter_bottleneck=64,
        hbph_init_input_scale=0.10,
        hbph_min_input_scale=0.05,
        hbph_max_input_scale=1.00,
        hbph_K=-1.0,
        hbph_dropout=0.0,
        hbph_init_from_classifier=True,

        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),

    train_cfg=dict(),

    test_cfg=dict(
        mode='slide',
        crop_size=(1024, 1024),
        stride=(768, 768)
    )
)


optimizer = dict(
    _delete_=True,
    type='AdamW',

    # 不动
    lr=0.00006,

    betas=(0.9, 0.999),
    weight_decay=0.01,

    paramwise_cfg=dict(
        custom_keys={

            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),

            # Original decoder
            'head': dict(lr_mult=10.0),

            # Our modules
            'bdfr': dict(lr_mult=10.0),

            'plla_adapter.class_embed': dict(
                lr_mult=10.0,
                decay_mult=0.0
            ),

            'plla_adapter.scale_raw': dict(
                lr_mult=2.0,
                decay_mult=0.0
            ),

            'plla_adapter': dict(
                lr_mult=5.0
            ),

            'hbmlr': dict(
                lr_mult=5.0
            ),
        }
    )
)


# Original SegMAN Cityscapes schedule: keep unchanged
lr_config = dict(
    _delete_=True,
    policy='poly',

    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,

    power=1.0,
    min_lr=0.0,
    by_epoch=False
)


# 你的batch设置保持不动
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4
)


# 我建议仍然2k eval一次
evaluation = dict(
    interval=2000,
    metric='mIoU',
    save_best='mIoU'
)

checkpoint_config = dict(
    by_epoch=False,
    interval=4000,
    max_keep_ckpts=3
)
