_base_ = [
    '../../_base_/models/segman.py',
    '../../_base_/datasets/ade20k_repeat.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# SegMAN-T + BDFR + boundary-aware rank-gated PLLA-v3 + HBPH auxiliary CE.
# 44-target experiment: BDFR/HBPH and the global training recipe are kept
# aligned with the verified 43.61 Full run; only PLLA is structurally upgraded.
# "From scratch" here follows the original SegMAN setup: train the segmentation
# model for 160K iterations with the ImageNet-pretrained SegMAN encoder.

norm_cfg = dict(type='SyncBN', requires_grad=True)

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
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,

        # A. BDFR: boundary-conditioned dual-frequency feature refinement.
        use_bdfr=True,
        bdfr_channels=32,
        bdfr_loss_weight=0.25,
        bdfr_dice_weight=1.0,
        bdfr_boundary_width=3,
        bdfr_auto_pos_weight=True,
        bdfr_pos_weight_max=10.0,
        bdfr_loss_warmup_iters=4000,
        bdfr_residual_gate_init=0.50,
        bdfr_residual_gate_scale=1.00,

        # B. PLLA-v3: boundary-aware rank-gated low-rank logit adapter.
        # BDFR boundary probability + semantic uncertainty jointly control
        # rank-wise corrections. class_embed is zero-init, so the adapter is
        # still an exact no-op at the beginning of training.
        use_plla_adapter=True,
        plla_rank=32,
        plla_hidden_channels=64,
        plla_scale_init=0.20,
        plla_max_scale=0.40,
        plla_gate_init_bias=-1.0,
        plla_detach_prob=True,

        # Keep old PLLA ablation switches off in the merged model.
        use_ucp_offset=False,
        use_dynamic_class_offset=False,

        # C. HBPH: training-only hyperbolic auxiliary CE branch.
        # It shapes the refined decoder feature during training and is not used
        # for inference logits. The weight is warmed up, then decayed late to
        # reduce train/test objective mismatch near convergence.
        use_hbph_aux=True,
        hbph_aux_loss_weight=0.25,
        hbph_aux_min_loss_weight=0.10,
        hbph_aux_warmup_iters=8000,
        hbph_aux_decay_start=120000,
        hbph_aux_decay_end=160000,
        hbph_aux_detach_feature=False,
        hbph_adapter_bottleneck=64,
        hbph_init_input_scale=0.10,
        hbph_min_input_scale=0.05,
        hbph_max_input_scale=1.00,
        hbph_K=-1.0,
        hbph_dropout=0.0,
        hbph_init_from_classifier=True,

        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={
        'pos_block': dict(decay_mult=0.0),
        'norm': dict(decay_mult=0.0),
        # This follows the original SegMAN config: decode_head parameters use
        # 10x LR. More specific keys are listed for clarity and future ablation.
        'head': dict(lr_mult=10.0),
        'bdfr': dict(lr_mult=10.0),

        # PLLA-v3: keep the verified 10x decoder LR, but do not decay the
        # class embedding or the scalar residual strength.
        'plla_adapter.class_embed': dict(lr_mult=10.0, decay_mult=0.0),
        'plla_adapter.scale_raw': dict(lr_mult=5.0, decay_mult=0.0),
        'plla_adapter': dict(lr_mult=10.0),

        # Keep HBPH exactly on the 43.61-version branch LR.
        'hbmlr': dict(lr_mult=10.0),
    })
)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# Keep this consistent with your original fair-comparison setting. If you train
# on multiple GPUs, adjust samples_per_gpu so the total batch remains 16.
data = dict(samples_per_gpu=4, workers_per_gpu=4)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
