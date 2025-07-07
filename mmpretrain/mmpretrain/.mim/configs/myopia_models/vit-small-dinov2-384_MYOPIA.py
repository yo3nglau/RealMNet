_base_ = [
    '../_base_/datasets/MYOPIA.py',
    '../_base_/default_runtime.py',
]

# load model pretrained on imagenet
pretrained = 'https://download.openmmlab.com/mmpretrain/v1.0/dinov2/' \
             'vit-small-p14_dinov2-pre_3rdparty_20230426-5641ca5a.pth'

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='dinov2-small',
        img_size=384,
        patch_size=14,
        layer_scale_init_value=1e-5,
        drop_path_rate=0.7,
        init_cfg=[
            dict(type='TruncNormal',
                 layer=['Conv2d', 'Linear'],
                 std=.02,
                 bias=0.),
            dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.),
            dict(type='Pretrained', checkpoint=pretrained, prefix='backbone'),
        ],
    ),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=7,
        in_channels=384,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_sigmoid=True),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.6),
        dict(type='CutMix', alpha=1.0),
    ], probs=[0.3, 0.3],
    ),
)

# schedule setting
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0),
        }),
    clip_grad=dict(max_norm=1., norm_type=2),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-2,
        by_epoch=True,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-6, by_epoch=True, begin=5),
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)
val_cfg = dict()
test_cfg = dict()

NOTES = 'Benchmarking, vit-small-dinov2-384, layer_scale_init_value=1e-5, drop_path_rate=0.7'

visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project='revision', notes=NOTES)),
    ]
)
