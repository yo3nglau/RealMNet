_base_ = [
    '../_base_/datasets/MYOPIA_380.py',
    '../_base_/default_runtime.py',
]

pretrained = "https://download.openmmlab.com/mmclassification/v0/efficientnet/" \
             "efficientnet-b4_3rdparty_8xb32_in1k_20220119-81fd4077.pth"

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EfficientNet',
        arch='b4',
        drop_path_rate=0.5,
        init_cfg=[
            dict(type='Kaiming', layer='Conv2d'),
            dict(
                type='Constant',
                layer=['_BatchNorm', 'GroupNorm'],
                val=1),
            dict(type='Pretrained', checkpoint=pretrained, prefix='backbone'),
        ],
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=7,
        in_channels=1792,
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
        lr=5e-5,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
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

NOTES = 'Benchmarking, EfficientNet-B4-380'

visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project='benchmarking', notes=NOTES)),
    ]
)
