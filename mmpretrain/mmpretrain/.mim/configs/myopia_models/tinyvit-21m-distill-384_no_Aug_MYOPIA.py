_base_ = [
    '../_base_/datasets/MYOPIA_no_SA.py',
    '../_base_/default_runtime.py',
]

# load model pretrained on imagenet
pretrained = 'https://download.openmmlab.com/mmclassification/v0/tinyvit/' \
             'tinyvit-21m_in21k-distill-pre_3rdparty_in1k-384px_20221021-65be6b3f.pth'

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='TinyViT',
        arch='21m',
        img_size=(384, 384),
        window_size=[12, 12, 24, 12],
        drop_path_rate=0.5,
        out_indices=(3,),
        gap_before_final_norm=True,
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
        in_channels=576,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_sigmoid=True),
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

NOTES = 'TinyViT-21m-384, MYOPIA_MLRUS-20, GPU 4090, CrossEntropyLoss, no Augmentation'

visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project='benchmarking', notes=NOTES)),
    ]
)
