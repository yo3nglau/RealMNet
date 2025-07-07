import os

# dataset settings
dataset_type = 'MYOPIA'
data_root = 'datasets/PSMM/'
resampling = 'MYOPIA_MLRUS-20'
image_set_dir_train = os.path.join(data_root, f'ImageSets/Main/{resampling}')
image_set_dir_val_test = os.path.join(data_root, f'ImageSets/Main')

data_preprocessor = dict(
    num_classes=7,
    # RGB format normalization parameters
    mean=[0, 0, 0],
    std=[255, 255, 255],
    # convert image from BGR to RGB
    to_rgb=True,
    # generate onehot-format labels for multi-label classification
    to_onehot=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomErasing', erase_prob=0.3, min_area_ratio=0.05, max_area_ratio=0.3, aspect_range=(0.3, 3.3)),
    dict(type='Albumentations',
         transforms=[
             dict(type='Affine', p=0.3, scale=[0.8, 1.2], rotate=15, shear=10),
             dict(type='HorizontalFlip', p=0.3),
             dict(type='VerticalFlip', p=0.1),
             dict(type='GaussianBlur', p=0.3, blur_limit=(7, 7), sigma_limit=(0.1, 1)),
             dict(type='ColorJitter', p=0.3, brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
             dict(type='GaussNoise', p=0.3, var_limit=(10.0, 50.0), per_channel=True),
         ],
    ),
    dict(type='PackInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        image_set_path=os.path.join(image_set_dir_train, f'{resampling}.csv'),
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        image_set_path=os.path.join(image_set_dir_val_test, 'val.csv'),
        split='val',
        pipeline=val_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        image_set_path=os.path.join(image_set_dir_val_test, 'test.csv'),
        split='test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# calculate precision_recall_f1 and mAP
val_evaluator = [
    dict(type='MultiLabelMetric', thr=0.5, collect_device='gpu'),
    dict(type='MultiLabelMetric', thr=0.6, prefix='multi-label-thr', collect_device='gpu'),
    dict(type='MultiLabelMetric', thr=0.7, prefix='multi-label-thr', collect_device='gpu'),
    dict(type='MultiLabelMetric', thr=0.8, prefix='multi-label-thr', collect_device='gpu'),
    dict(type='MultiLabelMetric', topk=2, prefix='multi-label-top2', collect_device='gpu'),
    dict(type='AveragePrecision', collect_device='gpu'),
    dict(type='AUC', collect_device='gpu'),
    dict(type='HammingLoss', collect_device='gpu'),
    dict(type='Coverage', collect_device='gpu'),
    dict(type='RankingLoss', collect_device='gpu'),
    dict(type='MultiLabelMetric', average='micro', prefix='PRF1_micro', collect_device='gpu'),
    dict(type='MultiLabelMetric', average=None, prefix='PRF1_class-wise', collect_device='gpu'),
    dict(type='AveragePrecision', average=None, prefix='mAP_class-wise', collect_device='gpu'),
    dict(type='AUC', average=None, prefix='AUC_class-wise', collect_device='gpu'),
    dict(type='BrierLoss', prefix='BrierLoss_class-wise', collect_device='gpu'),
]

# calculate precision_recall_f1 and mAP
test_evaluator = val_evaluator
