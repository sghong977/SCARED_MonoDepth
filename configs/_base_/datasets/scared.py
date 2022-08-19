# dataset settings Only for test
full_w, full_h = 600, 480 #1280, 720
crop_w, crop_h = 600, 480 #1280, 720
depth_scale,max_depth = 100, 192
dataset_type = 'SCAREDDataset'
data_root = '../scared/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size= (crop_w, crop_h)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='DepthLoadAnnotations'),
    #dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.0),
    #dict(type='RandomCrop', crop_size=(crop_w, crop_h)),
    dict(type='ColorAug', prob=0.5, gamma_range=[0.9, 1.1], brightness_range=[0.75, 1.25], color_range=[0.9, 1.1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'depth_gt']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(full_w, full_h)),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(full_w, full_h),
        flip=False, #True,
        #flip_direction='horizontal',
        transforms=[
            #dict(type='RandomFlip', direction='horizontal'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        depth_scale=depth_scale,
        split='train',
        pipeline=train_pipeline,
        #garg_crop=False,
        #eigen_crop=True,
        min_depth=1e-3,
        max_depth=max_depth),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        depth_scale=depth_scale,
        split='test',
        pipeline=test_pipeline,
        #garg_crop=False,
        #eigen_crop=True,
        min_depth=1e-3,
        max_depth=max_depth),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        min_depth=1e-3,
        max_depth=max_depth))


