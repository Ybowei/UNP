# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_multi_pipelines = dict(
    query=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ],
    support=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='CropResizeInstance', num_context_pixels=0, target_size=(256, 256)),
        dict(type='Normalize', **img_norm_cfg),
        #dict(type='GenerateMask', target_size=(224, 224)),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# classes splits are predefined in FewShotVOCDataset
data_root = '/data/huang1/PHD_Works/Few_Shot_Object_Detection/mmfewshot/data/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='NWayKShotDataset',
        num_support_ways=15,
        num_support_shots=1,
        one_support_shot_per_image=True,
        num_used_support_shots=200,
        save_dataset=False,
        dataset=dict(
            type='FewShotDIORDataset',
            ann_cfg=[
                dict(
                    type='ann_file',
                    ann_file=data_root + 'DIOR/ImageSets/Main/trainval.txt')
            ],
            img_prefix=data_root,
            multi_pipelines=train_multi_pipelines,
            classes=None,
            use_difficult=True,
            instance_wise=False,
            dataset_name='query_dataset'),
        support_dataset=dict(
            type='FewShotDIORDataset',
            ann_cfg=[
                dict(
                    type='ann_file',
                    ann_file=data_root + 'DIOR/ImageSets/Main/trainval.txt')
            ],
            img_prefix=data_root,
            multi_pipelines=train_multi_pipelines,
            classes=None,
            use_difficult=False,
            instance_wise=False,
            dataset_name='support_dataset')),
    val=dict(
        type='FewShotDIORDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'DIOR/ImageSets/Main/test.txt')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=None),
    test=dict(
        type='FewShotDIORDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'DIOR/ImageSets/Main/test.txt')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        classes=None),
    model_init=dict(
        copy_from_train_dataset=True,
        samples_per_gpu=16,
        workers_per_gpu=1,
        type='FewShotDIORDataset',
        ann_cfg=None,
        img_prefix=data_root,
        pipeline=train_multi_pipelines['support'],
        use_difficult=False,
        instance_wise=True,
        classes=None,
        dataset_name='model_init_dataset'))
evaluation = dict(interval=5000, metric='mAP')
