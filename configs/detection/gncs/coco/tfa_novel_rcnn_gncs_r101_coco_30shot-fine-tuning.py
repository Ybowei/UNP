_base_ = [
    '../../../_base_/datasets/nway_kshot_sampling_sty/few_shot_coco.py',
    '../../../_base_/schedules/schedule.py', '../../tfa_rcnn_gncs_r101_c4.py',
    '../../../_base_/default_runtime.py'
]
# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_multi_pipelines = dict(
    query=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='Resize',
            img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                       (1333, 768), (1333, 800)],
            keep_ratio=True,
            multiscale_mode='value'),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ],
    support=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='CropResizeInstance',
            num_context_pixels=16,
            target_size=(320, 320)),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
num_support_ways = 20   # coco setting
num_support_shots = 1
data_root = '/data/huang1/PHD_Works/Few_Shot_Object_Detection/mmfewshot/data/coco/'
data = dict(
    train=dict(
        type='NWayKShotSamDataset',
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        one_support_shot_per_image=False,
        num_used_support_shots=30,
        save_dataset=True,
        dataset = dict(
            type='FewShotCocoDefaultDataset',
            ann_cfg=[dict(method='TFA', setting='30SHOT')],
            img_prefix=data_root,
            multi_pipelines=train_multi_pipelines,
            num_novel_shots=30,
            num_base_shots=30,
            classes='ALL_CLASSES',
            instance_wise=False,
            dataset_name='query_dataset'
            ),
        support_dataset=dict(
            type='FewShotCocoDefaultDataset',
            ann_cfg=[dict(method='TFA_gncs', setting='30SHOT')],
            num_novel_shots=30,
            img_prefix=data_root,
            multi_pipelines=train_multi_pipelines,
            classes='NOVEL_CLASSES',
            instance_wise=False,
            dataset_name='support_dataset')),
    val=dict(classes='ALL_CLASSES'),
    test=dict(
        pipeline=test_pipeline,
        classes='ALL_CLASSES'))

model = dict(
    backbone=dict(depth=101, frozen_stages=3),
    frozen_parameters=['backbone', 'rpn_head', 'roi_head.shared_head'],    #  'backbone', 'rpn_head', 'roi_head.shared_head'
    roi_head=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        shared_head=dict(
            pretrained='open-mmlab://detectron2/resnet101_caffe',
            depth=101),
        bbox_head=dict(num_classes=80)))

evaluation = dict(interval=6000)
checkpoint_config = dict(interval=12000)
optimizer = dict(lr=0.0005)
lr_config = dict(warmup_iters=100, step=[84000])
runner = dict(max_iters=120000)
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/fsce/README.md for more details.
load_from = '/data/huang1/PHD_Works/Few_Shot_Object_Detection/mmfewshot/checkpoints/gncs/transfor/coco/base_model_random_init_bbox_head.pth'
work_dir = '/data/huang1/PHD_Works/Few_Shot_Object_Detection/mmfewshot/work_dirs/gncs_mmfewshot_split/transfor/coco/30shot'