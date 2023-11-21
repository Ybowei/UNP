_base_ = [
    '../../../../_base_/datasets/nway_kshot_sampling_sty/few_shot_voc.py',
    '../../../../_base_/schedules/schedule.py', '../../../tfa_rcnn_gncs_r101_c4.py',
    '../../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_multi_pipelines = dict(
    query=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='Resize',
            img_scale=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                       (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                       (1333, 736), (1333, 768), (1333, 800)],
            keep_ratio=True,
            multiscale_mode='value'),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
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
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data_root = '/data/huang1/PHD_Works/Few_Shot_Object_Detection/mmfewshot/data/VOCdevkit/'
data = dict(
    train=dict(
        dataset = dict(
            type='FewShotVOCDefaultDataset',
            ann_cfg=[dict(method='TFA', setting='SPLIT2_3SHOT')],
            num_novel_shots=3,
            num_base_shots=3,
            img_prefix=data_root,
            multi_pipelines=train_multi_pipelines,
            classes='ALL_CLASSES_SPLIT2',
            use_difficult=False,
            instance_wise=False,
            one_instance_per_image=False,
            dataset_name='query_dataset'
            ),
        support_dataset=dict(
            type='FewShotVOCDefaultDataset',
            ann_cfg=[dict(method='TFA_gncs', setting='SPLIT2_3SHOT')],
            num_novel_shots=3,
            img_prefix=data_root,
            multi_pipelines=train_multi_pipelines,
            classes='NOVEL_CLASSES_SPLIT2',
            use_difficult=False,
            instance_wise=False,
            one_instance_per_image=False,
            dataset_name='support_dataset')),
    val=dict(classes='ALL_CLASSES_SPLIT2'),
    test=dict(classes='ALL_CLASSES_SPLIT2'))

num_support_ways = 5   # voc setting
num_support_shots = 1
model = dict(
    backbone=dict(depth=101, frozen_stages=3),
    frozen_parameters=['backbone', 'rpn_head', 'roi_head.shared_head'],    #  'backbone', 'rpn_head', 'roi_head.shared_head'
    roi_head=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        shared_head=dict(
            pretrained='open-mmlab://detectron2/resnet101_caffe',
            depth=101),
        bbox_head=dict(num_classes=20)))

evaluation = dict(
    interval=2000,
    class_splits=['BASE_CLASSES_SPLIT2', 'NOVEL_CLASSES_SPLIT2'])
checkpoint_config = dict(interval=2000)
optimizer = dict(lr=0.0005)
lr_config = dict(warmup_iters=200, step=[6000,])
runner = dict(max_iters=8000)
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/fsce/README.md for more details.
load_from = '/data/huang1/PHD_Works/Few_Shot_Object_Detection/mmfewshot/checkpoints/gncs/transfor/voc/tfa_r101_c4_4xb2_voc_split2_base_training_random_init_bbox_head.pth'
work_dir = '/data/huang1/PHD_Works/Few_Shot_Object_Detection/mmfewshot/work_dirs/gncs_mmfewshot_split/transfor/voc/split2/3shot_ft_cls_head'