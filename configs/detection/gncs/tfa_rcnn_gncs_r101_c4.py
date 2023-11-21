_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_c4.py',
]
num_support_ways = 20   # coco setting
num_support_shots = 1
# model settings
norm_cfg = dict(type='BN', requires_grad=False)
pretrained = 'open-mmlab://detectron2/resnet101_caffe'
model = dict(
    type='TFA_GNCS',
    pretrained=pretrained,
    backbone=dict(depth=101, frozen_stages=1),
    roi_head=dict(
        type='NegCorRoIHead',
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots),
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(
                type='UNPSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
                floor_thr=0.1,
                num_bins=5,
                alpha=0.25,
                gamma=20,
                iou_thr=0.5
            ),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=6000,
            max_per_img=300,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
