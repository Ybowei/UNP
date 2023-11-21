_base_ = [
    '../../_base_/datasets/fine_tune_based/base_coco.py',
    '../../_base_/schedules/schedule.py',
    '../../_base_/models/faster_rcnn_r50_caffe_c4.py',
    '../../_base_/default_runtime.py'
]
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
lr_config = dict(warmup_iters=1000, step=[255000, 300000])
runner = dict(max_iters=360000)
# model settings
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(bbox_head=dict(num_classes=60)))
evaluation = dict(interval=60000, metric='bbox', classwise=True)
checkpoint_config = dict(interval=10000)
work_dir = '/data/huang1/PHD_Works/Few_Shot_Object_Detection/mmfewshot/work_dirs/coco/TFA_baseline/base/tfa_no_fpn'