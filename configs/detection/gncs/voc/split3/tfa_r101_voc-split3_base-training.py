_base_ = [
    '../../../_base_/datasets/fine_tune_based/base_voc.py',
    '../../../_base_/schedules/schedule.py',
    '../../../_base_/models/faster_rcnn_r50_caffe_c4.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
data = dict(
    train=dict(classes='BASE_CLASSES_SPLIT3'),
    val=dict(classes='BASE_CLASSES_SPLIT3'),
    test=dict(classes='BASE_CLASSES_SPLIT3'))
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
lr_config = dict(warmup_iters=100, step=[20000, 32000])
runner = dict(max_iters=36000)
# model settings
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(bbox_head=dict(num_classes=15)))
# using regular sampler can get a better base model
use_infinite_sampler = False
work_dir = '/data/huang1/PHD_Works/Few_Shot_Object_Detection/mmfewshot/work_dirs/voc/TFA_baseline/split3/base/tfa_no_fpn'
evaluation = dict(interval=6000, metric='mAP')