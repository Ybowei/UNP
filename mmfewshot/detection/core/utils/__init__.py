# Copyright (c) OpenMMLab. All rights reserved.
from .custom_hook import ContrastiveLossDecayHook
from .classes import COCO_CLASSES, VOC_CLASSES

__all__ = ['ContrastiveLossDecayHook', 'COCO_CLASSES', 'VOC_CLASSES']
