# Copyright (c) OpenMMLab. All rights reserved.
from .supervised_contrastive_loss import SupervisedContrastiveLoss
from .focal_loss import CEFocalLoss
#from .pisa_loss import isr_p, carl_loss

__all__ = ['SupervisedContrastiveLoss', 'CEFocalLoss']   # 'isr_p', 'carl_loss'
