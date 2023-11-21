# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseFewShotDataset, BaseXMLFewShotDataset
from .builder import build_dataloader, build_dataset
from .coco import COCO_SPLIT, FewShotCocoDataset
from .dataloader_wrappers import NWayKShotDataloader
from .dataset_wrappers import NWayKShotDataset, QueryAwareDataset, NWayKShotSamDataset
from .pipelines import CropResizeInstance, GenerateMask
from .utils import NumpyEncoder, get_copy_dataset_type
from .voc import VOC_SPLIT, FewShotVOCDataset, FewShotVOCDataset_Revision  #FewShotVOCDefaultDataset_Revision
from .dior import DIOR_SPLIT, FewShotDIORDataset
from .semi_dataset import SemiDataset
from .txt_style import TXTDataset
from .fewshot_semi_dataset import FewShotSemiDataset


"""=====================================================================================

Adding new sections - [ 'DIOR_SPLIT', 'FewShotDIORDataset', 'BaseXMLFewShotDataset'
                        

====================================================================================="""

__all__ = [
    'build_dataloader', 'build_dataset', 'QueryAwareDataset',
    'NWayKShotDataset', 'NWayKShotDataloader', 'BaseFewShotDataset',
    'FewShotVOCDataset', 'FewShotCocoDataset', 'CropResizeInstance',
    'GenerateMask', 'NumpyEncoder', 'COCO_SPLIT', 'VOC_SPLIT',
    'get_copy_dataset_type', 'DIOR_SPLIT', 'FewShotDIORDataset',
    'BaseXMLFewShotDataset', 'FewShotVOCDataset_Revision',  #'FewShotVOCDefaultDataset_Revision'
    'SemiDataset', 'TXTDataset', 'FewShotSemiDataset', 'NWayKShotSamDataset'
]
