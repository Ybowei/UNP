# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
import torch
import numpy as np
from torch.utils.data import Dataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.pipelines.formatting import to_tensor
from mmfewshot.detection.datasets.coco import FewShotCocoDataset
from mmcv.parallel import DataContainer as DC

@DATASETS.register_module()
class FewShotSemiDataset(Dataset):
    #CLASSES = COCO_CLASSES

    def __init__(self,
                 ann_cfg,
                 ann_cfg_u,
                 pipeline,
                 pipeline_share,
                 pipeline_imcom,
                 pipeline_com,
                 img_prefix,
                 img_prefix_u,
                 classes=None,
                 num_novel_shots=None,
                 num_base_shots=None,
                 ann_shot_filter=None,
                 min_bbox_area=None,
                 dataset_name=None,
                 test_mode=False):
        super().__init__()

        self.coco_imcomplete = FewShotCocoDataset(
            ann_cfg=ann_cfg, pipeline=pipeline, img_prefix=img_prefix,     # pipeline=pipeline_share
            classes=classes, num_novel_shots=num_novel_shots, num_base_shots=num_base_shots, ann_shot_filter=ann_shot_filter,
            min_bbox_area=min_bbox_area, dataset_name=dataset_name, test_mode=test_mode)
        self.coco_complete = FewShotCocoDataset(
            ann_cfg=ann_cfg_u, pipeline=pipeline, img_prefix=img_prefix_u,
            classes=classes, num_novel_shots=None, num_base_shots=None, ann_shot_filter=None,
            min_bbox_area=min_bbox_area, dataset_name=dataset_name, test_mode=test_mode)

        assert self.judge_img_same_order(), \
            f'{self.coco_imcomplete} and {self.coco_complete} should have same image index order.'

        self.CLASSES = self.coco_imcomplete.CLASSES
        self.pipeline_share = Compose(pipeline_share)
        self.pipeline_imcom = Compose(pipeline_imcom)
        self.pipeline_com = Compose(pipeline_com) if pipeline_com else None

        self.flag = self.coco_complete.flag  # not used

    def __len__(self):
        return len(self.coco_imcomplete)

    def __getitem__(self, idx):
        #idx = idx + 5
        results_imcom = self.coco_imcomplete[idx]
        results_com = self.coco_complete[idx]
        results_imcom, results_com = self.merge_split_ann(results_imcom, results_com)
        results = self.pipeline_imcom(results_imcom)
        results = self.get_share_pipe_img(results_com, results)
        if self.pipeline_com:
            results_com = self.pipeline_com(results_com)
            results.update({f'{key}_com': val for key, val in results_com.items()})
        return results

    def update_ann_file(self, ann_file):
        self.coco_complete.data_infos = self.coco_complete.load_annotations(ann_file)

    def judge_img_same_order(self):
        sign = False
        count = 0
        for i in range(0, len(self.coco_imcomplete.img_ids)):
            if self.coco_imcomplete.img_ids[i] == self.coco_complete.img_ids[i]:
                count += 1
            else:
                print(False)
                break
        if count == len(self.coco_imcomplete.img_ids):
            sign = True
            print('coco_imcomplete and coco_complete have same order of image index.')
        else:
            print(False)
        return sign

    def merge_split_ann(self, result_imcom, result_com):
        # merge teacher's img and bboxes to stedent's, processed the unified pipeline,
        # and split the teacher's img and bboxes

        result_com['gt_bboxes_imcom'] = result_imcom['gt_bboxes']
        result_com['bbox_fields'].append('gt_bboxes_imcom')
        # flip aug
        result_com = self.pipeline_share(result_com)
        # get teacher's processed img and bboxes
        result_imcom['img'] = result_com['img']
        result_imcom['gt_bboxes'] = result_com['gt_bboxes_imcom']
        result_imcom['flip'] = result_com['flip']
        result_imcom['flip_direction'] = result_com['flip_direction']
        return result_imcom, result_com

    def get_share_pipe_img(self, result_com, results):

        # prepare img for clip model to crop
        img_share_pipe = result_com['img'].copy()
        #img_share_pipe = np.ascontiguousarray(img_share_pipe.transpose(2, 0, 1))
        img_share_pipe = np.ascontiguousarray(img_share_pipe)
        results['img_share_pipe'] = DC(to_tensor(img_share_pipe), stack=False)
        return results



