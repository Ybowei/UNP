# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import nms_match

from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.transforms import bbox2roi
from mmdet.core.bbox.samplers.base_sampler import BaseSampler
from mmdet.core.bbox.samplers.sampling_result import SamplingResult
import math

@BBOX_SAMPLERS.register_module()
class GroupNegtiveRecalibrationSampler(BaseSampler):
    r"""Importance-based Sample Reweighting (ISR_N), described in `Prime Sample
    Attention in Object Detection <https://arxiv.org/abs/1904.04821>`_.

    Score hierarchical local rank (HLR) differentiates with RandomSampler in
    negative part. It firstly computes Score-HLR in a two-step way,
    then linearly maps score hlr to the loss weights.

    Args:
        num (int): Total number of sampled RoIs.
        pos_fraction (float): Fraction of positive samples.
        context (:class:`BaseRoIHead`): RoI head that the sampler belongs to.
        neg_pos_ub (int): Upper bound of the ratio of num negative to num
            positive, -1 means no upper bound.
        add_gt_as_proposals (bool): Whether to add ground truth as proposals.
        k (float): Power of the non-linear mapping.
        bias (float): Shift of the non-linear mapping.
        score_thr (float): Minimum score that a negative sample is to be
            considered as valid bbox.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 alpha=0.25,
                 beta=50.0,
                 gamma=20.0,
                 iou_thr=0.5,
                 **kwargs):
        super().__init__(num, pos_fraction, neg_pos_ub, add_gt_as_proposals)
        self.alpha = alpha
        self.gamma = gamma
        self.iou_thr = iou_thr


    @staticmethod
    def random_choice(gallery, num):
        """Randomly select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = 'cpu'
            gallery = torch.tensor(gallery, dtype=torch.long, device=device)
        perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0).flatten()
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self,
                    assign_result,
                    num_expected,
                    bboxes,
                    feats=None,
                    img_meta=None,
                    **kwargs):
        """Sample negative samples."""
        neg_inds = torch.nonzero(assign_result.gt_inds == 0).flatten()
        num_neg = neg_inds.size(0)
        if num_neg == 0:
            return neg_inds, None

        # filter out samples with the max score lower than score_thr
        max_overlaps = assign_result.max_overlaps
        max_overlaps_neg = max_overlaps[neg_inds]
        #len(max_overlaps_neg[(max_overlaps_neg >= 0) & (max_overlaps_neg < 0.2)])

        num_expected = min(num_neg, num_expected)
        rand_inds = torch.randperm(num_neg)[:num_expected]
        #neg_label_weights = max_overlaps.new_ones
        neg_label_weights = self.alpha + (1 - self.alpha) * torch.exp(-self.beta * torch.exp(-self.gamma * max_overlaps_neg))

        return neg_inds[rand_inds], neg_label_weights[rand_inds]

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               img_meta=None,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            tuple[:obj:`SamplingResult`, Tensor]: Sampling result and negative
                label weights.
        """
        bboxes = bboxes[:, :4]

        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
        if self.add_gt_as_proposals:
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds, neg_label_weights = self.neg_sampler._sample_neg(
            assign_result,
            num_expected_neg,
            bboxes,
            img_meta=img_meta,
            **kwargs)

        return SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                              assign_result, gt_flags), neg_label_weights
