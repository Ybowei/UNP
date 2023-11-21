# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import nms_match
import numpy as np
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.transforms import bbox2roi
from mmdet.core.bbox.samplers.base_sampler import BaseSampler
from mmdet.core.bbox.samplers.sampling_result import SamplingResult
import torch.nn.functional as F


@BBOX_SAMPLERS.register_module()
class GNRSampler(BaseSampler):
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
                 context,
                 floor_thr=0.1,
                 num_bins=3,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 alpha=0.25,
                 gamma=20.0,
                 iou_thr=0.5,
                 **kwargs):
        super().__init__(num, pos_fraction, neg_pos_ub, add_gt_as_proposals)

        self.floor_thr = floor_thr
        self.num_bins = num_bins
        self.context = context
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
                #device = torch.cuda.current_device()
                device = torch.cuda.device_count() - 1
            else:
                device = 'cpu'
            gallery = torch.tensor(gallery, dtype=torch.long, device=device)
        perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def normalize(self, feats, axis=-1):
        normal_feats = 20. * feats / (torch.norm(feats, 2, axis, keepdim=True).expand_as(feats) + 1e-12)
        return normal_feats

    def sample_via_interval(self, max_overlaps, full_set, num_expected):
        """Sample according to the iou interval.

        Args:
            max_overlaps (torch.Tensor): IoU between bounding boxes and ground
                truth boxes.
            full_set (set(int)): A full set of indices of boxes。
            num_expected (int): Number of expected samples。

        Returns:
            np.ndarray: Indices  of samples
        """
        max_iou = self.iou_thr
        iou_interval = max_iou / self.num_bins
        per_num_expected = int(num_expected / self.num_bins)

        sampled_inds = []
        for i in range(self.num_bins):
            start_iou = i * iou_interval     # problem
            end_iou = (i + 1) * iou_interval
            tmp_set = set(
                np.where(
                    np.logical_and(max_overlaps >= start_iou,
                                   max_overlaps < end_iou))[0])
            tmp_inds = list(tmp_set & full_set)
            if len(tmp_inds) > per_num_expected:
                tmp_sampled_set = self.random_choice(tmp_inds,
                                                     per_num_expected)
            else:
                tmp_sampled_set = np.array(tmp_inds, dtype=np.int)
            sampled_inds.append(tmp_sampled_set)

        sampled_inds_list = sampled_inds
        sampled_inds = np.concatenate(sampled_inds)
        if len(sampled_inds) < num_expected:
            num_extra = num_expected - len(sampled_inds)
            extra_inds = np.array(list(full_set - set(sampled_inds)))
            if len(extra_inds) > num_extra:
                extra_inds = self.random_choice(extra_inds, num_extra)
            sampled_inds_list.append(extra_inds)
            sampled_inds = np.concatenate([sampled_inds, extra_inds])
            sampled_inds_confusion = np.concatenate([sampled_inds_list[0], sampled_inds_list[-1]])
        else:
            sampled_inds_confusion = sampled_inds_list[0]

        return sampled_inds, sampled_inds_confusion

    def sample_confusion_via_iou(self, max_overlaps, full_set):
        """Sample according to the iou interval.

        Args:
            max_overlaps (torch.Tensor): IoU between bounding boxes and ground
                truth boxes.
            full_set (set(int)): A full set of indices of boxes。
            num_expected (int): Number of expected samples。

        Returns:
            np.ndarray: Indices  of samples
        """
        iou_thr = self.floor_thr
        start_iou = 0
        sampled_set = set(np.where(
            np.logical_and(max_overlaps >= start_iou, max_overlaps <= iou_thr))[0])
        confusion_sampled_inds = list(sampled_set & full_set)
        return np.array(confusion_sampled_inds)

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
                    query_feats,
                    support_proto,
                    feats=None,
                    img_meta=None,
                    **kwargs):
        """Sample negative samples. """
        neg_inds = torch.nonzero(assign_result.gt_inds == 0).flatten()
        num_neg = neg_inds.size(0)
        if num_neg == 0:
            return neg_inds, None

        max_overlaps = assign_result.max_overlaps.cpu().numpy()
        # balance sampling for negative samples
        neg_set = set(neg_inds.cpu().numpy())
        iou_sampling_neg_inds = list(neg_set)

        if len(iou_sampling_neg_inds) > num_expected:
            if self.num_bins >= 2:
                iou_sampled_inds, confusion_sampled_inds = self.sample_via_interval(
                    max_overlaps, neg_set, num_expected)
            else:
                iou_sampled_inds = self.random_choice(iou_sampling_neg_inds, num_expected)
                confusion_sampled_inds = self.sample_confusion_via_iou(max_overlaps, set(iou_sampled_inds))
        else:
            iou_sampled_inds = np.array(iou_sampling_neg_inds, dtype=np.int)
            confusion_sampled_inds = self.sample_confusion_via_iou(max_overlaps, set(iou_sampled_inds))

        # if len(iou_sampled_inds) < num_expected:
        #     num_extra = num_expected - len(iou_sampled_inds)
        #     extra_inds = np.array(list(neg_set - set(iou_sampled_inds)))
        #     if len(extra_inds) > num_extra:
        #         extra_inds = self.random_choice(extra_inds, num_extra)
        #     sampled_inds = np.concatenate((iou_sampled_inds, extra_inds))
        # else:
        sampled_inds = iou_sampled_inds

        inds = np.argsort(sampled_inds)
        aso = sampled_inds[inds]
        bina = np.searchsorted(aso[:-1], confusion_sampled_inds)
        inds_in_confusion_sampled_inds = np.where(confusion_sampled_inds == aso[bina])[0]
        inds_in_sampled_inds = inds[bina[inds_in_confusion_sampled_inds]]
        confusion_sample_index_tensor = torch.LongTensor(inds_in_sampled_inds).to(assign_result.gt_inds.device)
        sampled_inds = torch.from_numpy(sampled_inds).long().to(assign_result.gt_inds.device)
        confusion_sampled_inds = torch.from_numpy(confusion_sampled_inds).long().to(assign_result.gt_inds.device)
        # try:
        #     confusion_sampled_inds = torch.from_numpy(confusion_sampled_inds).long().to(assign_result.gt_inds.device)
        # except:
        #     print('num_neg is {}'.format(num_neg))
        #     print('the len of gt_inds is {}'.format(len(assign_result.gt_inds)))
        #     print(len(confusion_sampled_inds))

        if len(confusion_sampled_inds):
            with torch.no_grad():
                confusion_neg_bboxes = bboxes[confusion_sampled_inds]
                confusion_neg_rois = bbox2roi([confusion_neg_bboxes])
                confusion_neg_feats = self.context.extract_roi_feat([query_feats], confusion_neg_rois)
                if len(confusion_neg_feats.size()) > 2:
                    confusion_neg_feats = self.context.maxpool(confusion_neg_feats).view(confusion_neg_feats.shape[0], confusion_neg_feats.shape[1])
                support_proto = support_proto.view(support_proto.shape[0], support_proto.shape[1])
                confusion_neg_feats = self.normalize(confusion_neg_feats)
                support_proto = self.normalize(support_proto)
                #cls_score_u = torch.cdist(confusion_neg_feats_nor, support_proto_nor, p=2.0)
                cls_score_cos = F.cosine_similarity(confusion_neg_feats.unsqueeze(1), support_proto.unsqueeze(0), dim=-1)
                max_score, argmax_score = cls_score_cos[:, :-1].max(-1)
                #max_score, argmax_score = cls_score_cos.softmax(-1)[:, :-1].max(-1)
                neg_label_weights = max_score.new_ones(len(sampled_inds))
                confusion_neg_label_weights = max_score.new_ones(len(max_score))
                confusion_neg_label_weights = self.alpha + (1 - self.alpha) * \
                                              torch.exp(torch.exp(-self.gamma * torch.sub(confusion_neg_label_weights, max_score)))
                neg_label_weights_new = neg_label_weights.scatter(0, confusion_sample_index_tensor, confusion_neg_label_weights)
        else:
            neg_label_weights_new = sampled_inds.new_ones(len(sampled_inds))

        return sampled_inds, neg_label_weights_new


    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               query_feats,
               support_proto,
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
            query_feats,
            support_proto,
            img_meta=img_meta,
            **kwargs)

        return SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                              assign_result, gt_flags), neg_label_weights



