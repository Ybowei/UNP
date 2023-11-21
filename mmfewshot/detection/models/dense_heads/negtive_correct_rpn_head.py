# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple

import torch
from mmcv.runner import force_fp32
from mmcv.utils import ConfigDict
from mmdet.core import bbox2roi, images_to_levels, multi_apply, anchor_inside_flags, unmap
from mmdet.models import RPNHead
from mmdet.models.builder import HEADS, build_roi_extractor
from torch import Tensor
import torch.nn as nn
from mmfewshot.detection.models.utils import build_aggregator

@HEADS.register_module()
class NegCorRPNHead(RPNHead):
    """RPN head for `Attention RPN <https://arxiv.org/abs/1908.01998>`_.

    Args:
        num_support_ways (int): Number of sampled classes (pos + neg).
        num_support_shots (int): Number of shot for each classes.
        aggregation_layer (dict): Config of `aggregation_layer`.
        roi_extractor (dict): Config of `roi_extractor`.
    """

    def __init__(self,
                 num_support_ways: int,
                 num_support_shots: int,
                 roi_extractor: Dict = dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=14, sampling_ratio=0),
                     out_channels=1024,
                     featmap_strides=[16]),
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.num_support_ways = num_support_ways
        self.num_support_shots = num_support_shots
        assert roi_extractor is not None, \
            'missing config of roi_extractor.'

        self.roi_extractor = \
            build_roi_extractor(copy.deepcopy(roi_extractor))
        self.maxpool = nn.MaxPool2d(kernel_size=7, padding=0)

    def extract_roi_feat(self, feats: List[Tensor], rois: Tensor) -> Tensor:
        """Forward function.

        Args:
            feats (list[Tensor]): Input features with shape (N, C, H, W).
            rois (Tensor): with shape (m, 5).

         Returns:
            Tensor: RoI features with shape (N, C, H, W).
        """
        return self.roi_extractor(feats, rois)

    def forward_train(self,
                      query_feats: List[Tensor],
                      support_feats: List[Tensor],
                      query_gt_bboxes: List[Tensor],
                      query_img_metas: List[Dict],
                      support_gt_bboxes: List[Tensor],
                      query_gt_bboxes_ignore: Optional[List[Tensor]] = None,
                      proposal_cfg: Optional[ConfigDict] = None,
                      **kwargs) -> Tuple[Dict, List[Tuple]]:
        """Forward function in training phase.

        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W)..
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            query_gt_bboxes (list[Tensor]): List of ground truth bboxes of
                query image, each item with shape (num_gts, 4).
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: `img_shape`, `scale_factor`, `flip`, and may
                also contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            support_gt_bboxes (list[Tensor]): List of ground truth bboxes of
                support image, each item with shape (num_gts, 4).
            query_gt_bboxes_ignore (list[Tensor]): List of ground truth bboxes
                to be ignored of query image with shape (num_ignored_gts, 4).
                Default: None.
            proposal_cfg (:obj:`ConfigDict`): Test / postprocessing
                configuration. if None, test_cfg would be used. Default: None.

        Returns:
            tuple: loss components and proposals of each image.

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - proposal_list (list[Tensor]): Proposals of each image.
        """
        batch_size = len(query_img_metas)
        query_feat_list = []
        support_proto_list = []

        query_feat = query_feats[0]
        support_rois = bbox2roi([bboxes for bboxes in support_gt_bboxes])
        support_roi_feats = self.extract_roi_feat(support_feats, support_rois)
        support_roi_feats = self.maxpool(support_roi_feats)

        avg_support_feats = [
            support_roi_feats[i * self.num_support_shots:(i + 1) *
                              self.num_support_shots].mean([0, 2, 3],
                                                           keepdim=True)
            for i in range(
                support_roi_feats.size(0) // self.num_support_shots)
        ]

        avg_support_feats = torch.cat(avg_support_feats)
        for i in range(batch_size):
            query_feat_list.append(query_feat[i].unsqueeze(0))
            support_proto_list.append(avg_support_feats)

        outs = self(query_feats)
        query_gt_labels = kwargs['query_gt_labels']
        if query_gt_labels is None:
            loss_inputs = outs + (query_gt_bboxes, query_img_metas)
        else:
            loss_inputs = outs + (query_gt_bboxes, query_gt_labels, query_img_metas)

        losses = self.loss(*loss_inputs, gt_bboxes_ignore=query_gt_bboxes_ignore,
                           query_feat_list=query_feat_list, support_feat_list=support_proto_list)

        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas=query_img_metas, cfg=proposal_cfg)
            return losses, proposal_list


    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores: List[Tensor],
             bbox_preds: List[Tensor],
             gt_bboxes: List[Tensor],
             img_metas: List[Dict],
             gt_labels: Optional[List[Tensor]] = None,
             gt_bboxes_ignore: Optional[List[Tensor]] = None,
             query_feat_list: List[Tensor] = None,
             support_feat_list: List[Tensor] = None) -> Dict:
        """Compute losses of rpn head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
                Default: None.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss. Default: None
            pair_flags (list[bool]): Indicate predicted result is from positive
                pair or negative pair with shape (N). Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        # get anchors and training targets
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            query_feat_list=query_feat_list,
            support_feat_list=support_feat_list)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = num_total_pos + num_total_neg

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single Tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox)

    def simple_test(self,
                    query_feats: List[Tensor],
                    support_feat: Tensor,
                    query_img_metas: List[Dict],
                    rescale: bool = False) -> List[Tensor]:
        """Test function without test time augmentation.

        Args:
            query_feats (list[Tensor]): List of query features, each item with
                shape(N, C, H, W).
            support_feat (Tensor): Support features with shape (N, C, H, W).
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: `img_shape`, `scale_factor`, `flip`, and may
                also contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            rescale (bool): Whether to rescale the results.
                Default: False.

        Returns:
            List[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        # fuse support and query features
        feats = self.aggregation_layer(
            query_feat=query_feats[0], support_feat=support_feat)
        proposal_list = self.simple_test_rpn(feats, query_img_metas)
        if rescale:
            for proposals, meta in zip(proposal_list, query_img_metas):
                proposals[:, :4] /= proposals.new_tensor(meta['scale_factor'])

        return proposal_list

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    query_feat_list,
                    support_feat_list,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False,
):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all
                  images.
                - num_total_neg (int): Number of negative samples in all
                  images.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            query_feat_list,
            support_feat_list,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)


    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            query_feats_list,
                            support_feats_list,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result, neg_label_weights = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes, query_feats_list, support_feats_list)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if (len(neg_inds) > 0) & (len(neg_label_weights) > 0):
            label_weights[neg_inds] = neg_label_weights
        else:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)