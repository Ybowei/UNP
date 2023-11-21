# Copyright (c) OpenMMLab. All rights reserved.
"""Reshape the classification and regression layer for novel classes.

The bbox head from base training only supports `num_base_classes` prediction,
while in few shot fine-tuning it need to handle (`num_base_classes` +
`num_novel_classes`) classes. Thus, the layer related to number of classes
need to be reshaped.

The original implementation provides three ways to reshape the bbox head:

    - `combine`: combine two bbox heads from different models, for example,
        one model is trained with base classes data and another one is
        trained with novel classes data only.
    - `remove`: remove the final layer of the base model and the weights of
        the removed layer can't load from the base model checkpoint and
        will use random initialized weights for few shot fine-tuning.
    - `random_init`: create a random initialized layer (`num_base_classes` +
        `num_novel_classes`) and copy the weights of base classes from the
        base model.

Temporally, we only use this script in FSCE and TFA with `random_init`.
This part of code is modified from
https://github.com/ucbdrive/few-shot-object-detection/.

Example:
    # VOC base model
    python3 -m tools.detection.misc.initialize_bbox_head \
        --src1 work_dirs/tfa_r101_fpn_voc-split1_base-training/latest.pth \
        --method random_init \
        --save-dir work_dirs/tfa_r101_fpn_voc-split1_base-training
    # COCO base model
    python3 -m tools.detection.misc.initialize_bbox_head \
        --src1 work_dirs/tfa_r101_fpn_coco_base-training/latest.pth \
        --method random_init \
        --coco \
        --save-dir work_dirs/tfa_r101_fpn_coco_base-training
"""
import argparse
import os

import torch
from mmcv.runner.utils import set_random_seed

# COCO config
COCO_NOVEL_CLASSES = [
    1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72
]
COCO_BASE_CLASSES = [
    8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
    88, 89, 90
]
COCO_ALL_CLASSES = sorted(COCO_BASE_CLASSES + COCO_NOVEL_CLASSES)
COCO_IDMAP = {v: i for i, v in enumerate(COCO_ALL_CLASSES)}
COCO_TAR_SIZE = 80
# VOC config
VOC_TAR_SIZE = 20


def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--src1', type=str, help='Path to the main checkpoint')
    parser.add_argument(
        '--src2',
        type=str,
        default=None,
        help='Path to the secondary checkpoint. used when replace rpn layers of two checkpoints')
    parser.add_argument(
        '--save-dir', type=str, default=None, help='Save directory')
    parser.add_argument(
        '--method',
        choices=['combine', 'remove', 'random_init', 'replace'],
        required=True,
        help='Reshape method. combine = combine bbox heads from different '
        'checkpoints. remove = for fine-tuning on novel dataset, remove the '
        'final layer of the base detector. random_init = randomly initialize '
        'novel weights. replace = ')
    parser.add_argument(
        '--param-name',
        type=str,
        nargs='+',
        #default=['roi_head.bbox_head.fc_cls', 'roi_head.bbox_head.fc_reg'],
        default=['rpn_head.rpn_conv', 'rpn_head.rpn_cls', 'rpn_head.rpn_reg'],
        help='Target parameter names')
    parser.add_argument(
        '--tar-name',
        type=str,
        default='base_model',
        help='Name of the new checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    # Dataset
    parser.add_argument('--coco', action='store_true', help='For COCO models')
    parser.add_argument('--lvis', action='store_true', help='For LVIS models')
    return parser.parse_args()


def combine_checkpoints(param_name, is_weight, tar_size, checkpoint,
                        checkpoint2, args):
    """Combine base detector with novel detector.

    Feature extractor weights are from the base detector. Only the final layer
    weights are combined.
    """
    if not is_weight and param_name + '.bias' not in checkpoint['state_dict']:
        return
    if not is_weight and param_name + '.bias' not in checkpoint2['state_dict']:
        return
    weight_name = param_name + ('.weight' if is_weight else '.bias')
    pretrained_weight = checkpoint['state_dict'][weight_name]
    prev_cls = pretrained_weight.size(0)
    if 'fc_cls' in param_name:
        prev_cls -= 1
    if is_weight:
        feat_size = pretrained_weight.size(1)
        new_weight = torch.rand((tar_size, feat_size))
    else:
        new_weight = torch.zeros(tar_size)
    if args.coco:
        BASE_CLASSES = COCO_BASE_CLASSES
        IDMAP = COCO_IDMAP
        for i, c in enumerate(BASE_CLASSES):
            idx = i if args.coco else c
            if 'fc_cls' in param_name:
                new_weight[IDMAP[c]] = pretrained_weight[idx]
            else:
                new_weight[IDMAP[c] * 4:(IDMAP[c] + 1) * 4] = \
                    pretrained_weight[idx * 4:(idx + 1) * 4]
    else:
        new_weight[:prev_cls] = pretrained_weight[:prev_cls]

    checkpoint2_weight = checkpoint2['state_dict'][weight_name]
    if args.coco:
        NOVEL_CLASSES = COCO_NOVEL_CLASSES
        IDMAP = COCO_IDMAP
        for i, c in enumerate(NOVEL_CLASSES):
            if 'fc_cls' in param_name:
                new_weight[IDMAP[c]] = checkpoint2_weight[i]
            else:
                new_weight[IDMAP[c] * 4:(IDMAP[c] + 1) * 4] = \
                    checkpoint2_weight[i * 4:(i + 1) * 4]
        if 'fc_cls' in param_name:
            new_weight[-1] = pretrained_weight[-1]
    else:
        if 'fc_cls' in param_name:
            new_weight[prev_cls:-1] = checkpoint2_weight[:-1]
            new_weight[-1] = pretrained_weight[-1]
        else:
            new_weight[prev_cls:] = checkpoint2_weight
    checkpoint['state_dict'][weight_name] = new_weight
    return checkpoint

def replace_checkpoints(param_name, is_weight, checkpoint, checkpoint2, args):
    """Combine base detector with novel detector.

    Feature extractor weights are from the base detector. Only the final layer
    weights are combined.

    param_name: list

    """

    if not is_weight and param_name + '.bias' not in checkpoint['state_dict']:
        return
    if not is_weight and param_name + '.bias' not in checkpoint2['state_dict']:
        return
    weight_name = param_name + ('.weight' if is_weight else '.bias')
    steg1_weight = checkpoint['state_dict'][weight_name]
    steg2_weight = checkpoint2['state_dict'][weight_name]
    checkpoint['state_dict'][weight_name] = steg2_weight
    return checkpoint


def reset_checkpoint(checkpoint):
    if 'scheduler' in checkpoint:
        del checkpoint['scheduler']
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    if 'iteration' in checkpoint:
        checkpoint['iteration'] = 0


def main():
    args = parse_args()
    set_random_seed(args.seed)
    checkpoint = torch.load(args.src1)
    save_name = args.tar_name + f'_{args.method}_bbox_head.pth'
    save_dir = args.save_dir \
        if args.save_dir != '' else os.path.dirname(args.src1)
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    reset_checkpoint(checkpoint)

    if args.coco:
        TAR_SIZE = COCO_TAR_SIZE
    else:
        TAR_SIZE = VOC_TAR_SIZE

    if args.method == 'remove':
        # Remove parameters
        for param_name in args.param_name:
            del checkpoint['state_dict'][param_name + '.weight']
            if param_name + '.bias' in checkpoint['state_dict']:
                del checkpoint['state_dict'][param_name + '.bias']
    elif args.method == 'combine':
        checkpoint2 = torch.load(args.src2)
        tar_sizes = [TAR_SIZE + 1, TAR_SIZE * 4]
        for idx, (param_name,
                  tar_size) in enumerate(zip(args.param_name, tar_sizes)):
            combine_checkpoints(param_name, True, tar_size, checkpoint,
                                checkpoint2, args)
            combine_checkpoints(param_name, False, tar_size, checkpoint,
                                checkpoint2, args)
    elif args.method == 'replace':
        checkpoint2 = torch.load(args.src2)
        for idx, param_name in enumerate(args.param_name):
            replace_checkpoints(param_name, True, checkpoint, checkpoint2, args)
            replace_checkpoints(param_name, False, checkpoint, checkpoint2, args)
    else:
        raise ValueError(f'not support method: {args.method}')

    torch.save(checkpoint, save_path)
    print('save changed checkpoint to {}'.format(save_path))


if __name__ == '__main__':
    main()
