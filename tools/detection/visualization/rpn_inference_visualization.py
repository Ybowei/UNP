import os
import os.path as osp
import argparse
from pathlib import Path
import warnings
import torch
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmcv.image import tensor2imgs
from mmfewshot.detection.datasets import build_dataloader, build_dataset
from mmfewshot.detection.models import build_detector

def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('--config', help='train config file path',
                        default='/data/huang1/PHD_Works/Few_Shot_Object_Detection/mmfewshot/fsod_config/KnowDis/tfa_baseline/coco/vis/coco_10shot_vis_proposal.py')
    parser.add_argument('--checkpoint', help='checkpoint file',
                       default='/data/huang1/PHD_Works/Few_Shot_Object_Detection/mmfewshot/work_dirs/MINI/Motivation/TFA_base_and_mini/coco/10shot/ft_rpn/iter_160000.pth')
    # 以下三个pipeline排除,方便可视化
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['Normalize'],   # ['DefaultFormatBundle', 'Normalize', 'Collect']
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-interval',
        type=int,
        default=0,
        help='the interval of show (ms)')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.0,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type):
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.train
    if train_data_cfg.get('dataset', None) is not None:
        # voc数据集
        datasets = train_data_cfg['dataset']
        datasets['pipeline'] = [
            x for x in datasets.pipeline if x['type'] not in skip_type
        ]
    else:
        train_data_cfg['pipeline'] = [
            x for x in train_data_cfg.pipeline if x['type'] not in skip_type
        ]

    return cfg


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type)

    # currently only support single images testing
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    assert samples_per_gpu == 1, 'currently only support single images testing'
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model from a config file and a checkpoint file
    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    progress_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        filename = os.path.join(args.output_dir, Path(data['img_metas'].data[0][0]['ori_filename'].split('/')[-1])) if args.output_dir is not None else None
        img, img_metas = data['img'], data['img_metas']
        img = img.data[0].cuda()
        img_metas = img_metas.data[0]
        with torch.no_grad():
            result = model.module.simple_inference_for_rpn_vis(img, img_metas)
        imgs = tensor2imgs(img, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            ori_h, ori_w = img_meta['ori_shape'][:-1]
            img_show = mmcv.imresize(img_show, (ori_w, ori_h))

            if args.output_dir:
                out_file = osp.join(args.output_dir, img_meta['ori_filename'])
            else:
                out_file = None
            # bboxes = bboxes.detach().cpu().numpy()
            result = result[0].detach().cpu().numpy()

            model.module.show_proposal_result(
                img_show,
                result,
                score_thr=args.show_score_thr,
                show=args.show,
                out_file=out_file)



if __name__ == '__main__':
    main()