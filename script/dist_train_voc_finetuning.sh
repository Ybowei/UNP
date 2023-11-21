#!/usr/bin/env bash

GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29000}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

for shot in 1 2 3 5 10
do
  CONFIG=configs/detection/gncs/voc/split1/tfa_novel_rcnn_gncs_r101_voc-split1_${shot}shot-fine-tuning.py
  PYTHONPATH="$(dirname $0)/../../":$PYTHONPATH \
  python -m torch.distributed.launch \
      --nnodes=$NNODES \
      --node_rank=$NODE_RANK \
      --master_addr=$MASTER_ADDR \
      --nproc_per_node=$GPUS \
      --master_port=$PORT \
      $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
done