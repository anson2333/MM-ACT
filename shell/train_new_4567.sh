#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PATH="/mnt/pfs/scalelab2/yuhao/miniconda3/envs/mmact/bin:$PATH"

echo "4卡 ZeRO-3 + CPU Offload 训练"

cd /mnt/pfs/scalelab2/hms/MM-ACT

/mnt/pfs/scalelab2/yuhao/miniconda3/envs/mmact/bin/python -m accelerate.commands.launch \
  --config_file accelerate_configs/1_node_4_gpus_4567.yaml \
  --main_process_port 29500 \
  training/train_mmact_robotwin_mix.py \
  config=configs/mmact_robotwin_mix.yaml