#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7

# ZeRO-3优化
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "4卡 ZeRO-3 + CPU Offload 训练"

accelerate launch \
  --config_file accelerate_configs/1_node_4_gpus_4567.yaml \
  --main_process_port 29500 \
  training/train_mmact_robotwin_mix.py \
  config=configs/mmact_robotwin_mix.yaml