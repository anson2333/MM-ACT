export CUDA_VISIBLE_DEVICES=6,7
accelerate launch \
  --config_file accelerate_configs/1_node_2_gpus_67.yaml \
  training/train_mmact_robotwin_mix.py \
  config=configs/mmact_robotwin_mix.yaml
