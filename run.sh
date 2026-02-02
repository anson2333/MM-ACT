export CUDA_VISIBLE_DEVICES=0,1
accelerate launch \
  --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml \
  --main_process_port 1110 \
  training/train_mmact_libero_action.py \
  config=configs/mmact_libero_action.yaml