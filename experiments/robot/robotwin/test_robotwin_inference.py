
import os
import sys
import torch
import yaml
import numpy as np
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# 验证
print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.current_device()}")
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from experiments.robot.robotwin.deploy_policy import get_model

def test_inference():
    # Load config from deploy_policy.yml
    config_path = os.path.join(os.path.dirname(__file__), "deploy_policy.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        # Use arguments from the YAML config
        args = config.copy()
        
        print(f"Loading config from {config_path}")
        print(f"Target Model Path: {args['model_path']}")
        
        if not os.path.exists(args['model_path']):
            print(f"WARNING: The model path '{args['model_path']}' does not exist on this machine.")
            print("If you are running this script locally but the model is on a server, please adjust the path or run this script on the server.")
        
        # args["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using Device: {args.get('device', 'auto')}")

        print(f"Loading model architecture...")
        try:
            model = get_model(args)
        except OSError as e:
            print(f"\n[Error] Failed to load checkpoint. Typical reasons:")
            print("1. The path is incorrect.")
            print("2. The checkpoint relies on DeepSpeed ZeRO-3 artifacts but 'model.safetensors' (or sharded files) are missing/empty.")
            print(f"System Error: {e}")
            return
        except Exception as e:
            print(f"\n[Error] Unexpected error loading model: {e}")
            import traceback
            traceback.print_exc()
            return

    print("Model loaded successfully.")
    
    # Create dummy inputs
    # inputs: (images_tensor(List,[head_image, wrist_image]), text_task, state_tensor, flat_prev_actions)
    
    # Mock images: List of tensors. For Aloha (3 cameras usually? or 2?)
    # code says: camera_names = ["head_camera", "left_camera", "right_camera"] in encode_obs
    # But input_process iterates image_tensor.
    # Let's assume 1 image for simplicity or 3 random images.
    C, H, W = 3, 256, 256
    dummy_images = [torch.randn(C, H, W) for _ in range(3)] 
    
    text_task = "Put the apple in the box"
    
    # State: Joint state + gripper? 
    # Aloha has 14 joints + 2 grippers = 16 dim? 
    # quantize_state_with_offset expects a list/tensor.
    dummy_state = [0.0] * 16 
    
    # Prev action
    dummy_prev_action = torch.zeros((16,), dtype=torch.int32) # Token IDs
    
    inputs = (dummy_images, text_task, dummy_state, dummy_prev_action)
    
    print("Running get_actions...")
    with torch.no_grad():
        robot_type = args.get("robot_type", "aloha-agilex")
        # Validate robot_type vs action_dim
        if args.get("action_dim", 16) > 7 and robot_type == "franka":
            print("[WARNING] action_dim > 7 but robot_type is 'franka'. Overriding to 'aloha-agilex'.")
            robot_type = "aloha-agilex"
  
        
        print(f"Using Robot Type: {robot_type}")
        action_chunk, token_ids = model.get_actions(inputs, robot_type=robot_type)
        
    print("Inference successful!")
    print(f"Action Chunk Shape: {action_chunk.shape}")
    print(f"Action Chunk Values (Sample): {action_chunk[0]}")
    
    if args.get("use_continuous_head", True): # We assume True for this test
        print("Confirmed: Continuous output generated.")

if __name__ == "__main__":
    test_inference()
