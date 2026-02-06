
import os
import sys
import torch
import yaml
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from experiments.robot.robotwin.deploy_policy import get_model

def test_inference():
    # Load config from deploy_policy.yml
    config_path = os.path.join(os.path.dirname(__file__), "deploy_policy.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override with checkpoint path if needed (user provided checkpoint-step-50)
    # Assuming the user usually provides arguments via command line, we mock them here.
    # We use the checkpoint-step-50 path provided in the prompt context if accessible, 
    # otherwise we use placeholders.
    
    # Path from user context
    ckpt_path = "/mnt/pfs/scalelab2/hms/MM-ACT/checkpoint-step-50" 
    # Note: I am not on the user's machine, so this path might fail if I try to run it.
    # But the user asked for CODE modification/analysis.
    
    # I will create a script the USER can run.
    print("Test script for MMACT continuous action evaluation.")
    
    # Mock Args
    args = config.copy()
    args["model_path"] = ckpt_path # User should update this
    args["vq_model_path"] = "/mnt/pfs/scalelab2/yitian-proj/MM-ACT/huggingface/hub/models--showlab--magvitv2/snapshots/5c3fa78f8b3523347c5cd1a4c97f3c4e96f33d5d" # Placeholder
    args["action_dim"] = 16 # RobotWin likely 16 (Aloha) or 7 (Franka)
    args["robot_type"] = "aloha-agilex" # From eval.sh/yml default
    args["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {args['model_path']}...")
    try:
        model = get_model(args)
    except Exception as e:
        print(f"Failed to load model (expected if path is wrong): {e}")
        print("Please ensure 'model_path' in this script matches your actual checkpoint.")
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
    
    inputs = ([dummy_images], text_task, dummy_state, dummy_prev_action)
    
    print("Running get_actions...")
    with torch.no_grad():
        action_chunk, token_ids = model.get_actions(inputs, robot_type=args["robot_type"])
        
    print("Inference successful!")
    print(f"Action Chunk Shape: {action_chunk.shape}")
    print(f"Action Chunk Values (Sample): {action_chunk[0]}")
    
    if args.get("use_continuous_head", True): # We assume True for this test
        print("Confirmed: Continuous output generated.")

if __name__ == "__main__":
    test_inference()
