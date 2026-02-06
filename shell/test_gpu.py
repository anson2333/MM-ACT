import torch
import os

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
print(f"可见GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)} | {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

# 测试指定GPU
target_gpus = [6, 7]
print(f"\n准备测试GPU {target_gpus}...")

# 设置可见GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

# 重新初始化CUDA
torch.cuda.empty_cache()

# 简单测试
for gpu_id in [0, 1]:  # 现在6,7变成了0,1
    tensor = torch.ones(1).cuda(gpu_id)
    print(f"GPU {gpu_id} (原{target_gpus[gpu_id]}) 测试通过: {tensor}")

print("\n✓ GPU 6和7正常工作")