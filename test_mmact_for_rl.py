from typing import List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import torch
from transformers import AutoTokenizer

from transformers.generation import TopKLogitsWarper

from RLinf.rlinf.models.embodiment.modules.value_head import ValueHead
from RLinf.rlinf.models.embodiment.model_utils import (
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
)

from training.prompting_utils import UniversalPrompting
from models import MMACTModelLM, MAGVITv2
from training.utils import image_transform_tensor

class MMACTForRLActionPrediction(nn.Module):
    def __init__(
        self,
        model_path: str,
        vq_model_path: str,
        action_vocab_size=None,
        vocab_offset=None,
        device="cuda:0",
        timesteps=4,
        exec_steps=6,
        preprocessing_max_seq_length=1024,
        training_chunk_size=8,
        action_dim=16,
        robot_type: str = "franka",
        add_value_head: bool = True,
    ):  
        super().__init__()
        self.image_transform_tensor = image_transform_tensor
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left", local_files_only=True
        )
        self.device = device
        self.timesteps = timesteps
        self.exec_steps = exec_steps
        self.preprocessing_max_seq_length = preprocessing_max_seq_length
        self.training_chunk_size = training_chunk_size
        self.action_dim = action_dim
        self.model = MMACTModelLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(self.device)
        print("Finish loading checkpoint. Start loading vq-model.")
        self.vq_model = MAGVITv2.from_pretrained(vq_model_path).to(self.device)
        # self.vq_model.eval()
        # self.vq_model.requires_grad_(False)
        print("Finish loading vq-model.")
        self.vocab_offset = (
            vocab_offset
            if vocab_offset
            else self.model.config.vocab_size - self.model.config.action_vocab_size
        )
        self.action_vocab_size = (
            self.action_vocab_size
            if action_vocab_size
            else self.model.config.action_vocab_size
        )
        if robot_type == "franka":  # match training padding method
            max_action_prompt_len = (
                self.preprocessing_max_seq_length
                - self.training_chunk_size * (self.action_dim * 2)
                - 2
            )
        else:  # training total len - chunk_size * action_dim - <soa><eoa>
            max_action_prompt_len = (
                self.preprocessing_max_seq_length
                - self.training_chunk_size * self.action_dim
                - 2
            )

        self.uni_prompting = UniversalPrompting(
            self.tokenizer,
            special_tokens=(
                "<|soi|>",
                "<|eoi|>",
                "<|sov|>",
                "<|eov|>",
                "<|t2i|>",
                "<|mmu|>",
                "<|t2v|>",
                "<|v2v|>",
                "<|lvg|>",
                "<|mm2a|>",
                "<|soa|>",
                "<|eoa|>",
                "<|7dim|>",
                "<|14dim|>",
                "<|sostate|>",
                "<|eostate|>",
            ),
            ignore_id=-100,
            cond_dropout_prob=0.0,
            use_reserved_token=True,
            max_action_prompt_len=max_action_prompt_len,
        )
        if True:
            self.value_head = ValueHead(
                input_dim=4096,
                hidden_sizes=(512, 256, 128),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )
            self.value_head.to(dtype=torch.bfloat16, device=self.device)

    def image_process_for_generate(self, images_list):
        """
        输入 images_list: [(16, 3, 256, 256), (16, 3, 256, 256)] (主视角和手腕视角)
        返回: image_tokens 结构为 [Batch_0_tokens, Batch_1_tokens, ..., Batch_15_tokens]
        每个 Batch_i_tokens 是一个包含多个视角 Token 的列表。
        """
        # 1. 初始化结果列表，长度为 Batch Size (16)
        batch_size = images_list[0].shape[0]
        all_batch_tokens = [[] for _ in range(batch_size)]

        # 2. 遍历视角（例如先处理所有人的主视角，再处理所有人的手腕视角）
        for view_tensor in images_list:
            # view_tensor 形状: (16, 3, 256, 256)
            view_tensor = view_tensor.to(self.device)
            target_dtype = next(self.vq_model.parameters()).dtype 
            view_tensor = view_tensor.to(dtype=target_dtype)
            with torch.no_grad():
                # 利用 VQ-model 的 Batch 推理，一次性处理 16 张图
                # 假设 get_code 返回 (16, Token_Num)
                batch_tokens = self.vq_model.get_code(view_tensor) 
                
            # 加上文本词表偏移量
            batch_tokens = batch_tokens + len(self.uni_prompting.text_tokenizer)
            
            # 3. 将结果分发到对应的样本中
            # batch_tokens.cpu().unbind(0) 得到 16 个 token 向量
            for i, single_img_tokens in enumerate(batch_tokens.cpu().unbind(0)):
                all_batch_tokens[i].append(single_img_tokens)

        return all_batch_tokens


    def quantize_state_with_offset(self, values, bins: int = 1024):
        """
        输入 values: Tensor 形状 (16, 8)
        返回: List[Tensor], 长度 16, 每个 Tensor 形状 (8,)
        """
        # 1. 整体截断到 [-1, 1]
        values = torch.clamp(values, min=-1.0, max=1.0)

        # 2. 整体映射到 [0, bins-1] 并加上 offset
        # 这一步会自动对 (16, 8) 里的每个元素执行
        indices = torch.round((values + 1) / 2 * (bins - 1)).long() + self.vocab_offset

        # 3. 沿维度 0 (Batch 维度) 拆分
        # unbind(0) 会把 (16, 8) 的 Tensor 变成 16 个 (8,) 的 Tensor 组成的元组
        return list(indices.unbind(0))

    def dequantize_action_with_offset(
        self, action_tokens, bins: int = 1024
    ) -> torch.Tensor:
        action_tokens = action_tokens.to(torch.int).clamp(0, bins - 1)
        return (action_tokens / (bins - 1) * 2) - 1

    def input_process(self, inputs):
        images_tensor, text_tasks, state_tensor, prev_action_tokens = inputs
        batch_size = len(text_tasks)
        action_dim = [int(self.action_dim)] * batch_size
        prev_action_tokens = [prev_action_tokens + self.vocab_offset] * batch_size
        state_tokens = self.quantize_state_with_offset(state_tensor, bins=self.action_vocab_size)
        
        reshape_images_tensor = []
        for view in images_tensor:
            # 对一个视角下的 16 张图分别做 transform
            # unbind(0) 将 (16, 3, 256, 256) 拆成 16 个 (3, 256, 256) 的列表
            processed_batch = [self.image_transform_tensor(img) for img in view.unbind(0)]
            # 重新组装回 Tensor: (16, 3, 256, 256)
            reshape_images_tensor.append(torch.stack(processed_batch))
        image_tokens = self.image_process_for_generate(reshape_images_tensor)
        input_ids, attention_masks, prompt_ids = self.uni_prompting(
            (
                image_tokens,
                text_tasks,
                state_tokens,
                prev_action_tokens,
                action_dim,
                self.device,
                self.training_chunk_size,
            ),
            "mm2a_gen",
        )
        # 注意: 这里 prompt_ids[0] 是标量，因为做了 left padding 对齐
        return input_ids, attention_masks, prompt_ids[0]

    def get_actions(self, inputs):
        """
        Your inputs should include
        images_tensor(List,[head_image, wrist_image]), text_task, state_tensor,previous_action_tokens([] if not used in training)
        执行推理并返回动作以及用于 RL 计算的上下文信息
        """
        input_ids, attention_masks, prompt_id = self.input_process(inputs)
        
        # [修改点 1] 接收 action_generate 返回的 logits
        gen_token_ids, action_logits, prompt_hidden_states, step_map = self.model.action_generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            timesteps=self.timesteps,
            guidance_scale=0,
            chunk_size=self.training_chunk_size,
            action_dim=self.action_dim,
            prompt_id=prompt_id, # 注意：不要加括号
            uni_prompting=self.uni_prompting,
            temperature=0.0,
            action_vocab_size=self.action_vocab_size,
        )
        
        # gen_token_ids 应该是 0-1023 的 bin index (不含 offset)
        # 反量化得到物理动作
        action_chunk = self.dequantize_action_with_offset(
            gen_token_ids, bins=self.action_vocab_size
        ).view(gen_token_ids.shape[0], self.training_chunk_size, self.action_dim)

        # [修改点 2] 返回所有训练需要的上下文数据
        return (
            action_chunk,
            gen_token_ids.view(gen_token_ids.shape[0], self.training_chunk_size, self.action_dim),
            input_ids,
            attention_masks,
            prompt_id,
            action_logits,
            prompt_hidden_states,
            step_map,
        )
    
    def predict_action_batch(
            self,
            env_obs=None,
            calculate_values=True,  # 新增参数: 是否计算 Critic Value
            **kargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # 1. 准备输入数据
        images_tensor = [env_obs["images"], env_obs["wrist_images"]]
        text_tasks = env_obs["task_descriptions"]
        state_tensor = env_obs["states"]
        state_tensor[:, 3:6] = state_tensor[:, 3:6] / 5.0  # quickly fix bug in state
        flat_prev_actions_tensors = torch.tensor([])

        # 2. 执行推理
        # [修改 3] 接收 prompt_hidden_states
        (
            action_chunk, 
            token_ids, 
            prompt_input_ids, 
            prompt_attention_masks, 
            prompt_id, 
            action_logits, 
            prompt_hidden_states, # <--- 获取它
            step_map
        ) = self.get_actions(
            inputs=(
                images_tensor,
                text_tasks,
                state_tensor,
                flat_prev_actions_tensors,
            ),
        )

        # ================= Result 计算逻辑 =================
        
        # 3. 计算 LogProbs
        # ---------------------------------------------------------------------
        # token_ids: [Batch, Chunk, Dim] -> Flatten [Batch, Chunk*Dim]
        batch_size = token_ids.shape[0]
        flat_action_ids = token_ids.reshape(batch_size, -1)
        
        # [修改 4] 直接计算 LogProbs (移除 TopK 和 Temperature)
        # 这里的 action_logits 已经是 [Batch, Chunk*Dim, Action_Vocab_Size]
        action_logits = action_logits.permute(0, 2, 1)
        chunk_logprobs = compute_logprobs_from_logits(
            logits=action_logits, 
            target=flat_action_ids # 注意参数名通常是 labels 或 target，取决于你的 utils 实现
        ) 

        # 4. 计算 Values (State Value V(s))
        # ---------------------------------------------------------------------
        chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        # 注意：这里改为检查 self.value_head (因为 FSDP 适配时我们将它挂在 self 下)
        if calculate_values and hasattr(self, "value_head"):
            # [修改 5] 性能优化核心：直接复用 prompt_hidden_states
            # 不再进行额外的 forward pass，节省 50% 显存和计算
            
            # prompt_hidden_states: [Batch, Seq_Len, Hidden_Dim]
            # 我们提取 <soa> 位置的特征，也就是 prompt_id 所在的位置
            
            # 确保切片出来的 tensor 不需要梯度（Critic 只需要输入特征）
            # 虽然 inference 模式下通常自带 no_grad，但 detach 更保险
            state_hidden = prompt_hidden_states[:, prompt_id, :]
            
            # 必须确保输入精度是 bfloat16 (因为 value_head 已转为 bf16)
            if state_hidden.dtype != torch.bfloat16:
                 state_hidden = state_hidden.to(dtype=torch.bfloat16)

            chunk_values = self.value_head(state_hidden)

        # 5. 构造 Forward Inputs
        # ---------------------------------------------------------------------
        full_input_ids = prompt_input_ids.clone()
        
        num_action_tokens = flat_action_ids.shape[1]
        action_slice = slice(prompt_id + 1, prompt_id + 1 + num_action_tokens)
        
        full_input_ids[:, action_slice] = flat_action_ids + self.vocab_offset

        forward_inputs = {
            "input_ids": full_input_ids.cpu(),
            "attention_mask": prompt_attention_masks.cpu(),
            "action_tokens": token_ids.cpu(),
        }

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "step_map": step_map,
            "forward_inputs": forward_inputs,
        }
        
        # [修改 6] 返回 NumPy 数组给 EnvWorker
        return action_chunk.float().cpu().numpy(), result
    
    def forward(
        self,
        data: dict[str, torch.Tensor],
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = False,
        use_cache: bool = False,
    ):
        """
        PPO Training Forward Pass (Teacher Forcing)
        """
        # 1. 解包数据并移动到设备
        # input_ids 已经包含了 [Prompt ... <soa> ... Actions]
        input_ids = data["input_ids"].to(self.device)
        attention_mask = data["attention_mask"].to(self.device)
        # target_action_tokens 是相对索引 (0-1023)，用于计算 Loss
        target_action_tokens = data["action_tokens"].to(self.device)

        # 2. 模型前向传播 (Parallel)
        # 如果需要计算 Value，必须开启 output_hidden_states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=compute_values,
            use_cache=use_cache,
            return_dict=True
        )

        output_dict = {}

        # 3. 计算关键位置索引
        # 我们需要找到动作开始的位置 (<soa>)
        # input_ids 布局: [Prompt_Tokens ... <soa>, Act_1, Act_2, ..., Act_N]
        # 动作总长度 = chunk_size * action_dim
        num_action_tokens = self.training_chunk_size * self.action_dim
        
        # <soa> 的索引位置 = 总长度 - 动作长度 - 1
        prompt_end_idx = input_ids.shape[1] - num_action_tokens - 1

        # 4. 计算 Values (Critic)
        # 只需要 <soa> 位置的 hidden state，代表动作执行前的状态 V(s)
        if compute_values and hasattr(self, "value_head"):
            # 取最后一层 hidden state
            # Shape: [Batch, Hidden_Dim]
            soa_hidden = outputs.hidden_states[-1][:, prompt_end_idx, :]
            
            # 精度对齐 (ValueHead 是 bfloat16)
            if soa_hidden.dtype != torch.bfloat16:
                soa_hidden = soa_hidden.to(dtype=torch.bfloat16)
                
            output_dict["values"] = self.value_head(soa_hidden) # [B, 1]

        # 5. 计算 LogProbs & Entropy (Actor)
        if compute_logprobs or compute_entropy:
            # 逻辑：位置 i 的 Logits 预测位置 i+1 的 Token
            # 我们需要预测: Act_1, Act_2, ..., Act_N
            # 对应的输入位置是: <soa>, Act_1, ..., Act_{N-1}
            # 所以切片范围是: [prompt_end_idx : -1]
            
            # [Batch, Num_Action_Tokens, Vocab_Size]
            relevant_logits = outputs.logits[:, prompt_end_idx:-1, :]

            # 词表切片：只保留 Action Bins 部分
            # [Batch, Num_Action_Tokens, Action_Vocab_Size]
            action_logits = relevant_logits[..., self.vocab_offset : self.vocab_offset + self.action_vocab_size]

            # 维度转置 (适配 CrossEntropy): [B, Seq, Vocab] -> [B, Vocab, Seq]
            action_logits = action_logits.permute(0, 2, 1)

            if compute_logprobs:
                # 展平 Target 以匹配 Logits 维度
                flat_targets = target_action_tokens.reshape(input_ids.shape[0], -1)
                
                output_dict["logprobs"] = compute_logprobs_from_logits(
                    logits=action_logits,
                    target=flat_targets
                )

            if compute_entropy:
                output_dict["entropy"] = compute_entropy_from_logits(
                    logits=action_logits
                )

        return output_dict
    
if __name__ == "__main__":
    # ================= 配置区域 =================
    # 请将以下路径修改为你服务器上实际的模型路径
    MODEL_PATH = "/mnt/pfs/scalelab2/ch/MM-ACT/checkpoints/MM-ACT-10" 
    VQ_MODEL_PATH = "/mnt/pfs/scalelab2/yitian-proj/MM-ACT/huggingface/hub/models--showlab--magvitv2/snapshots/5c3fa78f8b3523347c5cd1a4c97f3c4e96f33d5d"
    
    # 检查环境是否支持 CUDA
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    # ===========================================

    print(f"🚀 [Test] 正在初始化 MMACTForRLActionPrediction (Device: {DEVICE})...")
    
    # 1. 实例化模型
    # 注意：根据你的 states shape (16, 8)，这里的 action_dim 应该设为 8 以匹配
    try:
        model = MMACTForRLActionPrediction(
            model_path=MODEL_PATH,
            vq_model_path=VQ_MODEL_PATH,
            device=DEVICE,
            action_dim=7,
            robot_type="franka",
            training_chunk_size=8,
        )
    except OSError as e:
        print(f"\n❌ 模型加载失败，请检查路径是否正确:\n{e}")
        sys.exit(1)

    # 2. 构造模拟输入数据 (Batch Size = 16)
    BATCH_SIZE = 16
    print(f"📦 [Test] 正在构造 Batch Size = {BATCH_SIZE} 的模拟 env_obs 数据...")

    # 模拟 uint8 类型的原始图像 (0-255)
    # 形状: (Batch, Channel, Height, Width)
    dummy_images = torch.randint(0, 256, (BATCH_SIZE, 3, 256, 256), dtype=torch.uint8)
    dummy_wrist_images = torch.randint(0, 256, (BATCH_SIZE, 3, 256, 256), dtype=torch.uint8)
    
    # 模拟 float32 类型的状态向量
    # 形状: (Batch, State_Dim) -> (16, 8)
    dummy_states = torch.randn((BATCH_SIZE, 8), dtype=torch.float32)

    # 模拟任务描述列表
    dummy_tasks = [
        "put the white mug on the plate and put the chocolate"
    ] * BATCH_SIZE

    env_obs = {
        "images": dummy_images,
        "wrist_images": dummy_wrist_images,
        "states": dummy_states,
        "task_descriptions": dummy_tasks
    }

    # 3. 打印输入信息以便核对
    print("\n📋 输入数据概览:")
    for k, v in env_obs.items():
        if isinstance(v, torch.Tensor):
            print(f"  - {k}: Tensor {tuple(v.shape)} | {v.dtype}")
        elif isinstance(v, list):
            print(f"  - {k}: List len={len(v)} | First: {v[0][:20]}...")

    # 4. 执行预测
    print("\n⚡️ [Test] 开始执行 predict_action_batch ...")
    try:
        # 这一步会测试:
        # 1. state 归一化 (x/5.0)
        # 2. 图像 Batch 预处理 (uint8 -> float -> norm)
        # 3. VQ-Model Batch 推理
        # 4. Prompt 组装与 LLM 生成
        action_chunk, result = model.predict_action_batch(env_obs)
        
        print("\n✅ 测试成功! 输出结果:")
        print("=" * 40)
        # 检查输出形状
        print(f"Output Action Chunk Shape: {action_chunk.shape}") 
        
        # 预期形状通常是 (chunk_size, action_dim) 或者 (Batch, chunk_size, action_dim)
        # 取决于你的 get_actions 实现返回的是单条还是 Batch
        
        # 检查数值范围 (归一化后的动作通常在 -1 到 1 之间)
        print(f"Action Values Range:       [{action_chunk.min().item():.3f}, {action_chunk.max().item():.3f}]")
        print("=" * 40)

    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()