from jetengine_ext.llm import LLM
from jetengine_ext.sampling_params import SamplingParams
import torch
import torch.nn.functional as F
import multiprocessing as mp
import random
from jinja2 import Template
from training.libero_action_dataset import ActionGenerationFromLeRobotDataset
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer
import os
import sys
import json
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Any
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.utils import get_config, mask_or_random_replace_action_tokens
from training.prompting_utils import UniversalPrompting
from models import MMACTModelLM, MAGVITv2, get_mask_schedule

from omegaconf import DictConfig, ListConfig, OmegaConf

def add_gumbel_noise(logits, temperature):
    """添加 Gumbel 噪声用于随机采样"""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    noise = (- torch.log(noise)) ** temperature
    return logits.exp() / noise


def get_num_transfer_tokens(mask_index, steps):
    """
    计算每个样本在每个迭代步中需要更新的 token 数量
    Args:
        mask_index: (B, L) 布尔张量，表示哪些位置是 mask
        steps: 当前的迭代步数
    Returns:
        (B, steps) 张量，每个样本在每步需要 unmask 的 token 数量
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)  # (B, 1)
    base = mask_num // steps
    remainder = mask_num % steps
    
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, 
        device=mask_index.device, 
        dtype=torch.int64
    ) + base
    
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    
    return num_transfer_tokens


def get_transfer_index(logits, temperature, target, mask_index, x, num_transfer_tokens, threshold=None):
    """
    根据置信度策略选择要 unmask 的位置
    Args:
        logits: (B, L, V) 模型输出的 logits
        temperature: Gumbel 噪声温度
        target: 置信度计算策略 ('confidence', 'margin_confidence', 'neg_entropy', 'random')
        mask_index: (B, L) 布尔张量，当前 mask 位置
        x: (B, L) 当前序列
        num_transfer_tokens: (B,) 或 标量，本步要 unmask 的数量
        threshold: 可选的置信度阈值
    Returns:
        x0: (B, L) 预测的 token ids
        transfer_index: (B, L) 布尔张量，表示要更新的位置
    """
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L)
    
    # 计算置信度
    if target == 'confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)  # (B, L, V)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
        )  # (B, L)
    elif target == 'margin_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        top2 = torch.topk(p, 2, dim=-1).values  # (B, L, 2)
        x0_p = top2[..., 0] - top2[..., 1]
    elif target == 'neg_entropy':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = -torch.sum(p * torch.log(p + 1e-10), dim=-1)
    elif target == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(f"Target strategy '{target}' not implemented")
    
    # 只在 mask 位置保留预测值
    x0 = torch.where(mask_index, x0, x)
    
    # 基于阈值的选择
    if threshold is not None:
        selected = mask_index & (x0_p >= threshold)
        has_mask = mask_index.any(dim=-1)
        none_sel = (~selected.any(dim=-1)) & has_mask
        
        if none_sel.any():
            masked_scores = x0_p.masked_fill(~mask_index, float("-inf"))
            best_idx = masked_scores.argmax(dim=-1)
            rows = torch.nonzero(none_sel, as_tuple=False).squeeze(-1)
            selected[rows, best_idx[rows]] = True
        
        return x0, selected
    
    # 基于 top-k 的选择
    confidence = x0_p.masked_fill(~mask_index, float("-inf"))
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    
    for j in range(confidence.shape[0]):
        k = int(
            num_transfer_tokens[j].item() 
            if torch.is_tensor(num_transfer_tokens[j]) 
            else num_transfer_tokens[j]
        )
        if k <= 0:
            continue
        _, sel = torch.topk(confidence[j], k=k)
        transfer_index[j, sel] = True
    
    return x0, transfer_index

@dataclass
class DiffusionOutput:
    sequences: torch.Tensor               # final result  (B, L_total)  (GPU)
    history:   List[torch.Tensor]         # all intermediate x (CPU)
    nfe:       int

def denoise_step_map_from_history(history, mask_id: int, sample_idx: int = 0):
    """从 history 构造 step_map，记录每个 token 首次被 unmask 的步数"""
    L = history[0].shape[1]        
    step_map = torch.zeros(L, dtype=torch.long)
    prev = torch.full((L,), mask_id, dtype=torch.long)

    for t, snap in enumerate(history, start=0): 
        cur = snap[sample_idx]        
        changed = (prev == mask_id) & (cur != mask_id)
        step_map[changed] = t
        prev = cur
        if (step_map == 0).sum() == 0:     
            break
    return step_map

@torch.no_grad()
def generate_action_with_step_tracking(
        model,
        input_ids,      # <left padding to 1024> + <sos> + <sot> + <text_tokens> + <eot> + <soi> + <image_tokens> + <eoi> + ... + 
                        # <state_tokens> + <soa> + <action_tokens>(all_masked) + <eoa> + <pad>(padding to 14 dimension action tokens)
        steps,
        action_token_length,
        temperature,
        target,
        mask_id,
        use_cache=True,
        unmask_threshold=None,
    ) -> DiffusionOutput:
    """
    Prefix-cache based diffusion rollout for action tokens.
    Signature mirrors llada_rl_rollout.generate_with_prefix_cache so callers can pass the same arguments.
    input_ids: (B, L0) tensor containing prompt token ids with masked action tail. 
    action_token_length: number of tokens to generate masked positions appended to prompt
    # TODO  -DELETE block_length: size of each decoding block
    # TODO  -DELETE further_horizon: extra window length to include beyond block
    Returns DiffusionOutput with sequences, history (snapshots), first_unmask_step and nfe.
    """
    # ensure input_ids is tensor on device
    if not torch.is_tensor(input_ids):
        raise ValueError("input_ids must be a torch.Tensor of token ids")
    
    B, L0 = input_ids.shape
    device = input_ids.device


    


    nfe = 0
    hist: List[torch.Tensor] = []

    blk = 0
    
    # TODO 下面是传入一张7维action的情况。如果是两张七维action，那么e = L0，s = L0 - action_token_length*2 - 1
    s = L0 - action_token_length*2 - 1  # the first token of action_tokens
    e = L0 - action_token_length - 1       # eoa_id
    
    x = input_ids

    # how many tokens to transfer per step within this block
    num_transfer = get_num_transfer_tokens((x[:, s:e] == mask_id), steps)

    # first full forward to build prefix cache (or full logits if no cache)
    if use_cache:
        out = model(x, use_cache=True)
        pkv = getattr(out, "past_key_values", None)
        # chop prefix out of pkv to keep cache small (retain entries up to s)
        if pkv is not None:
            new_pkv = tuple(tuple(t[:, :, :s] for t in layer) for layer in pkv)
            pkv = new_pkv
        logits = out.logits
    else:
        out = model(x, use_cache=False)
        logits = out.logits
        pkv = None

    mask_all = (x == mask_id)
    # only consider up to block end
    mask_all[:, e:] = 0

    # initial selection for this block (i == 0)
    x0, tr_idx = get_transfer_index(
        logits, temperature, target, mask_all, x, num_transfer[:, 0], unmask_threshold
    )

    # apply updates
    x[tr_idx] = x0[tr_idx]
    # mark first unmask step using nfe (current forward count)
    hist.append(x.clone().cpu())
    nfe += 1

    i = 1
    while True:
        # determine mask region for this iteration: 
        mask_all = (x == mask_id)

        if not mask_all.any():
            break
        
        if use_cache:
            # use cropped pkv and only pass window suffix to avoid recomputing prefix
            logits = model(x, past_key_values=pkv, use_cache=True).logits
            x0, tr_idx = get_transfer_index(
                logits, temperature, target, mask_all, x, num_transfer[:, i], unmask_threshold
            )
            x[tr_idx] = x0[tr_idx]
        else:
            # no cache: full forward then slice logits
            logits = model(x, use_cache=False).logits
            # slice suffix starting at s
            logits = logits[:, s:]
            x0, tr_idx = get_transfer_index(
                logits, temperature, target, mask_all[:,s:], x[:, s:], num_transfer[:, i], unmask_threshold
            )
            x[:,s:][tr_idx] = x0[tr_idx]

        hist.append(x.clone().cpu())
        nfe += 1

        # stop when this block has no remaining masks
        if (x[:, s:e] == mask_id).sum() == 0:
            break
        i += 1

    return DiffusionOutput(sequences=x, history=hist, nfe=nfe)

def prepare_inputs_and_labels_for_rollout(batch, vq_model, uni_prompting, mask_id, config, mask_schedule, action_vocab_size, device):
    """准备 rollout 用的输入，返回 masked action 和相关信息"""
    # return的是包含全masked的action token的input_ids
    
    images, texts, state_tokens, prev_action_tokens, action_tokens, action_dims = batch
    
    # 图像编码
    image_tokens = []
    for imgs in images:
        img_tokens = []
        for img in imgs:
            img = img.to(device)
            with torch.no_grad():
                tokens = vq_model.get_code(img.unsqueeze(0))[0]
            tokens = tokens + len(uni_prompting.text_tokenizer)
            img_tokens.append(tokens)
        image_tokens.append(img_tokens)
    
    action_tokens = [a.to(device) for a in action_tokens]
    
    # mask action tokens (用于 rollout)
    masked_action, labels, _, _ = mask_or_random_replace_action_tokens(
        action_tokens,
        mask_id,
        uni_prompting.pad_id,
        config,
        mask_schedule,
        action_vocab_size,
        is_train=False,  # rollout 时不使用训练模式
        seed=None,
    )
    
    masked_list, label_list = [], []
    for i, a in enumerate(action_tokens):
        seq_len = a.size(0)
        masked_list.append(masked_action[i, :seq_len].cpu())
        label_list.append(labels[i, :seq_len].cpu())
    
    input_ids, attention_masks, label_ids = uni_prompting(
        (
            image_tokens,
            texts,
            state_tokens,
            prev_action_tokens,
            masked_list,
            action_dims,
            label_list,
            device,
            config.training.chunk_size,
        ),
        "mm2a",
    )
    
    
    _, L0 = input_ids.shape
    x = input_ids
    action_token_length = 56 
    x[:,(L0-2*action_token_length -1):(L0-action_token_length-1)] = mask_id


    return {
        "input_ids": x.to(device),
        "attention_masks": attention_masks.to(device),
        "label_ids": label_ids.to(device),
        "original_action_tokens": action_tokens,  # 保留原始 action
        "texts": texts,
        "action_dims": action_dims,
    }

if __name__ == "__main__":

    config = get_config()
    print(OmegaConf.to_yaml(config))

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    project_name = config.experiment.project
    
    # 加载模型
    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.mmact.pretrained_model_path
    else:
        pretrained_model = os.path.join("..", project_name, "ckpt", config.model.optimized_name)
    
    print(f"Loading model from: {pretrained_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True, padding_side="left")
    model = MMACTModelLM.from_pretrained(
        pretrained_model, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    action_vocab_size = model.config.action_vocab_size
    mask_id = model.config.mask_token_id
    
    # 初始化 prompting
    max_action_prompt_len = (
        config.dataset.preprocessing.max_seq_length
        - config.training.chunk_size * 14
        - 2  # <soa><eoa>
    )
    
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_action_prompt_len=max_action_prompt_len,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", 
            "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
            "<|mm2a|>", "<|soa|>", "<|eoa|>", 
            "<|7dim|>", "<|14dim|>", 
            "<|sostate|>", "<|eostate|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=0.0,
        use_reserved_token=True,
        action_vocab_size=action_vocab_size,
    )
    
    # 加载 VQ model
    vq_cfg = config.model.vq_model
    if vq_cfg.type == "magvitv2":
        vq_model = MAGVITv2.from_pretrained(vq_cfg.vq_model_name).to(device)
    else:
        raise ValueError(f"Unsupported vq model type: {vq_cfg.type}")
    vq_model.eval()
    vq_model.requires_grad_(False)
    
    vocab_offset = model.config.vocab_size - model.config.action_vocab_size
    mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))
    
    if not hasattr(config.training, "min_masking_rate"):
        config.training.min_masking_rate = 0.0
    
    # 加载数据集
    if config.experiment.function == "train":
        dataset_file_paths = config.dataset.dataset_file_paths
        
        libero_dataset = [
            ActionGenerationFromLeRobotDataset(
                prev_action_size=6,
                chunk_size=config.training.chunk_size,
                vocab_offset=vocab_offset,
                action_vocab_size=action_vocab_size,
                dataset_fps=10,
                third_image_name="observation.images.image",
                wrist_image_name=["observation.images.wrist_image"],
                use_prev_action=config.training.use_prev_action,
                action_dim=7,
                use_norm=False,
                libero_dataset=True,
                repo_id=".",
                root=dataset_file_path,
                force_cache_sync=False,
                revision=None,
                download_videos=False,
            )
            for dataset_file_path in dataset_file_paths
        ]
        
        libero_mix_dataset = ConcatDataset(libero_dataset)
        dataloader = DataLoader(
            libero_mix_dataset,
            batch_size=config.training.batch_size,
            num_workers=config.training.dataprocess_nums,
            shuffle=True,
            collate_fn=ActionGenerationFromLeRobotDataset.collate_fn,
        )
        
        # Rollout 参数
        steps = config.rollout.steps
        temperature = config.rollout.temperature
        target = config.rollout.target
        use_cache = config.rollout.use_cache
        
        if config.rollout.remasking_strategy == "low_confidence_static":
            unmask_threshold = None
        else:
            unmask_threshold = config.rollout.dynamic_threshold
        
        # 存储 rollout 结果
        rollout_results = []
        
        print("Starting rollout...")
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Rollout Progress")):
            # 准备输入
            batch_data = prepare_inputs_and_labels_for_rollout(
                batch, vq_model, uni_prompting, mask_id, 
                config, mask_schedule, action_vocab_size, device
            )
            
            # 生成
            with torch.no_grad():
                out = generate_action_with_step_tracking(
                    model,
                    batch_data["input_ids"],
                    steps,
                    config.training.chunk_size * batch_data["action_dims"][0],
                    temperature,
                    target,
                    mask_id,
                    use_cache=use_cache,
                    unmask_threshold=unmask_threshold
                )
            
            # 解码并保存结果
            for i in range(out.sequences.size(0)):
                # 提取 action 部分的 step_map
                # 假设 action tokens 在序列末尾
                _, L0 = batch_data["input_ids"].shape
                action_token_length = config.training.chunk_size * batch_data["action_dims"][0]
                action_start_idx = L0-2*action_token_length -1
                eoa_idx = L0 - action_token_length - 1
                
                step_map_i = denoise_step_map_from_history(
                    out.history, mask_id=mask_id, sample_idx=i
                )
                action_step_map = step_map_i[action_start_idx:eoa_idx].tolist()
                
                result = {
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                    "text": batch_data["texts"][i],
                    "generated_action_tokens": out.sequences[i, action_start_idx:eoa_idx].cpu().tolist(),
                    "step_map": action_step_map,
                    "original_action_tokens": batch_data["original_action_tokens"][i].cpu().tolist(),
                    "nfe": out.nfe,
                }
                rollout_results.append(result)
            
            # 限制处理数量（可选）
            if hasattr(config.rollout, "num_task_per_step") and batch_idx >= config.rollout.num_task_per_step // config.training.batch_size:
                break
        
        # 保存结果
        output_dir = os.path.join("..", project_name, "temp_data")
        os.makedirs(output_dir, exist_ok=True)
            # 这是保存在哪里？
            # 答案：保存到上级目录的 project_name 文件夹下的 temp_data 子文件夹中。
            
        time_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        output_file = os.path.join(
            output_dir, 
            f"new_rollout_outputs_epoch_{config.experiment.current_epoch}_{time_str}.json"
        )
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(rollout_results, f, indent=2, ensure_ascii=False)
        
        print(f"Rollout completed. Results saved to {output_file}")
        print(f"Total samples processed: {len(rollout_results)}")
        
    else:
        raise NotImplementedError("Only train function is implemented now.")