from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)
from dataclasses import fields
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel, AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers import PretrainedConfig

from PIL import Image

from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
    ActivationCheckpointingStrategy,
)

from .modeling_llada import LLaDAModelLM
from .action_head import FlowMatchingActionHead
from .modeling_mmada import MMadaConfig, MMadaModelLM
from .sampling import cosine_schedule, mask_by_random_topk
from models.action_head import FlowMatchingActionHead

log = logging.getLogger(__name__)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Gumbel-max trick for sampling from categorical distributions.

    For MDM-style discrete diffusion, low-precision Gumbel max can hurt quality,
    so we keep this in float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

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


def get_transfer_index(logits, temperature, mask_index, x, num_transfer_tokens, target='confidence', threshold=None):
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

def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Pre-compute, for each sample and each diffusion step, how many tokens
    should be transitioned.

    Args:
        mask_index: (B, L) bool/0-1, 1 for tokens that can be updated.
        steps:     total number of reverse diffusion steps.

    Returns:
        (B, steps) int64 tensor with the number of tokens to transfer at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)  # (B, 1)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


class MMACTConfig(MMadaConfig):
    """
    Config for the MMACT model.

    This reuses MMadaConfig and adds a few MMACT-specific fields while
    changing the HF ``model_type`` so that AutoConfig can distinguish it.
    """

    model_type = "mmact"

    def __init__(
        self,
        use_continuous_head: bool = False,
        continuous_action_dim: int = 7,
        continuous_head_loss_weight: float = 1.0,
        **kwargs,
    ):
        self.use_continuous_head = use_continuous_head
        self.continuous_action_dim = continuous_action_dim
        self.continuous_head_loss_weight = continuous_head_loss_weight
        # Let the parent handle common fields
        super().__init__(**kwargs)


class MMACTModelLM(MMadaModelLM):
    """
    MMACT model.

    This class *inherits* from MMadaModelLM and adds the extra utilities from
    the modified MMada implementation (action decoding, multi-task
    forward, multi losses, etc.).  The original MMada implementation is left
    untouched.
    """

    config_class = MMACTConfig
    base_model_prefix = "model"

    def __init__(self, config: MMACTConfig, *args, **kwargs):
        # Extract custom arguments passed via from_pretrained kwargs
        # This prevents TypeError in parent LLaDAModelLM.__init__ which doesn't accept unknown kwargs
        use_continuous_head = kwargs.pop("use_continuous_head", None)
        continuous_action_dim = kwargs.pop("continuous_action_dim", None)
        flow_matching_hidden_dim = kwargs.pop("flow_matching_hidden_dim", None)
        
        # Update config if these values were provided
        if use_continuous_head is not None:
            config.use_continuous_head = use_continuous_head
        if continuous_action_dim is not None:
            config.continuous_action_dim = continuous_action_dim
        if flow_matching_hidden_dim is not None:
            config.flow_matching_hidden_dim = flow_matching_hidden_dim

        
        print(f"Initializing MMadaModelLM with config: {config}")
        # Reuse MMadaModelLM initialization
        super().__init__(config, *args, **kwargs)

        if getattr(config, "use_continuous_head", False):
            # We assume d_model is available in config, typically implied by the parent class
            hidden_size = config.d_model
            action_dim = getattr(config, "continuous_action_dim", 7)
            
            # Use FlowMatchingActionHead instead of simple MLP
            self.continuous_head = FlowMatchingActionHead(
                input_dim=hidden_size,  # Transformer hidden size
                hidden_dim=getattr(config, "flow_matching_hidden_dim", 4096),
                action_dim=action_dim
            )
            # Initialize weights? usually handled by post_init or let pytorch default
            # self.continuous_head.apply(self._init_weights) # if needed

    def _get_gen_indices(self, input_ids: torch.LongTensor) -> Optional[torch.Tensor]:
        # Helper to identify generation (Action) tokens for MoT routing
        mask_token_id = getattr(self.config, "mask_token_id", 126336)
        vocab_offset = getattr(self.config, "vocab_offset", 134656) 
        action_vocab_size = getattr(self.config, "action_vocab_size", 1024)
        
        # Action tokens are either MASK tokens or in the Action Vocab range
        is_gen = (input_ids == mask_token_id) | \
                 ((input_ids >= vocab_offset) & (input_ids < vocab_offset + action_vocab_size))
                 
        if not is_gen.any():
            return None
            
        # Return flattened indices
        is_gen_flat = is_gen.view(-1)
        gen_indices = torch.nonzero(is_gen_flat, as_tuple=True)[0]
        return gen_indices

    # ------------------------------------------------------------------
    # Action generation (MaskGIT-style over action tokens)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def action_generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        temperature: float = 1.0,
        prompt_id: Optional[int] = None,
        timesteps: int = 4,
        guidance_scale: float = 0.0,  # not used in our work
        noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
        generator: Optional[torch.Generator] = None,
        config: Optional[PretrainedConfig] = None,
        chunk_size: int = 24,
        action_dim: int = 7,
        mask_token_id: int = 126336,
        vocab_offset: int = 134656,
        action_vocab_size: int = 1024,
        **kwargs,
    ) -> tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]: # [修改 1] 返回增加 Hidden States
        """
        Parallel MaskGIT-style decoding for *action* tokens.
        Returns:
            sampled_ids: [Batch, Chunk*Dim]
            logits: [Batch, Chunk*Dim, Action_Vocab_Size]
            prompt_hidden_states: [Batch, Seq_Len, Hidden_Dim] (来自第一步推理)
        """
        if prompt_id is None:
            raise ValueError("action_generate requires `prompt_id` (index of <soa>).")

        num_action_tokens = chunk_size * action_dim
        num_new_special_tokens = 0
        
        B, L0 = input_ids.shape
        # 初始化 logits 变量，防止 timesteps=0 的极端情况（虽然一般不会发生）
        logits = None
        prompt_hidden_states = None # [修改 2] 初始化 hidden_states 容器
        
        if action_dim == 7:
            s = L0 - num_action_tokens*2 - 1  # the first token of action_tokens
            e = L0 - num_action_tokens - 1
        else: 
            s = L0 - num_action_tokens - 1
            e = L0 - 1
            
        num_transfer = get_num_transfer_tokens((input_ids[:, s:e] == mask_token_id), timesteps)

        hist: List[torch.Tensor] = []
        sampled_ids: List[torch.Tensor] = []
        action_step_map: List[torch.Tensor] = []
        
        attention_bias = (
            (attention_mask[:, :, None] & attention_mask[:, None, :])
            .bool()
            .unsqueeze(1)
        )
        gen_indices = self._get_gen_indices(input_ids)
        out = self(input_ids,
                   attention_bias = attention_bias,
                   output_hidden_states = True,
                   use_cache=False,
                   gen_indices=gen_indices)
        logits = out.logits
        prompt_hidden_states = out.hidden_states[-1]

        mask_all = (input_ids == mask_token_id)
        # only consider up to block end
        mask_all[:, e:] = 0

        x0, tr_idx = get_transfer_index(
            logits, temperature,  mask_all, input_ids, num_transfer[:, 0]
        )
        nfe = 0
        input_ids[tr_idx] = x0[tr_idx]
        hist.append(input_ids.clone().cpu())
        nfe +=1
        
        i = 1
        
        while True:
            mask_all = (input_ids == mask_token_id)
            
            if not mask_all.any():
                break
            
            gen_indices = self._get_gen_indices(input_ids)
            logits = self(input_ids, 
                          attention_bias = attention_bias,
                          output_hidden_states = True,
                          use_cache=False,
                          gen_indices=gen_indices).logits
            # slice suffix starting at s
            logits = logits[:, s:e]
            
            x0, tr_idx = get_transfer_index(
                logits, temperature, mask_all[:,s:e], input_ids[:, s:e], num_transfer[:, i]
            )
            input_ids[:,s:e][tr_idx] = x0[tr_idx]
            
            hist.append(input_ids.clone().cpu())
            nfe += 1
            
            if (input_ids[:, s:e] == mask_token_id).sum() == 0:
                break
            i += 1
        
        for i in range(B):
            step_map_i = denoise_step_map_from_history(
                hist, mask_id=mask_token_id, sample_idx=i
            )
            action_step_map.append(step_map_i[s:e])
            sampled_ids.append(input_ids[i,s:e].clone().cpu())
            
        sampled_ids = torch.stack(sampled_ids)
        action_step_map = torch.stack(action_step_map)
        
        logits = logits.to(sampled_ids.device)
        # [修改 5] 返回四个值
        return sampled_ids, logits, prompt_hidden_states, action_step_map

    # ------------------------------------------------------------------
    # multi-task forward with action / t2i / lm / mmu
    # ------------------------------------------------------------------
    def forward_process_vla(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        batch_size_t2i: int = 0,
        batch_size_lm: int = 0,
        batch_size_mmu: int = 0,
        batch_size_action: int = 0,
        max_image_seq_length: int = 128,
        max_action_prompt_len: int = 768,
        action_masks: Optional[torch.Tensor] = None,
        action_loss_type: str = "single",  # "single" | "multi" | "gaussian"
        p_mask_lm: Optional[torch.Tensor] = None,
        p_mask_mmu: Optional[torch.Tensor] = None,
        answer_lengths_mmu: Optional[torch.Tensor] = None,
        t2i_masks: Optional[torch.Tensor] = None,
        answer_lengths_lm: Optional[torch.Tensor] = None,
        action_err_token_len: int = 10,
        at_value: float = 1e-3,
        continuous_labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass that can handle four task types(three in use) in a single
        batch:

        - action generation (first ``batch_size_action`` samples)
        - text-to-image generation (next ``batch_size_t2i`` samples)
        - LM-only samples (next ``batch_size_lm`` samples)
        - multi-modal understanding (last ``batch_size_mmu`` samples)
        """
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        # -------- Build attention bias --------
        attention_bias = torch.ones(
            batch_size, 1, seq_len, seq_len, device=input_ids.device
        )

        # ACTION part (comes first)
        if batch_size_action > 0 and action_masks is not None:
            attention_bias_action = (
                (action_masks[:, :, None] & action_masks[:, None, :])
                .bool()
                .unsqueeze(1)
            )
            attention_bias[:batch_size_action] = attention_bias_action

        # T2I part (comes right after action part)
        if batch_size_t2i > 0 and t2i_masks is not None:
            attention_bias_t2i = (
                (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
            )
            attention_bias[batch_size_action : batch_size_action + batch_size_t2i] = (
                attention_bias_t2i
            )

        # logits = self(input_ids, attention_bias=attention_bias).logits
        gen_indices = self._get_gen_indices(input_ids)
        outputs = self(input_ids, attention_bias=attention_bias, output_hidden_states=True, gen_indices=gen_indices)
        logits = outputs.logits
        self.output_size = logits.shape[-1]

        # ------------------------------------------------------------------
        # Loss: ACTION
        # ------------------------------------------------------------------
        if batch_size_action > 0:
            action_logits = logits[
                :batch_size_action, max_action_prompt_len + 1 :
            ].contiguous()
            action_labels = labels[
                :batch_size_action, max_action_prompt_len + 1 :
            ].contiguous()

            if action_loss_type == "multi":
                loss_action = self.modified_ce_with_margin_rule(
                    action_logits,
                    action_labels,
                    ignore_index=-100,
                    action_err_token_len=action_err_token_len,
                )
            elif action_loss_type == "gaussian":
                loss_action = self.gaussian_soft_ce_window(
                    action_logits,
                    action_labels,
                    radius=int(action_err_token_len / 2.0),
                    at_value=at_value,
                    ignore_index=-100,
                )
            elif action_loss_type == "single":
                loss_action = F.cross_entropy(
                    action_logits.view(-1, self.output_size),
                    action_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                raise ValueError(f"Unsupported action_loss_type: {action_loss_type}")
        else:
            loss_action = torch.tensor(0.0, device=input_ids.device)

        # ------------------------------------------------------------------
        # Loss: Continuous Action (Flow Matching)
        # ------------------------------------------------------------------
        loss_continuous = torch.tensor(0.0, device=input_ids.device)
        if batch_size_action > 0 and getattr(self.config, "use_continuous_head", False) and continuous_labels is not None:
            # Extract hidden states: [B, Seq - Prompt - 1, Hidden]
            # Match the slice used for action_logits but shifted by 1 for causality
            # action_logits used [max_action_prompt_len + 1 :]
            # So hidden states should come from [max_action_prompt_len : -1] ??
            # Actually, outputs.hidden_states[-1] is full sequence.
            # We want the hidden states that PREDICT the action tokens.
            # continuous_labels are "ground truth" for the action.
            # We want to condition on the state *before* the action (or during).
            # If we condition on VLM hidden states, we typically use the full sequence of action tokens.
            
            # Use the same slice length as action_labels
            slice_start = max_action_prompt_len
            slice_end = logits.shape[1] - 1 # Assuming prediction for last token comes from second to last hidden state?
            # Or if logits are aligned with input_ids (usual HF way), logits[i] is pred for input[i+1].
            # logits has shape [B, SeqLen, V].
            # input_ids has shape [B, SeqLen].
            # If we used `action_logits` from `max_action_prompt_len + 1`...
            # The indices are `max_action_prompt_len + 1` to `End`.
            # The hidden states generating these are `max_action_prompt_len` to `End - 1`.
            
            hidden_states = outputs.hidden_states[-1]
            action_hidden_states = hidden_states[:batch_size_action, slice_start : -1].contiguous()
            
            # Verify shapes match
            # continuous_labels: [B, Chunk, Dim]
            # action_hidden_states: [B, T_action_tokens, Hidden]
            # T_action_tokens should equal Chunk * Dim (if 1 token per dim).
            # Or check config.
            
            # Trim hidden states to match required length (Chunk * Dim) if slightly larger
            # usually caused by having an extra special token (EOS) at end of slice
            required_len = continuous_labels.shape[1] * continuous_labels.shape[2] # Chunk * Dim
            if action_hidden_states.shape[1] > required_len:
                action_hidden_states = action_hidden_states[:, :required_len]
            
            
            # Sample noise
            return_dict = self.continuous_head.sample_noisy_actions(continuous_labels)
            noisy_actions = return_dict["noisy_actions"]
            timesteps = return_dict["timestep_embeddings"]
            target_flow = return_dict["flow"]
            
            # Predict
            velocity_pred = self.continuous_head.predict_velocity(
                action_hidden_states, noisy_actions, timesteps
            )
            
            loss_continuous = F.mse_loss(velocity_pred, target_flow)


        # ------------------------------------------------------------------
        # Loss: T2I
        # ------------------------------------------------------------------
        if batch_size_t2i > 0:
            start_t2i = batch_size_action
            end_t2i = start_t2i + batch_size_t2i
            loss_t2i = F.cross_entropy(
                logits[start_t2i:end_t2i, max_image_seq_length + 1 :]
                .contiguous()
                .view(-1, self.output_size),
                labels[start_t2i:end_t2i, max_image_seq_length + 1 :]
                .contiguous()
                .view(-1),
                ignore_index=-100,
            )
        else:
            loss_t2i = torch.tensor(0.0, device=input_ids.device)

        # ------------------------------------------------------------------
        # Loss: LM-only
        # ------------------------------------------------------------------
        if batch_size_lm > 0:
            masked_indices = input_ids == self.config.mask_token_id
            start_lm = batch_size_action + batch_size_t2i
            end_lm = start_lm + batch_size_lm

            masked_indices_lm = masked_indices[start_lm:end_lm]
            p_mask_lm = p_mask_lm.to(masked_indices_lm.device)

            per_token_loss = (
                F.cross_entropy(
                    logits[start_lm:end_lm][masked_indices_lm]
                    .contiguous()
                    .view(-1, self.output_size),
                    labels[start_lm:end_lm][masked_indices_lm].contiguous().view(-1),
                    ignore_index=-100,
                    reduction="none",
                )
                / p_mask_lm[masked_indices_lm]
            )

            if answer_lengths_lm is not None:
                answer_lengths_lm = answer_lengths_lm.to(masked_indices_lm.device)
                loss_lm = torch.sum(
                    per_token_loss / answer_lengths_lm[masked_indices_lm]
                ) / float(logits[start_lm:end_lm].shape[0])
            else:
                loss_lm = per_token_loss.sum() / float(
                    logits[start_lm:end_lm].shape[0] * logits[start_lm:end_lm].shape[1]
                )
        else:
            loss_lm = torch.tensor(0.0, device=input_ids.device)

        # ------------------------------------------------------------------
        # Loss: MMU
        # ------------------------------------------------------------------
        if batch_size_mmu > 0:
            masked_indices = input_ids == self.config.mask_token_id
            masked_indices_mmu = masked_indices[-batch_size_mmu:]
            p_mask_mmu = p_mask_mmu.to(masked_indices_mmu.device)
            answer_lengths_mmu = answer_lengths_mmu.to(masked_indices_mmu.device)

            per_token_loss = (
                F.cross_entropy(
                    logits[-batch_size_mmu:][masked_indices_mmu]
                    .contiguous()
                    .view(-1, self.output_size),
                    labels[-batch_size_mmu:][masked_indices_mmu].contiguous().view(-1),
                    ignore_index=-100,
                    reduction="none",
                )
                / p_mask_mmu[masked_indices_mmu]
            )

            loss_mmu = torch.sum(
                per_token_loss / answer_lengths_mmu[masked_indices_mmu]
            ) / float(logits[-batch_size_mmu:].shape[0])
        else:
            loss_mmu = torch.tensor(0.0, device=input_ids.device)

        return logits, loss_action, loss_t2i, loss_lm, loss_mmu, loss_continuous

    # ------------------------------------------------------------------
    # Soft loss variants used for action training
    # ------------------------------------------------------------------
    def modified_ce_with_margin_rule(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
        reduction: str = "mean",
        action_err_token_len: int = 5,
    ) -> torch.Tensor:
        """
        Margin-based variant of cross-entropy.
        基于 margin 的策略——若模型的 argmax 落在 GT 周围一定半径内，
        就把 argmax 的概率当作“正确”的 log-prob 来计算损失；否则仍使用 GT 的 log-prob
        If the argmax token is within ``action_err_token_len`` of the ground-truth id,
        we treat argmax as correct (use its log-prob); otherwise we use the
        ground-truth token's log-prob.
        """
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
        max_logprob, argmax_ids = torch.max(log_probs, dim=-1)  # [B, T], [B, T]

        valid = labels.ne(ignore_index)  # [B, T]
        safe_labels = labels.masked_fill(~valid, 0) # 为避免无效位置带来的索引问题，先把无效位置的索引换成 0

        true_logprob = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # 取出“真实 id 的 log-prob”

        use_max = argmax_ids.sub(safe_labels).abs().le(action_err_token_len) & valid
        chosen_logprob = torch.where(use_max, max_logprob, true_logprob)
            # 当模型 argmax id 与真实 id 的绝对差 <= action_err_token_len 时，
            # 把模型 argmax 的 log-prob 当作 chosen_logprob；否则用真实 id 的 log-prob。
        per_token_loss = -chosen_logprob.masked_fill(~valid, 0.0)

        if reduction == "mean":
            denom = valid.sum().clamp_min(1)
            return per_token_loss.sum() / denom
        if reduction == "sum":
            return per_token_loss.sum()
        if reduction == "none":
            return per_token_loss
        raise ValueError(f"Unsupported reduction: {reduction}")

    def gaussian_soft_ce_window(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        radius: int = 5,
        at_value: float = 1e-3,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        Soft cross-entropy over a local Gaussian window around the ground-truth id.
        在 GT 周围构造一个离散高斯分布作为软目标（GT 附近的 id 也应得概率质量），
        然后计算这个软目标与模型分布的交叉熵（即对 GT 及邻近 id 进行平滑监督）。
        Tokens within +/- ``radius`` of the GT id receive non-zero mass according
        to a discrete Gaussian kernel whose tail value at the window edge is
        approximately ``at_value``.
        对每个 token，构建一个软目标概率分布 q over tokens limited to GT±r，
        权重由离散高斯给出并归一化： q_k ∝ exp(-0.5 * (k - GT)^2 / sigma^2)
        目标损失是 KL(q || p) up to constants, 或等价地： - sum_k q_k log p_k. 
        因为 q 是“软标签”，模型被鼓励把概率质量放在 GT 及其邻近位置上，而不是把质量全部压在单点上。
        """
        B, T, V = logits.shape

        valid = labels.ne(ignore_index)  # [B, T]
        if valid.sum() == 0:
            return logits.sum() * 0.0

        sigma = radius / math.sqrt(2.0 * math.log(1.0 / max(at_value, 1e-12)))
        offsets = torch.arange(-radius, radius + 1, device=logits.device)  # [2r+1]
        base_w = torch.exp(-0.5 * (offsets.float() / max(sigma, 1e-6)) ** 2)  # [2r+1]

        idx = labels.unsqueeze(-1) + offsets  # [B, T, 2r+1]
        in_range = (idx >= 0) & (idx < V)
        in_range &= valid.unsqueeze(-1)

        idx_clamped = idx.clamp(0, V - 1)

        w = base_w.view(1, 1, -1).expand_as(idx).clone()
        w = torch.where(in_range, w, torch.zeros_like(w))

        z = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        w = w / z

        logprob = F.log_softmax(logits, dim=-1)  # [B, T, V]
        logprob_win = torch.gather(logprob, dim=-1, index=idx_clamped)  # [B, T, 2r+1]

        loss_pos = -(w * logprob_win).sum(dim=-1)  # [B, T]
        loss = (loss_pos * valid.float()).sum() / valid.float().sum()
        return loss

    def forward_process_action(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.Tensor,
        continuous_labels: Optional[torch.Tensor] = None,
        max_seq_length: int = 768,
        action_loss_type: Optional[str] = None,  # "single" | "multi" | "gaussian"
        action_err_token_len: int = 6,
        at_value: float = 0.3,
        eoa_token_id: int = 126099,
    ):
        """
        Standalone forward for action training only.
        """
        attention_bias_action = (
            (attention_masks[:, :, None] & attention_masks[:, None, :])     # (B,L,1)和(B,1,L)按位与运算得到(B,L,L)
            .bool()     
            .unsqueeze(1)   # (B,L,L)->(B,1,L,L)
        )   # 代表哪个token对哪个token有效，双向token-level的注意力可见矩阵
        
        output_hidden_states = getattr(self.config, "use_continuous_head", False) and (continuous_labels is not None)
        gen_indices = self._get_gen_indices(input_ids)
        outputs = self(input_ids, attention_bias=attention_bias_action, output_hidden_states=output_hidden_states, gen_indices=gen_indices)
        logits = outputs.logits

        self.output_size = logits.shape[-1]     # 输出词表的大小
        action_logits = logits[:, max_seq_length + 1 :].contiguous()
        action_labels = labels[:, max_seq_length + 1 :].contiguous()

        # 动作 token 空间通常是离散化的但有序（相邻 id 含义相近），因此直接的 plain cross-entropy（把全部概率压在单一 GT id 上）可能过于苛刻。
        # 下面两个loss计算分别实现了两种“宽容/平滑”策略
        if action_loss_type == "multi":
            loss_action = self.modified_ce_with_margin_rule(
                action_logits,
                action_labels,
                ignore_index=-100,
                action_err_token_len=action_err_token_len,
            )
        elif action_loss_type == "gaussian":
            loss_action = self.gaussian_soft_ce_window(
                action_logits,
                action_labels,
                ignore_index=-100,
                radius=int(action_err_token_len / 2.0),
                at_value=at_value,
            )
        else:
            # default: plain cross-entropy
            loss_action = F.cross_entropy(
                action_logits.view(-1, self.output_size),
                action_labels.view(-1),
                ignore_index=-100,
            )
        
        if output_hidden_states:
            hidden_states = outputs.hidden_states[-1]
            action_hidden = hidden_states[:, max_seq_length + 1 :].contiguous()
            
            # Flow Matching Logic
            return_dict = self.continuous_head.sample_noisy_actions(continuous_labels)
            noisy_actions = return_dict["noisy_actions"]
            timesteps = return_dict["timestep_embeddings"]
            target_flow = return_dict["flow"]
            
            # Trim hidden states to match required length (Chunk * Dim)
            required_len = noisy_actions.shape[1] * self.continuous_head.action_dim
            if action_hidden.shape[1] > required_len:
                action_hidden = action_hidden[:, :required_len]
                
            velocity_pred = self.continuous_head.predict_velocity(
                action_hidden, noisy_actions, timesteps
            )
            
            loss_cont = F.mse_loss(velocity_pred, target_flow)
            
            weight = getattr(self.config, "continuous_head_loss_weight", 1.0)
            loss_action = loss_action + weight * loss_cont

        return logits, loss_action

    # ------------------------------------------------------------------
    # MMU generation: single-block discrete diffusion
    # ------------------------------------------------------------------
    @torch.no_grad()
    def mmu_generate_single_block(
        self,
        idx: Optional[torch.LongTensor] = None,  # [L_p] or [1, L_p]
        sequence_ids: Optional[torch.LongTensor] = None,  # [L] or [1, L]
        prompt_mask: Optional[
            torch.LongTensor
        ] = None,  # [L] or [1, L], 1=prompt,0=to-gen
        gen_len: Optional[int] = None,
        steps: int = 64,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",  # or "random"
        mask_id: int = 126336,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Single-sample, *unpadded* single-block discrete diffusion generation.

        Two calling patterns:

        1) Provide only ``idx`` (prompt).  Then you *must* give ``gen_len``.
        2) Provide ``sequence_ids`` and ``prompt_mask``:
           - we treat the prefix where prompt_mask==1 as the prompt
           - by default we generate for the remaining positions
             (``L - prefix_len``) unless a smaller ``gen_len`` is passed.
        """
        use_pair = (sequence_ids is not None) and (prompt_mask is not None)
        use_idx = idx is not None
        if not (use_pair ^ use_idx):
            raise ValueError(
                "Provide either (idx) or (sequence_ids + prompt_mask), but not both."
            )

        # Normalize to [1, L] tensors and determine prefix length
        if use_idx:
            if idx.ndim == 1:
                idx = idx.unsqueeze(0)
            elif idx.ndim != 2 or idx.size(0) != 1:
                raise ValueError("`idx` must be 1D or [1, L].")
            device = idx.device
            prefix_len = idx.size(1)
            if gen_len is None:
                raise ValueError("When using raw `idx`, you must specify `gen_len`.")
            Tnew = int(gen_len)
        else:
            if sequence_ids.ndim == 1:
                sequence_ids = sequence_ids.unsqueeze(0)
            elif sequence_ids.ndim != 2 or sequence_ids.size(0) != 1:
                raise ValueError("`sequence_ids` must be 1D or [1, L].")

            if prompt_mask.ndim == 1:
                prompt_mask = prompt_mask.unsqueeze(0)
            elif prompt_mask.ndim != 2 or prompt_mask.size(0) != 1:
                raise ValueError("`prompt_mask` must be 1D or [1, L].")

            device = sequence_ids.device
            L = sequence_ids.size(1)
            prefix_len = int((prompt_mask == 1).sum().item())

            if prefix_len < 0 or prefix_len > L:
                raise ValueError("Invalid prefix length inferred from `prompt_mask`.")

            idx = sequence_ids[:, :prefix_len]
            tail_len = L - prefix_len
            if tail_len <= 0:
                Tnew = 0
            else:
                Tnew = int(tail_len if gen_len is None else min(gen_len, tail_len))

        # Build attention bias if provided
        if attention_mask is not None and (attention_mask == 0).any():
            attention_bias = (
                (attention_mask[:, :, None] & attention_mask[:, None, :])
                .bool()
                .unsqueeze(1)
            )
        else:
            attention_bias = None

        # Nothing to generate -> return prompt as-is
        if Tnew <= 0:
            meta = {
                "idx_width": prefix_len,
                "per_sample_gen_len": torch.tensor(
                    [0], device=device, dtype=torch.long
                ),
            }
            return idx, meta

        # Initialize sequence: prompt + Tnew <MASK>
        x = torch.full((1, prefix_len + Tnew), mask_id, dtype=idx.dtype, device=device)
        x[:, :prefix_len] = idx

        # Allowed window for generation
        allowed_abs = torch.zeros_like(x, dtype=torch.bool)
        allowed_abs[:, prefix_len : prefix_len + Tnew] = True

        # Precompute how many tokens to transfer at each step
        num_transfer_tokens = get_num_transfer_tokens(
            allowed_abs[:, prefix_len:], steps
        )  # [1, steps]

        for t in range(steps):
            # classifier-free guidance over the whole sequence
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[:, :prefix_len] = mask_id
                x_cat = torch.cat([x, un_x], dim=0)
                logits = self(x_cat).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1.0) * (logits - un_logits)
            else:
                logits = self(x, attention_bias=attention_bias).logits

            # Gumbel-max sampling
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # [1, L_all]

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand_like(x0, dtype=torch.float64)
            else:
                raise NotImplementedError(f"Unknown remasking mode: {remasking}")

            mask_index = x == mask_id
            selectable = allowed_abs & mask_index
            if not torch.any(selectable):
                break

            confidence = torch.full_like(x0_p, float("-inf"))
            confidence[selectable] = x0_p[selectable]

            k = int(num_transfer_tokens[0, t].item())
            k = min(k, int(selectable.sum().item()))
            if k > 0:
                _, sel = torch.topk(confidence[0], k)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                transfer_index[0, sel] = True
                x[transfer_index] = x0[transfer_index]

        meta = {
            "idx_width": prefix_len,
            "per_sample_gen_len": torch.tensor([Tnew], device=device, dtype=torch.long),
        }
        return x, meta


# Register with Hugging Face Auto* APIs
AutoConfig.register("mmact", MMACTConfig)
AutoModelForCausalLM.register(MMACTConfig, MMACTModelLM)
AutoModel.register(MMACTConfig, MMACTModelLM)
