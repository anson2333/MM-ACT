import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384 # 
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.5
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    mask_token_id: int = -1
    block_length: int = 4

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)

        # compatibility: some custom configs (LLaDA etc.) do not expose
        # `max_position_embeddings`. try several fallbacks.
        max_pos = getattr(self.hf_config, "max_position_embeddings", None)
        if max_pos is None:
            # common alternative names
            for k in ("n_positions", "max_seq_len", "n_ctx", "context_length"):
                max_pos = getattr(self.hf_config, k, None)
                if max_pos is not None:
                    break

        if max_pos is None:
            # last resort: keep existing default but warn
            try:
                import warnings
                warnings.warn(
                    f"hf_config for {self.model} has no max position attr; "
                    "falling back to Config.max_model_len (may be too small)."
                )
            except Exception:
                pass
            max_pos = self.max_model_len

        self.max_model_len = min(self.max_model_len, int(max_pos))
        assert self.max_num_batched_tokens >= self.max_model_len
        assert self.mask_token_id != -1, "Mask token ID must be set"
