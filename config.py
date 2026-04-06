from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    # Model
    model_name: str = "gpt2"
    lora_layer: int = 11  # last transformer layer — minimizes per-config compute
    lora_rank_q: int = 15
    lora_rank_v: int = 15
    lora_alpha: float = 25.0
    lora_seed: int = 42

    # Eval — predict last token from preceding context
    eval_text: str = "The quick brown fox jumps over the lazy dog"
    seq_len: int = 8  # tokens of context

    # Enumeration
    num_params: int = 40  # lora_rank_q + lora_rank_v
    num_gpus: int = 8
    config_batch_size: int = 262144  # configs evaluated in one batched forward pass
    dtype: torch.dtype = torch.float16
    checkpoint_interval: int = 1024  # batches between checkpoints

    # Storage
    output_dir: str = "/data/backups/rganapa/lora_landscape/results"

    def __post_init__(self):
        self.num_params = self.lora_rank_q + self.lora_rank_v
        self.total_configs = 2 ** self.num_params
