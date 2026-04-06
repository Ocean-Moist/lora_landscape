"""Configuration for coupled MLP adapter landscape benchmark."""

from dataclasses import dataclass, field
import torch


@dataclass
class CoupledConfig:
    # Model
    model_name: str = "gpt2"

    # Adapter architecture
    adapter_layers: tuple = (2, 6, 11)  # insertion points in transformer
    bottleneck_width: int = 4  # 768 -> w -> 768
    bits_per_adapter: int = 8  # free binary params per adapter
    adapter_alpha: float = 2.0  # scaling for adapter output
    adapter_seed: int = 42

    # Dataset — enough sequences for representational pressure, but small
    # enough that attention scores [B, 12, seq, seq] fit in GPU memory
    num_sequences: int = 4
    seq_len: int = 32  # 4*32 = 128 tokens total
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"

    # Enumeration
    num_gpus: int = 8
    config_batch_size: int = 256  # configs per batch in innermost loop
    dtype: torch.dtype = torch.float16

    # Storage
    output_dir: str = "/data/backups/rganapa/lora_landscape/results_coupled"

    def __post_init__(self):
        self.num_adapters = len(self.adapter_layers)
        self.num_params = self.num_adapters * self.bits_per_adapter  # 24
        self.total_configs = 2 ** self.num_params  # 2^24 = 16,777,216
        self.configs_per_layer = 2 ** self.bits_per_adapter  # 256
