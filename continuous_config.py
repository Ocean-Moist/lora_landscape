"""Configuration for continuous MLP adapter landscape with provable global optimum."""

from dataclasses import dataclass
import torch


@dataclass
class ContinuousConfig:
    # Model
    model_name: str = "gpt2"

    # Adapter architecture — same as coupled, but fewer free params
    adapter_layers: tuple = (2, 6, 11)
    bottleneck_width: int = 4
    params_per_adapter: int = 2  # 1 in W1, 1 in W2 → multiplicative coupling
    adapter_alpha: float = 2.0
    adapter_seed: int = 42

    # Continuous parameter range
    param_min: float = -1.0
    param_max: float = 1.0

    # Grid resolution per dimension
    n_grid: int = 32  # 32 points per param → 32^2=1024 per adapter

    # Dataset
    num_sequences: int = 4
    seq_len: int = 32

    # Enumeration
    num_gpus: int = 8
    dtype: torch.dtype = torch.float16

    # Storage
    output_dir: str = "/data/backups/rganapa/lora_landscape/results_continuous"

    def __post_init__(self):
        self.num_adapters = len(self.adapter_layers)
        self.num_params = self.num_adapters * self.params_per_adapter  # 6
        self.configs_per_adapter = self.n_grid ** self.params_per_adapter  # 1024
        self.total_configs = self.configs_per_adapter ** self.num_adapters  # 1024^3 ≈ 1.07B
        self.grid_values = None  # computed at runtime via torch.linspace

    @property
    def grid_spacing(self):
        return (self.param_max - self.param_min) / (self.n_grid - 1)
