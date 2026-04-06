"""Test different alpha values to find one that creates meaningful loss variation."""
import torch
from config import Config
from binary_lora import BinaryLoRAConfig, config_indices_to_binary
from model_setup import load_frozen_model, prepare_eval_data, PrecomputedState
from enumeration import batched_forward

cfg = Config()
device = torch.device("cuda:0")
model = load_frozen_model(device, cfg.dtype, cfg.model_name)
input_ids, labels = prepare_eval_data(cfg.model_name, cfg.eval_text, cfg.seq_len)
labels = labels.to(device)

B = 10000  # sample 10K random configs

for alpha in [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
    lora_cfg = BinaryLoRAConfig(
        768, cfg.lora_rank_q, cfg.lora_rank_v, alpha, cfg.lora_seed, device, cfg.dtype
    )
    state = PrecomputedState(model, input_ids, lora_cfg, cfg.lora_layer, device, cfg.dtype)

    # Random config indices
    indices = torch.randint(0, 2**cfg.num_params, (B,), device=device, dtype=torch.int64)
    binary = config_indices_to_binary(indices, cfg.num_params).to(cfg.dtype)
    m_Q = binary[:, :cfg.lora_rank_q]
    m_V = binary[:, cfg.lora_rank_q:]

    with torch.no_grad():
        losses = batched_forward(state, m_Q, m_V, lora_cfg.B_Q, lora_cfg.B_V, alpha, labels)

    lo, hi = losses.min().item(), losses.max().item()
    mu, std = losses.mean().item(), losses.std().item()
    print(f"alpha={alpha:>6.1f}  mean={mu:.3f}  std={std:.4f}  range=[{lo:.3f}, {hi:.3f}]  spread={hi-lo:.3f}")
