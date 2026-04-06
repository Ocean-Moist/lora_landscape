"""Benchmark throughput at different batch sizes."""
import torch
import time
from config import Config
from binary_lora import BinaryLoRAConfig, config_indices_to_binary
from model_setup import load_frozen_model, prepare_eval_data, PrecomputedState
from enumeration import batched_forward

cfg = Config()
device = torch.device("cuda:0")
model = load_frozen_model(device, cfg.dtype, cfg.model_name)
input_ids, labels = prepare_eval_data(cfg.model_name, cfg.eval_text, cfg.seq_len)
labels = labels.to(device)
lora_cfg = BinaryLoRAConfig(768, cfg.lora_rank_q, cfg.lora_rank_v, cfg.lora_alpha, cfg.lora_seed, device, cfg.dtype)
state = PrecomputedState(model, input_ids, lora_cfg, cfg.lora_layer, device, cfg.dtype)
del model
torch.cuda.empty_cache()

for bs in [32768, 65536, 131072, 262144, 524288]:
    try:
        indices = torch.arange(bs, device=device, dtype=torch.int64)
        binary = config_indices_to_binary(indices, cfg.num_params).to(cfg.dtype)
        m_Q = binary[:, :cfg.lora_rank_q]
        m_V = binary[:, cfg.lora_rank_q:]
        with torch.no_grad():
            _ = batched_forward(state, m_Q, m_V, lora_cfg.B_Q, lora_cfg.B_V, cfg.lora_alpha, labels)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            with torch.no_grad():
                losses = batched_forward(state, m_Q, m_V, lora_cfg.B_Q, lora_cfg.B_V, cfg.lora_alpha, labels)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / 5
        throughput = bs / elapsed
        mem = torch.cuda.max_memory_allocated() / 1e9
        lo = losses.min().item()
        hi = losses.max().item()
        print(f"bs={bs:>7d}  {elapsed*1000:>7.1f}ms  {throughput:>10,.0f} cfg/s  {mem:.1f}GB  loss=[{lo:.2f},{hi:.2f}]")
        torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        print(f"bs={bs:>7d}  FAILED: {e}")
        torch.cuda.reset_peak_memory_stats()

best_bs = 131072  # likely sweet spot
throughput_per_gpu = 131072 / 0.1  # placeholder, will be from actual measurement
print(f"\n--- Projections at best batch size ---")
print(f"8 GPUs for 2^40 configs: {2**40 / (throughput_per_gpu * 8) / 3600:.1f} hours (estimated)")
print(f"8 GPUs for 2^30 configs: {2**30 / (throughput_per_gpu * 8) / 3600:.4f} hours (estimated)")
