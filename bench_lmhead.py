"""Benchmark LM head matmul with different vocab padding."""
import torch
import torch.nn.functional as F
import time
from config import Config
from binary_lora import BinaryLoRAConfig, config_indices_to_binary
from model_setup import load_frozen_model, prepare_eval_data, PrecomputedState

cfg = Config()
device = torch.device("cuda:0")
model = load_frozen_model(device, cfg.dtype, cfg.model_name)
input_ids, labels = prepare_eval_data(cfg.model_name, cfg.eval_text, cfg.seq_len)
labels = labels.to(device)
lora_cfg = BinaryLoRAConfig(768, cfg.lora_rank_q, cfg.lora_rank_v, cfg.lora_alpha, cfg.lora_seed, device, cfg.dtype)
state = PrecomputedState(model, input_ids, lora_cfg, cfg.lora_layer, device, cfg.dtype)
del model
torch.cuda.empty_cache()

W = state.lm_head_weight  # [50257, 768]
B = 32768
h = torch.randn(B * 7, 768, device=device, dtype=cfg.dtype)

for vocab_size in [50257, 50304, 50432, 51200]:
    if vocab_size > 50257:
        W_test = F.pad(W, (0, 0, 0, vocab_size - 50257), value=-1e9)
    else:
        W_test = W

    # warmup
    _ = h @ W_test.T
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(10):
        _ = h @ W_test.T
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / 10

    flops = B * 7 * 768 * vocab_size * 2
    tflops = flops / elapsed / 1e12
    print(f"V={vocab_size:>6d}  {elapsed*1000:>7.1f}ms  {tflops:>6.1f} TFLOP/s")
