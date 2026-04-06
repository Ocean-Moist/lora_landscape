"""Push LR until optimizers break."""

import torch
import numpy as np
from config import Config
from binary_lora import BinaryLoRAConfig
from model_setup import load_frozen_model, prepare_eval_data, PrecomputedState
from optimizers import run_optimizer_benchmark, Muon

cfg = Config()
device = torch.device("cuda:0")
model = load_frozen_model(device, cfg.dtype, cfg.model_name)
input_ids, labels = prepare_eval_data(cfg.model_name, cfg.eval_text, cfg.seq_len)
labels = labels.to(device)
lora_cfg = BinaryLoRAConfig(768, cfg.lora_rank_q, cfg.lora_rank_v, cfg.lora_alpha, cfg.lora_seed, device, cfg.dtype)
state = PrecomputedState(model, input_ids, lora_cfg, cfg.lora_layer, device, cfg.dtype)
del model; torch.cuda.empty_cache()

rq, rv = cfg.lora_rank_q, cfg.lora_rank_v
RUNS = 5000
STEPS = 50
GMIN = 4.0469

configs = [
    # SGD — push past 10
    ("SGD lr=10", torch.optim.SGD, dict(lr=10, momentum=0.9)),
    ("SGD lr=20", torch.optim.SGD, dict(lr=20, momentum=0.9)),
    ("SGD lr=50", torch.optim.SGD, dict(lr=50, momentum=0.9)),
    ("SGD lr=100", torch.optim.SGD, dict(lr=100, momentum=0.9)),
    ("SGD lr=500", torch.optim.SGD, dict(lr=500, momentum=0.9)),
    ("SGD lr=1000", torch.optim.SGD, dict(lr=1000, momentum=0.9)),
    # SGD no momentum — cleaner signal
    ("SGD lr=50 no-mom", torch.optim.SGD, dict(lr=50, momentum=0.0)),
    ("SGD lr=100 no-mom", torch.optim.SGD, dict(lr=100, momentum=0.0)),
    ("SGD lr=500 no-mom", torch.optim.SGD, dict(lr=500, momentum=0.0)),
    ("SGD lr=1000 no-mom", torch.optim.SGD, dict(lr=1000, momentum=0.0)),
    # AdamW — push past 5
    ("AdamW lr=5", torch.optim.AdamW, dict(lr=5.0)),
    ("AdamW lr=10", torch.optim.AdamW, dict(lr=10.0)),
    ("AdamW lr=50", torch.optim.AdamW, dict(lr=50.0)),
    ("AdamW lr=100", torch.optim.AdamW, dict(lr=100.0)),
    ("AdamW lr=500", torch.optim.AdamW, dict(lr=500.0)),
    # Muon — push past 20
    ("Muon lr=20", Muon, dict(lr=20, momentum=0.95, rank_q=rq, rank_v=rv)),
    ("Muon lr=50", Muon, dict(lr=50, momentum=0.95, rank_q=rq, rank_v=rv)),
    ("Muon lr=100", Muon, dict(lr=100, momentum=0.95, rank_q=rq, rank_v=rv)),
    ("Muon lr=500", Muon, dict(lr=500, momentum=0.95, rank_q=rq, rank_v=rv)),
    ("Muon lr=1000", Muon, dict(lr=1000, momentum=0.95, rank_q=rq, rank_v=rv)),
]

print(f"{'Optimizer':<22s} {'Succ%':>6s} {'MedBest':>8s} {'MedSteps':>8s} {'Mean':>8s}")
print("-" * 58)

for name, cls, kwargs in configs:
    try:
        r = run_optimizer_benchmark(
            name, cls, kwargs, state, lora_cfg, labels,
            cfg.num_params, n_runs=RUNS, max_steps=STEPS, global_min_loss=GMIN,
            batch_chunk=2048,
        )
        succ_mask = r.best_losses < GMIN + 0.01
        med_steps = np.median(r.steps_to_best[succ_mask]) if succ_mask.any() else float("nan")
        print(f"{name:<22s} {r.success_rate:>5.1%} {r.median_best_loss:>8.4f} {med_steps:>8.1f} {r.mean_best_loss:>8.4f}")
    except Exception as e:
        print(f"{name:<22s} FAILED: {e}")
