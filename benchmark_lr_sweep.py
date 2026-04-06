"""Quick LR sweep — how fast can each optimizer reach the global min?"""

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
STEPS = 100
GMIN = 4.0469

configs = [
    # SGD — crank LR
    ("SGD lr=0.5", torch.optim.SGD, dict(lr=0.5, momentum=0.9)),
    ("SGD lr=1.0", torch.optim.SGD, dict(lr=1.0, momentum=0.9)),
    ("SGD lr=2.0", torch.optim.SGD, dict(lr=2.0, momentum=0.9)),
    ("SGD lr=5.0", torch.optim.SGD, dict(lr=5.0, momentum=0.9)),
    ("SGD lr=10.0", torch.optim.SGD, dict(lr=10.0, momentum=0.9)),
    # AdamW — crank LR
    ("AdamW lr=0.1", torch.optim.AdamW, dict(lr=0.1)),
    ("AdamW lr=0.5", torch.optim.AdamW, dict(lr=0.5)),
    ("AdamW lr=1.0", torch.optim.AdamW, dict(lr=1.0)),
    ("AdamW lr=2.0", torch.optim.AdamW, dict(lr=2.0)),
    ("AdamW lr=5.0", torch.optim.AdamW, dict(lr=5.0)),
    # Muon — crank LR
    ("Muon lr=1.0", Muon, dict(lr=1.0, momentum=0.95, rank_q=rq, rank_v=rv)),
    ("Muon lr=2.0", Muon, dict(lr=2.0, momentum=0.95, rank_q=rq, rank_v=rv)),
    ("Muon lr=5.0", Muon, dict(lr=5.0, momentum=0.95, rank_q=rq, rank_v=rv)),
    ("Muon lr=10.0", Muon, dict(lr=10.0, momentum=0.95, rank_q=rq, rank_v=rv)),
    ("Muon lr=20.0", Muon, dict(lr=20.0, momentum=0.95, rank_q=rq, rank_v=rv)),
]

print(f"{'Optimizer':<20s} {'Success%':>8s} {'MedBest':>8s} {'AvgSteps':>9s} {'Med Steps':>9s}")
print("-" * 60)

for name, cls, kwargs in configs:
    r = run_optimizer_benchmark(
        name, cls, kwargs, state, lora_cfg, labels,
        cfg.num_params, n_runs=RUNS, max_steps=STEPS, global_min_loss=GMIN,
        batch_chunk=2048,
    )
    med_steps = np.median(r.steps_to_best[r.best_losses < GMIN + 0.01]) if r.success_rate > 0 else float("nan")
    print(f"{name:<20s} {r.success_rate:>7.1%} {r.median_best_loss:>8.4f} {r.steps_to_best.mean():>9.1f} {med_steps:>9.1f}")
