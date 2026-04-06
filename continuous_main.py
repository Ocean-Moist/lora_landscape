"""Multi-GPU entry point for continuous adapter grid enumeration."""

import argparse
import os
import time
import torch
import torch.multiprocessing as mp

from continuous_config import ContinuousConfig
from continuous_adapter import ContinuousAdapterConfig
from coupled_data import load_eval_sequences
from continuous_enumeration import enumerate_continuous


def worker(gpu_id: int, cfg: ContinuousConfig):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    print(f"[GPU {gpu_id}] Loading model...")
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(cfg.model_name, torch_dtype=cfg.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)

    print(f"[GPU {gpu_id}] Loading eval data...")
    input_ids, labels = load_eval_sequences(
        cfg.model_name, cfg.num_sequences, cfg.seq_len, seed=42
    )
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    print(f"[GPU {gpu_id}] Building adapters...")
    adapters = []
    for i, layer_idx in enumerate(cfg.adapter_layers):
        adapters.append(ContinuousAdapterConfig(
            d_model=768,
            width=cfg.bottleneck_width,
            n_free=cfg.params_per_adapter,
            alpha=cfg.adapter_alpha,
            seed=cfg.adapter_seed + i * 1000,
            device=device,
            dtype=cfg.dtype,
        ))

    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"[GPU {gpu_id}] Starting enumeration...")
    t0 = time.time()
    enumerate_continuous(
        gpu_id=gpu_id,
        num_gpus=cfg.num_gpus,
        model=model,
        adapters=adapters,
        input_ids=input_ids,
        labels=labels,
        cfg=cfg,
        output_path=cfg.output_dir,
    )
    elapsed = time.time() - t0
    print(f"[GPU {gpu_id}] Finished in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Continuous adapter grid enumeration")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--n-grid", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    cfg = ContinuousConfig()
    cfg.num_gpus = args.gpus
    if args.n_grid:
        cfg.n_grid = args.n_grid
        cfg.__post_init__()
    if args.alpha:
        cfg.adapter_alpha = args.alpha
    if args.output_dir:
        cfg.output_dir = args.output_dir

    assert cfg.configs_per_adapter % cfg.num_gpus == 0, \
        f"{cfg.configs_per_adapter} must be divisible by {cfg.num_gpus}"

    print(f"Continuous MLP Adapter Grid Enumeration")
    print(f"  Adapters at layers: {cfg.adapter_layers}")
    print(f"  Params per adapter: {cfg.params_per_adapter}")
    print(f"  Grid: {cfg.n_grid} points per dim, {cfg.configs_per_adapter} per adapter")
    print(f"  Total configs: {cfg.total_configs:,}")
    print(f"  Parameter range: [{cfg.param_min}, {cfg.param_max}]")
    print(f"  Grid spacing: {cfg.grid_spacing:.4f}")
    print(f"  Alpha: {cfg.adapter_alpha}")
    print(f"  GPUs: {cfg.num_gpus}")

    if args.test:
        cfg.num_gpus = 1
        print(f"\nTest mode: 1 GPU")
        worker(0, cfg)
        return

    if cfg.num_gpus == 1:
        worker(0, cfg)
    else:
        mp.set_start_method("spawn", force=True)
        mp.spawn(worker, nprocs=cfg.num_gpus, args=(cfg,), join=True)


if __name__ == "__main__":
    main()
