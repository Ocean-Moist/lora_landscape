"""Optimizer benchmarking against certified continuous landscape ground truth.

No STE needed — parameters are continuous, gradients are exact.
Parallelized across GPUs: each GPU runs a different optimizer config.
"""

import argparse
import json
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from pathlib import Path

from continuous_config import ContinuousConfig
from continuous_adapter import ContinuousAdapterConfig
from coupled_data import load_eval_sequences
from continuous_certify import differentiable_forward, load_landscape, global_index_to_params


def run_single_benchmark(gpu_id, name, optimizer_cls, optimizer_kwargs,
                         cfg, global_min, n_runs=2000, max_steps=50,
                         chunk_size=128):
    """Run one optimizer config on one GPU."""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(cfg.model_name, torch_dtype=cfg.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)

    input_ids, labels = load_eval_sequences(cfg.model_name, cfg.num_sequences, cfg.seq_len)
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    adapters = []
    for i, layer_idx in enumerate(cfg.adapter_layers):
        adapters.append(ContinuousAdapterConfig(
            d_model=768, width=cfg.bottleneck_width, n_free=cfg.params_per_adapter,
            alpha=cfg.adapter_alpha, seed=cfg.adapter_seed + i * 1000,
            device=device, dtype=cfg.dtype,
        ))

    all_best = []
    all_steps = []

    for cs in range(0, n_runs, chunk_size):
        ce = min(cs + chunk_size, n_runs)
        sz = ce - cs

        latent = torch.nn.Parameter(
            torch.empty(sz, cfg.num_params, device=device, dtype=torch.float32).uniform_(
                cfg.param_min, cfg.param_max
            )
        )
        optimizer = optimizer_cls([latent], **optimizer_kwargs)

        best_losses = torch.full((sz,), float("inf"), device=device)
        steps_to_best = torch.zeros(sz, device=device, dtype=torch.long)

        for step in range(max_steps):
            optimizer.zero_grad()
            losses = differentiable_forward(model, adapters, input_ids, labels, latent, cfg)
            losses.sum().backward()
            optimizer.step()

            with torch.no_grad():
                latent.data.clamp_(cfg.param_min, cfg.param_max)
                improved = losses < best_losses
                best_losses = torch.where(improved, losses, best_losses)
                steps_to_best = torch.where(improved, step, steps_to_best)

        all_best.append(best_losses.detach().cpu().numpy())
        all_steps.append(steps_to_best.cpu().numpy())
        print(f"  [GPU {gpu_id}] {name}: {ce}/{n_runs}, best={best_losses.min():.4f}, "
              f"median={best_losses.median():.4f}")

    best = np.concatenate(all_best)
    steps = np.concatenate(all_steps)
    success = float(np.mean(np.isclose(best, global_min, atol=0.05)))

    return {
        "name": name,
        "success_rate": success,
        "median_best": float(np.median(best)),
        "mean_best": float(np.mean(best)),
        "min_best": float(best.min()),
        "mean_steps": float(steps.mean()),
        "best_loss_percentiles": {
            str(p): float(np.percentile(best, p))
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        },
    }


def worker(gpu_id, task_queue, result_dict, cfg, global_min, n_runs, max_steps):
    """Worker process: pull optimizer configs from queue, run them."""
    while True:
        try:
            idx, name, cls_name, kwargs = task_queue.get_nowait()
        except Exception:
            break

        # Reconstruct optimizer class from name
        if cls_name == "SGD":
            cls = torch.optim.SGD
        elif cls_name == "AdamW":
            cls = torch.optim.AdamW
        else:
            cls = torch.optim.AdamW

        print(f"[GPU {gpu_id}] Starting {name}...")
        t0 = time.time()
        result = run_single_benchmark(gpu_id, name, cls, kwargs, cfg,
                                       global_min, n_runs, max_steps)
        result["wall_time"] = time.time() - t0
        result_dict[idx] = result
        print(f"[GPU {gpu_id}] {name}: success={result['success_rate']:.2%}, "
              f"median={result['median_best']:.4f}, time={result['wall_time']:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--runs", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", type=str, default="continuous_benchmark.json")
    args = parser.parse_args()

    cfg = ContinuousConfig()
    cfg.output_dir = args.input_dir

    # Find global min without loading full landscape (too slow)
    # Just use the grid min from certification
    cert_path = Path(args.input_dir) / "certification.json"
    if cert_path.exists():
        with open(cert_path) as f:
            cert = json.load(f)
        global_min = cert.get("refinement", {}).get("best_loss", cert["grid_min_loss"])
    else:
        print("Loading landscape for global min...")
        losses = load_landscape(args.input_dir, args.num_gpus, cfg.total_configs)
        global_min = float(losses.min())
        del losses

    print(f"Global min: {global_min:.4f}")

    # Define all optimizer configs
    optimizer_configs = [
        ("SGD (lr=0.001)", "SGD", dict(lr=0.001)),
        ("SGD (lr=0.01)", "SGD", dict(lr=0.01)),
        ("SGD (lr=0.01, mom=0.9)", "SGD", dict(lr=0.01, momentum=0.9)),
        ("SGD (lr=0.1, mom=0.9)", "SGD", dict(lr=0.1, momentum=0.9)),
        ("AdamW (lr=0.001)", "AdamW", dict(lr=0.001)),
        ("AdamW (lr=0.01)", "AdamW", dict(lr=0.01)),
        ("AdamW (lr=0.1)", "AdamW", dict(lr=0.1)),
        ("AdamW (lr=0.5)", "AdamW", dict(lr=0.5)),
    ]

    # Create task queue and result dict
    mp.set_start_method("spawn", force=True)
    task_queue = mp.Queue()
    manager = mp.Manager()
    result_dict = manager.dict()

    for idx, (name, cls_name, kwargs) in enumerate(optimizer_configs):
        task_queue.put((idx, name, cls_name, kwargs))

    # Launch workers — one per GPU, each pulls tasks from queue
    n_workers = min(args.num_gpus, len(optimizer_configs))
    processes = []
    for gpu_id in range(n_workers):
        p = mp.Process(target=worker, args=(
            gpu_id, task_queue, result_dict, cfg, global_min, args.runs, args.steps
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Collect results in order
    all_results = {"global_min": global_min, "results": []}
    for idx in range(len(optimizer_configs)):
        if idx in result_dict:
            all_results["results"].append(dict(result_dict[idx]))

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Optimizer':<30s} {'Success%':>8s} {'MedBest':>8s} {'MinBest':>8s} {'AvgSteps':>9s}")
    print("-" * 90)
    for r in all_results["results"]:
        print(f"{r['name']:<30s} {r['success_rate']:>7.2%} "
              f"{r['median_best']:>8.4f} {r['min_best']:>8.4f} {r['mean_steps']:>9.1f}")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
