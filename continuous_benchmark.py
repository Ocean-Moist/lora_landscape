"""Optimizer benchmarking against certified continuous landscape ground truth.

No STE needed — parameters are continuous, gradients are exact.
This is directly relevant to real LLM training dynamics.
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

from continuous_config import ContinuousConfig
from continuous_adapter import ContinuousAdapterConfig
from coupled_data import load_eval_sequences
from continuous_certify import differentiable_forward, load_landscape, global_index_to_params


def run_benchmark(name, optimizer_cls, optimizer_kwargs, model, adapters,
                  input_ids, labels, cfg, n_runs=5000, max_steps=200,
                  global_min=0.0, chunk_size=256):
    """Run optimizer from random inits, track convergence."""
    device = input_ids.device
    all_best = []
    all_steps = []
    all_trajectories = []

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

            # Clamp to parameter range
            with torch.no_grad():
                latent.data.clamp_(cfg.param_min, cfg.param_max)

            with torch.no_grad():
                improved = losses < best_losses
                best_losses = torch.where(improved, losses, best_losses)
                steps_to_best = torch.where(improved, step, steps_to_best)

        all_best.append(best_losses.detach().cpu().numpy())
        all_steps.append(steps_to_best.cpu().numpy())
        print(f"  {name}: {ce}/{n_runs}, best={best_losses.min():.4f}, "
              f"median={best_losses.median():.4f}")

    best = np.concatenate(all_best)
    steps = np.concatenate(all_steps)
    success = float(np.mean(np.isclose(best, global_min, atol=0.01)))

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=str, default="continuous_benchmark.json")
    args = parser.parse_args()

    cfg = ContinuousConfig()
    cfg.output_dir = args.input_dir
    device = torch.device(f"cuda:{args.gpu}")

    # Load landscape for ground truth
    print("Loading landscape...")
    losses = load_landscape(args.input_dir, args.num_gpus, cfg.total_configs)
    global_min = float(losses.min())
    global_min_idx = int(np.argmin(losses))
    global_min_params = global_index_to_params(global_min_idx, cfg)
    print(f"  Global min: {global_min:.4f} at params {global_min_params}")

    # Load model
    print("Loading model...")
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(cfg.model_name, torch_dtype=cfg.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)

    input_ids, labels_data = load_eval_sequences(cfg.model_name, cfg.num_sequences, cfg.seq_len)
    input_ids = input_ids.to(device)
    labels_data = labels_data.to(device)

    adapters = []
    for i, layer_idx in enumerate(cfg.adapter_layers):
        adapters.append(ContinuousAdapterConfig(
            d_model=768, width=cfg.bottleneck_width, n_free=cfg.params_per_adapter,
            alpha=cfg.adapter_alpha, seed=cfg.adapter_seed + i * 1000,
            device=device, dtype=cfg.dtype,
        ))

    all_results = {"global_min": global_min, "results": []}

    # Define optimizers — no STE, exact continuous gradients
    configs = [
        # SGD variants
        ("SGD (lr=0.001)", torch.optim.SGD, dict(lr=0.001)),
        ("SGD (lr=0.01)", torch.optim.SGD, dict(lr=0.01)),
        ("SGD (lr=0.01, mom=0.9)", torch.optim.SGD, dict(lr=0.01, momentum=0.9)),
        ("SGD (lr=0.1, mom=0.9)", torch.optim.SGD, dict(lr=0.1, momentum=0.9)),
        # AdamW variants
        ("AdamW (lr=0.001)", torch.optim.AdamW, dict(lr=0.001)),
        ("AdamW (lr=0.01)", torch.optim.AdamW, dict(lr=0.01)),
        ("AdamW (lr=0.1)", torch.optim.AdamW, dict(lr=0.1)),
        ("AdamW (lr=0.5)", torch.optim.AdamW, dict(lr=0.5)),
        # L-BFGS (second order)
        # Note: L-BFGS needs closure, handled separately
    ]

    for name, cls, kwargs in configs:
        print(f"\n{name}...")
        t0 = time.time()
        result = run_benchmark(
            name, cls, kwargs, model, adapters, input_ids, labels_data, cfg,
            n_runs=args.runs, max_steps=args.steps, global_min=global_min,
        )
        result["wall_time"] = time.time() - t0
        print(f"  Success: {result['success_rate']:.2%}, "
              f"Median: {result['median_best']:.4f}, "
              f"Min: {result['min_best']:.4f}, "
              f"Steps: {result['mean_steps']:.1f}")
        all_results["results"].append(result)

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
