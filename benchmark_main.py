"""Run optimizer benchmarks against ground-truth ternary LoRA landscape."""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

from config import Config
from binary_lora import BinaryLoRAConfig
from model_setup import load_frozen_model, prepare_eval_data, PrecomputedState
from optimizers import run_optimizer_benchmark, Muon, compute_landscape_percentile


def load_landscape(input_dir: str, total_configs: int, num_gpus: int) -> np.ndarray:
    """Load the full loss landscape for percentile computation."""
    shard_size = total_configs // num_gpus
    shards = []
    for i in range(num_gpus):
        path = Path(input_dir) / f"losses_shard_{i}.npy"
        if path.exists():
            shard = np.memmap(path, dtype=np.float16, mode="r", shape=(shard_size,))
            shards.append(shard)
    if shards:
        return np.concatenate(shards).astype(np.float32)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--global-min", type=float, default=4.0469)
    parser.add_argument("--landscape-dir", type=str, default=None,
                        help="Directory with loss shards for percentile computation")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(f"cuda:{args.gpu}")

    # Load landscape for percentile computation
    landscape_dir = args.landscape_dir or cfg.output_dir
    print(f"Loading landscape from {landscape_dir}...")
    landscape = load_landscape(landscape_dir, cfg.total_configs, cfg.num_gpus)
    if landscape is not None:
        print(f"Loaded {len(landscape):,} configs for percentile computation")
    else:
        print("WARNING: No landscape data found, percentiles will be 0")

    print("Loading model and precomputing...")
    model = load_frozen_model(device, cfg.dtype, cfg.model_name)
    input_ids, labels = prepare_eval_data(cfg.model_name, cfg.eval_text, cfg.seq_len)
    labels = labels.to(device)

    lora_cfg = BinaryLoRAConfig(
        768, cfg.lora_rank_q, cfg.lora_rank_v, cfg.lora_alpha, cfg.lora_seed, device, cfg.dtype
    )
    state = PrecomputedState(model, input_ids, lora_cfg, cfg.lora_layer, device, cfg.dtype)
    del model
    torch.cuda.empty_cache()

    rq, rv = cfg.lora_rank_q, cfg.lora_rank_v

    # Define optimizers
    optimizers = [
        # SGD
        ("SGD (lr=0.1, mom=0.9)", torch.optim.SGD, dict(lr=0.1, momentum=0.9)),
        ("SGD (lr=0.5, mom=0.9)", torch.optim.SGD, dict(lr=0.5, momentum=0.9)),
        # AdamW
        ("AdamW (lr=0.01)", torch.optim.AdamW, dict(lr=0.01)),
        ("AdamW (lr=0.05)", torch.optim.AdamW, dict(lr=0.05)),
        ("AdamW (lr=0.1)", torch.optim.AdamW, dict(lr=0.1)),
        # MUON — sweep LR and momentum
        ("Muon (lr=0.02, mom=0.95)", Muon, dict(lr=0.02, momentum=0.95, rank_q=rq, rank_v=rv)),
        ("Muon (lr=0.05, mom=0.95)", Muon, dict(lr=0.05, momentum=0.95, rank_q=rq, rank_v=rv)),
        ("Muon (lr=0.1, mom=0.95)", Muon, dict(lr=0.1, momentum=0.95, rank_q=rq, rank_v=rv)),
        ("Muon (lr=0.2, mom=0.95)", Muon, dict(lr=0.2, momentum=0.95, rank_q=rq, rank_v=rv)),
        ("Muon (lr=0.5, mom=0.95)", Muon, dict(lr=0.5, momentum=0.95, rank_q=rq, rank_v=rv)),
        ("Muon (lr=1.0, mom=0.95)", Muon, dict(lr=1.0, momentum=0.95, rank_q=rq, rank_v=rv)),
        ("Muon (lr=0.5, mom=0.9)", Muon, dict(lr=0.5, momentum=0.9, rank_q=rq, rank_v=rv)),
        ("Muon (lr=0.5, mom=0.99)", Muon, dict(lr=0.5, momentum=0.99, rank_q=rq, rank_v=rv)),
    ]

    results = []
    for name, cls, kwargs in optimizers:
        print(f"\nBenchmarking {name}...")
        result = run_optimizer_benchmark(
            optimizer_name=name,
            optimizer_cls=cls,
            optimizer_kwargs=kwargs,
            state=state,
            lora_config=lora_cfg,
            labels=labels,
            num_params=cfg.num_params,
            n_runs=args.runs,
            max_steps=args.steps,
            global_min_loss=args.global_min,
        )

        # Compute landscape percentile of median best loss
        if landscape is not None:
            result.best_loss_landscape_percentile = compute_landscape_percentile(
                result.median_best_loss, landscape
            )

        print(result.summary())
        results.append(result)

    # Summary comparison
    print("\n" + "=" * 90)
    print(f"{'Optimizer':<30s} {'Success%':>8s} {'MedBest':>8s} {'Landscape%':>10s} {'AvgSteps':>9s}")
    print("-" * 90)
    for r in results:
        print(f"{r.optimizer_name:<30s} {r.success_rate:>7.2%} {r.median_best_loss:>8.4f} "
              f"{r.best_loss_landscape_percentile:>9.4f}% {r.steps_to_best.mean():>9.1f}")

    # Save results
    out = {
        "config": {
            "num_params": cfg.num_params,
            "num_levels": cfg.num_levels,
            "lora_alpha": cfg.lora_alpha,
            "n_runs": args.runs,
            "max_steps": args.steps,
            "global_min_loss": args.global_min,
        },
        "results": [],
    }
    for r in results:
        out["results"].append({
            "name": r.optimizer_name,
            "success_rate": r.success_rate,
            "mean_best_loss": r.mean_best_loss,
            "median_best_loss": r.median_best_loss,
            "landscape_percentile": r.best_loss_landscape_percentile,
            "mean_steps_to_best": float(r.steps_to_best.mean()),
            "best_loss_percentiles": {
                str(p): float(v) for p, v in
                zip([1, 5, 10, 25, 50, 75, 90, 95, 99],
                    list(map(float, np.percentile(r.best_losses, [1, 5, 10, 25, 50, 75, 90, 95, 99]))))
            },
        })

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
