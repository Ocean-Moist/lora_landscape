"""Run optimizer benchmarks against ground-truth ternary LoRA landscape."""

import argparse
import json
import torch
from pathlib import Path

from config import Config
from binary_lora import BinaryLoRAConfig
from model_setup import load_frozen_model, prepare_eval_data, PrecomputedState
from optimizers import run_optimizer_benchmark, Muon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10000, help="Random initializations per optimizer")
    parser.add_argument("--steps", type=int, default=200, help="Max optimizer steps per run")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--global-min", type=float, default=4.0469, help="Known global minimum loss")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(f"cuda:{args.gpu}")

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

    # Define optimizers to benchmark
    optimizers = [
        ("SGD (lr=0.1, mom=0.9)", torch.optim.SGD, dict(lr=0.1, momentum=0.9)),
        ("SGD (lr=0.5, mom=0.9)", torch.optim.SGD, dict(lr=0.5, momentum=0.9)),
        ("AdamW (lr=0.01)", torch.optim.AdamW, dict(lr=0.01)),
        ("AdamW (lr=0.05)", torch.optim.AdamW, dict(lr=0.05)),
        ("AdamW (lr=0.1)", torch.optim.AdamW, dict(lr=0.1)),
        ("Muon (lr=0.02)", Muon, dict(lr=0.02, momentum=0.95)),
        ("Muon (lr=0.05)", Muon, dict(lr=0.05, momentum=0.95)),
        ("Muon (lr=0.1)", Muon, dict(lr=0.1, momentum=0.95)),
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
        print(result.summary())
        results.append(result)

    # Summary comparison
    print("\n" + "=" * 70)
    print(f"{'Optimizer':<25s} {'Success%':>8s} {'MeanBest':>9s} {'MedBest':>9s} {'AvgSteps':>9s}")
    print("-" * 70)
    for r in results:
        print(f"{r.optimizer_name:<25s} {r.success_rate:>7.2%} {r.mean_best_loss:>9.4f} {r.median_best_loss:>9.4f} {r.steps_to_best.mean():>9.1f}")

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
            "mean_steps_to_best": float(r.steps_to_best.mean()),
            "best_loss_percentiles": {
                str(p): float(v) for p, v in
                zip([1, 5, 10, 25, 50, 75, 90, 95, 99],
                    list(map(float, __import__("numpy").percentile(r.best_losses, [1, 5, 10, 25, 50, 75, 90, 95, 99]))))
            },
        })

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
