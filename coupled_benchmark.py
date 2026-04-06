"""Optimizer benchmarking against coupled MLP adapter ground-truth landscape.

Uses STE for gradient-based optimizers. Also includes combinatorial search baselines.
"""

import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from coupled_config import CoupledConfig
from coupled_adapter import AdapterConfig, config_indices_to_binary
from coupled_data import load_eval_sequences
from coupled_enumeration import run_layers_batched as run_layers, run_layers_single


# ---- STE for binary quantization ----

class STEBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_binary(x):
    return STEBinary.apply(x)


# ---- Differentiable forward for coupled adapters ----

def differentiable_coupled_forward(
    model, adapters: list[AdapterConfig], input_ids: torch.Tensor,
    labels: torch.Tensor, latent: torch.Tensor, cfg,
) -> torch.Tensor:
    """Forward pass with STE quantization for all 3 adapters.

    Args:
        latent: [N_runs, 24] continuous params

    Returns:
        [N_runs] losses with grad
    """
    N = latent.shape[0]
    n_per = cfg.bits_per_adapter
    adapter_layers = cfg.adapter_layers
    dtype = cfg.dtype

    # Quantize via STE
    quantized = ste_binary(latent).to(dtype)

    # Flatten sequences
    N_seq, seq_per = input_ids.shape
    total_seq = N_seq * seq_per
    flat_ids = input_ids.reshape(1, -1)
    flat_labels = labels.reshape(-1)

    # Embeddings (shared across runs)
    with torch.no_grad():
        embeds = model.transformer.wte(flat_ids) + model.transformer.wpe(
            torch.arange(total_seq, device=input_ids.device).unsqueeze(0))
        embeds = model.transformer.drop(embeds).squeeze(0)  # [seq, d]

    h = embeds.unsqueeze(0).expand(N, -1, -1)  # [N, seq, d]

    # Run through model with adapters at each insertion point
    prev_layer = 0
    for adapter_idx, layer_idx in enumerate(adapter_layers):
        # Frozen layers before this adapter
        with torch.no_grad():
            h = run_layers(model, h, prev_layer, layer_idx)

        # Apply adapter with STE-quantized weights
        bits = quantized[:, adapter_idx * n_per:(adapter_idx + 1) * n_per]
        W1, W2 = adapters[adapter_idx].build_weights(bits)
        # MLP adapter
        bottleneck = torch.bmm(h, W1.transpose(1, 2))
        bottleneck = F.relu(bottleneck)
        out = torch.bmm(bottleneck, W2.transpose(1, 2))
        h = h + adapters[adapter_idx].alpha * out

        prev_layer = layer_idx

    # Run remaining layers (adapter_layers[-1] through end)
    with torch.no_grad():
        h = run_layers(model, h, adapter_layers[-1], adapter_layers[-1] + 1)
        # Final LN + LM head
        h = model.transformer.ln_f(h)

    # LM head with grad (through adapter params only)
    logits = h @ model.lm_head.weight.T  # [N, seq, vocab]
    chunk_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        flat_labels.unsqueeze(0).expand(N, -1).reshape(-1),
        reduction="none",
    )
    losses = chunk_losses.reshape(N, -1).mean(dim=1)
    return losses


# ---- MUON optimizer (adapted for flat 24-dim) ----

class MuonCoupled(torch.optim.Optimizer):
    """MUON with 24-dim params split into 3 adapter groups of 8."""

    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @staticmethod
    def _batch_newton_schulz(G, steps=5):
        a, b, c = (3.4445, -4.7750, 2.0315)
        norms = G.norm(dim=(1, 2), keepdim=True).clamp(min=1e-7)
        X = G / norms
        if min(G.shape[1], G.shape[2]) < 2:
            return X
        for _ in range(steps):
            A = X @ X.transpose(-1, -2)
            B = b * A + c * A @ A
            X = a * X + B @ X
        return X

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)

                if buf.dim() == 2 and buf.shape[1] == 24:
                    # Split into 3 groups of 8, reshape each to 2x4 for NS
                    N = buf.shape[0]
                    parts = []
                    for i in range(3):
                        chunk = buf[:, i*8:(i+1)*8].reshape(N, 2, 4)
                        orth = self._batch_newton_schulz(chunk, ns_steps)
                        parts.append(orth.reshape(N, 8))
                    update = torch.cat(parts, dim=1)
                else:
                    update = buf / (buf.norm() + 1e-8)

                p.add_(update, alpha=-lr)


# ---- Random search baseline ----

def random_search(losses: np.ndarray, num_params: int, n_runs: int,
                  budget: int, rng: np.random.RandomState) -> dict:
    """Random search: sample `budget` configs, return best."""
    total = len(losses)
    best_losses = np.full(n_runs, np.inf)
    steps_to_best = np.zeros(n_runs, dtype=np.int32)

    for step in range(budget):
        indices = rng.randint(0, total, n_runs)
        sampled = losses[indices]
        improved = sampled < best_losses
        best_losses[improved] = sampled[improved]
        steps_to_best[improved] = step

    return {
        "best_losses": best_losses,
        "steps_to_best": steps_to_best,
    }


# ---- Greedy local search (hill climbing) ----

def greedy_local_search(losses: np.ndarray, num_params: int, n_runs: int,
                        budget: int, rng: np.random.RandomState) -> dict:
    """Greedy hill climbing: start random, flip best-improving bit each step."""
    total = len(losses)
    current = rng.randint(0, total, n_runs)
    current_losses = losses[current].copy()
    best_losses = current_losses.copy()
    steps_to_best = np.zeros(n_runs, dtype=np.int32)

    for step in range(budget):
        # Try all bit flips, pick best
        best_neighbor_loss = np.full(n_runs, np.inf)
        best_neighbor_idx = current.copy()

        for k in range(num_params):
            neighbor = current ^ (1 << k)
            neighbor_loss = losses[neighbor]
            improved = neighbor_loss < best_neighbor_loss
            best_neighbor_loss[improved] = neighbor_loss[improved]
            best_neighbor_idx[improved] = neighbor[improved]

        # Move to best neighbor if it improves
        move = best_neighbor_loss < current_losses
        current[move] = best_neighbor_idx[move]
        current_losses[move] = best_neighbor_loss[move]

        improved = current_losses < best_losses
        best_losses[improved] = current_losses[improved]
        steps_to_best[improved] = step

        # If stuck, restart with small probability
        stuck = ~move
        restart = stuck & (rng.random(n_runs) < 0.1)
        new_starts = rng.randint(0, total, n_runs)
        current[restart] = new_starts[restart]
        current_losses[restart] = losses[current[restart]]

    return {
        "best_losses": best_losses,
        "steps_to_best": steps_to_best,
    }


# ---- Main benchmark ----

def run_gradient_benchmark(
    name: str, optimizer_cls, optimizer_kwargs: dict,
    model, adapters, input_ids, labels, cfg,
    n_runs: int = 5000, max_steps: int = 50,
    global_min_loss: float = 0.0, chunk_size: int = 512,
) -> dict:
    """Run gradient-based optimizer benchmark."""
    device = input_ids.device
    num_params = cfg.num_params

    all_best = []
    all_steps = []

    for cs in range(0, n_runs, chunk_size):
        ce = min(cs + chunk_size, n_runs)
        sz = ce - cs

        latent = torch.nn.Parameter(
            torch.empty(sz, num_params, device=device, dtype=torch.float32).uniform_(-1.5, 1.5)
        )
        optimizer = optimizer_cls([latent], **optimizer_kwargs)

        best_losses = torch.full((sz,), float("inf"), device=device)
        steps_to_best = torch.zeros(sz, device=device, dtype=torch.long)

        for step in range(max_steps):
            optimizer.zero_grad()
            losses = differentiable_coupled_forward(
                model, adapters, input_ids, labels, latent, cfg
            )
            losses.sum().backward()
            optimizer.step()

            with torch.no_grad():
                improved = losses < best_losses
                best_losses = torch.where(improved, losses, best_losses)
                steps_to_best = torch.where(improved, step, steps_to_best)

        all_best.append(best_losses.detach().cpu().numpy())
        all_steps.append(steps_to_best.cpu().numpy())
        print(f"  {name}: {ce}/{n_runs}, chunk best={best_losses.min():.4f}, "
              f"median={best_losses.median():.4f}")

    best = np.concatenate(all_best)
    steps = np.concatenate(all_steps)
    success = float(np.mean(np.isclose(best, global_min_loss, atol=0.01)))

    return {
        "name": name,
        "best_losses": best,
        "steps_to_best": steps,
        "success_rate": success,
        "median_best": float(np.median(best)),
        "mean_best": float(np.mean(best)),
        "mean_steps": float(steps.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", type=str, default="coupled_benchmark.json")
    args = parser.parse_args()

    cfg = CoupledConfig()
    cfg.output_dir = args.input_dir
    device = torch.device(f"cuda:{args.gpu}")

    # Load landscape for ground truth + combinatorial methods
    print("Loading landscape...")
    from coupled_diagnostics import load_landscape
    losses = load_landscape(args.input_dir, args.num_gpus, cfg.total_configs)
    global_min = float(losses.min())
    global_min_idx = int(np.argmin(losses))
    print(f"  Global min: {global_min:.4f} at index {global_min_idx}")

    # Load model
    print("Loading model...")
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(cfg.model_name, torch_dtype=cfg.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)

    input_ids, labels_data = load_eval_sequences(
        cfg.model_name, cfg.num_sequences, cfg.seq_len
    )
    input_ids = input_ids.to(device)
    labels_data = labels_data.to(device)

    adapters = []
    for i, layer_idx in enumerate(cfg.adapter_layers):
        adapters.append(AdapterConfig(
            d_model=768, width=cfg.bottleneck_width, n_free=cfg.bits_per_adapter,
            alpha=cfg.adapter_alpha, seed=cfg.adapter_seed + i * 1000,
            device=device, dtype=cfg.dtype,
        ))

    all_results = {"global_min": global_min, "total_configs": cfg.total_configs, "results": []}

    # ---- Combinatorial baselines (run on landscape array) ----
    rng = np.random.RandomState(42)

    print("\nRandom Search (50 evals)...")
    rs = random_search(losses, cfg.num_params, args.runs, 50, rng)
    rs_success = float(np.mean(np.isclose(rs["best_losses"], global_min, atol=0.01)))
    rs_result = {
        "name": "Random Search (50 evals)", "success_rate": rs_success,
        "median_best": float(np.median(rs["best_losses"])),
        "mean_steps": float(rs["steps_to_best"].mean()),
    }
    print(f"  Success: {rs_success:.2%}, Median: {np.median(rs['best_losses']):.4f}")
    all_results["results"].append(rs_result)

    print("\nGreedy Local Search (50 steps)...")
    gls = greedy_local_search(losses, cfg.num_params, min(args.runs, 5000), 50, rng)
    gls_success = float(np.mean(np.isclose(gls["best_losses"], global_min, atol=0.01)))
    gls_result = {
        "name": "Greedy Local Search (50 steps)", "success_rate": gls_success,
        "median_best": float(np.median(gls["best_losses"])),
        "mean_steps": float(gls["steps_to_best"].mean()),
    }
    print(f"  Success: {gls_success:.2%}, Median: {np.median(gls['best_losses']):.4f}")
    all_results["results"].append(gls_result)

    # ---- Gradient-based optimizers ----
    gradient_configs = [
        ("SGD (lr=0.5, mom=0.9)", torch.optim.SGD, dict(lr=0.5, momentum=0.9)),
        ("SGD (lr=2.0, mom=0.9)", torch.optim.SGD, dict(lr=2.0, momentum=0.9)),
        ("SGD (lr=10, mom=0.9)", torch.optim.SGD, dict(lr=10, momentum=0.9)),
        ("AdamW (lr=0.05)", torch.optim.AdamW, dict(lr=0.05)),
        ("AdamW (lr=0.5)", torch.optim.AdamW, dict(lr=0.5)),
        ("AdamW (lr=5.0)", torch.optim.AdamW, dict(lr=5.0)),
        ("AdamW (lr=50)", torch.optim.AdamW, dict(lr=50)),
        ("Muon (lr=0.1)", MuonCoupled, dict(lr=0.1, momentum=0.95)),
        ("Muon (lr=1.0)", MuonCoupled, dict(lr=1.0, momentum=0.95)),
        ("Muon (lr=10)", MuonCoupled, dict(lr=10, momentum=0.95)),
    ]

    for name, cls, kwargs in gradient_configs:
        print(f"\n{name}...")
        t0 = time.time()
        result = run_gradient_benchmark(
            name, cls, kwargs, model, adapters, input_ids, labels_data, cfg,
            n_runs=args.runs, max_steps=args.steps, global_min_loss=global_min,
        )
        elapsed = time.time() - t0
        result["wall_time"] = elapsed
        print(f"  Success: {result['success_rate']:.2%}, "
              f"Median: {result['median_best']:.4f}, "
              f"Steps: {result['mean_steps']:.1f}, "
              f"Time: {elapsed:.1f}s")
        # Don't serialize numpy arrays
        result_clean = {k: v for k, v in result.items()
                        if not isinstance(v, np.ndarray)}
        all_results["results"].append(result_clean)

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Optimizer':<35s} {'Success%':>8s} {'MedBest':>8s} {'AvgSteps':>9s}")
    print("-" * 90)
    for r in all_results["results"]:
        print(f"{r['name']:<35s} {r['success_rate']:>7.2%} "
              f"{r['median_best']:>8.4f} {r['mean_steps']:>9.1f}")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
