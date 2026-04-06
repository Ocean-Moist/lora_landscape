"""Interaction diagnostics for the coupled landscape.

Verifies that the landscape is genuinely non-additive before optimizer benchmarking.
Metrics: singleton R², pairwise interactions, gradient-sign predictability, local minima count.
"""

import argparse
import json
import numpy as np
from pathlib import Path


def load_landscape(output_dir: str, num_gpus: int, total_configs: int) -> np.ndarray:
    """Load all shards into a single float32 array."""
    shard_size = total_configs // num_gpus
    shards = []
    for i in range(num_gpus):
        path = Path(output_dir) / f"losses_shard_{i}.npy"
        shard = np.memmap(path, dtype=np.float16, mode="r", shape=(shard_size,))
        shards.append(np.array(shard, dtype=np.float32))
    return np.concatenate(shards)


def config_index_to_bits(index: int, num_params: int) -> np.ndarray:
    """Convert config index to binary {-1, +1} vector."""
    bits = np.zeros(num_params, dtype=np.float32)
    for k in range(num_params):
        bits[k] = ((index >> k) & 1) * 2 - 1
    return bits


def bits_to_index(bits: np.ndarray) -> int:
    """Convert {-1, +1} vector to config index."""
    idx = 0
    for k in range(len(bits)):
        if bits[k] > 0:
            idx |= (1 << k)
    return idx


def singleton_variance_explained(losses: np.ndarray, num_params: int) -> tuple[float, np.ndarray]:
    """Fit additive model and compute R².

    For each bit, compute mean loss when bit=+1 vs bit=-1.
    The additive prediction is: mean + sum(effect_i * bit_i).

    Returns (R², per-bit effects).
    """
    total = len(losses)
    global_mean = losses.mean()
    effects = np.zeros(num_params, dtype=np.float64)

    for k in range(num_params):
        # bit k is +1 when index has bit k set
        mask = np.arange(total) & (1 << k)
        plus_mean = losses[mask > 0].mean()
        minus_mean = losses[mask == 0].mean()
        effects[k] = (plus_mean - minus_mean) / 2

    # Compute additive predictions for a sample
    rng = np.random.RandomState(42)
    sample_size = min(1_000_000, total)
    sample_idx = rng.choice(total, sample_size, replace=False)

    predictions = np.full(sample_size, global_mean, dtype=np.float64)
    for k in range(num_params):
        bit_vals = ((sample_idx >> k) & 1).astype(np.float64) * 2 - 1
        predictions += effects[k] * bit_vals

    actual = losses[sample_idx].astype(np.float64)
    ss_res = np.sum((actual - predictions) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return r_squared, effects


def pairwise_interactions(losses: np.ndarray, num_params: int,
                          n_samples: int = 500_000) -> np.ndarray:
    """Estimate pairwise interaction strengths.

    I_ij = E[f] - E[f|flip_i] - E[f|flip_j] + E[f|flip_ij]
    Estimated by sampling random configs.

    Returns: [num_params, num_params] interaction matrix.
    """
    rng = np.random.RandomState(42)
    sample_idx = rng.randint(0, len(losses), n_samples)
    f_x = losses[sample_idx].astype(np.float64)

    interactions = np.zeros((num_params, num_params), dtype=np.float64)

    for i in range(num_params):
        flip_i = sample_idx ^ (1 << i)
        f_i = losses[flip_i].astype(np.float64)

        for j in range(i + 1, num_params):
            flip_j = sample_idx ^ (1 << j)
            flip_ij = sample_idx ^ (1 << i) ^ (1 << j)
            f_j = losses[flip_j].astype(np.float64)
            f_ij = losses[flip_ij].astype(np.float64)

            # Interaction: f(x) - f(flip_i) - f(flip_j) + f(flip_ij)
            interaction = np.mean(np.abs(f_x - f_i - f_j + f_ij))
            interactions[i, j] = interaction
            interactions[j, i] = interaction

    return interactions


def gradient_sign_predictability(losses: np.ndarray, num_params: int,
                                 n_samples: int = 10_000) -> dict:
    """From random starting points, predict optimum by gradient sign.

    Compute numerical gradient (via single bit flips), take sign,
    and measure Hamming distance to true optimum.
    """
    rng = np.random.RandomState(42)
    global_min_idx = int(np.argmin(losses))
    optimal_bits = config_index_to_bits(global_min_idx, num_params)

    sample_idx = rng.randint(0, len(losses), n_samples)
    hamming_distances = np.zeros(n_samples, dtype=np.int32)

    for s in range(n_samples):
        idx = sample_idx[s]
        current_loss = losses[idx]
        predicted_bits = config_index_to_bits(idx, num_params)

        # "Gradient" = loss decrease from flipping each bit
        for k in range(num_params):
            neighbor_idx = idx ^ (1 << k)
            neighbor_loss = losses[neighbor_idx]
            grad = neighbor_loss - current_loss
            # If flipping decreases loss, flip
            if grad < 0:
                predicted_bits[k] *= -1

        hamming = int(np.sum(predicted_bits != optimal_bits))
        hamming_distances[s] = hamming

    return {
        "mean_hamming": float(hamming_distances.mean()),
        "median_hamming": float(np.median(hamming_distances)),
        "pct_exact": float(np.mean(hamming_distances == 0)),
        "pct_within_1": float(np.mean(hamming_distances <= 1)),
        "pct_within_3": float(np.mean(hamming_distances <= 3)),
        "histogram": {int(k): int(v) for k, v in
                      zip(*np.unique(hamming_distances, return_counts=True))},
    }


def count_local_minima(losses: np.ndarray, num_params: int,
                       n_samples: int = 2_000_000) -> dict:
    """Count configs that are local minima (no single bit flip improves loss).

    Samples random configs and checks all neighbors.
    Extrapolates count from sample rate.
    """
    total = len(losses)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(total, min(n_samples, total), replace=False)

    local_min_count = 0
    local_min_losses = []

    for idx in sample_idx:
        current_loss = losses[idx]
        is_local_min = True
        for k in range(num_params):
            neighbor = idx ^ (1 << k)
            if losses[neighbor] < current_loss:
                is_local_min = False
                break
        if is_local_min:
            local_min_count += 1
            local_min_losses.append(float(current_loss))

    sample_rate = len(sample_idx) / total
    estimated_total = local_min_count / sample_rate if sample_rate > 0 else 0

    return {
        "found_in_sample": local_min_count,
        "sample_size": len(sample_idx),
        "estimated_total": int(estimated_total),
        "local_min_losses": sorted(local_min_losses)[:50],  # top 50 best
    }


def main():
    parser = argparse.ArgumentParser(description="Coupled landscape interaction diagnostics")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--num-params", type=int, default=24)
    parser.add_argument("--total-configs", type=int, default=2**24)
    parser.add_argument("--output", type=str, default="diagnostics.json")
    args = parser.parse_args()

    print(f"Loading landscape from {args.input_dir}...")
    losses = load_landscape(args.input_dir, args.num_gpus, args.total_configs)
    print(f"  Loaded {len(losses):,} configs")
    print(f"  Loss range: [{losses.min():.4f}, {losses.max():.4f}]")
    print(f"  Loss spread: {losses.max() - losses.min():.4f}")
    print(f"  Global min at index {np.argmin(losses)}: {losses.min():.4f}")

    results = {
        "landscape_stats": {
            "total_configs": len(losses),
            "loss_min": float(losses.min()),
            "loss_max": float(losses.max()),
            "loss_mean": float(losses.mean()),
            "loss_std": float(losses.std()),
            "loss_spread": float(losses.max() - losses.min()),
            "global_min_index": int(np.argmin(losses)),
        }
    }

    # 1. Singleton variance explained
    print("\n1. Singleton variance explained (additive R²)...")
    r2, effects = singleton_variance_explained(losses, args.num_params)
    print(f"  R² = {r2:.4f}")
    print(f"  Per-bit effects (sorted): {sorted(np.abs(effects))[::-1][:10]}")
    results["singleton_r2"] = r2
    results["singleton_effects"] = {
        "values": [float(e) for e in effects],
        "abs_sorted": [float(e) for e in sorted(np.abs(effects))[::-1]],
    }

    # 2. Pairwise interactions
    print("\n2. Pairwise interaction strengths...")
    interactions = pairwise_interactions(losses, args.num_params)
    median_singleton = float(np.median(np.abs(effects)))
    strong_pairs = np.sum(interactions > median_singleton)
    print(f"  Median singleton effect: {median_singleton:.6f}")
    print(f"  Pairs stronger than median singleton: {strong_pairs}")
    print(f"  Max pairwise: {interactions.max():.6f}")
    print(f"  Mean pairwise: {interactions[interactions > 0].mean():.6f}")
    results["pairwise"] = {
        "max_interaction": float(interactions.max()),
        "mean_interaction": float(interactions[interactions > 0].mean()),
        "pairs_above_median_singleton": int(strong_pairs),
        "top_10_pairs": [],
    }
    # Find top 10 pairs
    flat_idx = np.argsort(interactions.ravel())[::-1][:10]
    for fi in flat_idx:
        i, j = divmod(fi, args.num_params)
        if i < j:
            results["pairwise"]["top_10_pairs"].append({
                "bits": [int(i), int(j)],
                "strength": float(interactions[i, j]),
            })

    # 3. Gradient-sign predictability
    print("\n3. Gradient-sign predictability...")
    grad_results = gradient_sign_predictability(losses, args.num_params)
    print(f"  Mean Hamming to optimum: {grad_results['mean_hamming']:.2f}")
    print(f"  Median Hamming: {grad_results['median_hamming']:.1f}")
    print(f"  Exact match: {grad_results['pct_exact']:.2%}")
    print(f"  Within 3 flips: {grad_results['pct_within_3']:.2%}")
    results["gradient_predictability"] = grad_results

    # 4. Local minima count
    print("\n4. Local minima count...")
    minima = count_local_minima(losses, args.num_params)
    print(f"  Found {minima['found_in_sample']} in {minima['sample_size']:,} samples")
    print(f"  Estimated total: {minima['estimated_total']:,}")
    results["local_minima"] = minima

    # Verdict
    print("\n" + "=" * 60)
    print("DIAGNOSTIC VERDICT:")
    issues = []
    if r2 > 0.95:
        issues.append(f"R²={r2:.3f} (nearly additive — need <0.7)")
    elif r2 > 0.7:
        issues.append(f"R²={r2:.3f} (weakly non-additive — acceptable)")
    else:
        print(f"  ✓ R²={r2:.3f} — landscape has strong interactions")

    if grad_results["median_hamming"] < 3:
        issues.append(f"Median Hamming={grad_results['median_hamming']:.1f} (too predictable)")
    else:
        print(f"  ✓ Median Hamming={grad_results['median_hamming']:.1f} — gradients don't trivially solve it")

    if minima["estimated_total"] < 100:
        issues.append(f"~{minima['estimated_total']} local minima (need >100)")
    else:
        print(f"  ✓ ~{minima['estimated_total']} local minima — landscape has many basins")

    if issues:
        print(f"  ISSUES: {'; '.join(issues)}")
        print(f"  Consider: increase alpha, width, or add more adapter layers")
    else:
        print(f"  ALL CHECKS PASSED — landscape is suitable for optimizer benchmarking")

    results["verdict"] = {
        "passes": len(issues) == 0,
        "issues": issues,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
