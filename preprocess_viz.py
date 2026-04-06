"""Preprocess the 2^30 loss landscape into a 2D heightmap for Three.js visualization.

Key insight: we reorder bits by their marginal effect on loss before mapping
to 2D. This breaks the artificial periodicity from arbitrary bit ordering.
Strongest-effect bits become high-order bits, interleaved across x and y axes.
"""

import argparse
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path


def compute_bit_influence(all_losses, num_params, sample_size=1_000_000):
    """Compute each bit's marginal effect on loss.

    For each bit k, compare mean loss when bit k = +1 vs bit k = -1.
    """
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(all_losses), size=sample_size)
    losses = all_losses[indices]

    influence = np.zeros(num_params)
    for k in range(num_params):
        mask = (indices >> k) & 1  # 0 or 1
        mean_on = losses[mask == 1].mean()
        mean_off = losses[mask == 0].mean()
        influence[k] = abs(mean_on - mean_off)

    return influence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="viz/heightmap.bin")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--smooth", type=float, default=4.0, help="Gaussian sigma for smoothing")
    args = parser.parse_args()

    num_params = 30
    total = 2 ** num_params
    shard_size = total // 8
    half_params = num_params // 2  # 15 bits per axis

    print("Loading shards...")
    shards = []
    for i in range(8):
        path = Path(args.input_dir) / f"losses_shard_{i}.npy"
        shard = np.memmap(path, dtype=np.float16, mode="r", shape=(shard_size,))
        shards.append(shard)

    all_losses = np.concatenate(shards).astype(np.float32)
    print(f"Loaded {len(all_losses):,} losses, range [{all_losses.min():.3f}, {all_losses.max():.3f}]")

    # Step 1: Compute bit influence and reorder
    print("Computing bit influence...")
    influence = compute_bit_influence(all_losses, num_params)
    bit_order = np.argsort(-influence)  # strongest first
    print("Bit influence (sorted):")
    for i, b in enumerate(bit_order):
        axis = "x" if i % 2 == 0 else "y"
        proj = "Q" if b < 15 else "V"
        print(f"  bit {b:>2d} ({proj}[{b % 15:>2d}]) influence={influence[b]:.4f} -> {axis}-axis bit {i // 2}")

    # Interleave: even-ranked bits -> x-axis, odd-ranked bits -> y-axis
    x_bits = bit_order[0::2]  # 15 bits for x
    y_bits = bit_order[1::2]  # 15 bits for y

    # Step 2: Build remapping table
    # For a given (gx, gy) grid coordinate, we need the config index.
    # gx -> Gray code -> 15-bit binary -> assign to x_bits positions
    # gy -> Gray code -> 15-bit binary -> assign to y_bits positions

    full_res = 2 ** half_params  # 32768
    res = args.resolution
    block = full_res // res

    print(f"Building {res}x{res} heightmap (block-averaging {block}x{block} patches)...")
    heightmap = np.zeros((res, res), dtype=np.float64)

    for out_y in range(res):
        if out_y % 100 == 0:
            print(f"  row {out_y}/{res}")

        for gy_offset in range(block):
            gy = out_y * block + gy_offset
            # Gray code -> binary for y-axis bits
            gy_gray = gy ^ (gy >> 1)

            # Extract individual bits from gy_gray and place them at y_bits positions
            y_contrib = np.int64(0)
            for bit_pos in range(half_params):
                if (gy_gray >> bit_pos) & 1:
                    y_contrib |= np.int64(1) << int(y_bits[bit_pos])

            # Process x in bulk
            gx_all = np.arange(full_res, dtype=np.int64)
            gx_gray = gx_all ^ (gx_all >> 1)

            # Build config indices: place x bits at x_bits positions, OR with y_contrib
            config_indices = np.zeros(full_res, dtype=np.int64)
            for bit_pos in range(half_params):
                bit_vals = (gx_gray >> bit_pos) & 1
                config_indices |= bit_vals << int(x_bits[bit_pos])
            config_indices |= y_contrib

            losses = all_losses[config_indices]

            # Block-average into output columns
            losses_reshaped = losses.reshape(res, block)
            heightmap[out_y] += losses_reshaped.mean(axis=1)

        heightmap[out_y] /= block

    # Gaussian smooth
    if args.smooth > 0:
        print(f"Smoothing with sigma={args.smooth}")
        heightmap = gaussian_filter(heightmap, sigma=args.smooth)

    # Normalize to [0, 1]
    lo, hi = heightmap.min(), heightmap.max()
    heightmap_norm = ((heightmap - lo) / (hi - lo)).astype(np.float32)
    print(f"Heightmap range: [{lo:.3f}, {hi:.3f}]")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    heightmap_norm.tofile(out_path)
    print(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    meta = {
        "resolution": res,
        "loss_min": float(lo),
        "loss_max": float(hi),
        "loss_mean": float(heightmap.mean()),
        "loss_std": float(heightmap.std()),
        "total_configs": int(total),
        "num_params": num_params,
        "global_loss_min": float(all_losses.min()),
        "global_loss_max": float(all_losses.max()),
        "x_bits": x_bits.tolist(),
        "y_bits": y_bits.tolist(),
    }
    meta_path = out_path.parent / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
