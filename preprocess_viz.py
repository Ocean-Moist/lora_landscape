"""Preprocess the 2^30 loss landscape into a 2D heightmap for Three.js visualization.

Maps the 30-bit binary config space to a 2D grid using Gray code ordering,
so adjacent grid cells differ by exactly 1 bit (are Hamming neighbors).
Then downsamples and smooths for a clean terrain visualization.
"""

import argparse
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path


def gray_to_binary(g):
    """Convert Gray code to binary (works on numpy arrays)."""
    b = g.copy()
    mask = g >> 1
    while mask.any():
        b ^= mask
        mask >>= 1
    return b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="viz/heightmap.bin")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--smooth", type=float, default=2.0, help="Gaussian sigma for smoothing")
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

    # Build the 2D grid using Gray code ordering
    # Grid coordinate (gx, gy) maps to config index via Gray-to-binary conversion
    # This ensures adjacent grid cells are Hamming neighbors (differ by 1 bit)
    full_res = 2 ** half_params  # 32768
    res = args.resolution
    block = full_res // res

    print(f"Building {res}x{res} heightmap (block-averaging {block}x{block} patches)...")

    # Process in row chunks to manage memory
    # For each output row, we average over `block` rows of the full grid
    heightmap = np.zeros((res, res), dtype=np.float32)

    for out_y in range(res):
        if out_y % 100 == 0:
            print(f"  row {out_y}/{res}")
        # Gray-code rows in this output cell
        gy_start = out_y * block
        gy_end = gy_start + block

        row_accum = np.zeros(res, dtype=np.float64)
        for gy in range(gy_start, gy_end):
            # Convert Gray code row index to binary
            by = gy ^ (gy >> 1)  # binary-to-gray is g = b ^ (b >> 1), but we want gray-to-binary
            # Actually: gy IS the grid coordinate. We want the config bits.
            # Gray code: position gy in the grid corresponds to binary value where
            # successive positions differ by 1 bit.
            # Standard Gray code: gray(n) = n ^ (n >> 1)
            # So grid position gy corresponds to gray_code(gy) as the bit pattern
            # Actually we want: grid position gy -> bit pattern for the y-axis
            # Using standard Gray code: bit_y = gray(gy) = gy ^ (gy >> 1)
            bit_y = gy ^ (gy >> 1)

            # For each x block, compute the average loss
            for out_x_block in range(res):
                gx_start = out_x_block * block
                gx_end = gx_start + block
                # Convert all gx positions to bit patterns
                gx_range = np.arange(gx_start, gx_end, dtype=np.int64)
                bit_x = gx_range ^ (gx_range >> 1)  # Gray code
                # Full config index: bit_y in upper 15 bits, bit_x in lower 15 bits
                config_indices = (bit_y << half_params) | bit_x
                losses = all_losses[config_indices]
                row_accum[out_x_block] += losses.sum()

        heightmap[out_y] = row_accum / (block * block)

    # Gaussian smooth for the terrain look
    if args.smooth > 0:
        print(f"Smoothing with sigma={args.smooth}")
        heightmap = gaussian_filter(heightmap, sigma=args.smooth)

    # Normalize to [0, 1]
    lo, hi = heightmap.min(), heightmap.max()
    heightmap_norm = (heightmap - lo) / (hi - lo)
    print(f"Heightmap range: [{lo:.3f}, {hi:.3f}]")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    heightmap_norm.astype(np.float32).tofile(out_path)
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
    }
    meta_path = out_path.parent / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
