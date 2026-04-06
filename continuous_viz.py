"""Generate 2D slice visualizations of the 6D continuous landscape.

Extracts slices through the global minimum along pairs of dimensions,
outputs heatmap data for the Three.js visualizer.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from itertools import combinations


def load_landscape(output_dir: str, num_gpus: int, total_configs: int) -> np.ndarray:
    shard_size = total_configs // num_gpus
    shards = []
    for i in range(num_gpus):
        path = Path(output_dir) / f"losses_shard_{i}.npy"
        shard = np.memmap(path, dtype=np.float16, mode="r", shape=(shard_size,))
        shards.append(shard)
    return shards, shard_size


def global_index_to_grid_indices(index: int, n_grid: int, n_adapters: int, n_per: int):
    """Decompose flat index into per-dimension grid indices."""
    cpa = n_grid ** n_per
    indices = []
    remaining = index
    for _ in range(n_adapters):
        adapter_idx = remaining % cpa
        remaining //= cpa
        for _ in range(n_per):
            indices.append(adapter_idx % n_grid)
            adapter_idx //= n_grid
    return indices


def grid_indices_to_global(indices: list, n_grid: int, n_per: int):
    """Convert per-dimension grid indices back to flat global index."""
    n_adapters = len(indices) // n_per
    global_idx = 0
    cpa = n_grid ** n_per
    multiplier = 1
    for adapter in range(n_adapters):
        adapter_idx = 0
        dim_mult = 1
        for k in range(n_per):
            adapter_idx += indices[adapter * n_per + k] * dim_mult
            dim_mult *= n_grid
        global_idx += adapter_idx * multiplier
        multiplier *= cpa
    return global_idx


def extract_2d_slice(shards, shard_size, min_indices, dim_a, dim_b, n_grid, n_per):
    """Extract a 2D slice varying dim_a and dim_b, fixing all others at minimum."""
    n_adapters = len(min_indices) // n_per
    slice_data = np.zeros((n_grid, n_grid), dtype=np.float32)

    for ia in range(n_grid):
        for ib in range(n_grid):
            indices = list(min_indices)
            indices[dim_a] = ia
            indices[dim_b] = ib
            global_idx = grid_indices_to_global(indices, n_grid, n_per)
            shard = global_idx // shard_size
            offset = global_idx % shard_size
            slice_data[ia, ib] = float(shards[shard][offset])

    return slice_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--n-grid", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="viz_continuous")
    args = parser.parse_args()

    n_grid = args.n_grid
    n_per = 2
    n_adapters = 3
    n_params = n_adapters * n_per
    total_configs = (n_grid ** n_per) ** n_adapters

    print(f"Loading landscape ({total_configs:,} configs)...")
    shards, shard_size = load_landscape(args.input_dir, args.num_gpus, total_configs)

    # Find global minimum
    print("Finding global minimum...")
    min_loss = float("inf")
    min_idx = 0
    for s_id, shard in enumerate(shards):
        chunk = np.array(shard, dtype=np.float32)
        local_min_idx = int(np.argmin(chunk))
        local_min = float(chunk[local_min_idx])
        if local_min < min_loss:
            min_loss = local_min
            min_idx = s_id * shard_size + local_min_idx

    min_grid_indices = global_index_to_grid_indices(min_idx, n_grid, n_adapters, n_per)
    grid_values = np.linspace(-1, 1, n_grid)
    min_params = [float(grid_values[i]) for i in min_grid_indices]

    print(f"  Global min: {min_loss:.4f}")
    print(f"  Grid indices: {min_grid_indices}")
    print(f"  Params: {[f'{p:.3f}' for p in min_params]}")

    # Extract 2D slices for all 15 dimension pairs
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dim_names = ["L2_W1", "L2_W2", "L6_W1", "L6_W2", "L11_W1", "L11_W2"]
    slices_meta = []

    for dim_a, dim_b in combinations(range(n_params), 2):
        name = f"{dim_names[dim_a]}_vs_{dim_names[dim_b]}"
        print(f"  Extracting slice: {name}...")
        slice_data = extract_2d_slice(shards, shard_size, min_grid_indices,
                                      dim_a, dim_b, n_grid, n_per)

        # Save as binary float32
        slice_path = output / f"slice_{dim_a}_{dim_b}.bin"
        slice_data.astype(np.float32).tofile(str(slice_path))

        slices_meta.append({
            "dim_a": dim_a,
            "dim_b": dim_b,
            "name_a": dim_names[dim_a],
            "name_b": dim_names[dim_b],
            "file": f"slice_{dim_a}_{dim_b}.bin",
            "loss_min": float(slice_data.min()),
            "loss_max": float(slice_data.max()),
            "loss_range": float(slice_data.max() - slice_data.min()),
        })

    # Also save a 3D heightmap of the most interesting slice (widest loss range)
    best_slice = max(slices_meta, key=lambda s: s["loss_range"])
    print(f"\n  Most interesting slice: {best_slice['name_a']} vs {best_slice['name_b']}")
    print(f"    Loss range: {best_slice['loss_range']:.4f}")

    # Save metadata
    metadata = {
        "n_grid": n_grid,
        "n_params": n_params,
        "param_range": [-1.0, 1.0],
        "global_min_loss": min_loss,
        "global_min_params": min_params,
        "dim_names": dim_names,
        "slices": slices_meta,
        "best_slice": best_slice,
    }

    with open(output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Generate the heightmap for Three.js
    best_data = np.fromfile(str(output / best_slice["file"]), dtype=np.float32).reshape(n_grid, n_grid)
    # Normalize to [0, 1]
    normalized = (best_data - best_data.min()) / (best_data.max() - best_data.min())
    normalized.astype(np.float32).tofile(str(output / "heightmap.bin"))

    print(f"\nSaved {len(slices_meta)} slices + metadata to {output}/")
    print(f"Heightmap: {output}/heightmap.bin ({n_grid}×{n_grid} float32)")


if __name__ == "__main__":
    main()
