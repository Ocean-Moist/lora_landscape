"""Memory-mapped storage for the loss landscape.

Each GPU writes to its own shard. The full landscape is 2^40 float16 values (~2 TB).
"""

import json
import os
from pathlib import Path

import numpy as np


class LossStorage:
    """Write-side: one memmap shard per GPU."""

    def __init__(self, output_dir: str, gpu_id: int, total_configs: int, num_gpus: int):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.gpu_id = gpu_id
        self.shard_size = total_configs // num_gpus

        shard_path = self.dir / f"losses_shard_{gpu_id}.npy"
        if shard_path.exists():
            self.data = np.memmap(shard_path, dtype=np.float16, mode="r+", shape=(self.shard_size,))
        else:
            self.data = np.memmap(shard_path, dtype=np.float16, mode="w+", shape=(self.shard_size,))

        self.checkpoint_path = self.dir / f"checkpoint_{gpu_id}.json"

    def write_batch(self, offset: int, losses: np.ndarray):
        """Write a batch of losses at the given offset within this shard."""
        self.data[offset:offset + len(losses)] = losses
        if offset % (1024 * len(losses)) == 0:
            self.data.flush()

    def save_checkpoint(self, last_batch: int):
        with open(self.checkpoint_path, "w") as f:
            json.dump({"last_batch": last_batch, "gpu_id": self.gpu_id}, f)

    def load_checkpoint(self) -> int:
        """Returns the last completed batch index, or -1 if no checkpoint."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                return json.load(f)["last_batch"]
        return -1

    def flush(self):
        self.data.flush()


class LossLandscape:
    """Read-side: unified view over all shards."""

    def __init__(self, output_dir: str, num_gpus: int, total_configs: int):
        self.dir = Path(output_dir)
        self.num_gpus = num_gpus
        self.total_configs = total_configs
        self.shard_size = total_configs // num_gpus

        self.shards = []
        for i in range(num_gpus):
            path = self.dir / f"losses_shard_{i}.npy"
            self.shards.append(np.memmap(path, dtype=np.float16, mode="r", shape=(self.shard_size,)))

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            shard = index // self.shard_size
            offset = index % self.shard_size
            return self.shards[shard][offset]
        elif isinstance(index, np.ndarray):
            shards = index // self.shard_size
            offsets = index % self.shard_size
            result = np.empty(len(index), dtype=np.float16)
            for s in range(self.num_gpus):
                mask = shards == s
                if mask.any():
                    result[mask] = self.shards[s][offsets[mask]]
            return result
        raise TypeError(f"Unsupported index type: {type(index)}")

    def get_neighbor_loss(self, index: int, bit: int) -> np.float16:
        """Loss of the config reached by flipping one bit."""
        return self[index ^ (1 << bit)]

    def get_all_neighbor_losses(self, index: int, num_params: int = 40) -> np.ndarray:
        """Losses of all single-bit-flip neighbors."""
        neighbors = np.array([index ^ (1 << b) for b in range(num_params)], dtype=np.int64)
        return self[neighbors]

    def stream_chunks(self, chunk_size: int = 2**22):
        """Yield (start_index, chunk_array) over the full landscape."""
        for shard_id, shard in enumerate(self.shards):
            base = shard_id * self.shard_size
            for offset in range(0, self.shard_size, chunk_size):
                end = min(offset + chunk_size, self.shard_size)
                yield base + offset, shard[offset:end]
