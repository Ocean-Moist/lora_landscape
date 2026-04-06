"""Grid enumeration for continuous MLP adapters with tree-structured caching.

Same tree structure as binary coupled enumeration, but grid points are
continuous values in [-1, 1] instead of binary {-1, +1}.

32^6 = 1,073,741,824 configs (~1.07B) — comparable to the ternary LoRA case.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm

from continuous_adapter import ContinuousAdapterConfig, grid_indices_to_params
from coupled_enumeration import run_layers_single, run_layers_batched, compute_loss_batched
from coupled_adapter import apply_adapter_batched


def enumerate_continuous(
    gpu_id: int,
    num_gpus: int,
    model,
    adapters: list[ContinuousAdapterConfig],
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    cfg,
    output_path: str,
):
    """Enumerate all grid configs using layer-wise caching."""
    device = input_ids.device
    adapter_layers = cfg.adapter_layers
    n_per = cfg.params_per_adapter
    cpa = cfg.configs_per_adapter  # 1024

    N_seq, seq_per = input_ids.shape
    flat_ids = input_ids.reshape(1, -1)
    flat_labels = labels.reshape(-1)
    total_seq = N_seq * seq_per

    # All grid configs for one adapter level
    all_grid_params = grid_indices_to_params(
        torch.arange(cpa, device=device),
        n_per, cfg.n_grid, cfg.param_min, cfg.param_max,
    ).to(cfg.dtype)  # [1024, 2]

    # Last-token-per-sequence positions
    last_positions = torch.arange(seq_per - 2, total_seq, seq_per, device=device)

    # Split outermost loop across GPUs
    l0_per_gpu = cpa // num_gpus
    l0_start = gpu_id * l0_per_gpu
    l0_end = l0_start + l0_per_gpu
    configs_this_gpu = l0_per_gpu * cpa * cpa

    os.makedirs(output_path, exist_ok=True)
    shard_path = f"{output_path}/losses_shard_{gpu_id}.npy"
    losses_out = np.memmap(shard_path, dtype=np.float16, mode="w+",
                           shape=(configs_this_gpu,))

    # Step 1: Layers 0 to adapter_layers[0]-1
    with torch.no_grad():
        pos_ids = torch.arange(total_seq, device=device).unsqueeze(0)
        embeds = model.transformer.wte(flat_ids) + model.transformer.wpe(pos_ids)
        embeds = model.transformer.drop(embeds).squeeze(0)
        H_pre0 = run_layers_single(model, embeds, 0, adapter_layers[0])

    write_idx = 0
    pbar = tqdm(range(l0_start, l0_end), desc=f"GPU {gpu_id} L{adapter_layers[0]}",
                total=l0_per_gpu)

    for i0 in pbar:
        with torch.no_grad():
            params0 = all_grid_params[i0:i0+1]
            W1_0, W2_0 = adapters[0].build_weights(params0)
            H0 = apply_adapter_batched(H_pre0, W1_0, W2_0, adapters[0].alpha)
            H0 = H0.squeeze(0)
            H_pre1 = run_layers_single(model, H0, adapter_layers[0], adapter_layers[1])

        for i1 in range(cpa):
            with torch.no_grad():
                params1 = all_grid_params[i1:i1+1]
                W1_1, W2_1 = adapters[1].build_weights(params1)
                H1 = apply_adapter_batched(H_pre1, W1_1, W2_1, adapters[1].alpha)
                H1 = H1.squeeze(0)
                H_pre2 = run_layers_single(model, H1, adapter_layers[1], adapter_layers[2])

                # Batch all inner configs
                W1_2, W2_2 = adapters[2].build_weights(all_grid_params)
                H2 = apply_adapter_batched(H_pre2, W1_2, W2_2, adapters[2].alpha)
                H_final = run_layers_batched(model, H2, adapter_layers[2], adapter_layers[2] + 1)
                batch_losses = compute_loss_batched(model, H_final, flat_labels, last_positions)

            losses_out[write_idx:write_idx + cpa] = \
                batch_losses.cpu().numpy().astype(np.float16)
            write_idx += cpa

        if (i0 - l0_start) % 2 == 0:
            losses_out.flush()

    losses_out.flush()
    print(f"[GPU {gpu_id}] Done. Wrote {write_idx:,} losses to shard.")
