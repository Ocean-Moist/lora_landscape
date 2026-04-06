"""Layer-wise cached enumeration for coupled MLP adapters.

Exploits tree structure: cache intermediate states at each adapter insertion point.
- Layers 0 to adapter[0]-1: run once -> H_pre0
- Layer adapter[0] × 256 configs -> layers adapter[0]+1 to adapter[1]-1 -> H_pre1[256]
- Layer adapter[1] × 256 configs -> layers adapter[1]+1 to adapter[2]-1 -> H_pre2[65K]
- Layer adapter[2] × 256 configs -> remaining layers + LM head -> losses[16.7M]

With 4 seq × 32 tok = 128 tokens and batch=256, attention scores are [256, 12, 128, 128] = 96MB.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from coupled_adapter import AdapterConfig, apply_adapter_batched, config_indices_to_binary


def run_layers_single(model, hidden: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Run transformer layers [start, end) on hidden states. No batch dim.

    Args:
        hidden: [seq, d]
    Returns:
        [seq, d]
    """
    for i in range(start, end):
        layer = model.transformer.h[i]
        residual = hidden
        h = layer.ln_1(hidden)
        qkv = h @ layer.attn.c_attn.weight + layer.attn.c_attn.bias
        d = layer.attn.c_attn.weight.shape[0]
        nh = layer.attn.num_heads
        hd = d // nh
        Q, K, V = qkv.split(d, dim=-1)
        seq = hidden.shape[0]
        Q = Q.view(seq, nh, hd).transpose(0, 1)  # [nh, seq, hd]
        K = K.view(seq, nh, hd).transpose(0, 1)
        V = V.view(seq, nh, hd).transpose(0, 1)
        scores = (Q @ K.transpose(-2, -1)) * (hd ** -0.5)
        mask = torch.triu(torch.ones(seq, seq, device=hidden.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float("-inf"))
        attn_w = F.softmax(scores, dim=-1, dtype=torch.float32).to(hidden.dtype)
        attn_out = (attn_w @ V).transpose(0, 1).reshape(seq, d)
        attn_out = attn_out @ layer.attn.c_proj.weight + layer.attn.c_proj.bias
        hidden = residual + attn_out
        hidden = hidden + layer.mlp(layer.ln_2(hidden))
    return hidden


def run_layers_batched(model, hidden: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Run transformer layers [start, end) on batched hidden states.

    Args:
        hidden: [B, seq, d]
    Returns:
        [B, seq, d]
    """
    for i in range(start, end):
        layer = model.transformer.h[i]
        residual = hidden
        h = layer.ln_1(hidden)
        qkv = h @ layer.attn.c_attn.weight + layer.attn.c_attn.bias
        d = layer.attn.c_attn.weight.shape[0]
        nh = layer.attn.num_heads
        hd = d // nh
        B, seq = hidden.shape[:2]
        Q, K, V = qkv.split(d, dim=-1)
        Q = Q.view(B, seq, nh, hd).transpose(1, 2)  # [B, nh, seq, hd]
        K = K.view(B, seq, nh, hd).transpose(1, 2)
        V = V.view(B, seq, nh, hd).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) * (hd ** -0.5)
        mask = torch.triu(torch.ones(seq, seq, device=hidden.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float("-inf"))
        attn_w = F.softmax(scores, dim=-1, dtype=torch.float32).to(hidden.dtype)
        attn_out = (attn_w @ V).transpose(1, 2).reshape(B, seq, d)
        attn_out = attn_out @ layer.attn.c_proj.weight + layer.attn.c_proj.bias
        hidden = residual + attn_out
        hidden = hidden + layer.mlp(layer.ln_2(hidden))
    return hidden


# Keep old name for coupled_benchmark imports
run_layers = run_layers_batched


def compute_loss_batched(model, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute CE loss from final hidden states, all positions.

    Args:
        hidden: [B, seq, d]
        labels: [seq] flattened target tokens

    Returns:
        losses: [B]
    """
    B = hidden.shape[0]
    h = model.transformer.ln_f(hidden)

    # LM head: [B, seq, vocab]
    logits = h @ model.lm_head.weight.to(h.dtype).T

    # CE loss over all positions, averaged
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.unsqueeze(0).expand(B, -1).reshape(-1),
        reduction="none",
    ).reshape(B, -1).mean(dim=1)

    return losses


def enumerate_coupled(
    gpu_id: int,
    num_gpus: int,
    model,
    adapters: list[AdapterConfig],
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    cfg,
    output_path: str,
):
    """Enumerate all 2^24 configs using layer-wise caching."""
    device = input_ids.device
    adapter_layers = cfg.adapter_layers
    n_per = cfg.bits_per_adapter
    configs_per_layer = cfg.configs_per_layer  # 256

    # Flatten sequences: [N_seq, seq_len] -> [N_seq * seq_len]
    N_seq, seq_per = input_ids.shape
    flat_ids = input_ids.reshape(1, -1)  # [1, total_seq]
    flat_labels = labels.reshape(-1)      # [total_seq]
    total_seq = N_seq * seq_per

    # Precompute all 256 bit configs
    all_bits = config_indices_to_binary(
        torch.arange(configs_per_layer, device=device),
        n_per,
    ).to(cfg.dtype)  # [256, 8]

    # Split layer-0 (outermost) configs across GPUs
    l0_per_gpu = configs_per_layer // num_gpus
    l0_start = gpu_id * l0_per_gpu
    l0_end = l0_start + l0_per_gpu
    configs_this_gpu = l0_per_gpu * configs_per_layer * configs_per_layer

    # Memory-mapped output
    import os
    os.makedirs(output_path, exist_ok=True)
    shard_path = f"{output_path}/losses_shard_{gpu_id}.npy"
    losses_out = np.memmap(shard_path, dtype=np.float16, mode="w+",
                           shape=(configs_this_gpu,))

    # ---- Step 1: Embeddings + layers 0 to adapter_layers[0]-1 ----
    with torch.no_grad():
        pos_ids = torch.arange(total_seq, device=device).unsqueeze(0)
        embeds = model.transformer.wte(flat_ids) + model.transformer.wpe(pos_ids)
        embeds = model.transformer.drop(embeds).squeeze(0)  # [total_seq, d]

        H_pre0 = run_layers_single(model, embeds, 0, adapter_layers[0])
        # H_pre0: [total_seq, d]

    write_idx = 0
    pbar = tqdm(range(l0_start, l0_end), desc=f"GPU {gpu_id} L{adapter_layers[0]}",
                total=l0_per_gpu)

    for i0 in pbar:
        with torch.no_grad():
            # ---- Step 2: Apply adapter 0, run to next adapter ----
            bits0 = all_bits[i0:i0+1]  # [1, 8]
            W1_0, W2_0 = adapters[0].build_weights(bits0)
            H0 = apply_adapter_batched(H_pre0, W1_0, W2_0, adapters[0].alpha)
            H0 = H0.squeeze(0)  # [total_seq, d]

            # Run layers adapter[0] to adapter[1]-1
            H_pre1 = run_layers_single(model, H0, adapter_layers[0], adapter_layers[1])

        for i1 in range(configs_per_layer):
            with torch.no_grad():
                # ---- Step 3: Apply adapter 1, run to next adapter ----
                bits1 = all_bits[i1:i1+1]  # [1, 8]
                W1_1, W2_1 = adapters[1].build_weights(bits1)
                H1 = apply_adapter_batched(H_pre1, W1_1, W2_1, adapters[1].alpha)
                H1 = H1.squeeze(0)  # [total_seq, d]

                # Run layers adapter[1] to adapter[2]-1
                H_pre2 = run_layers_single(model, H1, adapter_layers[1], adapter_layers[2])

                # ---- Step 4: Batch all 256 adapter-2 configs ----
                W1_2, W2_2 = adapters[2].build_weights(all_bits)
                H2 = apply_adapter_batched(H_pre2, W1_2, W2_2, adapters[2].alpha)
                # H2: [256, total_seq, d]

                # Run layer adapter[2] (just 1 layer) batched
                H_final = run_layers_batched(model, H2, adapter_layers[2], adapter_layers[2] + 1)

                # Compute losses
                batch_losses = compute_loss_batched(model, H_final, flat_labels)

            losses_out[write_idx:write_idx + configs_per_layer] = \
                batch_losses.cpu().numpy().astype(np.float16)
            write_idx += configs_per_layer

        if (i0 - l0_start) % 2 == 0:
            losses_out.flush()

    losses_out.flush()
    print(f"[GPU {gpu_id}] Done. Wrote {write_idx:,} losses to shard.")
