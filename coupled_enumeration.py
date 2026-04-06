"""Layer-wise cached enumeration for coupled MLP adapters.

Exploits tree structure: cache intermediate states at each adapter insertion point.
- Layers 0-1: run once -> H1
- Layer 2 adapter × 256 configs -> run through layers 2-5 -> H5[256]
- Layer 6 adapter × 256 configs (per H5) -> run through layers 6-10 -> H10[65536]
- Layer 11 adapter × 256 configs (per H10) -> LM head -> losses[16.7M]
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from coupled_adapter import AdapterConfig, apply_adapter_batched, config_indices_to_binary


def run_layers(model, hidden: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Run transformer layers [start, end) on hidden states.

    Args:
        model: GPT2LMHeadModel
        hidden: [B, seq, d] or [seq, d]
        start: first layer index (inclusive)
        end: last layer index (exclusive)

    Returns:
        hidden: [B, seq, d] after running through layers
    """
    for i in range(start, end):
        layer = model.transformer.h[i]
        # GPT-2 block: LN1 -> Attn -> residual -> LN2 -> MLP -> residual
        residual = hidden
        h = layer.ln_1(hidden)

        # Self-attention
        qkv = h @ layer.attn.c_attn.weight + layer.attn.c_attn.bias
        d = layer.attn.c_attn.weight.shape[0]
        nh = layer.attn.num_heads
        hd = d // nh

        Q, K, V = qkv.split(d, dim=-1)

        # Reshape for multi-head
        shape = hidden.shape[:-1]  # [B, seq] or [seq]
        Q = Q.view(*shape, nh, hd).transpose(-3, -2)  # [..., nh, seq, hd]
        K = K.view(*shape, nh, hd).transpose(-3, -2)
        V = V.view(*shape, nh, hd).transpose(-3, -2)

        # Attention
        scale = hd ** -0.5
        scores = (Q @ K.transpose(-2, -1)) * scale
        seq_len = hidden.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float("-inf"))
        attn_w = F.softmax(scores, dim=-1, dtype=torch.float32).to(hidden.dtype)
        attn_out = attn_w @ V
        attn_out = attn_out.transpose(-3, -2).reshape(*shape, d)

        attn_out = attn_out @ layer.attn.c_proj.weight + layer.attn.c_proj.bias
        hidden = residual + attn_out

        # MLP
        hidden = hidden + layer.mlp(layer.ln_2(hidden))

    return hidden


def compute_loss(model, hidden: torch.Tensor, labels: torch.Tensor,
                 chunk_size: int = 8192) -> torch.Tensor:
    """Compute cross-entropy loss from final hidden states.

    Args:
        model: GPT2LMHeadModel (for ln_f and lm_head)
        hidden: [B, seq, d]
        labels: [N_seq, seq] — target token IDs

    Returns:
        losses: [B] — mean CE loss over all positions and sequences
    """
    B = hidden.shape[0]
    seq = hidden.shape[1]

    # Final layer norm
    h = model.transformer.ln_f(hidden)

    # LM head — compute loss over ALL positions for richer signal
    # labels shape: [N_seq, seq] — same for all configs
    # h shape: [B, seq, d] where seq = N_seq * seq_per_seq

    losses = torch.zeros(B, device=hidden.device, dtype=torch.float32)

    for cs in range(0, B, chunk_size):
        ce = min(cs + chunk_size, B)
        h_chunk = h[cs:ce]  # [chunk, seq, d]
        logits = h_chunk @ model.lm_head.weight.T  # [chunk, seq, vocab]

        # labels is [seq] (flattened), broadcast over chunk
        chunk_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.unsqueeze(0).expand(ce - cs, -1).reshape(-1),
            reduction="none",
        )
        losses[cs:ce] = chunk_losses.reshape(ce - cs, -1).mean(dim=1)

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
    """Enumerate all 2^24 configs using layer-wise caching.

    GPU parallelism: split the outermost loop (layer-2 configs) across GPUs.
    Each GPU handles configs_per_layer/num_gpus layer-2 configs and all
    downstream combinations.
    """
    device = input_ids.device
    adapter_layers = cfg.adapter_layers
    n_per = cfg.bits_per_adapter
    configs_per_layer = cfg.configs_per_layer  # 256

    # Flatten input: [N_seq, seq_len] -> [N_seq * seq_len]
    N_seq, seq_per = input_ids.shape
    flat_ids = input_ids.reshape(1, -1)  # [1, N_seq * seq_len]
    flat_labels = labels.reshape(-1)      # [N_seq * seq_len]

    total_seq = N_seq * seq_per

    # Precompute all 256 bit configs for one adapter level
    all_bits = config_indices_to_binary(
        torch.arange(configs_per_layer, device=device),
        n_per,
    ).to(cfg.dtype)  # [256, 8]

    # Split layer-2 configs across GPUs
    l2_per_gpu = configs_per_layer // num_gpus
    l2_start = gpu_id * l2_per_gpu
    l2_end = l2_start + l2_per_gpu
    configs_this_gpu = l2_per_gpu * configs_per_layer * configs_per_layer

    # Memory-mapped output
    shard_path = f"{output_path}/losses_shard_{gpu_id}.npy"
    losses_out = np.memmap(shard_path, dtype=np.float16, mode="w+",
                           shape=(configs_this_gpu,))

    # ---- Step 1: Run layers 0 to adapter_layers[0]-1 (once) ----
    with torch.no_grad():
        # Get embeddings
        embeds = model.transformer.wte(flat_ids) + model.transformer.wpe(
            torch.arange(total_seq, device=device).unsqueeze(0))
        embeds = model.transformer.drop(embeds)
        H0 = run_layers(model, embeds.squeeze(0), 0, adapter_layers[0])
        # H0: [total_seq, d_model]

    write_idx = 0
    total_inner = configs_per_layer * configs_per_layer
    pbar = tqdm(range(l2_start, l2_end), desc=f"GPU {gpu_id} L2",
                total=l2_per_gpu)

    for i2 in pbar:
        # ---- Step 2: Apply layer-2 adapter, run layers 2-5 ----
        with torch.no_grad():
            bits2 = all_bits[i2:i2+1]  # [1, 8]
            W1_2, W2_2 = adapters[0].build_weights(bits2)
            H2 = apply_adapter_batched(H0, W1_2, W2_2, adapters[0].alpha)
            H2 = H2.squeeze(0)  # [seq, d]
            H5 = run_layers(model, H2, adapter_layers[0], adapter_layers[1])
            # H5: [seq, d]

        for i6 in range(configs_per_layer):
            # ---- Step 3: Apply layer-6 adapter, run layers 6-10 ----
            with torch.no_grad():
                bits6 = all_bits[i6:i6+1]  # [1, 8]
                W1_6, W2_6 = adapters[1].build_weights(bits6)
                H6 = apply_adapter_batched(H5, W1_6, W2_6, adapters[1].alpha)
                H6 = H6.squeeze(0)  # [seq, d]
                H10 = run_layers(model, H6, adapter_layers[1], adapter_layers[2])
                # H10: [seq, d]

            # ---- Step 4: Apply layer-11 adapter + LM head for all 256 configs ----
            # Batch all 256 layer-11 configs at once
            with torch.no_grad():
                W1_11, W2_11 = adapters[2].build_weights(all_bits)
                # W1: [256, w, d], W2: [256, d, w]
                H11 = apply_adapter_batched(H10, W1_11, W2_11, adapters[2].alpha)
                # H11: [256, seq, d]

                # Run layer 11 block
                H_final = run_layers(model, H11, adapter_layers[2], adapter_layers[2] + 1)
                # H_final: [256, seq, d]

                # Compute losses
                batch_losses = compute_loss(model, H_final, flat_labels)
                # batch_losses: [256]

            losses_out[write_idx:write_idx + configs_per_layer] = batch_losses.cpu().numpy().astype(np.float16)
            write_idx += configs_per_layer

        if (i2 - l2_start) % 4 == 0:
            losses_out.flush()

    losses_out.flush()
    print(f"[GPU {gpu_id}] Done. Wrote {write_idx} losses.")
