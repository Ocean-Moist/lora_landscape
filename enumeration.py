"""Core batched forward pass: run layer 11 + LM head for a batch of binary LoRA configs."""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from binary_lora import config_indices_to_binary, apply_binary_lora_batched
from model_setup import PrecomputedState
from storage import LossStorage


def batched_forward(
    state: PrecomputedState,
    m_Q: torch.Tensor,   # [B, rank_q]
    m_V: torch.Tensor,   # [B, rank_v]
    B_Q: torch.Tensor,   # [rank_q, d_model]
    B_V: torch.Tensor,   # [rank_v, d_model]
    alpha: float,
    labels: torch.Tensor,  # [1, seq_len] or [seq_len]
    lm_chunk_size: int = 8192,
) -> torch.Tensor:
    """Forward pass from LoRA layer through LM head for B configs.

    The LM head matmul ([B*seq, 768] @ [768, 50257]) dominates memory.
    We chunk it to allow large B for the cheaper attention + MLP ops.

    Returns: [B] tensor of cross-entropy losses.
    """
    B = m_Q.shape[0]

    # 1. Compute LoRA deltas
    delta_Q, delta_V = apply_binary_lora_batched(
        state.proj_Q, state.proj_V, m_Q, m_V, B_Q, B_V, alpha
    )

    # 2. Form full Q and V: [B, seq, 768]
    Q = state.Q_base.unsqueeze(0) + delta_Q
    V = state.V_base.unsqueeze(0) + delta_V
    K = state.K.unsqueeze(0).expand(B, -1, -1)  # [B, seq, 768]

    # 3. Reshape for multi-head attention: [B, num_heads, seq, head_dim]
    seq = state.seq_len
    nh = state.num_heads
    hd = state.head_dim
    Q = Q.view(B, seq, nh, hd).transpose(1, 2)
    K = K.view(B, seq, nh, hd).transpose(1, 2)
    V = V.view(B, seq, nh, hd).transpose(1, 2)

    # 4. Attention — manual for short sequences (Flash Attention fails at large B with short seq)
    scale = hd ** -0.5
    scores = (Q @ K.transpose(-2, -1)) * scale  # [B, nh, seq, seq]
    # Causal mask
    mask = torch.triu(torch.ones(seq, seq, device=Q.device, dtype=torch.bool), diagonal=1)
    scores.masked_fill_(mask, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(Q.dtype)
    attn_out = attn_weights @ V  # [B, nh, seq, hd]
    attn_out = attn_out.transpose(1, 2).reshape(B, seq, nh * hd)

    # 5. c_proj + residual
    h = state.hidden.unsqueeze(0) + attn_out @ state.c_proj_weight + state.c_proj_bias

    # 6. LN2 + MLP + residual
    h = h + state.mlp(state.ln_2(h))

    # 7. Final layer norm
    h = state.ln_f(h)

    # 8. Chunked LM head + cross-entropy (avoids materializing full [B, seq-1, 50257] logits)
    h_shift = h[:, :-1, :].contiguous()  # [B, seq-1, 768]
    lab = labels.squeeze(0)[1:]  # [seq-1]
    seq_m1 = h_shift.shape[1]

    losses = torch.empty(B, device=h.device, dtype=torch.float32)
    for chunk_start in range(0, B, lm_chunk_size):
        chunk_end = min(chunk_start + lm_chunk_size, B)
        h_chunk = h_shift[chunk_start:chunk_end]  # [chunk, seq-1, 768]
        cs = chunk_end - chunk_start
        logits = h_chunk.reshape(cs * seq_m1, -1) @ state.lm_head_weight.T  # [chunk*seq_m1, vocab]
        loss = F.cross_entropy(
            logits,
            lab.repeat(cs),
            reduction="none",
        ).view(cs, seq_m1).mean(dim=1)
        losses[chunk_start:chunk_end] = loss

    return losses


def enumerate_gpu(
    gpu_id: int,
    num_gpus: int,
    state: PrecomputedState,
    lora_config,
    labels: torch.Tensor,
    storage: LossStorage,
    config_batch_size: int,
    num_params: int,
    total_configs: int,
):
    """Enumerate all configs assigned to this GPU."""
    device = state.hidden.device
    dtype = state.hidden.dtype

    configs_per_gpu = total_configs // num_gpus
    start = gpu_id * configs_per_gpu
    end = start + configs_per_gpu

    # Resume from checkpoint
    last_batch = storage.load_checkpoint()
    start_batch = last_batch + 1 if last_batch >= 0 else 0

    num_batches = (configs_per_gpu + config_batch_size - 1) // config_batch_size

    pbar = tqdm(
        range(start_batch, num_batches),
        desc=f"GPU {gpu_id}",
        initial=start_batch,
        total=num_batches,
    )

    for batch_idx in pbar:
        batch_start = start + batch_idx * config_batch_size
        batch_end = min(batch_start + config_batch_size, end)

        indices = torch.arange(batch_start, batch_end, device=device, dtype=torch.int64)
        binary = config_indices_to_binary(indices, num_params).to(dtype)

        m_Q = binary[:, :lora_config.rank_q]
        m_V = binary[:, lora_config.rank_q:]

        with torch.no_grad():
            losses = batched_forward(
                state, m_Q, m_V,
                lora_config.B_Q, lora_config.B_V,
                lora_config.alpha, labels,
            )

        offset = batch_idx * config_batch_size
        storage.write_batch(offset, losses.cpu().numpy().astype(np.float16))

        if batch_idx % 1024 == 0:
            storage.save_checkpoint(batch_idx)

    storage.flush()
    storage.save_checkpoint(num_batches - 1)
