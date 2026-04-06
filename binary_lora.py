"""Binary LoRA parameterization: A @ diag(m) @ B where m ∈ {-1, +1}^r.

Each binary parameter toggles one rank-1 perturbation direction.
The landscape over m is a genuine cross-section of the pretrained model's loss surface.
"""

import torch
import torch.nn as nn


class BinaryLoRAConfig:
    """Frozen random projections for binary LoRA on Q and V."""

    def __init__(self, d_model: int, rank_q: int, rank_v: int, alpha: float, seed: int, device: torch.device, dtype: torch.dtype):
        rng = torch.Generator(device="cpu").manual_seed(seed)

        # A: [d_model, rank], B: [rank, d_model] — frozen random directions
        # Normalize so perturbation magnitude is controlled by alpha alone
        self.A_Q = self._make_A(d_model, rank_q, rng).to(device=device, dtype=dtype)
        self.B_Q = self._make_B(rank_q, d_model, rng).to(device=device, dtype=dtype)
        self.A_V = self._make_A(d_model, rank_v, rng).to(device=device, dtype=dtype)
        self.B_V = self._make_B(rank_v, d_model, rng).to(device=device, dtype=dtype)
        self.alpha = alpha
        self.rank_q = rank_q
        self.rank_v = rank_v

    @staticmethod
    def _make_A(d_model: int, rank: int, rng: torch.Generator) -> torch.Tensor:
        A = torch.randn(d_model, rank, generator=rng)
        A = A / A.norm(dim=0, keepdim=True)  # unit-norm columns
        return A

    @staticmethod
    def _make_B(rank: int, d_model: int, rng: torch.Generator) -> torch.Tensor:
        B = torch.randn(rank, d_model, generator=rng)
        B = B / B.norm(dim=1, keepdim=True)  # unit-norm rows
        return B


def config_indices_to_binary(indices: torch.Tensor, num_params: int) -> torch.Tensor:
    """Convert integer config indices to binary vectors in {-1, +1}.

    Args:
        indices: [B] int64 tensor of config indices
        num_params: total binary params (e.g. 40)

    Returns:
        [B, num_params] float tensor with values in {-1, +1}
        Bits 0..rank_q-1 -> m_Q, bits rank_q..num_params-1 -> m_V
    """
    # bits[i, j] = (indices[i] >> j) & 1
    shifts = torch.arange(num_params, device=indices.device, dtype=torch.int64)
    bits = (indices.unsqueeze(1) >> shifts.unsqueeze(0)) & 1
    return bits.float() * 2 - 1  # {0,1} -> {-1,+1}


def apply_binary_lora_batched(
    proj_Q: torch.Tensor,  # [seq, rank_q] — precomputed hidden @ A_Q
    proj_V: torch.Tensor,  # [seq, rank_v] — precomputed hidden @ A_V
    m_Q: torch.Tensor,     # [B, rank_q] — binary config vectors for Q
    m_V: torch.Tensor,     # [B, rank_v] — binary config vectors for V
    B_Q: torch.Tensor,     # [rank_q, d_model]
    B_V: torch.Tensor,     # [rank_v, d_model]
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute batched LoRA deltas for Q and V projections.

    delta_Q[b] = (proj_Q * m_Q[b]) @ B_Q * alpha
               = proj_Q @ diag(m_Q[b]) @ B_Q * alpha

    Returns:
        delta_Q: [B, seq, d_model]
        delta_V: [B, seq, d_model]
    """
    # proj_Q: [seq, rank_q] -> [1, seq, rank_q]
    # m_Q:    [B, rank_q]   -> [B, 1, rank_q]
    # element-wise multiply -> [B, seq, rank_q]
    # matmul with B_Q [rank_q, d_model] -> [B, seq, d_model]
    delta_Q = (proj_Q.unsqueeze(0) * m_Q.unsqueeze(1)) @ B_Q * alpha
    delta_V = (proj_V.unsqueeze(0) * m_V.unsqueeze(1)) @ B_V * alpha
    return delta_Q, delta_V
