"""Quantized LoRA parameterization: A @ diag(m) @ B where m ∈ {-1, 0, +1}^r (ternary).

Each parameter toggles or disables one rank-1 perturbation direction.
The zero state creates genuine parameter interactions — a direction being "off"
changes how other directions affect loss.
"""

import torch


class BinaryLoRAConfig:
    """Frozen random projections for quantized LoRA on Q and V."""

    def __init__(self, d_model: int, rank_q: int, rank_v: int, alpha: float, seed: int, device: torch.device, dtype: torch.dtype):
        rng = torch.Generator(device="cpu").manual_seed(seed)

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
        A = A / A.norm(dim=0, keepdim=True)
        return A

    @staticmethod
    def _make_B(rank: int, d_model: int, rng: torch.Generator) -> torch.Tensor:
        B = torch.randn(rank, d_model, generator=rng)
        B = B / B.norm(dim=1, keepdim=True)
        return B


def config_indices_to_ternary(indices: torch.Tensor, num_params: int) -> torch.Tensor:
    """Convert integer config indices to ternary vectors in {-1, 0, +1}.

    Uses base-3 decomposition: digit d maps to d - 1, so {0,1,2} -> {-1,0,+1}.

    Args:
        indices: [B] int64 tensor of config indices
        num_params: total params (e.g. 19)

    Returns:
        [B, num_params] float tensor with values in {-1, 0, +1}
    """
    B = indices.shape[0]
    digits = torch.zeros(B, num_params, device=indices.device, dtype=torch.int64)
    remaining = indices.clone()
    for k in range(num_params):
        digits[:, k] = remaining % 3
        remaining //= 3
    return (digits - 1).float()  # {0,1,2} -> {-1,0,+1}


# Keep old name for compatibility
def config_indices_to_binary(indices: torch.Tensor, num_params: int) -> torch.Tensor:
    """Binary version for backwards compat. Use config_indices_to_ternary for ternary."""
    shifts = torch.arange(num_params, device=indices.device, dtype=torch.int64)
    bits = (indices.unsqueeze(1) >> shifts.unsqueeze(0)) & 1
    return bits.float() * 2 - 1


def apply_binary_lora_batched(
    proj_Q: torch.Tensor,
    proj_V: torch.Tensor,
    m_Q: torch.Tensor,
    m_V: torch.Tensor,
    B_Q: torch.Tensor,
    B_V: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute batched LoRA deltas. Works for any quantization (binary/ternary/4-level).

    delta_Q[b] = proj_Q @ diag(m_Q[b]) @ B_Q * alpha
    """
    delta_Q = (proj_Q.unsqueeze(0) * m_Q.unsqueeze(1)) @ B_Q * alpha
    delta_V = (proj_V.unsqueeze(0) * m_V.unsqueeze(1)) @ B_V * alpha
    return delta_Q, delta_V
