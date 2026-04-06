"""Binary MLP adapter with multiplicative coupling between free parameters.

Each adapter: h_out = h_in + alpha * W2 @ relu(W1 @ h_in)
W1: (w, 768), W2: (768, w) — mostly frozen random, with 8 free binary {-1, +1} entries
distributed across BOTH W1 and W2 to ensure multiplicative interaction.
"""

import torch
import torch.nn.functional as F


class AdapterConfig:
    """Frozen random weights + free parameter masks for one MLP adapter."""

    def __init__(self, d_model: int, width: int, n_free: int, alpha: float,
                 seed: int, device: torch.device, dtype: torch.dtype):
        self.d_model = d_model
        self.width = width
        self.n_free = n_free
        self.alpha = alpha

        rng = torch.Generator(device="cpu").manual_seed(seed)

        # Initialize W1 (width, d_model) and W2 (d_model, width) with proper scale
        scale1 = (2.0 / d_model) ** 0.5
        scale2 = (2.0 / width) ** 0.5
        W1_init = torch.randn(width, d_model, generator=rng) * scale1
        W2_init = torch.randn(d_model, width, generator=rng) * scale2

        # Select free parameter positions: split across W1 and W2
        # Put 4 in W1, 4 in W2 for maximum multiplicative coupling
        n_w1 = n_free // 2
        n_w2 = n_free - n_w1

        # Pick positions via deterministic random selection
        total_w1 = width * d_model
        total_w2 = d_model * width
        perm1 = torch.randperm(total_w1, generator=rng)[:n_w1]
        perm2 = torch.randperm(total_w2, generator=rng)[:n_w2]

        # Convert flat indices to (row, col) for W1 and W2
        self.free_idx_w1 = perm1  # flat indices into W1
        self.free_idx_w2 = perm2  # flat indices into W2
        self.n_w1 = n_w1
        self.n_w2 = n_w2

        # Store base weights (with free positions zeroed — they'll be set per-config)
        W1_flat = W1_init.reshape(-1)
        W2_flat = W2_init.reshape(-1)
        W1_flat[self.free_idx_w1] = 0.0
        W2_flat[self.free_idx_w2] = 0.0

        # Free params need LARGE scale relative to frozen weights to be impactful.
        # Each free param is one entry in W1 or W2. With small init scale (~0.05),
        # a binary flip barely registers. Use 10x init scale for clear signal.
        self.free_scale_w1 = torch.full((n_w1,), scale1 * 10.0)
        self.free_scale_w2 = torch.full((n_w2,), scale2 * 10.0)

        self.W1_base = W1_flat.reshape(width, d_model).to(device=device, dtype=dtype)
        self.W2_base = W2_flat.reshape(d_model, width).to(device=device, dtype=dtype)
        self.free_scale_w1 = self.free_scale_w1.to(device=device, dtype=dtype)
        self.free_scale_w2 = self.free_scale_w2.to(device=device, dtype=dtype)
        self.free_idx_w1 = self.free_idx_w1.to(device=device)
        self.free_idx_w2 = self.free_idx_w2.to(device=device)

    def build_weights(self, bits: torch.Tensor):
        """Build W1, W2 from binary config bits.

        Args:
            bits: [B, n_free] tensor with values in {-1, +1}

        Returns:
            W1: [B, width, d_model]
            W2: [B, d_model, width]
        """
        B = bits.shape[0]
        bits_w1 = bits[:, :self.n_w1]  # [B, n_w1]
        bits_w2 = bits[:, self.n_w1:]  # [B, n_w2]

        # Start from base weights (broadcast)
        W1 = self.W1_base.unsqueeze(0).expand(B, -1, -1).clone()  # [B, w, d]
        W2 = self.W2_base.unsqueeze(0).expand(B, -1, -1).clone()  # [B, d, w]

        # Insert scaled binary values at free positions
        W1_flat = W1.reshape(B, -1)
        W2_flat = W2.reshape(B, -1)

        vals_w1 = bits_w1 * self.free_scale_w1.unsqueeze(0)  # [B, n_w1]
        vals_w2 = bits_w2 * self.free_scale_w2.unsqueeze(0)  # [B, n_w2]

        W1_flat[:, self.free_idx_w1] = vals_w1
        W2_flat[:, self.free_idx_w2] = vals_w2

        W1 = W1_flat.reshape(B, self.width, self.d_model)
        W2 = W2_flat.reshape(B, self.d_model, self.width)
        return W1, W2


def apply_adapter_batched(h: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor,
                          alpha: float) -> torch.Tensor:
    """Apply MLP adapter to hidden states.

    Args:
        h: [B, seq, d_model] or [seq, d_model] (broadcast over B)
        W1: [B, width, d_model]
        W2: [B, d_model, width]
        alpha: scaling factor

    Returns:
        h_out: [B, seq, d_model]
    """
    if h.dim() == 2:
        h = h.unsqueeze(0)  # [1, seq, d]

    # h: [B, seq, d], W1.T: [B, d, w] -> bottleneck: [B, seq, w]
    bottleneck = torch.bmm(h.expand(W1.shape[0], -1, -1),
                           W1.transpose(1, 2))  # [B, seq, w]
    bottleneck = F.relu(bottleneck)

    # W2: [B, d, w] -> out: [B, seq, d]
    out = torch.bmm(bottleneck, W2.transpose(1, 2))  # [B, seq, d]

    return h.expand(W1.shape[0], -1, -1) + alpha * out


def config_indices_to_binary(indices: torch.Tensor, num_params: int) -> torch.Tensor:
    """Convert integer config indices to binary vectors in {-1, +1}.

    Args:
        indices: [B] int64 tensor
        num_params: total binary params

    Returns:
        [B, num_params] float tensor with values in {-1, +1}
    """
    shifts = torch.arange(num_params, device=indices.device, dtype=torch.int64)
    bits = (indices.unsqueeze(1) >> shifts.unsqueeze(0)) & 1
    return bits.float() * 2 - 1  # {0,1} -> {-1,+1}
