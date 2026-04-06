"""Continuous MLP adapter — same architecture as coupled, but 2 free params per adapter.

One free param in W1, one in W2, for multiplicative coupling within each adapter.
Cross-layer coupling comes from 3 adapters at separated layers.
"""

import torch
import torch.nn.functional as F


class ContinuousAdapterConfig:
    """Frozen random weights + 2 free continuous parameter positions per adapter."""

    def __init__(self, d_model: int, width: int, n_free: int, alpha: float,
                 seed: int, device: torch.device, dtype: torch.dtype):
        self.d_model = d_model
        self.width = width
        self.n_free = n_free
        self.alpha = alpha

        rng = torch.Generator(device="cpu").manual_seed(seed)

        scale1 = (2.0 / d_model) ** 0.5
        scale2 = (2.0 / width) ** 0.5
        W1_init = torch.randn(width, d_model, generator=rng) * scale1
        W2_init = torch.randn(d_model, width, generator=rng) * scale2

        # Place free params: half in W1, half in W2
        n_w1 = n_free // 2  # 1
        n_w2 = n_free - n_w1  # 1

        total_w1 = width * d_model
        total_w2 = d_model * width
        perm1 = torch.randperm(total_w1, generator=rng)[:n_w1]
        perm2 = torch.randperm(total_w2, generator=rng)[:n_w2]

        self.free_idx_w1 = perm1.to(device)
        self.free_idx_w2 = perm2.to(device)
        self.n_w1 = n_w1
        self.n_w2 = n_w2

        # Zero out free positions in base weights
        W1_flat = W1_init.reshape(-1)
        W2_flat = W2_init.reshape(-1)
        W1_flat[perm1] = 0.0
        W2_flat[perm2] = 0.0

        # Free param scale: 10x init for clear signal
        self.free_scale_w1 = torch.full((n_w1,), scale1 * 10.0).to(device=device, dtype=dtype)
        self.free_scale_w2 = torch.full((n_w2,), scale2 * 10.0).to(device=device, dtype=dtype)

        self.W1_base = W1_flat.reshape(width, d_model).to(device=device, dtype=dtype)
        self.W2_base = W2_flat.reshape(d_model, width).to(device=device, dtype=dtype)

        # Store spectral norms for Lipschitz certification
        self.W1_spec_norm = float(torch.linalg.norm(self.W1_base.float(), ord=2))
        self.W2_spec_norm = float(torch.linalg.norm(self.W2_base.float(), ord=2))

    def build_weights(self, params: torch.Tensor):
        """Build W1, W2 from continuous parameter values.

        Args:
            params: [B, n_free] tensor with values in [param_min, param_max]

        Returns:
            W1: [B, width, d_model]
            W2: [B, d_model, width]
        """
        B = params.shape[0]
        params_w1 = params[:, :self.n_w1]
        params_w2 = params[:, self.n_w1:]

        W1 = self.W1_base.unsqueeze(0).expand(B, -1, -1).clone()
        W2 = self.W2_base.unsqueeze(0).expand(B, -1, -1).clone()

        W1_flat = W1.reshape(B, -1)
        W2_flat = W2.reshape(B, -1)

        vals_w1 = params_w1 * self.free_scale_w1.unsqueeze(0)
        vals_w2 = params_w2 * self.free_scale_w2.unsqueeze(0)

        W1_flat[:, self.free_idx_w1] = vals_w1
        W2_flat[:, self.free_idx_w2] = vals_w2

        return W1_flat.reshape(B, self.width, self.d_model), \
               W2_flat.reshape(B, self.d_model, self.width)


def grid_indices_to_params(indices: torch.Tensor, n_params: int,
                           n_grid: int, param_min: float, param_max: float) -> torch.Tensor:
    """Map flat grid indices to continuous parameter vectors.

    For n_params=2 and n_grid=32: indices in [0, 1024) map to 2D grid.

    Args:
        indices: [B] int64
        n_params: params per adapter (2)
        n_grid: grid points per dim (32)

    Returns:
        [B, n_params] float tensor
    """
    grid = torch.linspace(param_min, param_max, n_grid, device=indices.device)
    params = torch.zeros(indices.shape[0], n_params, device=indices.device, dtype=torch.float32)
    remaining = indices.clone()
    for k in range(n_params):
        grid_idx = remaining % n_grid
        params[:, k] = grid[grid_idx]
        remaining //= n_grid
    return params
