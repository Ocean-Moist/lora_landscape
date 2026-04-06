"""Optimizer benchmarking against ground-truth ternary LoRA landscape.

Uses straight-through estimator (STE): maintain continuous latent params,
quantize to ternary for forward pass, pass gradients straight through.
All runs are batched for GPU efficiency.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

from binary_lora import BinaryLoRAConfig, apply_binary_lora_batched
from model_setup import PrecomputedState


# ---- Straight-Through Estimator for ternary quantization ----

class STETernary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Quantize to {-1, 0, +1}: round and clamp
        return torch.clamp(torch.round(x), -1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # straight-through


def ste_ternary(x):
    return STETernary.apply(x)


# ---- Differentiable forward pass (single step, batched across runs) ----

def differentiable_forward(
    state: PrecomputedState,
    latent: torch.Tensor,  # [N_runs, num_params] — continuous latent params
    lora_config: BinaryLoRAConfig,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Forward pass with STE quantization. Returns [N_runs] losses with grad."""
    # Quantize via STE, cast to model dtype
    quantized = ste_ternary(latent).to(state.hidden.dtype)
    m_Q = quantized[:, :lora_config.rank_q]
    m_V = quantized[:, lora_config.rank_q:]

    B = latent.shape[0]

    # LoRA deltas
    delta_Q, delta_V = apply_binary_lora_batched(
        state.proj_Q, state.proj_V, m_Q, m_V,
        lora_config.B_Q, lora_config.B_V, lora_config.alpha,
    )

    # Full Q, V, K
    Q = state.Q_base.unsqueeze(0) + delta_Q
    V = state.V_base.unsqueeze(0) + delta_V
    K = state.K.unsqueeze(0).expand(B, -1, -1)

    seq = state.seq_len
    nh = state.num_heads
    hd = state.head_dim

    Q = Q.view(B, seq, nh, hd).transpose(1, 2)
    K = K.view(B, seq, nh, hd).transpose(1, 2)
    V = V.view(B, seq, nh, hd).transpose(1, 2)

    # Attention
    scale = hd ** -0.5
    scores = (Q @ K.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(seq, seq, device=Q.device, dtype=torch.bool), diagonal=1)
    scores.masked_fill_(mask, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(Q.dtype)
    attn_out = attn_weights @ V
    attn_out = attn_out.transpose(1, 2).reshape(B, seq, nh * hd)

    # c_proj + residual
    h = state.hidden.unsqueeze(0) + attn_out @ state.c_proj_weight + state.c_proj_bias

    # LN2 + MLP + residual
    h = h + state.mlp(state.ln_2(h))

    # Final LN
    h = state.ln_f(h)

    # LM head — last position only
    h_last = h[:, -2, :]
    target = labels.squeeze(0)[-1]

    logits = h_last @ state.lm_head_weight.T
    losses = F.cross_entropy(logits, target.expand(B), reduction="none")
    return losses


# ---- MUON optimizer ----

class Muon(torch.optim.Optimizer):
    """MUON: Momentum with Orthogonalization via Newton-Schulz.

    For matrix-shaped momentum, applies polar decomposition approximation.
    For vector params, falls back to normalized momentum (equivalent behavior).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @staticmethod
    def _newton_schulz(G, steps=5):
        """Approximate the orthogonal factor of the polar decomposition."""
        # Only works for 2D tensors (matrices)
        if G.dim() != 2:
            return G / (G.norm() + 1e-8)

        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G / (G.norm() + 1e-7)
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        return X

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)

                # Orthogonalize: reshape to matrix if needed
                shape = buf.shape
                if buf.dim() == 1:
                    # Reshape vector to ~square matrix for Newton-Schulz
                    n = buf.shape[0]
                    # Find closest factorization
                    rows = int(n ** 0.5)
                    while n % rows != 0 and rows > 1:
                        rows -= 1
                    cols = n // rows
                    buf_mat = buf.view(rows, cols)
                    orth = self._newton_schulz(buf_mat, ns_steps)
                    update = orth.view(shape)
                else:
                    update = self._newton_schulz(buf.clone(), ns_steps)

                p.add_(update, alpha=-lr)


# ---- Benchmark runner ----

@dataclass
class BenchmarkResult:
    optimizer_name: str
    n_runs: int
    max_steps: int
    global_min_loss: float

    # Per-run results
    final_losses: np.ndarray    # [n_runs]
    best_losses: np.ndarray     # [n_runs] — best loss seen during run
    steps_to_best: np.ndarray   # [n_runs] — step at which best loss was found
    final_configs: np.ndarray   # [n_runs, num_params] — final ternary config

    @property
    def success_rate(self):
        """Fraction of runs that found the global optimum."""
        return float(np.mean(np.isclose(self.best_losses, self.global_min_loss, atol=0.01)))

    @property
    def mean_best_loss(self):
        return float(self.best_losses.mean())

    @property
    def median_best_loss(self):
        return float(np.median(self.best_losses))

    def summary(self):
        pcts = [10, 25, 50, 75, 90]
        pct_vals = np.percentile(self.best_losses, pcts)
        lines = [
            f"=== {self.optimizer_name} ===",
            f"  Runs: {self.n_runs}, Steps: {self.max_steps}",
            f"  Global min: {self.global_min_loss:.4f}",
            f"  Success rate (within 0.01): {self.success_rate:.2%}",
            f"  Best loss — mean: {self.mean_best_loss:.4f}, median: {self.median_best_loss:.4f}",
            f"  Best loss percentiles:",
        ]
        for p, v in zip(pcts, pct_vals):
            lines.append(f"    {p:>3d}th: {v:.4f}")
        lines.append(f"  Mean steps to best: {self.steps_to_best.mean():.1f}")
        return "\n".join(lines)


def run_optimizer_benchmark(
    optimizer_name: str,
    optimizer_cls,
    optimizer_kwargs: dict,
    state: PrecomputedState,
    lora_config: BinaryLoRAConfig,
    labels: torch.Tensor,
    num_params: int,
    n_runs: int = 10000,
    max_steps: int = 200,
    global_min_loss: float = 0.0,
    batch_chunk: int = 2048,  # process this many runs at a time (memory)
) -> BenchmarkResult:
    """Run an optimizer from random initializations and collect metrics."""

    device = state.hidden.device
    dtype = state.hidden.dtype
    all_final_losses = []
    all_best_losses = []
    all_steps_to_best = []
    all_final_configs = []

    for chunk_start in range(0, n_runs, batch_chunk):
        chunk_end = min(chunk_start + batch_chunk, n_runs)
        chunk_size = chunk_end - chunk_start

        # Random initialization: uniform in [-1.5, 1.5] so initial quantization covers all 3 values
        latent = torch.nn.Parameter(
            torch.empty(chunk_size, num_params, device=device, dtype=torch.float32).uniform_(-1.5, 1.5)
        )

        optimizer = optimizer_cls([latent], **optimizer_kwargs)

        best_losses = torch.full((chunk_size,), float("inf"), device=device)
        steps_to_best = torch.zeros(chunk_size, device=device, dtype=torch.long)

        for step in range(max_steps):
            optimizer.zero_grad()

            # Forward with STE
            losses = differentiable_forward(state, latent, lora_config, labels)
            total_loss = losses.sum()
            total_loss.backward()

            optimizer.step()

            # Track best
            with torch.no_grad():
                improved = losses < best_losses
                best_losses = torch.where(improved, losses, best_losses)
                steps_to_best = torch.where(improved, step, steps_to_best)

        # Final evaluation (clean, no grad)
        with torch.no_grad():
            final_losses = differentiable_forward(state, latent, lora_config, labels)
            final_configs = ste_ternary(latent)

        all_final_losses.append(final_losses.cpu().numpy())
        all_best_losses.append(best_losses.cpu().numpy())
        all_steps_to_best.append(steps_to_best.cpu().numpy())
        all_final_configs.append(final_configs.cpu().numpy())

        print(f"  {optimizer_name}: {chunk_end}/{n_runs} runs, "
              f"chunk best={best_losses.min():.4f}, median={best_losses.median():.4f}")

    return BenchmarkResult(
        optimizer_name=optimizer_name,
        n_runs=n_runs,
        max_steps=max_steps,
        global_min_loss=global_min_loss,
        final_losses=np.concatenate(all_final_losses),
        best_losses=np.concatenate(all_best_losses),
        steps_to_best=np.concatenate(all_steps_to_best),
        final_configs=np.concatenate(all_final_configs),
    )
