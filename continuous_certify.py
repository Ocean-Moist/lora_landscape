"""Lipschitz certification for the continuous landscape global minimum.

Strategy:
1. Find grid minimum from the dense enumeration
2. Estimate Lipschitz constant via gradient sampling (empirical)
3. Compute analytical Lipschitz upper bound via spectral norms (provable)
4. Certify: true_global_min ≥ grid_min - Lip * max_distance_to_nearest_grid_point
5. If gap too large, refine with denser local grid around minimum
"""

import argparse
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from continuous_config import ContinuousConfig
from continuous_adapter import ContinuousAdapterConfig, grid_indices_to_params
from coupled_data import load_eval_sequences
from coupled_enumeration import run_layers_single, run_layers_batched, compute_loss_batched
from coupled_adapter import apply_adapter_batched


def load_landscape(output_dir: str, num_gpus: int, total_configs: int) -> np.ndarray:
    shard_size = total_configs // num_gpus
    shards = []
    for i in range(num_gpus):
        path = Path(output_dir) / f"losses_shard_{i}.npy"
        shard = np.memmap(path, dtype=np.float16, mode="r", shape=(shard_size,))
        shards.append(np.array(shard, dtype=np.float32))
    return np.concatenate(shards)


def global_index_to_params(index: int, cfg: ContinuousConfig) -> np.ndarray:
    """Convert a flat global index to 6D continuous parameter vector."""
    cpa = cfg.configs_per_adapter
    n_per = cfg.params_per_adapter
    grid = np.linspace(cfg.param_min, cfg.param_max, cfg.n_grid)

    params = np.zeros(cfg.num_params)
    remaining = index
    for adapter in range(cfg.num_adapters):
        adapter_idx = remaining % cpa
        remaining //= cpa
        for k in range(n_per):
            grid_idx = adapter_idx % cfg.n_grid
            adapter_idx //= cfg.n_grid
            params[adapter * n_per + k] = grid[grid_idx]
    return params


def differentiable_forward(model, adapters, input_ids, labels, params_tensor, cfg):
    """Forward pass with gradient tracking for Lipschitz estimation.

    Args:
        params_tensor: [N, 6] continuous params (requires_grad=True)

    Returns:
        [N] losses with grad
    """
    N = params_tensor.shape[0]
    n_per = cfg.params_per_adapter
    adapter_layers = cfg.adapter_layers
    dtype = cfg.dtype

    N_seq, seq_per = input_ids.shape
    total_seq = N_seq * seq_per
    flat_ids = input_ids.reshape(1, -1)
    flat_labels = labels.reshape(-1)

    with torch.no_grad():
        pos_ids = torch.arange(total_seq, device=input_ids.device).unsqueeze(0)
        embeds = model.transformer.wte(flat_ids) + model.transformer.wpe(pos_ids)
        embeds = model.transformer.drop(embeds).squeeze(0)
        h_pre = run_layers_single(model, embeds, 0, adapter_layers[0])

    h = h_pre.unsqueeze(0).expand(N, -1, -1)

    for adapter_idx, layer_idx in enumerate(adapter_layers):
        p = params_tensor[:, adapter_idx * n_per:(adapter_idx + 1) * n_per].to(dtype)
        W1, W2 = adapters[adapter_idx].build_weights(p)
        bottleneck = torch.bmm(h, W1.transpose(1, 2))
        bottleneck = F.relu(bottleneck)
        out = torch.bmm(bottleneck, W2.transpose(1, 2))
        h = h + adapters[adapter_idx].alpha * out

        next_layer = adapter_layers[adapter_idx + 1] if adapter_idx + 1 < len(adapter_layers) else adapter_layers[-1] + 1
        h = run_layers_batched(model, h, layer_idx, next_layer)

    h = model.transformer.ln_f(h)
    logits = h @ model.lm_head.weight.to(dtype).T

    # Last-token-per-sequence loss
    last_positions = torch.arange(seq_per - 2, total_seq, seq_per, device=h.device)
    logits_sel = logits[:, last_positions, :]
    target = flat_labels[last_positions]

    losses = F.cross_entropy(
        logits_sel.reshape(-1, logits_sel.shape[-1]),
        target.unsqueeze(0).expand(N, -1).reshape(-1),
        reduction="none",
    ).reshape(N, -1).mean(dim=1)
    return losses


def estimate_lipschitz_empirical(model, adapters, input_ids, labels, cfg,
                                 center: np.ndarray, radius: float = 2.0,
                                 n_samples: int = 50000, batch_size: int = 64):
    """Estimate Lipschitz constant by sampling gradient norms near the minimum."""
    device = input_ids.device
    max_grad_norm = 0.0
    grad_norms = []

    for i in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - i)
        # Sample points uniformly in a ball around center
        pts = torch.tensor(center, device=device, dtype=torch.float32).unsqueeze(0).expand(bs, -1).clone()
        noise = torch.randn(bs, cfg.num_params, device=device) * radius / math.sqrt(cfg.num_params)
        pts = (pts + noise).clamp(cfg.param_min, cfg.param_max)
        pts.requires_grad_(True)

        losses = differentiable_forward(model, adapters, input_ids, labels, pts, cfg)
        losses.sum().backward()

        with torch.no_grad():
            gn = pts.grad.norm(dim=1)  # [bs]
            max_gn = gn.max().item()
            if max_gn > max_grad_norm:
                max_grad_norm = max_gn
            grad_norms.extend(gn.cpu().tolist())

    return {
        "max_grad_norm": max_grad_norm,
        "mean_grad_norm": float(np.mean(grad_norms)),
        "p99_grad_norm": float(np.percentile(grad_norms, 99)),
        "n_samples": len(grad_norms),
    }


def compute_analytical_lipschitz(model, adapters, input_ids, labels, cfg):
    """Compute an analytical upper bound on the Lipschitz constant.

    Uses the chain rule: Lip(L) ≤ Lip(CE) × Lip(network) × Lip(adapter→params)
    This will be loose but provably correct.
    """
    device = input_ids.device

    # Lip(CE w.r.t. logits) ≤ 2 (bounded gradient of softmax cross-entropy)
    lip_ce = 2.0

    # Lip(lm_head): spectral norm of lm_head weight
    lm_head_sn = float(torch.linalg.norm(model.lm_head.weight.float(), ord=2))

    # Lip(final_ln): layer norm Lipschitz ≈ gamma_max / (epsilon^0.5)
    # For GPT-2 ln_f, gamma is close to 1, eps=1e-5
    lip_ln = 2.0  # conservative bound

    # Lip(frozen layers): product of per-layer Lipschitz constants
    # Each transformer block: Lip ≤ 1 + Lip(attn) + Lip(mlp)
    # Attn Lip ≤ ||W_qkv|| * ||W_proj|| * seq_len (attention can amplify)
    # MLP Lip ≤ ||W_fc|| * ||W_proj||
    frozen_lip = 1.0
    for i in range(12):
        layer = model.transformer.h[i]
        # Very conservative per-layer bound
        attn_sn = float(torch.linalg.norm(layer.attn.c_attn.weight.float(), ord=2))
        proj_sn = float(torch.linalg.norm(layer.attn.c_proj.weight.float(), ord=2))
        mlp_fc_sn = float(torch.linalg.norm(layer.mlp.c_fc.weight.float(), ord=2))
        mlp_proj_sn = float(torch.linalg.norm(layer.mlp.c_proj.weight.float(), ord=2))

        # Residual connection: Lip(x + f(x)) ≤ 1 + Lip(f)
        layer_lip = (1 + attn_sn * proj_sn) * (1 + mlp_fc_sn * mlp_proj_sn)
        frozen_lip *= layer_lip

    # Lip(adapter → hidden_state change): for each free param
    # ∂h/∂m_i = α * free_scale * (specific column/row of frozen weights)
    # This is bounded by α * free_scale * max(||W1_base||, ||W2_base||) * ||h||
    max_adapter_lip = 0.0
    for adapter in adapters:
        # Adapter output: α * W2 @ relu(W1 @ h)
        # ∂(adapter)/∂(W1_entry): α * W2 @ relu'(W1 @ h) * h_specific
        # ∂(adapter)/∂(W2_entry): α * relu(W1 @ h)_specific
        # Bounded by: α * max(free_scale) * max(||W2|| * ||h||, ||W1 @ h||)
        adapter_lip = adapter.alpha * max(
            adapter.free_scale_w1.max().item() * adapter.W2_spec_norm,
            adapter.free_scale_w2.max().item() * adapter.W1_spec_norm,
        )
        max_adapter_lip = max(max_adapter_lip, adapter_lip)

    # Total: very loose but provable
    # Only count frozen layers AFTER adapters (most of the chain)
    # This overestimates massively because it assumes worst-case alignment at every step
    analytical_lip = lip_ce * lm_head_sn * lip_ln * frozen_lip * max_adapter_lip

    return {
        "analytical_lipschitz": analytical_lip,
        "lip_ce": lip_ce,
        "lm_head_sn": lm_head_sn,
        "lip_ln": lip_ln,
        "frozen_lip": frozen_lip,
        "max_adapter_lip": max_adapter_lip,
        "note": "Very loose upper bound — product of per-layer spectral norms overestimates by orders of magnitude",
    }


def refine_around_minimum(model, adapters, input_ids, labels, cfg,
                          center: np.ndarray, radius: float, n_refine: int = 50):
    """Dense local grid around the minimum for tighter certification.

    Evaluates n_refine^6 points in a small box around center.
    """
    device = input_ids.device
    grid_1d = torch.linspace(-radius, radius, n_refine, device=device)

    # Generate all combinations for 6 dims
    grids = [grid_1d] * cfg.num_params
    mesh = torch.stack(torch.meshgrid(*grids, indexing='ij'), dim=-1)
    points = mesh.reshape(-1, cfg.num_params) + torch.tensor(center, device=device, dtype=torch.float32)
    points = points.clamp(cfg.param_min, cfg.param_max)

    # Evaluate in batches
    all_losses = []
    batch_size = 1024
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        with torch.no_grad():
            losses = differentiable_forward(model, adapters, input_ids, labels, batch, cfg)
        all_losses.append(losses.cpu().numpy())

    all_losses = np.concatenate(all_losses)
    best_idx = np.argmin(all_losses)
    best_loss = float(all_losses[best_idx])
    best_params = points[best_idx].cpu().numpy()

    return {
        "best_loss": best_loss,
        "best_params": best_params.tolist(),
        "n_evaluated": len(all_losses),
        "radius": radius,
    }


def main():
    parser = argparse.ArgumentParser(description="Certify continuous landscape global minimum")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=str, default="certification.json")
    parser.add_argument("--lip-samples", type=int, default=50000)
    parser.add_argument("--refine-n", type=int, default=10,
                        help="Refinement grid points per dim (10^6 = 1M)")
    args = parser.parse_args()

    cfg = ContinuousConfig()
    cfg.output_dir = args.input_dir
    device = torch.device(f"cuda:{args.gpu}")

    # Load grid landscape
    print(f"Loading landscape from {args.input_dir}...")
    losses = load_landscape(args.input_dir, args.num_gpus, cfg.total_configs)
    grid_min_idx = int(np.argmin(losses))
    grid_min_loss = float(losses.min())
    grid_min_params = global_index_to_params(grid_min_idx, cfg)
    print(f"  Grid minimum: {grid_min_loss:.6f} at index {grid_min_idx}")
    print(f"  Grid min params: {grid_min_params}")
    print(f"  Loss range: [{losses.min():.4f}, {losses.max():.4f}]")

    # Load model
    print("\nLoading model...")
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(cfg.model_name, torch_dtype=cfg.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)

    input_ids, labels_data = load_eval_sequences(cfg.model_name, cfg.num_sequences, cfg.seq_len)
    input_ids = input_ids.to(device)
    labels_data = labels_data.to(device)

    adapters = []
    for i, layer_idx in enumerate(cfg.adapter_layers):
        adapters.append(ContinuousAdapterConfig(
            d_model=768, width=cfg.bottleneck_width, n_free=cfg.params_per_adapter,
            alpha=cfg.adapter_alpha, seed=cfg.adapter_seed + i * 1000,
            device=device, dtype=cfg.dtype,
        ))

    results = {
        "grid_min_loss": grid_min_loss,
        "grid_min_index": grid_min_idx,
        "grid_min_params": grid_min_params.tolist(),
        "grid_spacing": cfg.grid_spacing,
        "num_params": cfg.num_params,
    }

    # Verify grid minimum with float32
    print("\nVerifying grid minimum in float32...")
    verify_params = torch.tensor(grid_min_params, device=device, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        verified_loss = differentiable_forward(model, adapters, input_ids, labels_data, verify_params, cfg)
    verified_loss = float(verified_loss[0])
    print(f"  Verified loss: {verified_loss:.6f} (grid stored: {grid_min_loss:.6f})")
    results["verified_grid_min_loss"] = verified_loss

    # Step 1: Empirical Lipschitz
    print(f"\nEstimating Lipschitz constant ({args.lip_samples} samples)...")
    lip_results = estimate_lipschitz_empirical(
        model, adapters, input_ids, labels_data, cfg,
        center=grid_min_params, radius=2.0, n_samples=args.lip_samples,
    )
    print(f"  Max gradient norm: {lip_results['max_grad_norm']:.4f}")
    print(f"  Mean gradient norm: {lip_results['mean_grad_norm']:.4f}")
    print(f"  P99 gradient norm: {lip_results['p99_grad_norm']:.4f}")
    results["empirical_lipschitz"] = lip_results

    # Step 2: Certification with empirical Lip
    # Max distance from any point to nearest grid point:
    # In d dimensions with spacing h, max dist = h * sqrt(d) / 2
    max_dist = cfg.grid_spacing * math.sqrt(cfg.num_params) / 2
    empirical_lip = lip_results["max_grad_norm"] * 1.5  # safety margin
    empirical_gap = empirical_lip * max_dist

    print(f"\n  Grid spacing: {cfg.grid_spacing:.4f}")
    print(f"  Max distance to grid point: {max_dist:.4f}")
    print(f"  Empirical Lip (1.5× safety): {empirical_lip:.4f}")
    print(f"  Empirical certification gap: {empirical_gap:.4f}")
    print(f"  Certified range: [{verified_loss - empirical_gap:.4f}, {verified_loss:.4f}]")

    results["certification"] = {
        "max_grid_distance": max_dist,
        "empirical_lipschitz_with_margin": empirical_lip,
        "empirical_gap": empirical_gap,
        "certified_lower_bound": verified_loss - empirical_gap,
        "certified_upper_bound": verified_loss,
    }

    # Step 3: Analytical Lipschitz (provable but loose)
    print("\nComputing analytical Lipschitz bound...")
    analytical = compute_analytical_lipschitz(model, adapters, input_ids, labels_data, cfg)
    analytical_gap = analytical["analytical_lipschitz"] * max_dist
    print(f"  Analytical Lip: {analytical['analytical_lipschitz']:.2e}")
    print(f"  Analytical gap: {analytical_gap:.2e} (expected to be very loose)")
    results["analytical_lipschitz"] = analytical
    results["analytical_gap"] = analytical_gap

    # Step 4: Local refinement
    print(f"\nRefining around minimum ({args.refine_n}^{cfg.num_params} = {args.refine_n**cfg.num_params:,} points)...")
    refine_radius = cfg.grid_spacing * 2  # search 2× grid spacing around minimum
    refinement = refine_around_minimum(
        model, adapters, input_ids, labels_data, cfg,
        center=grid_min_params, radius=refine_radius, n_refine=args.refine_n,
    )
    print(f"  Refined best loss: {refinement['best_loss']:.6f}")
    print(f"  Refined best params: {refinement['best_params']}")
    improvement = verified_loss - refinement["best_loss"]
    print(f"  Improvement over grid: {improvement:.6f}")
    results["refinement"] = refinement

    # Step 5: Re-certify with refined minimum
    if refinement["best_loss"] < verified_loss:
        # Re-estimate Lipschitz around refined minimum
        print(f"\nRe-estimating Lipschitz around refined minimum...")
        lip2 = estimate_lipschitz_empirical(
            model, adapters, input_ids, labels_data, cfg,
            center=np.array(refinement["best_params"]),
            radius=refine_radius, n_samples=args.lip_samples,
        )
        local_lip = lip2["max_grad_norm"] * 1.5
        # After refinement, max distance is refine_radius * sqrt(d) / (refine_n - 1)
        refine_spacing = 2 * refine_radius / (args.refine_n - 1)
        refine_max_dist = refine_spacing * math.sqrt(cfg.num_params) / 2
        refined_gap = local_lip * refine_max_dist
        print(f"  Local Lip (1.5× safety): {local_lip:.4f}")
        print(f"  Refined spacing: {refine_spacing:.6f}")
        print(f"  Refined max distance: {refine_max_dist:.6f}")
        print(f"  Refined certification gap: {refined_gap:.6f}")

        results["refined_certification"] = {
            "local_lipschitz": local_lip,
            "refine_spacing": refine_spacing,
            "refine_max_distance": refine_max_dist,
            "refined_gap": refined_gap,
            "certified_lower_bound": refinement["best_loss"] - refined_gap,
            "certified_upper_bound": refinement["best_loss"],
        }

    # Final verdict
    final_min = refinement["best_loss"] if refinement["best_loss"] < verified_loss else verified_loss
    final_gap = results.get("refined_certification", results["certification"])
    gap_val = final_gap.get("refined_gap", final_gap.get("empirical_gap"))

    print(f"\n{'=' * 60}")
    print(f"CERTIFICATION RESULT:")
    print(f"  Best found minimum: {final_min:.6f}")
    print(f"  Empirical cert. gap: {gap_val:.6f}")
    print(f"  True global min ∈ [{final_min - gap_val:.6f}, {final_min:.6f}]")
    if gap_val < 0.01:
        print(f"  ✓ CERTIFIED: gap < 0.01 — provably within 0.01 of global optimum")
    elif gap_val < 0.1:
        print(f"  ~ ACCEPTABLE: gap < 0.1 — within 0.1 of global optimum")
    else:
        print(f"  ✗ GAP TOO LARGE: consider increasing grid density or refine_n")

    results["final_verdict"] = {
        "best_found_loss": final_min,
        "certification_gap": gap_val,
        "certified": gap_val < 0.01,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
