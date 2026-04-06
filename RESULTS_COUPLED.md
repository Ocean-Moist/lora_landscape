# Coupled MLP Adapter Landscape: Results

## Motivation

The previous ternary LoRA landscape (3^19 = 1.16B configs) was **near-additive** — each parameter acted as an approximately independent switch, and AdamW solved it in 6 steps at 99.9% success. The additive structure came from: single frozen layer, independent random directions, single 8-token prompt, and weak perturbation. A useful optimizer benchmark needs genuine parameter interactions.

## Architecture

**Coupled MLP adapters** replace diagonal-gated LoRA with tiny MLP bottleneck adapters at 3 separated transformer layers:

```
Base model: GPT-2 Small (124M params, frozen)
Adapter: h_out = h_in + α · W2 @ ReLU(W1 @ h_in)
  W1: (4 × 768), W2: (768 × 4)
  8 free binary {-1, +1} parameters per adapter, split across W1 and W2
Insertion points: layers 2, 6, 11 (early / mid / late)
Total free params: 24 binary → 2^24 = 16,777,216 configurations
α = 2.0, free param scale = 10× init
```

**Why this creates interactions:** Parameters in W1 control feature extraction into the bottleneck; parameters in W2 control how bottleneck activations write back. W2's effect depends entirely on what W1 has done — genuine epistasis. Cross-layer coupling arises through the residual stream: layer 2's adapter modifies representations that layer 6 reads, which modifies what layer 11 sees.

**Dataset:** 4 sequences × 32 tokens from WikiText-2, cross-entropy loss at last token per sequence.

## Enumeration

**Layer-wise cached tree enumeration** exploits the adapter insertion structure:

1. Layers 0–1 run once → cached hidden state (128 tokens)
2. 256 layer-2 adapter configs → each run through layers 2–5 → 256 cached states
3. Per cached state: 256 layer-6 configs → layers 6–10 → 65,536 states
4. Per state: 256 layer-11 configs → layer 11 + LM head → 16.7M loss values

**Performance:** 16,777,216 configs in **55 seconds on 8×H100**. Each GPU processes 32 layer-2 configs (2M total configs per GPU). Last-token-per-sequence loss avoids materializing a 3.3GB logits tensor in the inner loop.

## Landscape Statistics

| Metric | Value |
|---|---|
| Total configs | 16,777,216 |
| Loss range | [8.656, 14.938] |
| Loss spread | 6.281 nats |
| Loss mean ± std | 11.59 ± 1.08 |
| Global minimum | 8.656 at config 16,621,610 |

## Interaction Diagnostics

### Singleton Variance Explained (Additive R²)

The additive model (each bit independently contributes to loss) explains **R² = 0.745** of variance. This means **25.5% of loss variance comes from parameter interactions** — a massive improvement over the near-additive LoRA landscape (R² ≈ 0.99).

Top singleton effects by magnitude:

| Bit | Effect | Adapter |
|---|---|---|
| 12 | 0.638 | Adapter 1 (layer 6) |
| 22 | 0.481 | Adapter 2 (layer 11) |
| 6 | 0.267 | Adapter 0 (layer 2) |
| 2 | 0.220 | Adapter 0 (layer 2) |
| 11 | 0.175 | Adapter 1 (layer 6) |
| 19 | 0.164 | Adapter 2 (layer 11) |

Influence spans ~270× (0.638 to 0.002) — even wider than the LoRA landscape's 400×.

### Pairwise Interactions

266 of 276 possible pairs have interaction strength above the median singleton effect.

Top interactions:

| Pair | Strength | Adapters |
|---|---|---|
| (12, 22) | **1.266** | Layer 6 × Layer 11 |
| (2, 6) | 0.732 | Layer 2 × Layer 2 |
| (11, 12) | 0.478 | Layer 6 × Layer 6 |
| (2, 12) | 0.427 | Layer 2 × Layer 6 |
| (4, 12) | 0.401 | Layer 2 × Layer 6 |

The strongest interaction (bits 12 and 22, cross-layer) is **2× larger than the strongest singleton effect**. This is genuine multiplicative coupling: the optimal value for a layer-11 parameter depends on what layer-6 has done.

Mean pairwise interaction: 0.077 (2× the median singleton of 0.037).

### Gradient-Sign Predictability

From 10,000 random starting points, computing one gradient and taking the sign:

| Metric | Value |
|---|---|
| Mean Hamming to optimum | **8.24 / 24** |
| Median Hamming | 8.0 |
| Exact match (Hamming = 0) | 0.00% |
| Within 1 flip | 0.04% |
| Within 3 flips | 0.99% |

A single gradient evaluation gets **one-third of the bits wrong**. The distribution peaks sharply at 8 wrong bits — gradient-based methods need multi-step refinement, not one-shot sign-reading.

### Local Minima

| Metric | Value |
|---|---|
| Found in 2M samples | 15 |
| Estimated total | **~125** |
| Best local min (non-global) | 8.664 |
| Worst local min | 10.805 |

The landscape has two clusters of local minima:
- **Near-optimal cluster** (8.66–8.88): 9 minima within 0.22 of global min
- **High-loss cluster** (10.67–10.80): 6 minima trapped at 2+ nats above global min

## Optimizer Benchmarking

### Setup

All methods given 50 steps/evaluations. Gradient methods use STE (straight-through estimator) for binary quantization. Success = within 0.01 of global minimum (8.656).

### Results

| Optimizer | Success% | Median Best | Mean Steps |
|---|---|---|---|
| **Greedy Local Search** | **81.20%** | **8.664** | 20.7 |
| Random Search (50 evals) | 0.00% | 9.273 | 24.3 |
| SGD (lr=0.5, mom=0.9) | 0.00% | 9.102 | 32.5 |
| SGD (lr=2.0, mom=0.9) | 0.00% | 9.102 | 27.1 |
| SGD (lr=10, mom=0.9) | 0.00% | 9.102 | 25.3 |
| AdamW (lr=0.05) | 0.00% | 9.219 | 39.4 |
| AdamW (lr=0.5) | 0.00% | 9.102 | 27.8 |
| AdamW (lr=5.0) | 0.00% | 9.102 | 28.5 |
| AdamW (lr=50) | 0.00% | 9.102 | 19.9 |
| Muon (lr=0.1) | 0.00% | 9.336 | 36.6 |
| Muon (lr=1.0) | 0.00% | 9.125 | 30.0 |
| Muon (lr=10) | 0.00% | 9.133 | 28.7 |

### Analysis

**Greedy local search dominates.** By evaluating all 24 single-bit neighbors at each step and moving to the best improvement, it reaches the global minimum 81% of the time in ~21 steps. The remaining 19% gets trapped in the high-loss local minima cluster (~10.7).

**All gradient methods fail completely.** Every STE-based optimizer — SGD, AdamW, Muon across all learning rates — converges to a median loss of 9.10, approximately 0.45 above the global minimum. This represents a specific local basin that gradient-based methods cannot escape.

**Why gradient methods fail:**

1. **STE noise at interaction boundaries.** The straight-through estimator passes gradients through `sign()` as if it were identity. This works when the landscape is additive (each gradient component independently tells you the right direction). With strong interactions, the STE gradient at a point reflects the *local* loss slope, not the *interaction-adjusted* optimal direction. Flipping bit 12 may be beneficial only if bit 22 is simultaneously flipped — but gradients can't represent this conditional structure.

2. **The 9.10 basin.** All gradient methods converge to nearly the same loss regardless of learning rate. This suggests a large, flat basin in continuous latent space that maps to a suboptimal discrete configuration. The STE quantization boundary creates a barrier that smooth gradient descent cannot cross — it would need to simultaneously move multiple latent dimensions past their sign-flip thresholds, which requires coordinated movement that first-order methods don't perform.

3. **AdamW's advantage is gone.** In the additive landscape, AdamW's per-parameter scaling handled the 400× influence spread. Here, scaling doesn't help because the difficulty isn't magnitude mismatch — it's conditional dependencies between parameters. The right value for one parameter depends on others, and no per-coordinate adaptive rule captures this.

4. **Muon is worst.** Newton-Schulz orthogonalization, which normalizes the gradient direction, actively destroys the magnitude information needed to navigate even the additive component of the landscape.

**Why greedy search succeeds:**

Greedy local search evaluates actual discrete neighbor losses (no STE approximation). At each step, it sees the true effect of flipping each bit, including all interaction effects. The 24-wide neighborhood is small enough to exhaustively evaluate. The landscape has a clear basin of attraction: from most starting points, the greedy path leads to the global minimum in ~21 steps through the near-optimal minima cluster.

## Comparison: Additive vs Coupled Landscapes

| Property | Additive (LoRA) | Coupled (MLP) |
|---|---|---|
| Parameterization | A @ diag(m) @ B | W2 @ ReLU(W1 @ h) |
| Layers | 1 (layer 11) | 3 (layers 2, 6, 11) |
| Total params | 19 ternary | 24 binary |
| Search space | 1.16B | 16.8M |
| Additive R² | ~0.99 | 0.745 |
| Local minima | ~1 | ~125 |
| Gradient-sign Hamming | ~0 | 8.0 |
| Best gradient optimizer | AdamW (99.9% success, 6 steps) | None (0% success) |
| Best overall method | AdamW | Greedy Local Search (81%) |

## Implications

1. **STE is the bottleneck, not the optimizer.** All gradient methods hit the same wall regardless of optimizer sophistication. The lossy STE approximation converts a non-additive discrete landscape into a smooth surrogate that doesn't preserve the interaction structure. Better optimizers on this smooth surrogate don't help.

2. **Combinatorial search outperforms gradient descent.** When the landscape has genuine parameter interactions, exhaustive local evaluation (24 neighbors per step) beats sophisticated gradient-based optimization. The overhead of evaluating discrete neighbors is small for low-dimensional problems.

3. **Multi-layer coupling creates real difficulty.** The strongest pairwise interaction (1.27) is between bits in different adapters (layers 6 and 11). This cross-layer conditional dependence is what gradient methods fundamentally cannot navigate — it requires information that isn't in any single-point gradient.

4. **The benchmark produces meaningful optimizer separation.** Unlike the additive landscape where all methods clustered between 90-100% success, the coupled landscape separates methods into distinct tiers: greedy search (81%), gradient methods (0% but decent local optima), random search (0%, poor optima).

## Enumeration Details

- **Time:** 55 seconds on 8×H100 GPUs
- **Throughput:** ~305K configs/sec/GPU
- **Storage:** 8 × 2M × 2 bytes = 32MB total (vs 2.3GB for ternary LoRA)
- **Benchmark time:** ~136s per optimizer configuration (5000 runs × 50 steps)
