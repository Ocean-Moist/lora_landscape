# Exhaustive Loss Landscape Enumeration for Optimizer Benchmarking

## Overview

We build exact ground-truth loss landscapes for GPT-2 by exhaustively evaluating every possible configuration of small adapter modules. Three progressively harder landscapes test what makes optimization difficult in transformers.

| Landscape | Params | Search Space | Enumeration Time | Key Finding |
|---|---|---|---|---|
| Additive LoRA | 19 ternary | 1.16B | 4 min | AdamW solves in 6 steps (too easy) |
| Coupled Binary MLP | 24 binary | 16.8M | 55 sec | All gradient methods fail; greedy search wins |
| Continuous MLP | 6 continuous | 1.07B grid | 43 min | AdamW finds certified optimum (63% success) |

All experiments on 8×H100 80GB. Base model: GPT-2 Small (124M params, frozen).

---

## Landscape 1: Additive Ternary LoRA

### Setup

- **Parameterization**: `A @ diag(m) @ B` on layer 11's Q and V projections
- **A, B**: frozen random unit-norm projections; **m**: ternary {-1, 0, +1}
- **Rank**: Q=10, V=9 → 19 free parameters, α=25.0
- **Loss**: CE on last-token prediction, single 8-token prompt
- **Configs**: 3^19 = 1,162,261,467

### Landscape Properties

| Metric | Value |
|---|---|
| Loss range | [4.047, 6.371] (2.32 nat spread) |
| Additive R² | ~0.99 |
| Local minima | ~1 |
| Parameter influence range | 440× (0.353 to 0.0008) |

The landscape is **near-additive**: each parameter acts as an approximately independent switch. Top 5 of 19 parameters account for most loss variation.

### Optimizer Results (10K runs, 200 steps)

| Optimizer | Success% | Notes |
|---|---|---|
| AdamW (lr=50) | **99.9%** | 6 steps. Per-param scaling handles 440× influence spread |
| SGD (lr=100, mom=0.9) | 89.5% | Hard ceiling — can't normalize across magnitudes |
| Muon (lr=1.0) | 24.1% | Orthogonalization destroys magnitude info |

**Verdict:** Too easy. Near-additive structure reduces optimization to 19 independent classification problems. A single gradient evaluation suffices.

---

## Landscape 2: Coupled Binary MLP Adapters

### Setup

MLP bottleneck adapters at 3 separated layers create multiplicative coupling:

```
Adapter: h_out = h_in + α · W2 @ ReLU(W1 @ h_in)
W1: (4×768), W2: (768×4), 8 free binary params split across W1/W2
Layers: 2, 6, 11 — cross-layer coupling via residual stream
Total: 24 binary params → 2^24 = 16,777,216 configs
α = 2.0, free param scale = 10× init
Dataset: 4 sequences × 32 tokens from WikiText-2
```

Enumeration uses **tree-structured layer caching**: layers 0-1 cached once, then 256 layer-2 configs branch into 256 layer-6 configs branch into 256 layer-11 configs. Each GPU handles a subtree.

### Interaction Diagnostics

| Metric | Additive LoRA | Coupled MLP | Target |
|---|---|---|---|
| Additive R² | ~0.99 | **0.745** | < 0.7 |
| Pairwise interactions > median singleton | — | **266 / 276** | > 50 |
| Gradient-sign Hamming to optimum | ~0 | **8.0 / 24** | > 3 |
| Local minima (estimated) | ~1 | **~125** | > 100 |

The strongest pairwise interaction (bits 12 and 22, cross-layer between adapters at layers 6 and 11) has strength **1.266** — twice the strongest singleton effect (0.638). The landscape has two clusters of local minima: a near-optimal cluster (8.66-8.88) and a high-loss trap cluster (10.67-10.80).

### Optimizer Results (5K runs, 50 steps)

| Optimizer | Success% | Median Best |
|---|---|---|
| **Greedy Local Search** | **81.2%** | **8.664** |
| Random Search (50 evals) | 0.0% | 9.273 |
| SGD (all LRs) | 0.0% | 9.102 |
| AdamW (all LRs) | 0.0% | 9.102 |
| Muon (all LRs) | 0.0% | 9.125–9.336 |

**Complete inversion from Landscape 1.** Every gradient method (SGD, AdamW, Muon) across all learning rates converges to the same 9.10 basin — 0.45 nats above the global minimum. Greedy bit-flip search, which evaluates actual discrete neighbors, succeeds 81% of the time.

**Why gradient methods fail:** The STE passes gradients through `sign()` as identity. This works when parameters are independent (Landscape 1) but fails when the optimal value for bit 12 depends on bit 22. STE gradients reflect local slopes, not interaction-adjusted directions. All gradient methods hit the same wall because they're optimizing the same smooth STE surrogate, which doesn't preserve the discrete interaction structure.

**Why greedy search succeeds:** It evaluates true discrete neighbor losses, seeing all interaction effects. The 24-wide neighborhood is small enough to exhaustively check each step.

---

## Landscape 3: Continuous MLP Adapters (with Lipschitz Certification)

### Setup

Same adapter architecture as Landscape 2, but with **continuous** free parameters:

```
Same 3 adapters at layers 2, 6, 11
2 free continuous params per adapter (1 in W1, 1 in W2)
Total: 6 continuous params in [-1, 1]
Grid: 32 points per dim → 32^6 = 1,073,741,824 configs
```

### Lipschitz Certification

Since a grid can miss narrow basins between points, we certify proximity to the true global minimum:

1. **Grid evaluation**: 1.07B configs in 43 minutes on 8×H100
2. **Grid minimum**: 10.289 (found at params [0.81, -0.81, 1.0, -0.94, 0.68, -1.0])
3. **Local refinement**: 12^6 = 2.99M points around minimum → improved to 10.297
4. **Empirical Lipschitz**: sampled 50K gradient norms, max = 2.01, with 1.5× safety margin = 3.01
5. **Certification gap**: Lip × max_grid_distance = 3.01 × 0.079 = **0.023**

**Result: true global min ∈ [10.274, 10.297]** — certified within 0.023 nats.

The analytical Lipschitz bound (product of layer spectral norms) gives 3.03×10^72 — astronomically loose, illustrating why rigorous certification of neural network optima is essentially impossible at scale.

### Landscape Properties

| Metric | Value |
|---|---|
| Loss range | [10.289, 12.766] (2.48 nat spread) |
| Grid spacing | 0.0645 |
| Certification gap | 0.023 |
| Most variable slice | L6_W1 vs L6_W2 (1.90 nat range) |

2D slices through the 6D landscape at the optimum are smooth — approximately bilinear (one param in W1, one in W2). The continuous landscape lacks the sharp features of the discrete case: no local minima, no quantization barriers, no STE noise.

### Optimizer Results (2K runs, 50 steps, parallelized across 8 GPUs)

| Optimizer | Success% | Median Best | Min Best | Avg Steps |
|---|---|---|---|---|
| **AdamW (lr=0.5)** | **63.3%** | **10.297** | **10.281** | **18.7** |
| **AdamW (lr=0.1)** | **63.4%** | **10.297** | **10.289** | **26.7** |
| SGD (lr=0.1, mom=0.9) | 59.8% | 10.320 | 10.289 | 34.8 |
| SGD (lr=0.01, mom=0.9) | 7.6% | 10.531 | 10.297 | 45.7 |
| AdamW (lr=0.01) | 2.5% | 10.656 | 10.289 | 47.6 |
| SGD (lr=0.01) | 0.1% | 10.820 | 10.320 | 44.7 |

**AdamW recovers.** With exact continuous gradients (no STE), AdamW finds the certified optimum 63% of the time in 19 steps. SGD with momentum is competitive at high LR (60%). Low learning rates universally fail — insufficient movement in 50 steps.

---

## Cross-Landscape Comparison

| | Additive LoRA | Coupled Binary | Continuous |
|---|---|---|---|
| **What makes it hard** | Nothing (additive) | STE × interactions | Low LR / few steps |
| **Best method** | AdamW (99.9%) | Greedy search (81.2%) | AdamW (63.4%) |
| **Gradient methods** | Excellent | Complete failure | Good |
| **Key bottleneck** | Magnitude spread | Discrete interactions | Step budget |

## Conclusions

### 1. STE is the critical failure point, not landscape topology

The coupled binary landscape has genuine interactions (R²=0.745, 125 local minima, cross-layer epistasis). But making the same landscape continuous eliminates all difficulty for gradient methods. The complexity that defeats optimizers isn't in the loss surface geometry — it's in the STE approximation that converts a non-additive discrete problem into a smooth surrogate that doesn't preserve interaction structure.

### 2. Adaptive learning rates matter for heterogeneous influence

Across all three landscapes, AdamW's per-parameter scaling consistently outperforms SGD when parameter influence spans orders of magnitude. Muon's orthogonalization consistently hurts by erasing this magnitude information.

### 3. Combinatorial search beats gradients for discrete problems

When the search space is discrete and has genuine interactions, exhaustive neighborhood evaluation (greedy local search) outperforms all gradient-through-STE methods. This is relevant for quantization-aware training, neural architecture search, and MoE routing.

### 4. Certification is feasible at small scale, impossible at large scale

At 6 dimensions, we certify the global minimum within 0.023 nats using empirical Lipschitz bounds. The analytical bound (3×10^72) shows that rigorous certification via spectral norm products is useless even for tiny modifications to a 124M-param model. For real LLMs with billions of parameters, we fundamentally cannot verify proximity to the global optimum. Scaling laws (more compute = lower loss, always) suggest current training is demonstrably not near optimal.

### 5. Continuous transformer landscapes are smooth

The 2D slices through our 6D continuous landscape are smooth bowls and saddles — no sharp features, no local minima. This is consistent with the overparameterization picture: the loss surface through a small number of adapter parameters is well-conditioned because the base model already provides a good operating point. The interesting optimization challenges in real training come from scale (navigating 10^11 dimensions), not from landscape pathology in any low-dimensional subspace.

## Reproduction

All code at: https://github.com/Ocean-Moist/lora_landscape

```bash
# Additive LoRA (4 min on 8×H100)
uv run python enumerate_main.py --gpus 8

# Coupled binary (55 sec on 8×H100)  
uv run python coupled_main.py --gpus 8

# Continuous grid (43 min on 8×H100)
uv run python continuous_main.py --gpus 8

# Certification
uv run python continuous_certify.py --input-dir results_continuous --gpu 0

# Benchmarks
uv run python benchmark_main.py --gpu 0
uv run python coupled_benchmark.py --input-dir results_coupled --gpu 0
uv run python continuous_benchmark.py --input-dir results_continuous --num-gpus 8
```
