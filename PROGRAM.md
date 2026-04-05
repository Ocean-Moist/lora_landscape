# Exhaustive Landscape Enumeration via Binary LoRA: A Ground-Truth Benchmark for Optimizer Research

## Motivation

Optimizer research suffers from a fundamental problem: we never know the global optimum. When we compare Adam, MUON, SGD, or any new optimizer, we measure wall-clock-to-target-loss — but the target is arbitrary, and we have no idea how far any optimizer is from the true best solution. Without ground truth, we can't distinguish "this optimizer is good" from "this optimizer is good enough that we stopped looking."

Exhaustive enumeration solves this. If we can evaluate every possible parameter configuration and record every loss value, we get the complete landscape — every global optimum, every local minimum, every basin, every saddle point. We can then benchmark optimizers against absolute ground truth.

The challenge is making the enumerable space realistic. A toy problem with known optima is useless if its landscape doesn't resemble real training. We need a space that is simultaneously small enough to enumerate and shaped by a real model's geometry.

## Key Insight: LoRA as a Low-Dimensional Slice Through a Real Landscape

Take a pretrained transformer and freeze all parameters. Attach a LoRA adapter with a small number of binary parameters. The loss landscape over those parameters is not synthetic — it is a genuine cross-section of the pretrained model's loss surface. The curvature, feature interactions, and conditioning are all inherited from the real transformer. The optimizer interacts with this landscape using the exact same update rules it would use in full training. AdamW on 40 LoRA parameters is the same algorithm as AdamW on 400 million parameters.

This means optimizer rankings measured on the enumerable slice have a plausible path to transferring to full-scale training, because the landscape geometry and the optimizer mechanics are both real.

## Method

### Step 1: Setup

- Select a pretrained transformer (e.g., a small GPT-2 or similar).
- Freeze all model parameters.
- Attach a LoRA adapter to one or two weight matrices (e.g., the attention Q and V projections in a single layer).
- Constrain LoRA parameters to binary values {-1, +1}, with rank chosen to yield ~40 total parameters.
- Select a downstream task with a fixed small evaluation set (e.g., a classification task with a few hundred examples).

### Step 2: Exhaustive Enumeration

- Enumerate all 2^40 ≈ 10^12 binary configurations.
- For each configuration, run a forward pass on the evaluation set and record the loss.
- Store the complete mapping: configuration → loss.

#### Compute Budget

Each forward pass through a small transformer on a small batch costs roughly 10^6 — 10^7 FLOPs. At 10^12 configurations, total compute is ~10^18 — 10^19 FLOPs. Eight H100s provide ~10^19 FLOPs in a few hours. This is tight but feasible, with exact runtime depending on model size, batch size, and evaluation set.

Parallelization is straightforward: configurations are independent, so the workload distributes trivially across GPUs with no communication overhead.

### Step 3: Landscape Analysis

With the complete loss map in hand, characterize the landscape:

- **Global optima.** How many configurations achieve the minimum loss? Are they clustered or scattered?
- **Basin structure.** Define basins by local connectivity (configurations reachable by single bit flips that monotonically decrease loss). How many basins exist? How large are they? How deep?
- **Connectivity.** Can you walk between global optima via low-loss paths (single bit flips that stay below a threshold)? This tests whether the overparameterization-induced "connected basin" phenomenon exists at this scale.
- **Hamming distance structure.** How does loss correlate with Hamming distance from the nearest optimum? Is there a smooth basin around good solutions, or is the landscape totally uncorrelated?
- **Local minimum depth.** How many local minima exist and how far are they from global optima in loss value? This determines whether optimizers that escape local minima have an advantage.

### Step 4: Optimizer Benchmarking

Run each optimizer from many random initializations (e.g., 10,000 runs each). Measure:

- **Success rate.** Fraction of runs that reach a global optimum.
- **Steps to optimum.** Among successful runs, how many update steps were required.
- **Best loss found.** Among unsuccessful runs, how close did the optimizer get.
- **Basin coverage.** Which basins does each optimizer tend to find? Do some optimizers reliably find the deepest basin while others get trapped in shallow ones?

#### Adapting Continuous Optimizers to Binary Weights

The LoRA parameters are binary, but the optimizers are continuous. Use the straight-through estimator: maintain continuous latent variables, compute gradients with respect to them, apply the optimizer's update rule in continuous space, but snap to binary (sign function) for the forward pass. This is standard practice for binary neural networks and preserves the optimizer's full mechanics — momentum, adaptive learning rates, weight decay — while operating over a discrete search space.

#### Optimizers to Test

- SGD with momentum
- AdamW
- MUON
- Any new or experimental optimizer

All optimizers receive the same gradient signal (straight-through on binary weights) and the same hyperparameter tuning budget.

#### Baselines

- Random search (sample configurations uniformly, keep the best)
- Greedy local search (start at a random configuration, flip the bit that most improves loss, repeat until stuck)
- Simulated annealing
- Evolutionary strategies (population of configurations, mutate + select)

These establish whether gradient-based optimizers provide any advantage over gradient-free methods on this landscape.

### Step 5: Scaling Validation

This is the critical step. Increase the LoRA rank to yield more parameters:

| Parameters | Quantization | Configurations | Enumerable? |
|---|---|---|---|
| 40 | Binary | ~10^12 | Yes |
| 60 | Binary | ~10^18 | Borderline |
| 100 | Binary | ~10^30 | No |
| 256 | Continuous (float16) | ∞ | No |

At each scale, run the same optimizer comparison. At 40 parameters, you have ground truth. At 100+, you don't — you only have relative comparisons between optimizers.

**The key question: do optimizer rankings at 40 parameters predict rankings at 256 parameters?**

If yes, you've built a cheap, ground-truth proxy for optimizer evaluation. Test new optimizers at the enumerable scale, predict their large-scale behavior, confirm with a few expensive runs.

If no, document where and why the rankings diverge. This is itself valuable — it tells you which properties of an optimizer matter only at scale and which are already visible in small problems.

## What This Can and Cannot Tell You

### What it can tell you

- Which optimizers are best at navigating a real transformer loss landscape cross-section with verified ground truth.
- Whether gradient-based optimizers meaningfully outperform gradient-free search on this landscape.
- Whether the landscape has the basin structure, connectivity, and overparameterization properties that theory predicts.
- Whether optimizer rankings transfer across scale within the LoRA setting.

### What it cannot tell you

- Whether LoRA landscape geometry is representative of full-parameter training. LoRA constrains updates to a low-rank subspace, which may have systematically different curvature than full weight space. Optimizer rankings on LoRA may not predict rankings on full fine-tuning.
- Whether binary weight landscapes are representative of continuous weight landscapes. The straight-through estimator introduces bias, and the discrete landscape may have different basin structure than the continuous one. The ternary and 4-level cases should also be explored.
- Whether results on a single layer's LoRA transfer to multi-layer LoRA or full model adaptation.

### Mitigations

- Test multiple LoRA attachment points (different layers, different matrices) to check consistency.
- Test binary, ternary, and 4-level quantization to assess sensitivity to weight resolution.
- Compare LoRA optimizer rankings against known results from full-scale training to validate transfer.

## Why Not Other Approaches?

### Busy Beaver

The Busy Beaver function has known optima from exhaustive enumeration, but its landscape is structurally alien to NN training: no data distribution, no stochasticity, no overparameterization, no smooth basins. Optimizer rankings on BB would not transfer to real networks.

### Synthetic Loss Functions

Constructing a function with transformer-like Hessian spectra gives the right static geometry but misses the dynamic coevolution of the landscape and the optimizer during training. The landscape changes as weights update — a static synthetic function can't capture this.

### NAS-Bench

NAS-Bench exhaustively enumerates architectures, not weights. It benchmarks architecture search methods, not weight optimizers. The optimizer (typically AdamW) is held fixed. It answers a different question.

### Small Networks Trained from Scratch

A 40-parameter MLP trained on XOR has an enumerable weight space but its landscape bears no resemblance to transformer training. The curvature structure, conditioning, and basin geometry are totally different. Optimizer rankings would not transfer.

## Deliverables

1. **Complete loss landscape dataset.** All 2^40 configurations and their losses for a specific (model, LoRA config, task) triple. Released publicly for reproducibility.
2. **Landscape analysis.** Basin count, basin sizes, connectivity structure, Hamming distance correlations, local minimum census.
3. **Optimizer rankings with ground truth.** Success rates, convergence speeds, and basin preferences for each optimizer, benchmarked against known global optima.
4. **Scaling transfer analysis.** Whether rankings at 40 parameters predict rankings at 100+ parameters.
5. **Open source code.** Enumeration pipeline, landscape analysis tools, optimizer benchmarking harness.