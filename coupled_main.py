"""Multi-GPU entry point for coupled MLP adapter landscape enumeration."""

import argparse
import os
import time
import torch
import torch.multiprocessing as mp

from coupled_config import CoupledConfig
from coupled_adapter import AdapterConfig
from coupled_data import load_eval_sequences
from coupled_enumeration import enumerate_coupled


def worker(gpu_id: int, cfg: CoupledConfig):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    print(f"[GPU {gpu_id}] Loading model...")
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(cfg.model_name, torch_dtype=cfg.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)

    print(f"[GPU {gpu_id}] Loading eval data...")
    input_ids, labels = load_eval_sequences(
        cfg.model_name, cfg.num_sequences, cfg.seq_len, seed=42
    )
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    print(f"[GPU {gpu_id}] Building adapters...")
    adapters = []
    for i, layer_idx in enumerate(cfg.adapter_layers):
        adapter = AdapterConfig(
            d_model=768,
            width=cfg.bottleneck_width,
            n_free=cfg.bits_per_adapter,
            alpha=cfg.adapter_alpha,
            seed=cfg.adapter_seed + i * 1000,  # different seed per adapter
            device=device,
            dtype=cfg.dtype,
        )
        adapters.append(adapter)

    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"[GPU {gpu_id}] Starting enumeration...")
    t0 = time.time()
    enumerate_coupled(
        gpu_id=gpu_id,
        num_gpus=cfg.num_gpus,
        model=model,
        adapters=adapters,
        input_ids=input_ids,
        labels=labels,
        cfg=cfg,
        output_path=cfg.output_dir,
    )
    elapsed = time.time() - t0
    print(f"[GPU {gpu_id}] Finished in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Coupled MLP adapter landscape enumeration")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-sequences", type=int, default=None)
    parser.add_argument("--test", action="store_true", help="Quick test on 1 GPU")
    args = parser.parse_args()

    cfg = CoupledConfig()
    cfg.num_gpus = args.gpus
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.alpha is not None:
        cfg.adapter_alpha = args.alpha
    if args.width is not None:
        cfg.bottleneck_width = args.width
    if args.num_sequences is not None:
        cfg.num_sequences = args.num_sequences

    # Configs per layer must be divisible by num_gpus
    assert cfg.configs_per_layer % cfg.num_gpus == 0, \
        f"256 configs_per_layer must be divisible by {cfg.num_gpus} GPUs"

    print(f"Coupled MLP Adapter Landscape Enumeration")
    print(f"  Adapters at layers: {cfg.adapter_layers}")
    print(f"  Bottleneck width: {cfg.bottleneck_width}")
    print(f"  Bits per adapter: {cfg.bits_per_adapter}")
    print(f"  Total params: {cfg.num_params}, configs: {cfg.total_configs:,}")
    print(f"  Sequences: {cfg.num_sequences} × {cfg.seq_len} tokens")
    print(f"  Alpha: {cfg.adapter_alpha}")
    print(f"  GPUs: {cfg.num_gpus}")

    if args.test:
        cfg.num_gpus = 1
        print(f"\nTest mode: 1 GPU")
        worker(0, cfg)
        return

    if cfg.num_gpus == 1:
        worker(0, cfg)
    else:
        mp.set_start_method("spawn", force=True)
        mp.spawn(worker, nprocs=cfg.num_gpus, args=(cfg,), join=True)


if __name__ == "__main__":
    main()
