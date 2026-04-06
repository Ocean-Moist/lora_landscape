"""Multi-GPU entry point for exhaustive binary LoRA enumeration."""

import argparse
import torch
import torch.multiprocessing as mp

from config import Config
from binary_lora import BinaryLoRAConfig
from model_setup import load_frozen_model, prepare_eval_data, PrecomputedState
from enumeration import enumerate_gpu
from storage import LossStorage


def worker(gpu_id: int, cfg: Config):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    print(f"[GPU {gpu_id}] Loading model...")
    model = load_frozen_model(device, cfg.dtype, cfg.model_name)

    input_ids, labels = prepare_eval_data(cfg.model_name, cfg.eval_text, cfg.seq_len)
    labels = labels.to(device)

    lora_cfg = BinaryLoRAConfig(
        d_model=768,
        rank_q=cfg.lora_rank_q,
        rank_v=cfg.lora_rank_v,
        alpha=cfg.lora_alpha,
        seed=cfg.lora_seed,
        device=device,
        dtype=cfg.dtype,
    )

    print(f"[GPU {gpu_id}] Precomputing layer 0-{cfg.lora_layer - 1} activations...")
    state = PrecomputedState(model, input_ids, lora_cfg, cfg.lora_layer, device, cfg.dtype)

    # Free the full model — we only need the precomputed state
    del model
    torch.cuda.empty_cache()

    storage = LossStorage(cfg.output_dir, gpu_id, cfg.total_configs, cfg.num_gpus)

    print(f"[GPU {gpu_id}] Starting enumeration...")
    enumerate_gpu(
        gpu_id=gpu_id,
        num_gpus=cfg.num_gpus,
        state=state,
        lora_config=lora_cfg,
        labels=labels,
        storage=storage,
        config_batch_size=cfg.config_batch_size,
        num_params=cfg.num_params,
        total_configs=cfg.total_configs,
    )
    print(f"[GPU {gpu_id}] Done.")


def main():
    parser = argparse.ArgumentParser(description="Exhaustive binary LoRA enumeration")
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--batch-size", type=int, default=4096, help="Configs per batch")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--test", action="store_true", help="Quick test: 10 batches on 1 GPU")
    args = parser.parse_args()

    cfg = Config()
    cfg.num_gpus = args.gpus
    cfg.config_batch_size = args.batch_size
    if args.output_dir:
        cfg.output_dir = args.output_dir

    if args.test:
        cfg.num_gpus = 1
        # Override total_configs for quick test
        cfg.total_configs = cfg.config_batch_size * 10
        print(f"Test mode: {cfg.total_configs} configs on 1 GPU")
        worker(0, cfg)
        return

    if cfg.num_gpus == 1:
        worker(0, cfg)
    else:
        mp.set_start_method("spawn", force=True)
        mp.spawn(worker, nprocs=cfg.num_gpus, args=(cfg,), join=True)


if __name__ == "__main__":
    main()
