"""Load pretrained GPT-2, freeze, and precompute everything before the LoRA layer."""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from binary_lora import BinaryLoRAConfig


def load_frozen_model(device: torch.device, dtype: torch.dtype, model_name: str = "gpt2") -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def prepare_eval_data(model_name: str = "gpt2", eval_text: str = "The quick brown fox jumps over the lazy", seq_len: int = 8):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokens = tokenizer.encode(eval_text)
    if len(tokens) < seq_len + 1:
        raise ValueError(f"eval_text must tokenize to at least {seq_len + 1} tokens, got {len(tokens)}")
    input_ids = torch.tensor(tokens[:seq_len], dtype=torch.long).unsqueeze(0)  # [1, seq_len]
    labels = torch.tensor(tokens[1:seq_len + 1], dtype=torch.long).unsqueeze(0)  # [1, seq_len]
    return input_ids, labels


class PrecomputedState:
    """Cached intermediate activations from layers 0..(lora_layer-1).

    By caching the hidden state entering the LoRA layer and the base Q/K/V,
    we only need to run from the LoRA layer forward for each config.
    """
    def __init__(
        self,
        model: GPT2LMHeadModel,
        input_ids: torch.Tensor,
        lora_config: BinaryLoRAConfig,
        lora_layer: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        with torch.no_grad():
            # Use model's native forward to get hidden states at each layer
            # output_hidden_states[i] = output of layer i-1 (0 = embedding output)
            outputs = model.transformer(
                input_ids.to(device),
                output_hidden_states=True,
            )
            # hidden_states[0] = embedding, [1] = after layer 0, ..., [lora_layer] = after layer lora_layer-1
            hidden = outputs.hidden_states[lora_layer]  # [1, seq, 768]
            self.hidden = hidden.squeeze(0)  # [seq, 768]

            # Extract the LoRA layer's attention weights
            layer = model.transformer.h[lora_layer]
            attn = layer.attn

            # GPT-2 c_attn is Conv1D: output = input @ weight + bias
            # weight: [768, 2304], bias: [2304]
            ln1_out = layer.ln_1(hidden).squeeze(0)  # [seq, 768]

            qkv = ln1_out @ attn.c_attn.weight + attn.c_attn.bias  # [seq, 2304]
            d = attn.c_attn.weight.shape[0]  # 768
            self.Q_base = qkv[:, :d]        # [seq, 768]
            self.K = qkv[:, d:2*d]           # [seq, 768]
            self.V_base = qkv[:, 2*d:]       # [seq, 768]

            # Precompute projections for LoRA: hidden @ A
            self.proj_Q = ln1_out @ lora_config.A_Q  # [seq, rank_q]
            self.proj_V = ln1_out @ lora_config.A_V  # [seq, rank_v]

            # Store layer weights needed for the rest of the forward pass
            self.c_proj_weight = attn.c_proj.weight  # [768, 768]
            self.c_proj_bias = attn.c_proj.bias      # [768]
            self.ln_2 = layer.ln_2
            self.mlp = layer.mlp
            self.ln_f = model.transformer.ln_f
            self.lm_head_weight = model.lm_head.weight  # [vocab, 768]

            self.num_heads = attn.num_heads  # 12
            self.head_dim = d // attn.num_heads  # 64
            self.seq_len = input_ids.shape[1]
