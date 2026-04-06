"""Load real text sequences for coupled landscape evaluation."""

import torch
from transformers import GPT2Tokenizer


def load_eval_sequences(model_name: str = "gpt2", num_sequences: int = 16,
                        seq_len: int = 64, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Load tokenized sequences from wikitext for evaluation.

    Returns:
        input_ids: [num_sequences, seq_len] — context tokens
        labels: [num_sequences, seq_len] — shifted targets
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Use a deterministic text source — just use the tokenizer on a known corpus
    # We'll grab text from wikitext via datasets
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Concatenate all text and tokenize
    all_text = "\n".join([t for t in ds["text"] if len(t.strip()) > 0])
    all_tokens = tokenizer.encode(all_text)

    # Extract non-overlapping sequences of length seq_len + 1
    # (need +1 for shifted labels)
    stride = seq_len + 1
    sequences = []
    for i in range(0, len(all_tokens) - stride, stride):
        sequences.append(all_tokens[i:i + stride])
        if len(sequences) >= num_sequences * 4:
            break

    # Deterministic shuffle and select
    import random
    rng = random.Random(seed)
    rng.shuffle(sequences)
    sequences = sequences[:num_sequences]

    tokens = torch.tensor(sequences, dtype=torch.long)  # [N, seq_len+1]
    input_ids = tokens[:, :-1]  # [N, seq_len]
    labels = tokens[:, 1:]      # [N, seq_len]

    return input_ids, labels
