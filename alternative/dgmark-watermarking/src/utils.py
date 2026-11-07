"""
Utility functions for watermarking text generation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, Optional, Union


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits

    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def top_k_sampling_with_logging(logits: torch.Tensor, k: int, tokenizer=None) -> torch.Tensor:
    B, L, V = logits.shape

    values, indices = torch.topk(logits, k=k, dim=-1)  # [B, L, k]
    probs = F.softmax(values, dim=-1)                  # [B, L, k]

    sampled_idx = torch.multinomial(probs.view(-1, k), num_samples=1).squeeze(-1)  # [B * L]
    chosen_tokens = indices.view(-1, k)[torch.arange(B * L, device=logits.device), sampled_idx]
    chosen_tokens = chosen_tokens.view(B, L)  # [B, L]

    return chosen_tokens


def log_topk_for_selected(sampled_x0: torch.Tensor, logits: torch.Tensor, k: int,
                         tokenizer, transfer_index: torch.Tensor) -> None:

    B, L = sampled_x0.shape
    _, indices = torch.topk(logits, k=k, dim=-1)
    values = torch.gather(logits, -1, indices)
    probs = F.softmax(values, dim=-1)

    for b in range(B):
        for l in range(L):
            if transfer_index[b, l]:
                print(f"\n[Transfer Position b={b}, pos={l}]")
                for rank in range(k):
                    token_id = indices[b, l, rank].item()
                    prob = probs[b, l, rank].item()
                    token_str = tokenizer.decode([token_id]) if tokenizer else f"<{token_id}>"
                    print(f"  - Top-{rank+1}: Token {token_str} (id={token_id}), prob={prob:.4f}")
                chosen_id = sampled_x0[b, l].item()
                chosen_str = tokenizer.decode([chosen_id]) if tokenizer else f"<{chosen_id}>"
                print(f"  -> Selected token: {chosen_str} (id={chosen_id})")


def calculate_watermark_stats(generated_ids: list, prompt_len: int, eot_token_id: int = 126081,
                            private_key: Optional[Union[int, str]] = None) -> Tuple[int, int, float, int]:

    # Find EOT cutoff point
    try:
        cutoff_index = generated_ids.index(eot_token_id)
        trimmed_ids = generated_ids[:cutoff_index + 1]
        print(f"Cutoff at index: {cutoff_index} / Total generated: {len(generated_ids)}")
    except ValueError:
        trimmed_ids = generated_ids
        print(f"{eot_token_id} not found. Using full sequence of length {len(generated_ids)}.")

    # Calculate watermark matches using key-based compliance
    matched_count = 0
    for j, token_id in enumerate(trimmed_ids):
        real_pos = prompt_len + j + 1  
        if check_watermark_compliance(real_pos, token_id, private_key):
            matched_count += 1

    match_ratio = matched_count / len(trimmed_ids) if trimmed_ids else 0.0

    return matched_count, len(generated_ids), match_ratio, len(trimmed_ids)


def generate_key_sequence(private_key: Union[int, str], max_length: int = 10000) -> list:

    if isinstance(private_key, str):
        seed = sum(ord(c) for c in private_key)
    else:
        seed = int(private_key)

    # Use key as random seed for reproducibility
    rng = random.Random(seed)

    return [rng.randint(0, 1) for _ in range(max_length)]


_key_sequence_cache = {}


def get_key_based_parity(position: int, private_key: Optional[Union[int, str]] = None) -> int:

    if private_key is None:
        return position % 2

    cache_key = str(private_key)
    if cache_key not in _key_sequence_cache:
        _key_sequence_cache[cache_key] = generate_key_sequence(private_key, max_length=10000)

    key_sequence = _key_sequence_cache[cache_key]

    sequence_index = (position - 1) % len(key_sequence)
    return key_sequence[sequence_index]


def check_watermark_compliance(position: int, token_id: int, private_key: Optional[Union[int, str]] = None) -> bool:

    expected_parity = get_key_based_parity(position, private_key)

    if private_key is None:
        token_parity = token_id % 2
    else:
        cache_key = str(private_key)
        if cache_key not in _key_sequence_cache:
            _key_sequence_cache[cache_key] = generate_key_sequence(private_key, max_length=10000)

        key_sequence = _key_sequence_cache[cache_key]
        sequence_index = (position - 1) % len(key_sequence)
        key_bit = key_sequence[sequence_index]

        base_parity = token_id % 2
        token_parity = base_parity ^ key_bit  

    return expected_parity == token_parity