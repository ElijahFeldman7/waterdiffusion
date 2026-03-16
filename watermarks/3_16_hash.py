from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
from scipy.stats import binom


class SimpleRemaskWatermark:

    @staticmethod
    @lru_cache(maxsize=1 << 22)
    def _fnv1a(data: bytes) -> int:
        FNV_PRIME = 0x01000193
        h = 0x811c9dc5
        for c in data:
            h ^= c
            h = (h * FNV_PRIME) & 0xFFFFFFFF
        return h

    @staticmethod
    @lru_cache(maxsize=1 << 22)
    def pair_rule_follows(
        key: str,
        token_a: int,
        token_b: int,
        context_sum: int,
    ) -> bool:
        a = int(token_a)
        b = int(token_b)
        lo, hi = (a, b) if a < b else (b, a)

        sep = b'\xff'
        data = (
            key.encode('utf-8') + sep +
            str(context_sum).encode('utf-8') + sep +
            str(lo).encode('utf-8') + sep +
            str(hi).encode('utf-8')
        )
        seed = SimpleRemaskWatermark._fnv1a(data)

        rng = np.random.default_rng(seed)
        bit = int(rng.integers(0, 2))

        if a == b:
            return bool(bit)
        if a < b:
            return bit == 1
        return bit == 0

    @staticmethod
    def compute_sum_hash(
        tokens: List[int],
        center: int,
        context_window: List[int],
        vocab_size: int,
    ) -> int:
        total = 0
        for offset in context_window:
            j = center + offset
            if 0 <= j < len(tokens):
                total += tokens[j]
        return total % vocab_size

    def __init__(
        self,
        key: str = "",
        delta: float = 2.0,
        context_window: List[int] = None,
        vocab_size: int = 32000,
    ):
        self.key = key
        self.delta = delta
        self.context_window = context_window or [-2, -1, 1, 2]
        self.vocab_size = vocab_size

    @torch.no_grad()
    def apply_watermark(
        self,
        logits: torch.Tensor,
        current_text: torch.Tensor,
        mask_index: torch.Tensor,
        step: int,
        confidence: torch.Tensor,
        x0: torch.Tensor,
        key: str,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len = confidence.shape

        confidence_clamped = torch.clamp(confidence.clone(), 0.0001, 0.9999)
        logits_conf = torch.log(confidence_clamped / (1 - confidence_clamped))
        watermark_bias = torch.zeros_like(confidence)

        weights = torch.where(mask_index, confidence, torch.ones_like(confidence))

        for b in range(batch_size):
            token_list = [
                int(x0[b, k].item()) if mask_index[b, k]
                else int(current_text[b, k].item())
                for k in range(seq_len)
            ]

            for i in range(seq_len):
                if not mask_index[b, i]:
                    continue

                proposed_token = token_list[i]
                count = 0.0

                for j in self.context_window:
                    neighbor_index = i + j
                    if 0 <= neighbor_index < seq_len:
                        other_token = token_list[neighbor_index]

                        context_sum = SimpleRemaskWatermark.compute_sum_hash(
                            token_list, neighbor_index,
                            self.context_window, self.vocab_size
                        )

                        if j < 0:
                            is_green = SimpleRemaskWatermark.pair_rule_follows(
                                self.key, other_token, proposed_token, context_sum
                            )
                        else:
                            is_green = SimpleRemaskWatermark.pair_rule_follows(
                                self.key, proposed_token, other_token, context_sum
                            )

                        if is_green:
                            count += weights[b, neighbor_index].item()

                watermark_bias[b, i] = count * self.delta

        modified_logits = torch.where(
            mask_index,
            logits_conf + watermark_bias,
            logits_conf,
        )

        modified_confidence = torch.sigmoid(modified_logits)

        modified_confidence = torch.where(
            mask_index,
            modified_confidence,
            torch.full_like(modified_confidence, -float('inf')),
        )

        return modified_confidence

    def detect(
        self,
        text: str,
        tokenizer: Any,
        num_timestamps: Optional[int] = None,
        unmasked_tokens_per_timestamp: Optional[Union[int, float]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        tokens_tensor = tokenizer.encode(text, return_tensors="pt")[0]
        tokens = (
            tokens_tensor.cpu().tolist()
            if isinstance(tokens_tensor, torch.Tensor)
            else tokens_tensor
        )

        satisfied_pairs = 0
        total_pairs_checked = 0
        seq_len = len(tokens)
        p_value_threshold = kwargs.get("p_value_threshold", 0.05)

        for i in range(seq_len):
            for j in self.context_window:
                neighbor_index = i + j
                if 0 <= neighbor_index < seq_len:
                    total_pairs_checked += 1

                    context_sum = SimpleRemaskWatermark.compute_sum_hash(
                        tokens, neighbor_index,
                        self.context_window, self.vocab_size
                    )

                    if j < 0:
                        if SimpleRemaskWatermark.pair_rule_follows(
                            self.key, tokens[neighbor_index], tokens[i], context_sum
                        ):
                            satisfied_pairs += 1
                    elif j > 0:
                        if SimpleRemaskWatermark.pair_rule_follows(
                            self.key, tokens[i], tokens[neighbor_index], context_sum
                        ):
                            satisfied_pairs += 1

        satisfaction_rate = (
            satisfied_pairs / total_pairs_checked
            if total_pairs_checked > 0 else 0.0
        )

        if total_pairs_checked == 0:
            p_value = 1.0
        else:
            p_value = float(
                1 - binom.cdf(satisfied_pairs - 1, total_pairs_checked, 0.5)
            )

        return {
            "is_watermarked": p_value < p_value_threshold,
            "score": p_value,
            "satisfied_pairs": satisfied_pairs,
            "total_pairs_checked": total_pairs_checked,
            "satisfaction_rate": float(satisfaction_rate),
            "original_threshold": p_value_threshold,
            "num_timestamps": num_timestamps,
            "unmasked_tokens_per_timestamp": unmasked_tokens_per_timestamp,
        }