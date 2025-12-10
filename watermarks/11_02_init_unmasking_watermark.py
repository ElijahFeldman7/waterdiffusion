from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
import hashlib
from scipy.stats import norm

class SimpleRemaskWatermark:

    @staticmethod
    def pair_rule_follows(key: str, token_a: int, token_b: int) -> bool:
        a = int(token_a)
        b = int(token_b)
        lo, hi = (a, b) if a < b else (b, a)
        h = hashlib.sha256()
        h.update(key.encode('utf-8'))
        h.update(b'|')
        if a == b:
            h.update(str(a).encode('utf-8'))
        else:
            h.update(str(lo).encode('utf-8'))
            h.update(b'|')
            h.update(str(hi).encode('utf-8'))
        v = int.from_bytes(h.digest()[:2], 'big')
        bit = v & 1
        if a == b:
            return bit
        if a < b:
            return 1 if bit == 1 else 0
        return 1 if bit == 0 else 0

    def __init__(
        self,
        key: str = "",
        delta: float = 2.0,
        context_window: List[int] = None,
    ):
        self.key = key
        self.delta = delta
        self.context_window = context_window or [-2, -1, 1, 2]

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
        **kwargs
    ) -> torch.Tensor:
        modified_confidence = confidence.clone()
        batch_size, seq_len = confidence.shape
        for b in range(batch_size):
            for i in range(seq_len):
                proposed_token = int(x0[b, i].item())
                count = 0.0
                for j in self.context_window:
                    neighbor_index = i + j
                    if 0 <= neighbor_index < seq_len:
                        if mask_index[b, neighbor_index]:
                            other_token = int(x0[b, neighbor_index].item())
                            if j < 0:
                                if SimpleRemaskWatermark.pair_rule_follows(self.key, other_token, proposed_token): count += confidence[b, neighbor_index]
                            else:
                                if SimpleRemaskWatermark.pair_rule_follows(self.key, proposed_token, other_token): count += confidence[b, neighbor_index]
                        else:
                            other_token = int(current_text[b, neighbor_index].item())
                            if j < 0:
                                if SimpleRemaskWatermark.pair_rule_follows(self.key, other_token, proposed_token): count += 1
                            else:
                                if SimpleRemaskWatermark.pair_rule_follows(self.key, proposed_token, other_token): count += 1

                if mask_index[b, i]: modified_confidence[b, i] += count * self.delta
                else: modified_confidence[b, i] = -np.inf
                
        return modified_confidence
    
    def detect(
        self,
        text: str,
        tokenizer: Any,
        num_timestamps: Optional[int] = None,
        unmasked_tokens_per_timestamp: Optional[Union[int, float]] = None,
        **kwargs
    ) -> Dict[str, Any]:

        tokens_tensor = tokenizer.encode(text, return_tensors="pt")[0]
        tokens = tokens_tensor.cpu().tolist() if isinstance(tokens_tensor, torch.Tensor) else tokens_tensor

        satisfied_pairs = 0
        total_pairs_checked = 0
        seq_len = len(tokens)
        p_value_threshold = kwargs.get("p_value_threshold", 0.05)
        
        fpr_correction_factor = kwargs.get("fpr_correction_factor", 1.0)
        p_value_threshold_corrected = p_value_threshold / fpr_correction_factor

        seen_pairs = set()

        for i in range(seq_len):
            for j in self.context_window:
                neighbor_index = i + j
                if 0 <= neighbor_index < seq_len:
                    pair_key = tuple(sorted((i, neighbor_index)))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    total_pairs_checked += 1
                    if j < 0:
                        if SimpleRemaskWatermark.pair_rule_follows(self.key, tokens[neighbor_index], tokens[i]): satisfied_pairs += 1
                    if j > 0:
                        if SimpleRemaskWatermark.pair_rule_follows(self.key, tokens[i], tokens[neighbor_index]): satisfied_pairs += 1

        satisfaction_rate = satisfied_pairs / total_pairs_checked if total_pairs_checked > 0 else 0.0

        null_satis_rate = 0.5
        null_satis_rate_mean = 0.5
        std_dev = np.sqrt(total_pairs_checked * null_satis_rate * (1 - null_satis_rate))
        if std_dev == 0: z_score = 0.0
        else: z_score = (satisfied_pairs - total_pairs_checked * null_satis_rate_mean) / std_dev
        
        p_value = float(1 - norm.cdf(z_score))
        is_watermarked = p_value < p_value_threshold_corrected

        return {
            "is_watermarked": is_watermarked,
            "score": p_value,                
            "z_score": float(z_score),
            "satisfied_pairs": satisfied_pairs,
            "total_pairs_checked": total_pairs_checked,
            "satisfaction_rate": float(satisfaction_rate),
            "threshold": p_value_threshold_corrected, # Return the corrected threshold
            "original_threshold": p_value_threshold,
            "fpr_correction_factor": fpr_correction_factor,
            "num_timestamps": num_timestamps,
            "unmasked_tokens_per_timestamp": unmasked_tokens_per_timestamp,
        }