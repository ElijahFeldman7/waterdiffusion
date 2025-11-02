from typing import Any, Dict, List

import torch
import numpy as np
import hashlib
from scipy.stats import norm

from watermarks.watermark import BaseDiffusionWatermark
from watermarks.utils import pair_rule_follows


class SimpleRemaskWatermark(BaseDiffusionWatermark):

    def pair_rule_follows(key: str, token_a: int, token_b: int) -> bool:
        a = int(token_a)
        b = int(token_b)
        if a == b:
            return False
        lo, hi = (a, b) if a < b else (b, a)
        h = hashlib.sha256()
        h.update(key.encode('utf-8'))
        h.update(b'|')
        h.update(str(lo).encode('utf-8'))
        h.update(b'|')
        h.update(str(hi).encode('utf-8'))
        v = int.from_bytes(h.digest()[:2], 'big')
        bit = v & 1
        if a < b:
            return bit == 1
        else:
            return bit == 0



    def __init__(
        self,
        config: Dict[str, Any],
        key: str = "",
        delta: float = 2.0,
        context_window: List[int] = None,
    ):
        super().__init__(config)
        self.key = key
        self.delta = delta
        self.context_window = context_window or [-2, -1, 1, 2]
    
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
                count = 0
                for j in self.context_window:
                    neighbor_index = i + j
                    if 0 <= neighbor_index < seq_len and not mask_index[b, neighbor_index]:
                        other_token = int(current_text[b, neighbor_index].item())
                        if j < 0 and pair_rule_follows(self.key, other_token, proposed_token):
                            count += 1
                        if j > 0 and pair_rule_follows(self.key, proposed_token, other_token):
                            count += 1

                if mask_index[b, i]:
                    modified_confidence[b, i] += count * self.delta
                else:
                    modified_confidence[b, i] = -np.inf
        return modified_confidence
    
    def detect(
        self,
        text: str,  
        tokenizer: Any,
        **kwargs
    ) -> Dict[str, Any]:

        tokens_tensor = tokenizer.encode(text, return_tensors="pt")[0]
        tokens = tokens_tensor.cpu().tolist() if isinstance(tokens_tensor, torch.Tensor) else tokens_tensor

        satisfied_pairs = 0
        total_pairs_checked = 0
        seq_len = len(tokens)
        p_value_threshold = kwargs.get("p_value_threshold", 0.05)

        for i in range(seq_len):
            for j in self.context_window:
                neighbor_index = i + j
                if 0 <= neighbor_index < seq_len:
                    total_pairs_checked += 1
                    if j < 0 and pair_rule_follows(self.key, tokens[neighbor_index], tokens[i]):
                        satisfied_pairs += 1
                    if j > 0 and pair_rule_follows(self.key, tokens[i], tokens[neighbor_index]):
                        satisfied_pairs += 1

        if total_pairs_checked == 0:
            return {
                "is_watermarked": False,
                "score": 1.0,
                "z_score": 0.0,
                "satisfied_pairs": 0,
                "total_pairs_checked": 0,
                "satisfaction_rate": 0.0,
                "threshold": p_value_threshold,
                "error": "No relevant token pairs found in text",
            }

        satisfaction_rate = satisfied_pairs / total_pairs_checked
        expected_rate = 0.5
        std_error = np.sqrt(expected_rate * (1 - expected_rate) / total_pairs_checked)
        z_score = (satisfaction_rate - expected_rate) / std_error if std_error > 0 else 0.0

        p_value = float(1 - norm.cdf(z_score))
        is_watermarked = p_value < p_value_threshold

        return {
            "is_watermarked": is_watermarked,
            "score": p_value,
            "z_score": float(z_score),
            "satisfied_pairs": satisfied_pairs,
            "total_pairs_checked": total_pairs_checked,
            "satisfaction_rate": float(satisfaction_rate),
            "threshold": p_value_threshold,
        }
        
    
