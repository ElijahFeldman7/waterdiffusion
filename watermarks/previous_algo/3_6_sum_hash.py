from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
from scipy.stats import norm, binom

class SimpleRemaskWatermark:
    def fast_pair_hash(self, i: int, j: int, vocab_size: int) -> int:
        pair_key = (min(i, j) * vocab_size + max(i, j)) & 0xFFFFFFFF
        h = pair_key ^ 0x811c9dc5
        h = (h * 0x01000193) & 0xFFFFFFFF
        h ^= h >> 16
        h = (h * 0x85ebca6b) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 0xc2b2ae35) & 0xFFFFFFFF
        h ^= h >> 16
        return h % vocab_size
    def _fast_key_hash(self, key: str, vocab_size: int) -> int:
        h = 0x811c9dc5
        for c in key.encode():
            h ^= c
            h = (h * 0x01000193) & 0xFFFFFFFF
        return h % vocab_size
    @staticmethod
    def pair_rule_follows(i: int, j: int, token_a: int, token_b: int, vocab_size: int, key_int: int) -> bool:
        pair_hash = self.fast_pair_hash(i, j, vocab_size)
        hash_val = (token_a + token_b + pair_hash + key_int) % vocab_size
        return (hash_val % 2) == 0
    def __init__(
        self,
        key: str = "",
        delta: float = 2.0,
        context_window: List[int] = None,
        vocab_size: int = 157184,
    ):
        self.key = key
        self.delta = delta
        self.context_window = context_window or[-2, -1, 1, 2]
        self.vocab_size = vocab_size
        self.key_int = self._fast_key_hash(key, vocab_size)

    @torch.no_grad()
    def apply_watermark(self, logits: torch.Tensor, current_text: torch.Tensor,
                   mask_index: torch.Tensor, step: int, confidence: torch.Tensor,
                   x0: torch.Tensor, **kwargs) -> torch.Tensor:
    
        modified_confidence = confidence.clone()
        batch_size, seq_len = confidence.shape
        x0_probs = torch.softmax(logits, dim=-1)
    
        for b in range(batch_size):
            for i in range(seq_len):
                if not mask_index[b, i]:
                    modified_confidence[b, i] = -np.inf
                    continue
            
                prob_dist_i = x0_probs[b, i] 
                count = 0.0
            
                for j_offset in self.context_window:
                    j = i + j_offset
                    if 0 <= j < seq_len:
                        if mask_index[b, j]:
                            neighbor_probs = x0_probs[b, j]
                            weight = confidence[b, j].item()
                        else:
                            known_token = int(current_text[b, j].item())
                            neighbor_probs = torch.zeros(self.vocab_size, device=confidence.device)
                            neighbor_probs[known_token] = 1.0
                            weight = 1.0
                    
                        v_indices = torch.arange(self.vocab_size, device=confidence.device)
                        green_mask = torch.zeros(self.vocab_size, device=confidence.device)
                    
                        for v_idx in v_indices:
                            pair_hash = self.fast_pair_hash(i, j, self.vocab_size)
                            hash_val_base = (v_idx.item() + pair_hash + self.key_int) % self.vocab_size
                        
                            neighbor_tokens = torch.arange(self.vocab_size, device=confidence.device)
                            neighbor_hash_vals = (v_idx.item() + neighbor_tokens + pair_hash + self.key_int) % self.vocab_size
                            is_green = (neighbor_hash_vals % 2) == 0
                        
                            green_mask[v_idx] = torch.sum(neighbor_probs[is_green]).item()
                    
                        fft_i = torch.fft.fft(prob_dist_i)
                        fft_green = torch.fft.fft(green_mask)
                        prob_is_green = torch.real(torch.fft.ifft(fft_i * fft_green)).sum().item()
                    
                        count += (prob_is_green * weight)       
                prob = confidence[b, i].item()
                prob = max(min(prob, 0.9999), 0.0001)
                logit = np.log(prob / (1 - prob))
                watermark_bias = count * self.delta
                modified_logit = logit + watermark_bias
                modified_prob = 1 / (1 + np.exp(-modified_logit))
                modified_confidence[b, i] = modified_prob
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
        
        for i in range(seq_len):
            for j in self.context_window:
                neighbor_index = i + j
                if 0 <= neighbor_index < seq_len:
                    total_pairs_checked += 1
                    if SimpleRemaskWatermark.pair_rule_follows(
                        self.key_int, tokens[neighbor_index], tokens[i], self.vocab_size
                    ): 
                        satisfied_pairs += 1

        satisfaction_rate = satisfied_pairs / total_pairs_checked if total_pairs_checked > 0 else 0.0

        null_satis_rate = 0.5
        if total_pairs_checked == 0: 
            p_value = 1.0 
        else: 
            p_value = float(1 - binom.cdf(satisfied_pairs - 1, total_pairs_checked, null_satis_rate))
        
        is_watermarked = p_value < p_value_threshold

        return {
            "is_watermarked": is_watermarked,
            "score": p_value,                
            "satisfied_pairs": satisfied_pairs,
            "total_pairs_checked": total_pairs_checked,
            "satisfaction_rate": float(satisfaction_rate),
            "original_threshold": p_value_threshold,
            "num_timestamps": num_timestamps,
            "unmasked_tokens_per_timestamp": unmasked_tokens_per_timestamp,
        }