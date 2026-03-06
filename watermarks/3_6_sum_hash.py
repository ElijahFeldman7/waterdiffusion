from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
import hashlib
from scipy.stats import norm, binom

class SimpleRemaskWatermark:

    @staticmethod
    def pair_rule_follows(key_int: int, token_a: int, token_b: int, vocab_size: int) -> bool:
        hash_val = (int(token_a) + int(token_b) + key_int) % vocab_size
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
        
        self.key_int = int(hashlib.sha256(key.encode('utf-8')).hexdigest(), 16) % self.vocab_size

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
        
        x0_probs = torch.softmax(logits, dim=-1)
        
        v_indices = torch.arange(self.vocab_size, device=confidence.device)
        green_mask = ((v_indices + self.key_int) % 2) == 0
        
        for b in range(batch_size):
            for i in range(seq_len):
                if not mask_index[b, i]:
                    modified_confidence[b, i] = -np.inf
                    continue
                
                prob_dist_i = x0_probs[b, i] 
                count = 0.0
                
                for j in self.context_window:
                    neighbor_index = i + j
                    if 0 <= neighbor_index < seq_len:
                        
                        if mask_index[b, neighbor_index]:
                            prob_dist_neighbor = x0_probs[b, neighbor_index]
                            weight = confidence[b, neighbor_index].item()
                        else:
                            known_token = int(current_text[b, neighbor_index].item())
                            prob_dist_neighbor = torch.zeros(self.vocab_size, device=confidence.device)
                            prob_dist_neighbor[known_token] = 1.0
                            weight = 1.0
                        fft_i = torch.fft.fft(prob_dist_i)
                        fft_neighbor = torch.fft.fft(prob_dist_neighbor)
                        
                        sum_distribution = torch.fft.ifft(fft_i * fft_neighbor).real
                        
                        prob_is_green = torch.sum(sum_distribution[green_mask]).item()
                        
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