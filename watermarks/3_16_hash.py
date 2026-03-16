import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union
from scipy.stats import binom

class SimpleRemaskWatermark:
    def __init__(
        self,
        key: str = "",
        delta: float = 2.0,
        context_window: List[int] = None,
        vocab_size: int = 157184,
    ):
        self.vocab_size = vocab_size
        self.key_int = self._fast_key_hash(key, vocab_size)
        self.delta = delta
        self.context_window = context_window or [-2, -1, 1, 2]

    def _fast_key_hash(self, key: str, vocab_size: int) -> int:
        h = 0x811c9dc5
        for c in key.encode():
            h ^= c
            h = (h * 0x01000193) & 0xFFFFFFFF
        return h % vocab_size

    def _get_base_hash(self, a: int, b: int) -> int:
        val_min, val_max = (a, b) if a <= b else (b, a)
        combined = (val_min * self.vocab_size + val_max) ^ self.key_int
        h = (combined * 0x45d9f3b) & 0xFFFFFFFF
        h = ((h >> 16) ^ h) * 0x45d9f3b
        return h % 2

    def phi(self, a: int, b: int) -> int:
        h_ab = self._get_base_hash(a, b)
        return h_ab if a <= b else 1 - h_ab

    @torch.no_grad()
    def apply_watermark(
        self, 
        logits: torch.Tensor, 
        current_text: torch.Tensor,
        mask_index: torch.Tensor, 
        confidence: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        modified_confidence = confidence.clone()
        batch_size, seq_len = confidence.shape
        x0_tokens = torch.argmax(logits, dim=-1)
    
        for b in range(batch_size):
            for i in range(seq_len):
                if not mask_index[b, i]:
                    modified_confidence[b, i] = -1e9 
                    continue
            
                current_token_i = int(x0_tokens[b, i].item())
                sigma_i = 0.0
            
                for j_offset in self.context_window:
                    j = i + j_offset
                    if 0 <= j < seq_len:
                        is_masked = mask_index[b, j]
                        v_j = int(x0_tokens[b, j].item() if is_masked else current_text[b, j].item())
                        w_j = float(confidence[b, j].item() if is_masked else 1.0)
                        
                        valid = self.phi(v_j, current_token_i) if j_offset < 0 else self.phi(current_token_i, v_j)
                        sigma_i += w_j * valid

                c_i = torch.clamp(confidence[b, i], 1e-4, 1 - 1e-4)
                logit_i = torch.log(c_i / (1 - c_i))
                modified_confidence[b, i] = torch.sigmoid(logit_i + (self.delta * sigma_i))

        return modified_confidence

    def detect(
        self,
        text_or_tokens: Union[str, List[int]],
        tokenizer: Any = None,
        p_threshold: float = 0.05,
        **kwargs
    ) -> Dict[str, Any]:
        if isinstance(text_or_tokens, str):
            tokens = tokenizer.encode(text_or_tokens, add_special_tokens=False)
        else:
            tokens = text_or_tokens

        total_checked = 0
        green_count = 0
        seq_len = len(tokens)
        forward_offsets = [o for o in self.context_window if o > 0]
        
        for i in range(seq_len):
            for offset in forward_offsets:
                j = i + offset
                if j < seq_len:
                    total_checked += 1
                    if self.phi(tokens[i], tokens[j]) == 1:
                        green_count += 1
        
        p_val = 1.0 - binom.cdf(green_count - 1, total_checked, 0.5) if total_checked > 0 else 1.0
        
        return {
            "is_watermarked": p_val < p_threshold,
            "p_value": float(p_val),
            "green_fraction": green_count / total_checked if total_checked > 0 else 0.0,
            "total_pairs": total_checked
        }