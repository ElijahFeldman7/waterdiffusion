import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import norm
from dtaidistance import dtw

class Watermark:
    def __init__(
        self,
        key: str = "xooks",
        delta: float = 8.0,      
        frequency: float = 0.1,  
        vocab_size: int = 50257, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.key = key
        self.delta = delta
        self.frequency = frequency
        self.vocab_size = vocab_size
        self.device = device

        rng = np.random.RandomState(self._hash_string(key))
        self.token_values = torch.from_numpy(
            rng.uniform(-1, 1, size=(vocab_size,))
        ).to(device).float()
        
        self.null_stats_cache = {}

    def _hash_string(self, s: str) -> int:
        import hashlib
        return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**32)

    @torch.no_grad()
    def apply_watermark(
        self,
        logits: torch.Tensor,
        current_text: torch.Tensor,
        mask_index: torch.Tensor,
        step: int,
        confidence: torch.Tensor,
        x0: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len = confidence.shape

        positions = torch.arange(seq_len, device=self.device).float()
        carrier = torch.sin(2 * math.pi * self.frequency * positions) 
        
        safe_x0 = torch.clamp(x0, 0, self.vocab_size - 1)
        safe_current = torch.clamp(current_text, 0, self.vocab_size - 1)
        
        proposed_tokens = torch.where(mask_index, safe_x0, safe_current)
        proposed_values = self.token_values[proposed_tokens] 
        alignment = proposed_values * carrier.unsqueeze(0)
        
        watermark_bias = alignment * self.delta
        
        eps = 1e-6
        conf_clamped = torch.clamp(confidence, eps, 1.0 - eps)
        logits_conf = torch.log(conf_clamped / (1 - conf_clamped))
        
        modified_logits = torch.where(
            mask_index,
            logits_conf + watermark_bias,
            logits_conf
        )
        
        modified_confidence = torch.sigmoid(modified_logits)
        
        modified_confidence = torch.where(
            mask_index,
            modified_confidence,
            torch.full_like(modified_confidence, -float('inf'))
        )
        
        return modified_confidence
