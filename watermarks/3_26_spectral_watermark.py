import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from scipy.stats import norm


class SpectralWatermark:

    def __init__(
        self,
        key: str = "xooks",
        delta: float = 4.0,
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

    def detect(
        self,
        text: str,
        tokenizer: Any,
        **kwargs,
    ) -> Dict[str, Any]:
        tokens_tensor = tokenizer.encode(text, return_tensors="pt")[0].to(self.device)
        seq_len = tokens_tensor.shape[0]
        if seq_len < 10:
            return {"is_watermarked": False, "score": 0.0, "p_value": 1.0}

        safe_tokens = torch.clamp(tokens_tensor, 0, self.vocab_size - 1)
        v = self.token_values[safe_tokens] 
        
        positions = torch.arange(seq_len, device=self.device).float()
        carrier = torch.sin(2 * math.pi * self.frequency * positions)
        
        v_norm = (v - v.mean()) / (v.std() + 1e-8)
        c_norm = (carrier - carrier.mean()) / (carrier.std() + 1e-8)
        
        V = torch.fft.fft(v_norm)
        C = torch.fft.fft(c_norm)
        corr = torch.fft.ifft(V * torch.conj(C)).real
        
        max_corr = torch.max(corr).item() / seq_len
        
        z_score = max_corr * math.sqrt(seq_len)
        
        p_value = 1.0 - norm.cdf(z_score)
        
        p_threshold = kwargs.get("p_value_threshold", 0.01)

        return {
            "is_watermarked": bool(p_value < p_threshold),
            "score": float(z_score),
            "p_value": float(p_value),
            "max_correlation": float(max_corr),
            "sequence_length": seq_len,
        }
