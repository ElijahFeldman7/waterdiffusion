import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import norm
from dtaidistance import dtw

class DWTWatermark:

    def __init__(
        self,
        key: str = "xooks",
        delta: float = 8.0,
        frequency: float = 0.045,
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
        batch_size, seq_len = logits.shape[:2]

        positions = torch.arange(seq_len, device=self.device).float()
        carrier = torch.sin(2 * math.pi * self.frequency * positions)

        watermark_bias = torch.outer(carrier, self.token_values) * self.delta

        mask_expanded = mask_index.unsqueeze(-1)

        modified_logits = torch.where(
            mask_expanded,
            logits + watermark_bias.unsqueeze(0),
            logits,
        )

        return modified_logits


    def _get_null_stats(self, seq_len: int, window: int) -> Tuple[float, float]:
        cache_key = (seq_len, window)
        if cache_key in self.null_stats_cache:
            return self.null_stats_cache[cache_key]

        sim_rng = torch.Generator()
        sim_rng.manual_seed(42 + seq_len)
        null_v = torch.empty((1000, seq_len)).uniform_(-1, 1, generator=sim_rng)

        positions = torch.arange(seq_len).float()
        carrier = torch.sin(2 * math.pi * self.frequency * positions)
        c_norm = np.ascontiguousarray(
            ((carrier - carrier.mean()) / (carrier.std() + 1e-8)).cpu().numpy().astype(np.double)
        )

        distances = []
        for i in range(1000):
            v = null_v[i]
            v_norm = np.ascontiguousarray(
                ((v - v.mean()) / (v.std() + 1e-8)).cpu().numpy().astype(np.double)
            )
            distances.append(dtw.distance(v_norm, c_norm, window=window))

        mu = np.mean(distances)
        sigma = np.std(distances)
        self.null_stats_cache[cache_key] = (mu, sigma)
        return mu, sigma

    def detect(
        self,
        text: str,
        tokenizer: Any,
        p_value_threshold: float = 0.01,
        **kwargs,
    ) -> Dict[str, Any]:
        tokens_tensor = tokenizer.encode(text, return_tensors="pt")[0].to(self.device)
        seq_len = tokens_tensor.shape[0]

        if seq_len < 10:
            return {"is_watermarked": False, "score": 0.0, "p_value": 1.0}

        safe_tokens = torch.clamp(tokens_tensor, 0, self.vocab_size - 1)
        v = self.token_values[safe_tokens]
        v_norm = np.ascontiguousarray(
            ((v - v.mean()) / (v.std() + 1e-8)).cpu().numpy().astype(np.double)
        )

        positions = torch.arange(seq_len, device=self.device).float()
        carrier = torch.sin(2 * math.pi * self.frequency * positions)
        c_norm = np.ascontiguousarray(
            ((carrier - carrier.mean()) / (carrier.std() + 1e-8)).cpu().numpy().astype(np.double)
        )

        window = 10
        distance = dtw.distance(v_norm, c_norm, window=window)

        mu, sigma = self._get_null_stats(seq_len, window)
        z_score = (distance - mu) / sigma
        p_value = norm.cdf(z_score)
        is_watermarked = bool(p_value < p_value_threshold)

        return {
            "is_watermarked": is_watermarked,
            "score": distance,
            "z_score": z_score,
            "p_value": p_value,
        }