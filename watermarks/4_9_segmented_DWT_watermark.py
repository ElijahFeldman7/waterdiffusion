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

    def _safe_torch_normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        mean = x.mean()
        std = x.std(unbiased=False)
        if not torch.isfinite(std) or std.item() < 1e-8:
            return torch.zeros_like(x)
        return (x - mean) / std

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

    def _compute_segmented_distance(self, v_norm: np.ndarray, seq_len: int) -> float:
        """
        Segmented Phase-Search with Zero-Window.
        - Block size of 50 guarantees survival against massive paragraph edits.
        - window=0 strictly prevents random noise from stretching to lower its score.
        """
        block_size = 50
        safe_frequency = max(float(self.frequency), 1e-8)
        period = max(1, int(round(1 / safe_frequency))) 
        
        num_blocks = math.ceil(seq_len / block_size)
        if num_blocks == 0:
            return 0.0
            
        total_distance = 0.0
        
        for b in range(num_blocks):
            start = b * block_size
            end = min(start + block_size, seq_len)
            v_block = np.ascontiguousarray(v_norm[start:end])
            if not np.isfinite(v_block).all():
                v_block = np.nan_to_num(v_block, nan=0.0, posinf=0.0, neginf=0.0)
            
            best_block_dist = float('inf')
            
            for phase_shift in range(period):
                positions = torch.arange(end - start).float() + phase_shift
                carrier = torch.sin(2 * math.pi * self.frequency * positions)
                
                c_norm = self._safe_torch_normalize(carrier)
                c_block = np.ascontiguousarray(c_norm.cpu().numpy().astype(np.double))
                
                dist = dtw.distance(v_block, c_block, window=0)
                
                if np.isfinite(dist) and dist < best_block_dist:
                    best_block_dist = dist

            if not np.isfinite(best_block_dist):
                best_block_dist = float(np.linalg.norm(v_block))
                    
            total_distance += best_block_dist
            
        return total_distance

    def _get_null_stats(self, seq_len: int) -> Tuple[float, float]:
        cache_key = (seq_len, "segmented_zero_window")
        if cache_key in self.null_stats_cache:
            return self.null_stats_cache[cache_key]
            
        sim_rng = torch.Generator()
        sim_rng.manual_seed(42 + seq_len)
        null_v = torch.empty((1000, seq_len)).uniform_(-1, 1, generator=sim_rng)
        
        distances = []
        for i in range(1000):
            v = null_v[i]
            v_norm = np.ascontiguousarray(self._safe_torch_normalize(v).cpu().numpy().astype(np.double))
            
            dist = self._compute_segmented_distance(v_norm, seq_len)
            if np.isfinite(dist):
                distances.append(dist)

        if len(distances) == 0:
            mu, sigma = 0.0, 1.0
            self.null_stats_cache[cache_key] = (mu, sigma)
            return mu, sigma
            
        dist_arr = np.asarray(distances, dtype=np.double)
        mu = float(np.mean(dist_arr))
        sigma = float(np.std(dist_arr))
        if (not np.isfinite(sigma)) or sigma < 1e-8:
            sigma = 1.0
        if not np.isfinite(mu):
            mu = 0.0
        
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
        if seq_len < 10: return {"is_watermarked": False, "score": 0.0, "p_value": 1.0}

        safe_tokens = torch.clamp(tokens_tensor, 0, self.vocab_size - 1)
        v = self.token_values[safe_tokens] 
        
        v_norm = np.ascontiguousarray(self._safe_torch_normalize(v).cpu().numpy().astype(np.double))
        
        distance = self._compute_segmented_distance(v_norm, seq_len)
        mu, sigma = self._get_null_stats(seq_len)

        if not np.isfinite(distance):
            return {
                "is_watermarked": False,
                "score": float("inf"),
                "z_score": 0.0,
                "p_value": 1.0,
            }
        
        z_score = (distance - mu) / max(sigma, 1e-8)
        p_value = float(norm.cdf(z_score))
        if not np.isfinite(p_value):
            p_value = 1.0
        
        is_watermarked = bool(p_value < p_value_threshold)
        
        return {
            "is_watermarked": is_watermarked,
            "score": distance,       
            "z_score": z_score,       
            "p_value": p_value, 
        }