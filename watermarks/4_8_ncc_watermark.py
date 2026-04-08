import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import norm, chi2
from dtaidistance import dtw


class NCCDTWWatermark:
    def __init__(
        self,
        key: str = "xooks",
        delta: float = 4.0,
        frequencies: List[float] = [0.031, 0.045, 0.067, 0.091],
        vocab_size: int = 50257,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtw_window: int = 4,
    ):
        self.key = key
        self.delta = delta
        self.frequencies = frequencies
        self.vocab_size = vocab_size
        self.device = device
        self.dtw_window = dtw_window

        rng = np.random.RandomState(self._hash_string(key))
        self.token_values = torch.from_numpy(
            rng.uniform(-1, 1, size=(vocab_size,))
        ).to(device).float()

        self.null_stats_cache = {}

    def _hash_string(self, s: str) -> int:
        import hashlib
        return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**32)


    def _build_carriers(self, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len, device=self.device).float()
        carriers = []
        for f in self.frequencies:
            c = torch.sin(2 * math.pi * f * positions)
            c = (c - c.mean()) / (c.std() + 1e-8)
            carriers.append(c)
        return torch.stack(carriers)                   

    def _primary_carrier(self, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len, device=self.device).float()
        c = torch.sin(2 * math.pi * self.frequencies[0] * positions)
        return (c - c.mean()) / (c.std() + 1e-8)


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

        carriers    = self._build_carriers(seq_len)          
        combined    = carriers.sum(dim=0)                 
        combined    = combined / (combined.std() + 1e-8)

        safe_x0      = torch.clamp(x0, 0, self.vocab_size - 1)
        safe_current = torch.clamp(current_text, 0, self.vocab_size - 1)

        proposed_tokens = torch.where(mask_index, safe_x0, safe_current)
        proposed_values = self.token_values[proposed_tokens]               
        alignment       = proposed_values * combined.unsqueeze(0)        
        watermark_bias  = alignment * self.delta

        eps          = 1e-6
        conf_clamped = torch.clamp(confidence, eps, 1.0 - eps)
        logits_conf  = torch.log(conf_clamped / (1 - conf_clamped))

        modified_logits     = torch.where(mask_index,
                                          logits_conf + watermark_bias,
                                          logits_conf)
        modified_confidence = torch.sigmoid(modified_logits)
        modified_confidence = torch.where(mask_index,
                                          modified_confidence,
                                          torch.full_like(modified_confidence, -float('inf')))
        return modified_confidence


    def _ncc_detect(
        self,
        v: torch.Tensor,
        seq_len: int,
        p_value_threshold: float,
    ) -> Dict:
        """
        for each frequency k, compute the NCC z-score:

            Z_k = (1/T) * sum_t(v_norm_t * c_k_t) * sqrt(T)
                ~ N(0,1) under H0
                ~ N(rho_k * sqrt(T), 1) under H1

        combine K p-values with Fisher's method:
            chi2_stat = -2 * sum_k(log(p_k))  ~  chi2(2K) under H0
        """
        v_norm   = (v - v.mean()) / (v.std() + 1e-8)       
        carriers = self._build_carriers(seq_len)             

        z_scores, p_values = [], []
        for k in range(len(self.frequencies)):
            ncc  = (v_norm * carriers[k]).mean()           
            z    = float(ncc * math.sqrt(seq_len))
            p    = 1.0 - norm.cdf(z)                    
            z_scores.append(z)
            p_values.append(p)

        K            = len(self.frequencies)
        fisher_stat  = -2.0 * sum(math.log(max(p, 1e-300)) for p in p_values)
        combined_p   = 1.0 - chi2.cdf(fisher_stat, df=2 * K)
        is_watermarked = bool(combined_p < p_value_threshold)

        return {
            "stage":            "ncc",
            "is_watermarked":   is_watermarked,
            "combined_p_value": combined_p,
            "fisher_statistic": fisher_stat,
            "per_carrier_z":    z_scores,
            "per_carrier_p":    p_values,
        }


    def _get_null_stats(self, seq_len: int) -> Tuple[float, float]:
        cache_key = (seq_len, self.dtw_window)
        if cache_key in self.null_stats_cache:
            return self.null_stats_cache[cache_key]

        sim_rng = torch.Generator()
        sim_rng.manual_seed(42 + seq_len)
        null_v = torch.empty((1000, seq_len)).uniform_(-1, 1, generator=sim_rng)

        c      = self._primary_carrier(seq_len)
        c_norm = np.ascontiguousarray(c.cpu().numpy().astype(np.double))

        distances = []
        for i in range(1000):
            vi     = null_v[i]
            vi_norm = np.ascontiguousarray(
                ((vi - vi.mean()) / (vi.std() + 1e-8)).numpy().astype(np.double)
            )
            distances.append(dtw.distance(vi_norm, c_norm, window=self.dtw_window))

        mu, sigma = float(np.mean(distances)), float(np.std(distances))
        self.null_stats_cache[cache_key] = (mu, sigma)
        return mu, sigma

    def _dtw_detect(
        self,
        v: torch.Tensor,
        seq_len: int,
        p_value_threshold: float,
    ) -> Dict:

        c      = self._primary_carrier(seq_len)
        v_norm = np.ascontiguousarray(
            ((v - v.mean()) / (v.std() + 1e-8)).cpu().numpy().astype(np.double)
        )
        c_norm = np.ascontiguousarray(c.cpu().numpy().astype(np.double))

        distance       = dtw.distance(v_norm, c_norm, window=self.dtw_window)
        mu, sigma      = self._get_null_stats(seq_len)
        z_score        = (distance - mu) / (sigma + 1e-8)
        p_value        = float(norm.cdf(z_score))              
        is_watermarked = bool(p_value < p_value_threshold)

        return {
            "stage":          "dtw",
            "is_watermarked": is_watermarked,
            "score":          distance,
            "z_score":        z_score,
            "p_value":        p_value,
        }


    def detect(
        self,
        text: str,
        tokenizer: Any,
        p_value_threshold: float = 0.01,
        bonferroni: bool = True,
        **kwargs,
    ) -> Dict:
        tokens_tensor = tokenizer.encode(text, return_tensors="pt")[0].to(self.device)
        seq_len       = tokens_tensor.shape[0]

        if seq_len < 10:
            return {"is_watermarked": False, "stage": "skipped",
                    "reason": "sequence too short"}

        adjusted_threshold = p_value_threshold / 2 if bonferroni else p_value_threshold

        safe_tokens = torch.clamp(tokens_tensor, 0, self.vocab_size - 1)
        v = self.token_values[safe_tokens] 

        ncc_result = self._ncc_detect(v, seq_len, adjusted_threshold)

        if ncc_result["is_watermarked"]:
            return {**ncc_result, "dtw_result": None}

        dtw_result = self._dtw_detect(v, seq_len, adjusted_threshold)

        return {
            **dtw_result,
            "ncc_result": ncc_result,
        }