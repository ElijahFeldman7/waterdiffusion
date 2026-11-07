import math
from typing import Any, Dict

import numpy as np
import torch
from scipy.stats import norm

def _i64(c: int) -> int:
    return c - (1 << 64) if c >= (1 << 63) else c

_MIX1 = _i64(0xBF58476D1CE4E5B9)
_MIX2 = _i64(0x94D049BB133111EB)
_ROLE = [_i64(c) for c in (
    0xD6E8FEB86659FD93, 0xA3B195354A39B70D, 0x9E6C63D0876A9F4F,
    0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x27D4EB2F165667C5,
)]

def _lsr(x: torch.Tensor, k: int) -> torch.Tensor:
    return (x >> k) & _i64((1 << (64 - k)) - 1)

def _mix(x: torch.Tensor) -> torch.Tensor:
    x = x ^ _lsr(x, 30)
    x = x * _MIX1
    x = x ^ _lsr(x, 27)
    x = x * _MIX2
    x = x ^ _lsr(x, 31)
    return x

class DWTWatermark:
    def __init__(
        self,
        key: str = "xooks",
        kl_budget: float = 0.10,      
        delta_max: float = 10.0,      
        context_window: int = 1,      
        temperature: float = 1.0,
        vocab_size: int = 50257,
        min_tokens: int = 16,
        chunk_size: int = 128,
        bisect_iters: int = 40,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        delta: float = None,
        frequency: float = 0.045,
    ):
        self.key = key
        if delta is not None and kl_budget == 0.10:
            self.kl_budget = float(delta)
        else:
            self.kl_budget = float(kl_budget)
            
        self.delta_max = float(delta_max)
        self.window = int(context_window)
        self.temperature = float(temperature)
        self.vocab_size = int(vocab_size)
        self.min_tokens = int(min_tokens)
        self.chunk_size = int(chunk_size)
        self.bisect_iters = int(bisect_iters)
        self.device = device

        import hashlib
        self.key_int = _i64(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
        idx = torch.arange(self.vocab_size, dtype=torch.int64, device=device)
        self._idx_mixed = (idx + 1) * _ROLE[0]


    def _context_hashes(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.to(torch.int64)
        h = torch.full_like(ids, self.key_int)
        for j in range(1, self.window + 1):
            prev = torch.roll(ids, shifts=j, dims=-1).clone()
            prev[..., :j] = -j
            h = _mix(h ^ ((prev + 2) * _ROLE[j]))
        return _mix(h)

    def _get_gaussian_scores(self, mixed: torch.Tensor) -> torch.Tensor:
        u = (_lsr(mixed, 11).to(torch.float64) * (2 ** -53)).to(torch.float32)
        u = torch.clamp(u, 1e-7, 1 - 1e-7)
        s = torch.special.ndtri(u)
        return s

    def _scores_all_tokens(self, h: torch.Tensor) -> torch.Tensor:
        y = _mix(h.unsqueeze(-1) ^ self._idx_mixed.unsqueeze(0))
        return self._get_gaussian_scores(y)

    def _scores_for_tokens(self, h: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        y = _mix(h ^ ((tok.to(torch.int64) + 1) * _ROLE[0]))
        return self._get_gaussian_scores(y)


    @staticmethod
    def _kl_of_delta(d: torch.Tensor, p0: torch.Tensor, s: torch.Tensor) -> torch.Tensor:        
        log_p0 = torch.log(torch.clamp(p0, 1e-12))
        log_q_unnorm = log_p0 + d * s
        logZ = torch.logsumexp(log_q_unnorm, dim=-1, keepdim=True)
        q = torch.exp(log_q_unnorm - logZ)
        kl = torch.sum(q * (d * s - logZ), dim=-1)
        return kl

    def _solve_delta(self, p0: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        P = p0.shape[0]
        lo = torch.zeros((P, 1), device=p0.device)
        hi = torch.full((P, 1), self.delta_max, device=p0.device)
        
        within = self._kl_of_delta(hi, p0, s) <= self.kl_budget
        
        for _ in range(self.bisect_iters):
            mid = 0.5 * (lo + hi)
            kl = self._kl_of_delta(mid, p0, s)
            too_high = kl > self.kl_budget
            hi = torch.where(too_high.unsqueeze(1), mid, hi)
            lo = torch.where(too_high.unsqueeze(1), lo, mid)
            
        final_d = torch.where(within.unsqueeze(1), torch.full_like(lo, self.delta_max), lo)
        return final_d.squeeze(1)
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
        B, S, V_logits = logits.shape
        V = min(V_logits, self.vocab_size)

        ctx_ids = torch.where(mask_index, x0, current_text) if x0 is not None else current_text
        hashes = self._context_hashes(ctx_ids)  

        modified = logits.clone()
        for b in range(B):
            pos = mask_index[b].nonzero(as_tuple=True)[0]
            pos = pos[pos >= self.window]
            if pos.numel() == 0:
                continue
            for chunk in pos.split(self.chunk_size):
                s = self._scores_all_tokens(hashes[b, chunk])[:, :V]      
                p0 = torch.softmax(logits[b, chunk, :V].float() / self.temperature, dim=-1)
                
                d = self._solve_delta(p0, s)                               
                modified[b, chunk, :V] += (self.temperature * d).unsqueeze(1) * s
        return modified


    def detect(
        self,
        text: str,
        tokenizer: Any,
        p_value_threshold: float = 0.01,
        **kwargs,
    ) -> Dict[str, Any]:
        tokens = tokenizer.encode(text, return_tensors="pt")[0].to(self.device)
        tokens = torch.clamp(tokens.to(torch.int64), 0, self.vocab_size - 1)
        S = int(tokens.shape[0])
        
        if S < max(self.min_tokens, self.window + 1):
            return {"is_watermarked": False, "score": 0.0,
                    "z_score": 0.0, "p_value": 1.0, "n_scored": 0}

        hashes = self._context_hashes(tokens.unsqueeze(0))[0]
        s = self._scores_for_tokens(hashes, tokens)
        
        valid = torch.arange(S, device=self.device) >= self.window
        hashes, toks, s = hashes[valid], tokens[valid], s[valid]

        pair_key = _mix(hashes ^ ((toks + 1) * _ROLE[1]))
        _, first_idx = np.unique(pair_key.cpu().numpy(), return_index=True)
        s_unique = s.cpu().numpy()[np.sort(first_idx)]

        n = int(s_unique.shape[0])
        if n < self.min_tokens:
            return {"is_watermarked": False, "score": 0.0,
                    "z_score": 0.0, "p_value": 1.0, "n_scored": n}
                    
        score_sum = float(s_unique.sum())
        z_score = float(score_sum / math.sqrt(n))
        p_value = float(norm.sf(z_score))
        
        return {
            "is_watermarked": bool(p_value < p_value_threshold),
            "score": score_sum,
            "n_scored": n,
            "z_score": z_score,
            "p_value": p_value,
        }
