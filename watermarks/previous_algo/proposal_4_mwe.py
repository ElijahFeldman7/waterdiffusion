"""
Proposal 4 — MWE: Multi-Window hash Ensemble with chi-bar-squared aggregation.

Embeds k independent context-keyed antipodal channels simultaneously, each
using a *different context window length* w_j and an independent key salt.
The per-channel tilt is delta / sqrt(k), so the total per-token KL matches a
single channel at tilt delta (small-tilt regime).  Detection computes one
exact-null z-score per channel and aggregates with the one-sided
"chi-bar-squared" statistic X = sum_j max(z_j, 0)^2, whose null distribution
is the known finite mixture  P(X > x) = sum_m C(k,m) 2^{-k} P(chi^2_m > x)
— exact given per-channel normality, no Monte Carlo (R4).

Why an ensemble: attacks degrade channels *differently*.  A substitution of a
fraction eps of tokens destroys a w-window term with probability
1 - (1-eps)^{w+1}: short windows survive heavy editing, long windows are
harder to reverse-engineer (an oracle attack must map a context space of size
V^w) and more secure against spoofing.  The chi-bar aggregation is adaptive —
it automatically weights toward whichever channels survived — at a known,
bounded omnibus cost when nothing was attacked.  R1/R3 hold per channel as in
Proposal 1.  Dependencies: torch, numpy, scipy.
"""

import math
from typing import Any, Dict, List

import numpy as np
import torch
from scipy.stats import chi2, norm

# --------------------------- hashing primitives ---------------------------

def _i64(c: int) -> int:
    return c - (1 << 64) if c >= (1 << 63) else c


_MIX1 = _i64(0xBF58476D1CE4E5B9)
_MIX2 = _i64(0x94D049BB133111EB)
_ROLE = [_i64(c) for c in (
    0xD6E8FEB86659FD93, 0xA3B195354A39B70D, 0x9E6C63D0876A9F4F,
    0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x27D4EB2F165667C5,
    0x94D049BB133111EB, 0xFF51AFD7ED558CCD, 0xC4CEB9FE1A85EC53,
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
        delta: float = 2.0,            
        windows: List[int] = (1, 2, 3, 4),
        vocab_size: int = 50257,
        min_tokens: int = 16,
        chunk_size: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        frequency: float = 0.045,      
    ):
        self.key = key
        self.delta = float(delta)
        self.windows = list(windows)
        self.k = len(self.windows)
        assert max(self.windows) <= len(_ROLE) - 2
        self.delta_ch = self.delta / math.sqrt(self.k)  
        self.vocab_size = int(vocab_size)
        self.min_tokens = int(min_tokens)
        self.chunk_size = int(chunk_size)
        self.device = device

        import hashlib
        self.key_int = _i64(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
        self.salts = [
            _i64(int(hashlib.sha256(f"{key}|channel{j}".encode())
                     .hexdigest()[:16], 16))
            for j in range(self.k)
        ]
        idx = torch.arange(self.vocab_size, dtype=torch.int64, device=device)
        self._idx_mixed = (idx + 1) * _ROLE[0]


    def _context_hashes(self, ids: torch.Tensor, w: int, salt: int) -> torch.Tensor:
        ids = ids.to(torch.int64)
        h = torch.full_like(ids, self.key_int ^ salt)
        for j in range(1, w + 1):
            prev = torch.roll(ids, shifts=j, dims=-1).clone()
            prev[..., :j] = -j
            h = _mix(h ^ ((prev + 2) * _ROLE[j]))
        return _mix(h)

    def _scores_all_tokens(self, h: torch.Tensor) -> torch.Tensor:
        y = _mix(h.unsqueeze(-1) ^ self._idx_mixed.unsqueeze(0))
        return _lsr(y, 63).to(torch.float32) * 2.0 - 1.0

    def _scores_for_tokens(self, h: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        y = _mix(h ^ ((tok.to(torch.int64) + 1) * _ROLE[0]))
        return _lsr(y, 63).to(torch.float32) * 2.0 - 1.0

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

        ctx_ids = torch.where(mask_index, x0, current_text) if x0 is not None \
            else current_text

        modified = logits.clone()
        for b in range(B):
            pos_all = mask_index[b].nonzero(as_tuple=True)[0]
            if pos_all.numel() == 0:
                continue
            for j, w in enumerate(self.windows):
                hashes = self._context_hashes(ctx_ids[b:b + 1], w, self.salts[j])[0]
                pos = pos_all[pos_all >= w]   
                for chunk in pos.split(self.chunk_size):
                    s = self._scores_all_tokens(hashes[chunk])        
                    modified[b, chunk, :V] += self.delta_ch * s[:, :V]
        return modified
    @staticmethod
    def _chi_bar_sf(x: float, k: int) -> float:
        if x <= 0:
            return 1.0
        p = 0.0
        for m in range(1, k + 1):
            p += math.comb(k, m) * (0.5 ** k) * float(chi2.sf(x, df=m))
        return p

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
        if S < max(self.min_tokens, max(self.windows) + 1):
            return {"is_watermarked": False, "score": 0.0,
                    "z_score": 0.0, "p_value": 1.0, "channel_z": []}

        channel_z = []
        for j, w in enumerate(self.windows):
            hashes = self._context_hashes(tokens.unsqueeze(0), w, self.salts[j])[0]
            s = self._scores_for_tokens(hashes, tokens)
            valid = torch.arange(S, device=self.device) >= w
            h_v, t_v, s_v = hashes[valid], tokens[valid], s[valid]

            pair_key = _mix(h_v ^ ((t_v + 1) * _ROLE[len(_ROLE) - 1]))
            _, first_idx = np.unique(pair_key.cpu().numpy(), return_index=True)
            s_u = s_v.cpu().numpy()[np.sort(first_idx)]
            n = int(s_u.shape[0])
            if n < self.min_tokens:
                continue
            kk = int((s_u > 0).sum())
            z = (kk - n / 2.0 - 0.5) / math.sqrt(n / 4.0)
            channel_z.append(z)

        if not channel_z:
            return {"is_watermarked": False, "score": 0.0,
                    "z_score": 0.0, "p_value": 1.0, "channel_z": []}

        X = float(sum(max(z, 0.0) ** 2 for z in channel_z))
        p_value = self._chi_bar_sf(X, len(channel_z))
        z_equiv = float(norm.isf(min(max(p_value, 1e-300), 1.0)))

        return {
            "is_watermarked": bool(p_value < p_value_threshold),
            "score": X,
            "channel_z": [float(z) for z in channel_z],
            "z_score": z_equiv,
            "p_value": float(p_value),
        }