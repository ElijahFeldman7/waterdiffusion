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
    0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,
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
        delta: float = 2.5,
        frequency: float = 0.125,    
        anchor_bits: int = 3,         
        vocab_size: int = 50257,
        min_tokens: int = 24,
        chunk_size: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.key = key
        self.delta = float(delta)
        self.frequency = float(frequency)
        self.anchor_bits = int(anchor_bits)
        self.vocab_size = int(vocab_size)
        self.min_tokens = int(min_tokens)
        self.chunk_size = int(chunk_size)
        self.device = device

        import hashlib
        self.key_int = _i64(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))

        idx = torch.arange(self.vocab_size, dtype=torch.int64, device=device)
        self._idx_mixed = (idx + 1) * _ROLE[0]


    def _anchor_mask(self, ids: torch.Tensor) -> torch.Tensor:
        prev = torch.roll(ids, 1, dims=-1).clone()
        prev[..., 0] = -1
        h = _mix(_mix(torch.full_like(ids, self.key_int) ^ ((prev + 2) * _ROLE[1]))
                 ^ ((ids + 2) * _ROLE[2]))
        return (h & ((1 << self.anchor_bits) - 1)) == 0

    def _segment_state(self, ids: torch.Tensor):
        S = ids.shape[0]
        anchors = self._anchor_mask(ids.unsqueeze(0))[0]            
        idx = torch.arange(S, device=ids.device)
        last_anchor = torch.where(anchors, idx, torch.full_like(idx, -1))
        last_anchor = torch.cummax(last_anchor, dim=0).values
        tau = torch.where(last_anchor >= 0, idx - last_anchor, idx)  # (S,)

        prev = torch.roll(ids, 1, dims=-1).clone(); prev[0] = -1
        bigram_h = _mix(_mix(torch.full_like(ids, self.key_int)
                             ^ ((prev + 2) * _ROLE[3])) ^ ((ids + 2) * _ROLE[4]))
        seg_hash = torch.where(
            last_anchor >= 0,
            bigram_h[last_anchor.clamp(min=0)],
            torch.full_like(bigram_h, self.key_int),
        )
        return tau, seg_hash

    def _carrier(self, tau: torch.Tensor, seg_hash: torch.Tensor) -> torch.Tensor:
        phi = (_lsr(seg_hash, 32) % 4096).to(torch.float32) * (2 * math.pi / 4096.0)
        return torch.sin(2 * math.pi * self.frequency * tau.to(torch.float32) + phi)

    def _scores_all_tokens(self, seg_hash: torch.Tensor) -> torch.Tensor:
        y = _mix(seg_hash.unsqueeze(-1) ^ self._idx_mixed.unsqueeze(0))
        return _lsr(y, 63).to(torch.float32) * 2.0 - 1.0

    def _scores_for_tokens(self, seg_hash: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        y = _mix(seg_hash ^ ((tok.to(torch.int64) + 1) * _ROLE[0]))
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
            tau, seg_hash = self._segment_state(ctx_ids[b].to(torch.int64))
            carrier = self._carrier(tau, seg_hash)                  
            pos = mask_index[b].nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                continue
            for chunk in pos.split(self.chunk_size):
                s = self._scores_all_tokens(seg_hash[chunk])       
                modified[b, chunk, :V] += self.delta * carrier[chunk].unsqueeze(1) * s[:, :V]
        return modified
    def detect(
        self,
        text: str,
        tokenizer: Any,
        p_value_threshold: float = 0.01,
        conservative: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        tokens = tokenizer.encode(text, return_tensors="pt")[0].to(self.device)
        tokens = torch.clamp(tokens.to(torch.int64), 0, self.vocab_size - 1)
        S = int(tokens.shape[0])
        if S < self.min_tokens:
            return {"is_watermarked": False, "score": 0.0,
                    "z_score": 0.0, "p_value": 1.0, "n_scored": 0}

        tau, seg_hash = self._segment_state(tokens)
        carrier = self._carrier(tau, seg_hash)                       # (S,)
        s = self._scores_for_tokens(seg_hash, tokens)                # (S,) +/-1

        pair_key = _mix(seg_hash ^ ((tokens + 1) * _ROLE[1])).cpu().numpy()
        c_np = carrier.cpu().numpy().astype(np.float64)
        s_np = s.cpu().numpy().astype(np.float64)
        _, inv = np.unique(pair_key, return_inverse=True)
        n_groups = int(inv.max()) + 1
        g = np.zeros(n_groups); np.add.at(g, inv, c_np)              # group weights
        s_rep = np.zeros(n_groups); s_rep[inv] = s_np

        T = float((s_rep * g).sum())
        var = float((g ** 2).sum())
        if var <= 1e-12 or n_groups < self.min_tokens:
            return {"is_watermarked": False, "score": 0.0,
                    "z_score": 0.0, "p_value": 1.0, "n_scored": n_groups}

        z = T / math.sqrt(var)
        if conservative:
            p_value = float(min(1.0, math.exp(-0.5 * z * z)))
        else:
            p_value = float(norm.sf(z))

        return {
            "is_watermarked": bool(p_value < p_value_threshold),
            "score": T,
            "n_scored": n_groups,
            "z_score": float(z),
            "p_value": p_value,
        }