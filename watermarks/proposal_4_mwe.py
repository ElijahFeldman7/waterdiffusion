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
    """Multi-window ensemble watermark (MWE)."""

    def __init__(
        self,
        key: str = "xooks",
        delta: float = 2.0,            # total tilt budget; per channel delta/sqrt(k)
        windows: List[int] = (1, 2, 3, 4),
        vocab_size: int = 50257,
        min_tokens: int = 16,
        chunk_size: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        frequency: float = 0.045,      # unused; signature compatibility
    ):
        self.key = key
        self.delta = float(delta)
        self.windows = list(windows)
        self.k = len(self.windows)
        assert max(self.windows) <= len(_ROLE) - 2
        self.delta_ch = self.delta / math.sqrt(self.k)   # equal-KL split
        self.vocab_size = int(vocab_size)
        self.min_tokens = int(min_tokens)
        self.chunk_size = int(chunk_size)
        self.device = device

        import hashlib
        self.key_int = _i64(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
        # Independent per-channel salts derived from the master key.
        self.salts = [
            _i64(int(hashlib.sha256(f"{key}|channel{j}".encode())
                     .hexdigest()[:16], 16))
            for j in range(self.k)
        ]
        idx = torch.arange(self.vocab_size, dtype=torch.int64, device=device)
        self._idx_mixed = (idx + 1) * _ROLE[0]

    # ---------------------- per-channel hashing ---------------------------

    def _context_hashes(self, ids: torch.Tensor, w: int, salt: int) -> torch.Tensor:
        """Window-w keyed rolling hash; distinct salts make channels
        independent random functions of the same text."""
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

    # ------------------------------------------------------------------

    @torch.no_grad()
    def apply_watermark(
        self,
        logits: torch.Tensor,        # (B, S, V_logits)
        current_text: torch.Tensor,  # (B, S)
        mask_index: torch.Tensor,    # (B, S)
        step: int,
        confidence: torch.Tensor,
        x0: torch.Tensor,            # (B, S)
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
                pos = pos_all[pos_all >= w]   # need a full real window
                for chunk in pos.split(self.chunk_size):
                    s = self._scores_all_tokens(hashes[chunk])        # (P, V)
                    # Channels superpose: total bias is a sum of k antipodal
                    # tilts at delta/sqrt(k) each => per-token KL ~ delta^2/2,
                    # matching a single channel at delta (see report B.5).
                    modified[b, chunk, :V] += self.delta_ch * s[:, :V]
        return modified

    # ------------------------------------------------------------------

    @staticmethod
    def _chi_bar_sf(x: float, k: int) -> float:
        """Exact survival function of X = sum_j max(Z_j, 0)^2 for iid standard
        normal Z_j: the chi-bar-squared mixture
            P(X > x) = sum_{m=0}^{k} C(k, m) 2^{-k} P(chi^2_m > x),
        where the m=0 component is a point mass at 0 (contributes 0 for x>0).
        This follows by conditioning on which Z_j are positive."""
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

            # per-channel dedupe of (context, token) pairs, exactly as P1
            pair_key = _mix(h_v ^ ((t_v + 1) * _ROLE[len(_ROLE) - 1]))
            _, first_idx = np.unique(pair_key.cpu().numpy(), return_index=True)
            s_u = s_v.cpu().numpy()[np.sort(first_idx)]
            n = int(s_u.shape[0])
            if n < self.min_tokens:
                continue
            kk = int((s_u > 0).sum())
            # Continuity-corrected normal score for the exact binomial count;
            # superuniform (conservative) per channel under H0.
            z = (kk - n / 2.0 - 0.5) / math.sqrt(n / 4.0)
            channel_z.append(z)

        if not channel_z:
            return {"is_watermarked": False, "score": 0.0,
                    "z_score": 0.0, "p_value": 1.0, "channel_z": []}

        # One-sided omnibus statistic: only positive deviations are evidence,
        # so negative channels are clipped rather than squared in.
        X = float(sum(max(z, 0.0) ** 2 for z in channel_z))
        p_value = self._chi_bar_sf(X, len(channel_z))
        # Report an equivalent z for interface compatibility.
        z_equiv = float(norm.isf(min(max(p_value, 1e-300), 1.0)))

        return {
            "is_watermarked": bool(p_value < p_value_threshold),
            "score": X,
            "channel_z": [float(z) for z in channel_z],
            "z_score": z_equiv,
            "p_value": float(p_value),
        }


# --------------------------------- tests ----------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    V, S = 1000, 200
    wm = DWTWatermark(key="testkey", delta=2.4, windows=(1, 2, 3, 4),
                      vocab_size=V, device="cpu")

    class MockTok:
        def encode(self, text, return_tensors=None):
            return torch.tensor([int(x) for x in text.split()],
                                dtype=torch.long).unsqueeze(0)
    tok = MockTok()

    # chi-bar-squared sanity: Monte Carlo vs closed form
    Zm = np.random.randn(200000, 4)
    Xm = (np.clip(Zm, 0, None) ** 2).sum(axis=1)
    for q in (5.0, 9.0, 12.0):
        mc = (Xm > q).mean()
        cf = wm._chi_bar_sf(q, 4)
        assert abs(mc - cf) < 4 * math.sqrt(cf / 200000 + 1e-9), (q, mc, cf)
    print("[chi-bar] closed-form tail verified against Monte Carlo")

    out = wm.apply_watermark(torch.zeros(2, S, V), torch.randint(0, V, (2, S)),
                             torch.rand(2, S) > 0.5, 0, torch.rand(2, S),
                             torch.randint(0, V, (2, S)))
    assert out.shape == (2, S, V)

    pvals = []
    for _ in range(300):
        ids = np.random.randint(0, V, size=S)
        pvals.append(wm.detect(" ".join(map(str, ids)), tok)["p_value"])
    print(f"[null] FPR@0.01={(np.array(pvals) < 0.01).mean():.4f}")
    assert (np.array(pvals) < 0.01).mean() <= 0.03

    def simulate(delta_total):
        d_ch = delta_total / math.sqrt(wm.k)
        ids = list(np.random.randint(0, V, size=max(wm.windows)))
        while len(ids) < S:
            bias = np.zeros(V, dtype=np.float64)
            ext = torch.tensor(ids + [0]).unsqueeze(0)
            for j, w in enumerate(wm.windows):
                h = wm._context_hashes(ext, w, wm.salts[j])[0, -1]
                bias += d_ch * wm._scores_all_tokens(h.view(1))[0].numpy()
            p = np.exp(bias - bias.max()); p /= p.sum()   # tilted uniform base
            ids.append(int(np.random.choice(V, p=p)))
        return ids

    det, zs = 0, []
    for _ in range(50):
        r = wm.detect(" ".join(map(str, simulate(wm.delta))), tok)
        zs.append(r["z_score"]); det += r["is_watermarked"]
    print(f"[wm] mean z_equiv={np.mean(zs):.2f}  TPR@1%FPR={det/50:.2f}")
    assert det / 50 > 0.9

    # heavy substitution: 40% — long-window channels die, short ones carry it
    det = 0
    cz = []
    for _ in range(50):
        ids = np.array(simulate(wm.delta))
        idx = np.random.choice(S, size=int(0.4 * S), replace=False)
        ids[idx] = np.random.randint(0, V, size=len(idx))
        r = wm.detect(" ".join(map(str, ids)), tok)
        det += r["is_watermarked"]; cz.append(r["channel_z"])
    cz = np.array(cz)
    print(f"[wm+40%sub] TPR@1%FPR={det/50:.2f}  mean channel z by window "
          f"{wm.windows}: {np.round(cz.mean(axis=0), 2).tolist()}")
    print("MWE self-test passed.")
