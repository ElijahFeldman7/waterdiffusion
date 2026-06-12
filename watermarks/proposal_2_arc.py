"""
Proposal 2 — ARC: Anchor-Resynchronized Carrier.

Keeps the spirit of the baseline's spectral design — a sinusoidal carrier
modulating token scores — but replaces the absolute position index with a
*content clock*.  Keyed "anchor" events (a pseudo-random predicate on token
bigrams) partition the text into segments; the carrier phase counts tokens
since the last anchor, and each segment gets (i) its own keyed phase offset
and (ii) its own keyed +/-1 token-score vector.  Insertions and deletions
desynchronise the carrier only until the next anchor, where the clock resets
(R3).  Because both the anchors and the per-segment score vectors are derived
from local content through a keyed hash, no static vocabulary partition exists
(R1).  Detection is a matched filter against the reconstructed carrier with a
closed-form (conditionally exact-variance) Gaussian null plus a Hoeffding
fallback that is provably conservative at any FPR (R4).

Dependencies: torch, numpy, scipy only.
"""

import math
from typing import Any, Dict

import numpy as np
import torch
from scipy.stats import norm

# --------------------------- hashing primitives ---------------------------

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
    """Anchor-resynchronized sinusoidal watermark (ARC)."""

    def __init__(
        self,
        key: str = "xooks",
        delta: float = 2.5,
        frequency: float = 0.125,     # cycles per token of content clock
        anchor_bits: int = 3,         # P(anchor) = 2^-anchor_bits, mean seg len 8
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

    # ------------------------------------------------------------------
    # Content clock: anchors, segments, phases
    # ------------------------------------------------------------------

    def _anchor_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """(B,S) bool: position t is an anchor iff the keyed hash of the
        bigram (x_{t-1}, x_t) has its low `anchor_bits` bits all zero.
        Content-keyed, so anchors travel with the text under shifts."""
        prev = torch.roll(ids, 1, dims=-1).clone()
        prev[..., 0] = -1
        h = _mix(_mix(torch.full_like(ids, self.key_int) ^ ((prev + 2) * _ROLE[1]))
                 ^ ((ids + 2) * _ROLE[2]))
        return (h & ((1 << self.anchor_bits) - 1)) == 0

    def _segment_state(self, ids: torch.Tensor):
        """Compute, for a single sequence (S,):
        - tau:      (S,) tokens elapsed since the most recent anchor (clock)
        - seg_hash: (S,) keyed hash identifying the segment each position is in
                    (= hash of the anchor bigram that opened the segment)
        Positions before the first anchor get seg_hash = key (a 'segment 0'
        whose clock is the absolute index; it carries signal but is the only
        part of the text not shift-protected)."""
        S = ids.shape[0]
        anchors = self._anchor_mask(ids.unsqueeze(0))[0]            # (S,)
        idx = torch.arange(S, device=ids.device)
        # last anchor index at or before t (-1 if none yet)
        last_anchor = torch.where(anchors, idx, torch.full_like(idx, -1))
        last_anchor = torch.cummax(last_anchor, dim=0).values
        tau = torch.where(last_anchor >= 0, idx - last_anchor, idx)  # (S,)

        # Segment id: keyed hash of the opening anchor bigram.
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
        """sin(2*pi*f*tau + phi_seg) with a keyed per-segment phase offset.
        The phase offset prevents an adversary from inferring clock state by
        averaging across segments, and decorrelates segments under H0."""
        phi = (_lsr(seg_hash, 32) % 4096).to(torch.float32) * (2 * math.pi / 4096.0)
        return torch.sin(2 * math.pi * self.frequency * tau.to(torch.float32) + phi)

    def _scores_all_tokens(self, seg_hash: torch.Tensor) -> torch.Tensor:
        """(P,) segment hashes -> (P, V) +/-1 scores: the vocabulary partition
        is rekeyed in every segment (R1)."""
        y = _mix(seg_hash.unsqueeze(-1) ^ self._idx_mixed.unsqueeze(0))
        return _lsr(y, 63).to(torch.float32) * 2.0 - 1.0

    def _scores_for_tokens(self, seg_hash: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        y = _mix(seg_hash ^ ((tok.to(torch.int64) + 1) * _ROLE[0]))
        return _lsr(y, 63).to(torch.float32) * 2.0 - 1.0

    # ------------------------------------------------------------------
    # Generation-time bias
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

        # Reconstruct the anchor structure from the best current estimate of
        # the final text (committed tokens where available, x0 elsewhere).
        ctx_ids = torch.where(mask_index, x0, current_text) if x0 is not None \
            else current_text

        modified = logits.clone()
        for b in range(B):
            tau, seg_hash = self._segment_state(ctx_ids[b].to(torch.int64))
            carrier = self._carrier(tau, seg_hash)                  # (S,)
            pos = mask_index[b].nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                continue
            for chunk in pos.split(self.chunk_size):
                s = self._scores_all_tokens(seg_hash[chunk])        # (P, V)
                # bias[t, v] = delta * carrier_t * s_seg(t)(v): the carrier
                # amplitude-modulates a per-segment antipodal partition.
                modified[b, chunk, :V] += self.delta * carrier[chunk].unsqueeze(1) * s[:, :V]
        return modified

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

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

        # --- group exact duplicates -------------------------------------
        # Under H0 the random-function model makes s iid Rademacher across
        # *distinct* (seg_hash, token) pairs; identical pairs share one
        # Rademacher.  Group them: T = sum_j s_j * g_j with g_j the summed
        # carrier weight of group j, so Var[T | H0, text] = sum_j g_j^2
        # exactly — no Monte Carlo null estimation is needed.
        pair_key = _mix(seg_hash ^ ((tokens + 1) * _ROLE[1])).cpu().numpy()
        c_np = carrier.cpu().numpy().astype(np.float64)
        s_np = s.cpu().numpy().astype(np.float64)
        _, inv = np.unique(pair_key, return_inverse=True)
        n_groups = int(inv.max()) + 1
        g = np.zeros(n_groups); np.add.at(g, inv, c_np)              # group weights
        # one representative score per group (all members share it)
        s_rep = np.zeros(n_groups); s_rep[inv] = s_np

        T = float((s_rep * g).sum())
        var = float((g ** 2).sum())
        if var <= 1e-12 or n_groups < self.min_tokens:
            return {"is_watermarked": False, "score": 0.0,
                    "z_score": 0.0, "p_value": 1.0, "n_scored": n_groups}

        z = T / math.sqrt(var)
        if conservative:
            # Hoeffding: T is a weighted sum of independent Rademachers, hence
            # sub-Gaussian with proxy variance exactly `var`:
            # P(T >= x) <= exp(-x^2 / (2 var)).  Valid at any n, any weights.
            p_value = float(min(1.0, math.exp(-0.5 * z * z)))
        else:
            # CLT p-value; Berry-Esseen error <= C * sum|g|^3 / var^{3/2},
            # small because |g_j| <= a few units while var grows like n/2.
            p_value = float(norm.sf(z))

        return {
            "is_watermarked": bool(p_value < p_value_threshold),
            "score": T,
            "n_scored": n_groups,
            "z_score": float(z),
            "p_value": p_value,
        }


# --------------------------------- tests ----------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    V, S = 1000, 240
    wm = DWTWatermark(key="testkey", delta=2.5, vocab_size=V, device="cpu")

    class MockTok:
        def encode(self, text, return_tensors=None):
            return torch.tensor([int(x) for x in text.split()],
                                dtype=torch.long).unsqueeze(0)
    tok = MockTok()

    # signature / shape smoke test
    B = 2
    out = wm.apply_watermark(torch.zeros(B, S, V), torch.randint(0, V, (B, S)),
                             torch.rand(B, S) > 0.5, 0, torch.rand(B, S),
                             torch.randint(0, V, (B, S)))
    assert out.shape == (B, S, V)

    # null calibration
    pvals = []
    for _ in range(400):
        ids = np.random.randint(0, V, size=S)
        pvals.append(wm.detect(" ".join(map(str, ids)), tok)["p_value"])
    pvals = np.array(pvals)
    print(f"[null] mean p={pvals.mean():.3f}  FPR@0.01={(pvals < 0.01).mean():.4f}")
    assert (pvals < 0.01).mean() <= 0.03

    # power: sequential simulation of the tilted base LM.  The clock at step t
    # is fully determined by the prefix, so left-to-right simulation is exact.
    def simulate(delta):
        ids = [int(np.random.randint(0, V))]
        while len(ids) < S:
            t = len(ids)
            ext = torch.tensor(ids + [0], dtype=torch.int64)
            tau, seg_hash = wm._segment_state(ext)
            # position t's clock/segment depend only on tokens < t... except
            # the anchor predicate uses (x_{t-1}, x_t).  Conditioning on the
            # candidate token would change tau_t; the generator (like the
            # diffusion sampler with x0 estimates) uses the pre-token state,
            # which only misassigns the (rare) anchor positions themselves.
            c_t = wm._carrier(tau[t:t + 1], seg_hash[t:t + 1])[0]
            sc = wm._scores_all_tokens(seg_hash[t:t + 1])[0]
            p = torch.softmax(delta * c_t * sc, dim=0).numpy()
            ids.append(int(np.random.choice(V, p=p)))
        return ids

    zs, det = [], 0
    for _ in range(60):
        r = wm.detect(" ".join(map(str, simulate(wm.delta))), tok)
        zs.append(r["z_score"]); det += r["is_watermarked"]
    print(f"[wm]   mean z={np.mean(zs):.2f}  TPR@1%FPR={det/60:.2f}")
    assert det / 60 > 0.9

    # insert/delete robustness: 10% deletions
    det = 0
    for _ in range(60):
        ids = np.array(simulate(wm.delta))
        keep = np.sort(np.random.choice(S, size=int(0.9 * S), replace=False))
        det += wm.detect(" ".join(map(str, ids[keep])), tok)["is_watermarked"]
    print(f"[wm+10%del] TPR@1%FPR={det/60:.2f}")

    # 20% substitution
    det = 0
    for _ in range(60):
        ids = np.array(simulate(wm.delta))
        idx = np.random.choice(S, size=int(0.2 * S), replace=False)
        ids[idx] = np.random.randint(0, V, size=len(idx))
        det += wm.detect(" ".join(map(str, ids)), tok)["is_watermarked"]
    print(f"[wm+20%sub] TPR@1%FPR={det/60:.2f}")
    print("ARC self-test passed.")
