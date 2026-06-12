"""
Proposal 1 — CKAS: Context-Keyed Antipodal Scoring with an exact binomial detector.

Drop-in replacement for the baseline DWTWatermark.  The static token_values
vector and the absolute-position sinusoidal carrier are both removed.  At each
position t the +/-1 "colour" of every vocabulary token is derived from a keyed
hash of the w previous tokens, so the vocabulary partition is different at
every position and cannot be learned without the key (R1).  Because the score
of a token depends only on its local context window and not on its absolute
index, the detection statistic is invariant to position shifts away from edit
boundaries (R3).  Detection is the exact likelihood-ratio test for the implied
signal model, with an exact Binomial(n, 1/2) null (R4), and requires only the
text and a tokenizer (R5).

Dependencies: torch, numpy, scipy only (dtaidistance no longer needed).
"""

import math
from typing import Any, Dict, Tuple

import numpy as np
import torch
from scipy.stats import binom, norm

# ---------------------------------------------------------------------------
# 64-bit keyed hashing primitives (splitmix64 finaliser), fully vectorised in
# torch int64.  torch integer arithmetic wraps modulo 2^64 (two's complement),
# which is exactly the behaviour splitmix64 requires.
# ---------------------------------------------------------------------------

def _i64(c: int) -> int:
    """Map an unsigned 64-bit constant to its signed two's-complement value
    so it can be stored in a torch.int64 scalar."""
    return c - (1 << 64) if c >= (1 << 63) else c


_GOLDEN = _i64(0x9E3779B97F4A7C15)
_MIX1 = _i64(0xBF58476D1CE4E5B9)
_MIX2 = _i64(0x94D049BB133111EB)
# Independent odd multipliers used to separate the roles of "context token at
# offset j" and "candidate token" inside the hash input.
_ROLE = [_i64(c) for c in (
    0xD6E8FEB86659FD93, 0xA3B195354A39B70D, 0x9E6C63D0876A9F4F,
    0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x27D4EB2F165667C5,
    0x94D049BB133111EB, 0xFF51AFD7ED558CCD,
)]


def _lsr(x: torch.Tensor, k: int) -> torch.Tensor:
    """Logical (zero-fill) right shift for int64 tensors.  torch's >> is an
    arithmetic shift, so we mask off the sign-extended high bits."""
    return (x >> k) & _i64((1 << (64 - k)) - 1)


def _mix(x: torch.Tensor) -> torch.Tensor:
    """splitmix64 finaliser: a bijective avalanche function on 64-bit words.
    Modelled as a random function in the security analysis."""
    x = x ^ _lsr(x, 30)
    x = x * _MIX1
    x = x ^ _lsr(x, 27)
    x = x * _MIX2
    x = x ^ _lsr(x, 31)
    return x


class DWTWatermark:
    """Context-keyed antipodal watermark (CKAS).

    apply_watermark keeps the baseline signature; detect needs only
    (text, tokenizer)."""

    def __init__(
        self,
        key: str = "xooks",
        delta: float = 2.0,
        context_window: int = 2,  # w=1 maximises edit robustness, w>=3 maximises partition secrecy; 2 is the compromise (see report Section D)
        vocab_size: int = 50257,
        min_tokens: int = 16,
        chunk_size: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        # kept for signature compatibility with the baseline; unused:
        frequency: float = 0.045,
    ):
        self.key = key
        self.delta = float(delta)
        self.window = int(context_window)
        assert 1 <= self.window <= len(_ROLE) - 1, "context_window out of range"
        self.vocab_size = int(vocab_size)
        self.min_tokens = int(min_tokens)
        self.chunk_size = int(chunk_size)
        self.device = device

        # 64-bit key digest.  All randomness flows through this value; an
        # adversary without it faces a PRF-indistinguishability problem.
        import hashlib
        self.key_int = _i64(
            int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
        )

        # Candidate-token index vector, pre-mixed with its role constant so the
        # per-position score reduces to one _mix call over a broadcasted sum.
        idx = torch.arange(self.vocab_size, dtype=torch.int64, device=device)
        self._idx_mixed = (idx + 1) * _ROLE[0]

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def _context_hashes(self, ids: torch.Tensor) -> torch.Tensor:
        """Keyed rolling hash of the w tokens preceding each position.

        ids: (..., S) int64 token ids.  Returns (..., S) int64 hashes.
        Position t's hash depends only on tokens t-w .. t-1 (and the key),
        never on t itself -> shift invariance of the score sequence (R3).
        Positions t < w hash a deterministic pad value for the missing slots;
        the detector simply skips those positions.
        """
        ids = ids.to(torch.int64)
        h = torch.full_like(ids, self.key_int)
        for j in range(1, self.window + 1):
            prev = torch.roll(ids, shifts=j, dims=-1).clone()
            # Pad out-of-range slots with a constant; these positions are
            # excluded from detection so the exact pad value is irrelevant.
            prev[..., :j] = -j
            # Absorb the token at offset -j under its own role multiplier.
            h = _mix(h ^ ((prev + 2) * _ROLE[j]))
        return _mix(h)

    def _scores_all_tokens(self, h: torch.Tensor) -> torch.Tensor:
        """(P,) hashes -> (P, V) float32 scores in {-1, +1}.

        s(v | h) = 2 * topbit(mix(h XOR role(v))) - 1.  Under the random
        function model these are iid Rademacher across distinct (h, v)."""
        y = _mix(h.unsqueeze(-1) ^ self._idx_mixed.unsqueeze(0))
        return _lsr(y, 63).to(torch.float32) * 2.0 - 1.0

    def _scores_for_tokens(self, h: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        """Pointwise score of the realised token at each position: (S,) -> (S,)."""
        y = _mix(h ^ ((tok.to(torch.int64) + 1) * _ROLE[0]))
        return _lsr(y, 63).to(torch.float32) * 2.0 - 1.0

    # ------------------------------------------------------------------
    # Generation-time bias
    # ------------------------------------------------------------------

    @torch.no_grad()
    def apply_watermark(
        self,
        logits: torch.Tensor,        # (B, S, V_logits)
        current_text: torch.Tensor,  # (B, S) partially decoded ids
        mask_index: torch.Tensor,    # (B, S) True where still masked
        step: int,
        confidence: torch.Tensor,
        x0: torch.Tensor,            # (B, S) model's current full-sequence estimate
        **kwargs,
    ) -> torch.Tensor:
        B, S, V_logits = logits.shape
        V = min(V_logits, self.vocab_size)

        # Best available estimate of every position's eventual context: the
        # committed token where decoded, the model's x0 prediction where still
        # masked.  Committed neighbours dominate as decoding progresses, so
        # the generation-time hash converges to the detection-time hash.
        if x0 is not None:
            ctx_ids = torch.where(mask_index, x0, current_text)
        else:
            ctx_ids = current_text

        hashes = self._context_hashes(ctx_ids)  # (B, S)

        modified = logits.clone()
        for b in range(B):
            pos = mask_index[b].nonzero(as_tuple=True)[0]
            # Skip positions whose context window would index before 0; their
            # generation/detection hashes can disagree if a prompt is later
            # stripped, so we never spend bias there.
            pos = pos[pos >= self.window]
            if pos.numel() == 0:
                continue
            # Chunk over positions: the (P, V) int64 intermediate is the only
            # large allocation (chunk_size * V * 8 bytes).
            for chunk in pos.split(self.chunk_size):
                s = self._scores_all_tokens(hashes[b, chunk])  # (P, V) +/-1
                modified[b, chunk, :V] += self.delta * s[:, :V]
        return modified

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

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

        hashes = self._context_hashes(tokens.unsqueeze(0))[0]      # (S,)
        s = self._scores_for_tokens(hashes, tokens)                 # (S,) +/-1

        # Drop positions without a full real context window.
        valid = torch.arange(S, device=self.device) >= self.window
        hashes, toks, s = hashes[valid], tokens[valid], s[valid]

        # Deduplicate repeated (context-hash, token) pairs: identical pairs
        # produce identical scores, so repeats are perfectly correlated and
        # would invalidate the iid Rademacher null.  Counting each distinct
        # pair once restores exact independence (cf. KGW repetition masking).
        pair_key = _mix(hashes ^ ((toks + 1) * _ROLE[1]))
        _, first_idx = np.unique(pair_key.cpu().numpy(), return_index=True)
        s_unique = s.cpu().numpy()[np.sort(first_idx)]

        n = int(s_unique.shape[0])
        if n < self.min_tokens:
            return {"is_watermarked": False, "score": 0.0,
                    "z_score": 0.0, "p_value": 1.0, "n_scored": n}

        k = int((s_unique > 0).sum())  # number of "green" (s=+1) tokens

        # Exact one-sided binomial p-value: under H0, K ~ Binomial(n, 1/2)
        # exactly (random-function model), so the 1% FPR threshold needs no
        # Monte Carlo estimation at all.
        p_value = float(binom.sf(k - 1, n, 0.5))
        z_score = float((k - n / 2.0) / math.sqrt(n / 4.0))

        return {
            "is_watermarked": bool(p_value < p_value_threshold),
            "score": float(k),
            "n_scored": n,
            "z_score": z_score,
            "p_value": p_value,
        }


# ---------------------------------------------------------------------------
# Self-test: null calibration (FPR) and power under simulated generation.
# ---------------------------------------------------------------------------
"""
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    V, S, TRIALS = 1000, 200, 400
    wm = DWTWatermark(key="testkey", delta=2.0, context_window=3,
                      vocab_size=V, device="cpu")

    class MockTok:
        def encode(self, text, return_tensors=None):
            ids = torch.tensor([int(x) for x in text.split()], dtype=torch.long)
            return ids.unsqueeze(0)

    tok = MockTok()

    # --- shape / signature smoke test on the diffusion interface -----------
    B = 2
    logits = torch.zeros(B, S, V)
    cur = torch.randint(0, V, (B, S))
    mask = torch.rand(B, S) > 0.5
    x0 = torch.randint(0, V, (B, S))
    out = wm.apply_watermark(logits, cur, mask, 0, torch.rand(B, S), x0)
    assert out.shape == (B, S, V)
    assert torch.all(out[~mask] == 0), "unmasked rows must be untouched"
    changed_pos = mask.clone()
    changed_pos[:, : wm.window] = False  # first w positions are never biased
    assert torch.all(out[changed_pos].abs() == wm.delta), "bias must be +/-delta"

    # --- null calibration: random text, p-values should be superuniform ----
    pvals = []
    for _ in range(TRIALS):
        ids = np.random.randint(0, V, size=S)
        r = wm.detect(" ".join(map(str, ids)), tok)
        pvals.append(r["p_value"])
    pvals = np.array(pvals)
    fpr_01 = (pvals < 0.01).mean()
    print(f"[null] mean p={pvals.mean():.3f}  FPR@0.01={fpr_01:.4f}")
    assert fpr_01 <= 0.03, "null miscalibrated"

    # --- power: simulate sequential sampling from the tilted distribution --
    def simulate_watermarked(delta):
        ids = list(np.random.randint(0, V, size=wm.window))  # unbiased seed
        for t in range(wm.window, S):
            ctx = torch.tensor(ids, dtype=torch.int64).unsqueeze(0)
            h = wm._context_hashes(
                torch.tensor(ids + [0], dtype=torch.int64).unsqueeze(0)
            )[0, -1]  # hash for the next position (depends only on prefix)
            sc = wm._scores_all_tokens(h.view(1))[0]  # (V,)
            logp = delta * sc
            p = torch.softmax(logp, dim=0).numpy()    # tilted uniform base LM
            ids.append(int(np.random.choice(V, p=p)))
        return ids

    zs, det = [], 0
    for _ in range(60):
        ids = simulate_watermarked(wm.delta)
        r = wm.detect(" ".join(map(str, ids)), tok)
        zs.append(r["z_score"]); det += r["is_watermarked"]
    print(f"[wm]   mean z={np.mean(zs):.2f}  TPR@1%FPR={det/60:.2f}")
    assert det / 60 > 0.95

    # --- robustness: 20% random substitution -------------------------------
    det = 0
    for _ in range(60):
        ids = np.array(simulate_watermarked(wm.delta))
        idx = np.random.choice(S, size=int(0.2 * S), replace=False)
        ids[idx] = np.random.randint(0, V, size=len(idx))
        r = wm.detect(" ".join(map(str, ids)), tok)
        det += r["is_watermarked"]
    print(f"[wm+20%sub] TPR@1%FPR={det/60:.2f}")

    # --- robustness: 10% random deletion (position shift) -------------------
    det = 0
    for _ in range(60):
        ids = np.array(simulate_watermarked(wm.delta))
        keep = np.sort(np.random.choice(S, size=int(0.9 * S), replace=False))
        r = wm.detect(" ".join(map(str, ids[keep])), tok)
        det += r["is_watermarked"]
    print(f"[wm+10%del] TPR@1%FPR={det/60:.2f}")
    print("CKAS self-test passed.")
"""