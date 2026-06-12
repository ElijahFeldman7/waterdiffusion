"""
Proposal 3 — KLB: KL-Budgeted Exponential Tilting of context-keyed scores.

Ports the theoretically correct piece of ZurichWatermark — per-position
KL-constrained perturbation solved by bisection — into the context-keyed
antipodal framework.  Instead of adding a fixed +/-delta to every token's
logit, each masked position t receives the *largest* tilt delta_t such that
KL(P_t^wm || P_t^base) <= kl_budget exactly.  Because the perturbation is a
two-level (antipodal) tilt, the KL has a closed form in the single scalar
g_t = P_base(green | context_t), so the bisection is a cheap scalar root-find
vectorised over positions — no gradients, no energy functional, no top-k
approximation are needed (the gradient direction of Zurich's energy functional
collapses to the antipodal score vector itself in this model; see report).

This gives: (i) a hard per-position distortion guarantee, (ii) no wasted
budget at low-entropy positions where a fixed delta would burn KL for no
detectable signal, and (iii) bounded distortion at positions where a fixed
delta would overshoot.  The detector is the same exact Binomial(n, 1/2) test
as Proposal 1 (it cannot observe delta_t — an honestly stated mismatch with
the weighted-LRT ideal, analysed in the report).

R1/R3 hold via the context-keyed scores; R4 holds via the exact binomial
null; R5 signatures are preserved.  Dependencies: torch, numpy, scipy.
"""

import math
from typing import Any, Dict

import numpy as np
import torch
from scipy.stats import binom

# --------------------------- hashing primitives ---------------------------

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
    """KL-budgeted context-keyed antipodal watermark (KLB)."""

    def __init__(
        self,
        key: str = "xooks",
        kl_budget: float = 0.10,      # nats per position: KL(P_wm || P_base) cap
        delta_max: float = 4.0,       # tilt cap; prevents blow-up at g -> {0,1}
        context_window: int = 2,  # w=1 maximises edit robustness, w>=3 maximises partition secrecy; 2 is the compromise (see report Section D)
        temperature: float = 1.0,     # sampling temperature used by the decoder
        vocab_size: int = 50257,
        min_tokens: int = 16,
        chunk_size: int = 128,
        bisect_iters: int = 40,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        # signature compatibility with the baseline (unused):
        delta: float = None,
        frequency: float = 0.045,
    ):
        self.key = key
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

    # ---- context-keyed scoring (identical construction to Proposal 1) ----

    def _context_hashes(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.to(torch.int64)
        h = torch.full_like(ids, self.key_int)
        for j in range(1, self.window + 1):
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

    # ------------------------- KL machinery --------------------------------

    @staticmethod
    def _kl_of_delta(d: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Exact KL(P_wm || P_base) for the antipodal tilt, as a function of
        the tilt d >= 0 and the base green mass g = P_base(s = +1).

        With q(v) = p(v) e^{d s_v} / Z and Z = g e^d + (1-g) e^{-d}:
            E_q[s]        = (g e^d - (1-g) e^{-d}) / Z
            KL(q || p)    = d * E_q[s] - log Z
        All terms are computed in log space for numerical stability."""
        g = g.clamp(1e-9, 1 - 1e-9)
        log_g, log_1mg = torch.log(g), torch.log1p(-g)
        logZ = torch.logaddexp(log_g + d, log_1mg - d)
        # E_q[s] = exp(log_g + d - logZ) - exp(log_1mg - d - logZ)
        e_q_s = torch.exp(log_g + d - logZ) - torch.exp(log_1mg - d - logZ)
        return d * e_q_s - logZ

    def _solve_delta(self, g: torch.Tensor) -> torch.Tensor:
        """Vectorised bisection: largest d in [0, delta_max] with
        KL(d; g) <= kl_budget.  KL is continuous, increasing in d on [0, inf)
        with KL(0) = 0, so the root is unique and bisection converges
        geometrically: after k iters the bracket has width delta_max * 2^-k
        (40 iters => ~4e-12, far below float32 resolution)."""
        lo = torch.zeros_like(g)
        hi = torch.full_like(g, self.delta_max)
        # If even delta_max stays within budget (typical at g near 0 or 1,
        # where tilting barely moves the distribution), keep delta_max.
        within = self._kl_of_delta(hi, g) <= self.kl_budget
        for _ in range(self.bisect_iters):
            mid = 0.5 * (lo + hi)
            too_high = self._kl_of_delta(mid, g) > self.kl_budget
            hi = torch.where(too_high, mid, hi)
            lo = torch.where(too_high, lo, mid)
        return torch.where(within, torch.full_like(g, self.delta_max), lo)

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
        hashes = self._context_hashes(ctx_ids)  # (B, S)

        modified = logits.clone()
        for b in range(B):
            pos = mask_index[b].nonzero(as_tuple=True)[0]
            pos = pos[pos >= self.window]
            if pos.numel() == 0:
                continue
            for chunk in pos.split(self.chunk_size):
                s = self._scores_all_tokens(hashes[b, chunk])[:, :V]      # (P,V)
                # Base distribution this decoder will actually sample from.
                p0 = torch.softmax(
                    logits[b, chunk, :V].float() / self.temperature, dim=-1)
                # g_t = base probability mass on the green half at position t.
                g = (p0 * (s > 0).float()).sum(dim=-1)                    # (P,)
                # Largest tilt meeting the per-position KL budget, exactly.
                d = self._solve_delta(g)                                  # (P,)
                # Tilting logits by (temperature * d * s) tilts the sampled
                # distribution by exp(d * s): softmax((l + T d s)/T)
                # = p0 * exp(d s) / Z.
                modified[b, chunk, :V] += (self.temperature * d).unsqueeze(1) * s
        return modified

    # ------------------------------------------------------------------

    def detect(
        self,
        text: str,
        tokenizer: Any,
        p_value_threshold: float = 0.01,
        **kwargs,
    ) -> Dict[str, Any]:
        """Identical exact binomial detector as Proposal 1: the per-position
        delta_t is not observable from text alone, so the detector treats all
        scored positions equally.  The null is unaffected (still exactly
        Binomial(n, 1/2)); only the alternative's per-token mean varies."""
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
        k = int((s_unique > 0).sum())
        p_value = float(binom.sf(k - 1, n, 0.5))
        z_score = float((k - n / 2.0) / math.sqrt(n / 4.0))
        return {
            "is_watermarked": bool(p_value < p_value_threshold),
            "score": float(k),
            "n_scored": n,
            "z_score": z_score,
            "p_value": p_value,
        }


# --------------------------------- tests ----------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    V, S = 1000, 200
    wm = DWTWatermark(key="testkey", kl_budget=0.10, vocab_size=V, device="cpu")

    class MockTok:
        def encode(self, text, return_tensors=None):
            return torch.tensor([int(x) for x in text.split()],
                                dtype=torch.long).unsqueeze(0)
    tok = MockTok()

    # KL machinery correctness: brute-force check of the closed form.
    g = torch.tensor([0.5, 0.1, 0.9, 0.01])
    d = torch.tensor([1.0, 2.0, 0.5, 3.0])
    for gi, di in zip(g, d):
        p = torch.tensor([gi, 1 - gi])
        q = p * torch.exp(di * torch.tensor([1.0, -1.0]))
        q = q / q.sum()
        kl_ref = float((q * torch.log(q / p)).sum())
        kl = float(DWTWatermark._kl_of_delta(di.view(1), gi.view(1)))
        assert abs(kl - kl_ref) < 1e-5, (kl, kl_ref)
    # bisection meets the budget exactly (where not capped)
    dsol = wm._solve_delta(torch.tensor([0.3, 0.5, 0.7]))
    kl = DWTWatermark._kl_of_delta(dsol, torch.tensor([0.3, 0.5, 0.7]))
    assert torch.all((kl - wm.kl_budget).abs() < 1e-4)
    print(f"[kl] closed form + bisection verified; delta at g=0.5: {dsol[1]:.3f}")

    # Non-uniform base LM: Zipfian, so g varies across contexts and the
    # budget-capping actually engages.
    ranks = np.arange(1, V + 1)
    base_p = (1.0 / ranks); base_p /= base_p.sum()

    def simulate(kl_budget):
        wm.kl_budget = kl_budget
        ids = list(np.random.randint(0, V, size=wm.window))
        kls = []
        while len(ids) < S:
            h = wm._context_hashes(torch.tensor(ids + [0]).unsqueeze(0))[0, -1]
            sc = wm._scores_all_tokens(h.view(1))[0].numpy()
            # context-shuffled Zipf so g genuinely varies position to position
            perm = np.random.permutation(V)
            p0 = base_p[perm]
            g = torch.tensor([float(p0[sc > 0].sum())])
            d = float(wm._solve_delta(g))
            q = p0 * np.exp(d * sc); q /= q.sum()
            kls.append(float((q * np.log(q / p0)).sum()))
            ids.append(int(np.random.choice(V, p=q)))
        return ids, float(np.mean(kls))

    pvals = []
    for _ in range(300):
        ids = np.random.randint(0, V, size=S)
        pvals.append(wm.detect(" ".join(map(str, ids)), tok)["p_value"])
    print(f"[null] FPR@0.01={(np.array(pvals) < 0.01).mean():.4f}")

    det, zs, spent = 0, [], []
    for _ in range(50):
        ids, kl_mean = simulate(0.10)
        spent.append(kl_mean)
        r = wm.detect(" ".join(map(str, ids)), tok)
        zs.append(r["z_score"]); det += r["is_watermarked"]
    print(f"[wm kl=0.10] mean z={np.mean(zs):.2f}  TPR@1%FPR={det/50:.2f}  "
          f"realised KL/token={np.mean(spent):.4f} (cap respected)")
    assert det / 50 > 0.9 and np.mean(spent) <= 0.10 + 1e-6
    print("KLB self-test passed.")
