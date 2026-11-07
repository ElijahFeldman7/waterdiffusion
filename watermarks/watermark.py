"""
Joint token + order watermark for masked diffusion LMs (LLaDA-style decoding).

Formulation
-----------
One decoding step makes two decisions:
    token channel : v_i ~ q_i(. | s)  for every masked position i
    order channel : commit the positions j ~ pi(. | s, v)
The base process uses q_i = p_i (model) and pi = pi_0 (confidence remasking).

We maximise the expected green count of the final text under a KL budget. The KL of the
output is bounded by the KL of the trajectory, which splits into per-step terms:
    KL(Q_x || P_x) <= sum_t E[ sum_i KL(q_i || p_i) + KL(pi || pi_0) ].
The Lagrangian solution is one exponential tilt per channel:
    q_i*  ∝ p_i  * exp(delta  * alpha_i)       alpha_i = Gloaguen et al. energy gradient
    pi*(j) ∝ pi_0(j) * exp(lambda * Delta_j)   Delta_j = marginal expected-green gain of
                                                committing j's proposal now vs. later
The theory ties delta and lambda through a single multiplier. Here they are separate
knobs, so we can trace the empirical Pareto frontier.

Nested special cases (same green list and same detector for all):
    order_strength = 0  ->  Gloaguen et al. ("Zurich") token-channel watermark
    delta = 0           ->  order-only watermark (dgMARK-like, but context-hashed)
    both > 0            ->  joint watermark

The dgMARK baseline (Hong & No, ICML 2026) is reproduced with the authors' own
compliance function, from alternative/dgmark-watermarking/src/utils.py.
NOTE: in that file, when a private_key is given, compliance reduces to
"token_id is even", independent of key and position (the key bit cancels).
We therefore default to private_key=None, i.e. parity(token) == parity(position),
which is the authors' default.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZURICH_SRC = PROJECT_ROOT / "alternative" / "zurich" / "src"
DGMARK_UTILS = PROJECT_ROOT / "alternative" / "dgmark-watermarking" / "src" / "utils.py"

if str(ZURICH_SRC) not in sys.path:
    sys.path.insert(0, str(ZURICH_SRC))

from dlm_watermark.watermarks.diffusion_watermark import OurWatermark as _ZurichCore  # noqa: E402

MASK_TOKEN_ID = 126336


class _VocabShim:
    def __init__(self, vocab_size: int):
        self._n = int(vocab_size)

    def get_vocab(self):
        return {str(i): i for i in range(self._n)}


# --------------------------------------------------------------------------------------
# Context-hashed green list: token channel (Zurich) + order channel (ours)
# --------------------------------------------------------------------------------------
class JointWatermark:
    """
    Args:
        vocab_size: len(tokenizer.get_vocab()), the same value Zurich uses.
        delta: token-channel strength (Zurich's delta. Zurich divides it by |C|+1 internally).
        order_strength: lambda, the order-channel tilt in log-confidence units per
            expected green token.
        kernel: context offsets. The order channel supports a single offset (default [-1]).
        gamma: green-list fraction.
        topk: top-k used by Zurich's energy computation.
        belief_topk: top-k truncation of the belief distributions used for Delta.
    """

    def __init__(
        self,
        vocab_size: int,
        delta: float = 0.0,
        order_strength: float = 0.0,
        kernel: Sequence[int] = (-1,),
        gamma: float = 0.25,
        topk: int = 50,
        belief_topk: int = 50,
        n_iter: int = 1,
        device: str = "cuda",
        mask_token_id: int = MASK_TOKEN_ID,
    ):
        self.delta = float(delta)
        self.order_strength = float(order_strength)
        self.kernel = list(kernel)
        self.gamma = float(gamma)
        self.belief_topk = int(belief_topk)
        self.device = device
        self.vocab_size = int(vocab_size)

        if self.order_strength > 0 and len(self.kernel) != 1:
            raise ValueError("Order channel currently supports a single context offset, e.g. kernel=[-1].")
        self.offset = int(self.kernel[0])

        self.core = _ZurichCore(
            delta=self.delta,
            enforce_kl=False,
            convolution_kernel=self.kernel,
            greenlist_type="bernoulli",
            greenlist_params={"gamma": self.gamma},
            topk=topk,
            n_iter=n_iter,
            seeding_scheme="sumhash",
            tokenizer=_VocabShim(self.vocab_size),
            device=device,
        )
        self.core.set_mask_token(mask_token_id)
        self.greenlist = self.core.greenlist

    # ---------------- token channel ----------------
    @torch.no_grad()
    def token_logits(self, x: torch.LongTensor, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """x: (1,S), logits: (1,S,V) float32. Returns the tilted logits (1,S,V).

        At temperature 0, Zurich's code divides by zero. The energy is then computed at T=1
        (the model's own distribution), and the tilt is applied before argmax.
        """
        if self.delta <= 0:
            return logits
        self.core.set_temperature(temperature if temperature > 0 else 1.0)
        wm_logits, _ = self.core.watermark_logits(x, logits)
        return wm_logits

    # ---------------- order channel ----------------
    def _belief(self, rows: torch.LongTensor, x_row: torch.LongTensor, mask_row: torch.BoolTensor,
                logits: torch.Tensor, belief_temp: float):
        """Top-k belief distribution over the final token at each row.
        Committed rows -> one-hot on the committed token. Masked rows -> top-k of softmax(logits/T)."""
        K = self.belief_topk
        V = self.vocab_size
        lg = logits[rows, :V].float() / belief_temp
        p = torch.softmax(lg, dim=-1)
        vals, ids = torch.topk(p, K, dim=-1)
        vals = vals / vals.sum(-1, keepdim=True).clamp_min(1e-12)

        committed = ~mask_row[rows]
        if committed.any():
            onehot_ids = torch.zeros_like(ids)
            onehot_ids[:, 0] = x_row[rows]
            onehot_vals = torch.zeros_like(vals)
            onehot_vals[:, 0] = 1.0
            ids = torch.where(committed[:, None], onehot_ids, ids)
            vals = torch.where(committed[:, None], onehot_vals, vals)
        return ids, vals

    def _G(self, h: torch.LongTensor, v: torch.LongTensor) -> torch.Tensor:
        # sumhash with one context token: hash = that token id
        return self.greenlist.lookup(h, v).float()

    @torch.no_grad()
    def order_gain(
        self,
        cand: torch.LongTensor,        # (m,) absolute positions of masked candidates
        proposals: torch.LongTensor,   # (m,) proposed token at each candidate
        x_row: torch.LongTensor,       # (S,) current sequence
        mask_row: torch.BoolTensor,    # (S,) True where masked
        logits: torch.Tensor,          # (S,V) logits the tokens are sampled from (tilted if delta>0)
        resp_start: int,
        resp_end: int,
        belief_temp: float = 1.0,
    ) -> torch.Tensor:
        """
        Delta_i = E[#green | commit v_i at i now] - E[#green | i stays masked]

        With a single offset c, green_j = G(x_{j+c}, x_j). Committing i affects:
          (A) its own green, green_i = G(x_{i+c}, x_i)
          (B) the green of j = i - c, which uses x_i as its hash, green_j = G(x_i, x_j)
        Only positions the detector scores (inside the response) are counted.
        """
        c = self.offset
        S = x_row.shape[0]
        v = proposals.clamp(0, self.vocab_size - 1)

        qi_ids, qi_p = self._belief(cand, x_row, mask_row, logits, belief_temp)  # masked -> top-k

        # (A) own green: context at i + c
        ctx = cand + c
        valid_a = (ctx >= resp_start) & (ctx < resp_end)
        ctx_c = ctx.clamp(0, S - 1)
        h_ids, h_p = self._belief(ctx_c, x_row, mask_row, logits, belief_temp)
        g1 = self._G(h_ids, v[:, None])                                   # (m,K)  G(h, v_i)
        g2 = self._G(h_ids[:, :, None], qi_ids[:, None, :])               # (m,K,K) G(h, v')
        exp_g2 = (g2 * qi_p[:, None, :]).sum(-1)                          # (m,K)  E_{v'~q_i} G(h, v')
        gain_a = (h_p * (g1 - exp_g2)).sum(-1) * valid_a.float()

        # (B) neighbour j = i - c uses x_i as its hash
        j = cand - c
        valid_b = (j >= resp_start) & (j < resp_end)
        j_c = j.clamp(0, S - 1)
        j_ids, j_p = self._belief(j_c, x_row, mask_row, logits, belief_temp)
        g3 = self._G(v[:, None], j_ids)                                   # (m,K)  G(v_i, x_j)
        g4 = self._G(qi_ids[:, :, None], j_ids[:, None, :])               # (m,K,K) G(h', x_j)
        exp_g4 = (g4 * qi_p[:, :, None]).sum(1)                           # (m,K)  E_{h'~q_i} G(h', x_j)
        gain_b = (j_p * (g3 - exp_g4)).sum(-1) * valid_b.float()

        return gain_a + gain_b

    # ---------------- detection ----------------
    @torch.no_grad()
    def detect_ids(self, resp_ids: torch.LongTensor) -> Dict:
        """Zurich detector: binomial test on deduplicated (context, token) green indicators."""
        n_ctx = max(abs(k) for k in self.kernel)
        if resp_ids.numel() <= n_ctx + 1:
            return {"p_value": 1.0, "z_score": 0.0, "green_rate": float("nan"), "n_scored": 0}
        out = self.core.detect(input_ids=resp_ids.to(self.device).view(1, -1))
        colors = np.asarray(out["token_color"], dtype=float)
        mask = np.asarray(out["mask"], dtype=bool)
        n = int(mask.sum())
        return {
            "p_value": float(out["p_value"]),
            "z_score": float(out["z_score"]),
            "green_rate": float(colors[mask].mean()) if n else float("nan"),
            "n_scored": n,
        }


# --------------------------------------------------------------------------------------
# dgMARK baseline (authors' compliance function)
# --------------------------------------------------------------------------------------
class DgMark:
    def __init__(self, private_key: Optional[str] = None):
        spec = importlib.util.spec_from_file_location("dgmark_utils", str(DGMARK_UTILS))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._check = mod.check_watermark_compliance
        self.private_key = private_key

    def compliant(self, abs_positions: torch.LongTensor, tokens: torch.LongTensor) -> torch.BoolTensor:
        # generation.py calls check_watermark_compliance(pos + 1, token, key) with pos an absolute index
        flags = [bool(self._check(int(p) + 1, int(t), self.private_key))
                 for p, t in zip(abs_positions.tolist(), tokens.tolist())]
        return torch.tensor(flags, dtype=torch.bool, device=abs_positions.device)

    def detect_ids(self, resp_ids: torch.LongTensor, prompt_len: int) -> Dict:
        """Match ratio with real_pos = prompt_len + j + 1, as in detection.py. One-sided binomial test vs 1/2."""
        n = int(resp_ids.numel())
        if n == 0:
            return {"p_value": 1.0, "z_score": 0.0, "match_rate": float("nan"), "n_scored": 0}
        pos = torch.arange(prompt_len, prompt_len + n)
        m = int(self.compliant(pos, resp_ids.cpu()).sum())
        z = (m - n / 2) / math.sqrt(n / 4)
        p = float(stats.binom.sf(m - 1, n, 0.5))
        return {"p_value": p, "z_score": float(z), "match_rate": m / n, "n_scored": n}


# --------------------------------------------------------------------------------------
# Unified LLaDA decoding loop (one loop for every method => identical sampler)
# --------------------------------------------------------------------------------------
def _num_transfer_tokens(n_masked: int, steps: int) -> List[int]:
    base, rem = divmod(n_masked, steps)
    return [base + (1 if i < rem else 0) for i in range(steps)]


def _gumbel(shape, device, dtype=torch.float64):
    u = torch.rand(shape, device=device, dtype=dtype).clamp_(1e-20, 1 - 1e-12)
    return -torch.log(-torch.log(u))


@torch.no_grad()
def generate(
    model,
    prompt_ids: torch.LongTensor,             # (1,P)
    *,
    kind: str = "none",                       # "none" | "joint" | "dgmark"
    wm: Optional[JointWatermark] = None,
    dg: Optional[DgMark] = None,
    steps: int = 300,
    gen_length: int = 300,
    block_length: int = 25,
    temperature: float = 0.5,
    remasking: str = "low_confidence",        # "low_confidence" | "random"
    sched_tau: float = 0.0,                   # >0: Plackett-Luce (Gumbel-top-k) scheduler, applied to all methods
    belief_temp: Optional[float] = None,
    mask_id: int = MASK_TOKEN_ID,
):
    """Returns (x, diag). x: (1, P+gen_length). diag: per-sequence diagnostics."""
    device = prompt_ids.device
    P = prompt_ids.shape[1]
    S = P + gen_length
    x = torch.full((1, S), mask_id, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    n_blocks = gen_length // block_length
    assert steps % n_blocks == 0, "steps must be divisible by the number of blocks"
    spb = steps // n_blocks
    if belief_temp is None:
        belief_temp = temperature if temperature > 0 else 1.0

    use_token = kind == "joint" and wm is not None and wm.delta > 0
    use_order = kind == "joint" and wm is not None and wm.order_strength > 0
    diag = {"token_kl_sum": 0.0, "sel_kl_sum": 0.0, "n_committed": 0, "n_steps": 0,
            "order_changed": 0, "gain_committed_sum": 0.0}

    for b in range(n_blocks):
        b0, b1 = P + b * block_length, P + (b + 1) * block_length
        n_tr = _num_transfer_tokens(int((x[0, b0:b1] == mask_id).sum()), spb)
        for step in range(spb):
            k = n_tr[step]
            mask_row = x[0] == mask_id
            cand = (mask_row[b0:b1].nonzero(as_tuple=True)[0] + b0)
            if k == 0 or cand.numel() == 0:
                continue
            k = min(k, cand.numel())

            logits = model(x).logits.float()                              # (1,S,Vl)
            samp_logits = wm.token_logits(x, logits, temperature) if use_token else logits
            samp_logits, logits = samp_logits[0], logits[0]              # (S,Vl)

            # proposals at the candidate positions (same Gumbel-max sampler for every method)
            lc = samp_logits[cand].double()
            if temperature > 0:
                lc = lc + temperature * _gumbel(lc.shape, device)
            v = lc.argmax(-1)                                            # (m,)

            # base remasking score (Zurich and LLaDA use the sampling logits for confidence)
            if remasking == "low_confidence":
                logp = torch.log_softmax(samp_logits[cand].double(), -1)
                base = logp.gather(-1, v[:, None]).squeeze(-1)           # log confidence
            elif remasking == "random":
                base = torch.log(torch.rand(cand.numel(), device=device, dtype=torch.float64))
            else:
                raise NotImplementedError(remasking)

            score = base.clone()
            if kind == "dgmark":
                comp = dg.compliant(cand, v)
                score = score + 1e4 * comp.double()                     # matched first, then by confidence
            elif use_order:
                gain = wm.order_gain(cand, v, x[0], mask_row, samp_logits, P, S, belief_temp).double()
                score = score + wm.order_strength * gain

            # diagnostics: selection KL (Plackett-Luce first pick) for the softened scheduler
            if sched_tau > 0 and (use_order or kind == "dgmark"):
                lp_w = torch.log_softmax(score / sched_tau, -1)
                lp_0 = torch.log_softmax(base / sched_tau, -1)
                diag["sel_kl_sum"] += float((lp_w.exp() * (lp_w - lp_0)).sum())

            noisy = score + sched_tau * _gumbel(score.shape, device) if sched_tau > 0 else score
            sel = torch.topk(noisy, k).indices
            if use_order or kind == "dgmark":
                base_noisy = base + sched_tau * _gumbel(base.shape, device) if sched_tau > 0 else base
                base_sel = set(torch.topk(base_noisy, k).indices.tolist())
                diag["order_changed"] += int(len(set(sel.tolist()) - base_sel))
                if use_order:
                    diag["gain_committed_sum"] += float(gain[sel].sum())

            pos = cand[sel]
            if use_token:
                bt = belief_temp
                lq = torch.log_softmax(samp_logits[pos].double() / bt, -1)
                lp = torch.log_softmax(logits[pos].double() / bt, -1)
                diag["token_kl_sum"] += float((lq.exp() * (lq - lp)).sum())
            x[0, pos] = v[sel]
            diag["n_committed"] += int(pos.numel())
            diag["n_steps"] += 1

    return x, diag


def trim_response(resp_ids: torch.LongTensor, stop_ids: Sequence[int]) -> torch.LongTensor:
    """Cut at the first stop/eos/mask token."""
    stop = torch.tensor(list(stop_ids), device=resp_ids.device)
    hit = torch.isin(resp_ids, stop).nonzero(as_tuple=True)[0]
    return resp_ids[: int(hit[0])] if hit.numel() else resp_ids
