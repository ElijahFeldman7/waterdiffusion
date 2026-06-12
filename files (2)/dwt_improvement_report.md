# DWT Watermark Improvement: Four Independent Proposals

Each proposal below is a self-contained redesign of `DWTWatermark`, implemented in its own script with a preserved `apply_watermark(logits, current_text, mask_index, step, confidence, x0, **kwargs)` signature and a self-contained `detect(text, tokenizer)` returning `is_watermarked`, `z_score`, `p_value`. All four were implemented, self-tested, and benchmarked head-to-head against the baseline at matched per-token KL. Empirical numbers quoted below come from those runs (V=1,000, 200-token sequences, N=80–300 trials per cell; full harness in `benchmark_vs_baseline.py`).

| Proposal | Script |
|---|---|
| 1 — CKAS: Context-Keyed Antipodal Scoring | `proposal_1_ckas.py` |
| 2 — ARC: Anchor-Resynchronized Carrier | `proposal_2_arc.py` |
| 3 — KLB: KL-Budgeted Exponential Tilting | `proposal_3_klb.py` |
| 4 — MWE: Multi-Window Ensemble | `proposal_4_mwe.py` |

---

## Proposal 1: CKAS — Context-Keyed Antipodal Scoring with an Exact Binomial Detector

### A — Conceptual Design

Replace both the static `token_values` vector and the absolute-position sinusoidal carrier with a single primitive: a keyed pseudorandom function (PRF) that maps `(key, w preceding tokens, candidate token)` to a score `s ∈ {−1, +1}`. Generation tilts every masked position's logits by `+delta` toward its context-local "+1" half of the vocabulary; detection recomputes the same scores from the observed text and runs an exact one-sided binomial test on the count of +1 tokens. This simultaneously eliminates the static partition (the ±1 assignment of every token changes with every context — R1), replaces the position-absolute carrier with a content-derived reference that survives insertions and deletions (R3), replaces the 1,000-sample Monte Carlo null with an exact finite-sample null (R4), and replaces DTW — which is far from the likelihood-ratio test for this signal model — with the exact LRT (R2).

The sinusoid is deliberately removed rather than repaired: modulating signal strength by `sin(2πft)` spends KL at full rate at carrier peaks and wastes positions at zero crossings, while contributing nothing to detection power that a matched filter could not extract from a constant carrier. CKAS spends the same KL uniformly and detects with the statistic that is provably optimal for the resulting model.

### B — Mathematical Derivation

**1. Signal model.** Let `h_t = PRF_key(x_{t−w}, …, x_{t−1})` be a 64-bit keyed hash of the `w`-token context (splitmix64 finalizer over key-salted token IDs; `w = 2` by default). Define the antipodal score

```
s(v; h_t) = 2·bit(PRF(h_t, v)) − 1  ∈ {−1, +1},  with  P[s = +1] = 1/2
```

independently (in the random-oracle/PRF model) across distinct `(h_t, v)` pairs. The watermarked conditional is the exponential tilt of the base LM:

```
P1(x_t = v | ctx) ∝ P0(v | ctx) · exp(δ · s(v; h_t))
```

The bias at position `t` is determined by the key and the *realized local context only* — under a context-agnostic adversary, `I(s(v; h_t); v) = 0` because the PRF output is an independent fair coin for every fresh context. R1 is satisfied not just heuristically but in the formal sense the task requests.

**2. Detection statistic from the likelihood ratio.** For observed tokens `x_{1:n}` with recomputable scores `s_t = s(x_t; h_t)`:

```
log P1/P0 = Σ_t [ δ·s_t − log Z_t ],   Z_t = g_t·e^{+δ} + (1−g_t)·e^{−δ}
```

where `g_t = Σ_{v: s(v;h_t)=+1} P0(v|ctx)` is the green mass. Conditional on the contexts, `log Z_t` does not depend on the observed token, so the LRT statistic is exactly `T = Σ_t s_t` — equivalently the count `K = #{t : s_t = +1}`. Because the tilted family has monotone likelihood ratio in `K`, the one-sided binomial test is **uniformly most powerful** for this family by the Karlin–Rubin theorem. Two caveats are stated rather than hidden: (i) the LRT conditions on contexts, ignoring the feedback by which watermarked tokens at `t` alter the context hash at `t+1` — this affects power under H1 but not the size of the test; (ii) under H1 the realized `g_t` distribution shifts slightly; the test remains exact-level under H0 regardless.

**3. Null distribution — exact, no Monte Carlo.** Under H0 the text is independent of the key, so each `s_t` is an independent Rademacher variable in the PRF model and

```
K | H0  ~  Binomial(n, 1/2)        (after deduplication, see below)
p = P[Binomial(n, 1/2) ≥ k] = binom.sf(k−1, n, 1/2)
```

This is exact for every `n`, so the 1% FPR threshold is set with zero estimation error — directly resolving R4. Repeated `(h_t, x_t)` pairs are not independent (the same PRF query returns the same bit), so the detector deduplicates identical (context-hash, token) pairs before counting, exactly as ZurichWatermark's `mask_repetitions` does; `n` is the number of *distinct* pairs. Measured null FPR at nominal 1%: 0.005 (n=200 trials) — consistent with exactness.

**4. SNR scaling.** Under H1 with tilt δ at green mass `g`, the mean of `s_t` is

```
E[s_t] = (g·e^δ − (1−g)·e^{−δ}) / Z − (2g − 1)
       = tanh(δ)                          at g = 1/2
       ≈ 4g(1−g)·δ                        for small δ
```

Taking expectation over the entropy of the base LM, `E[g] = 1/2` and `E[4g(1−g)] = 1 − Σ_v p_v² = 1 − e^{−H₂}` (Rényi-2 entropy), so

```
SNR = E[T|H1]/√Var[T|H0] ≈ √n · δ · (1 − e^{−H₂})
```

— scaling as `√n·δ`, independent of vocabulary size `V`, and degrading gracefully (not catastrophically) on low-entropy text. The baseline's DTW statistic has no closed-form SNR; empirically (matched-filter idealization, below) the baseline at best matches this √n·δ scaling and DTW realizes substantially less of it.

**5. Bias efficiency and operating point.** Per-position KL of the tilt at `g = 1/2`:

```
KL(P1‖P0) = δ·tanh(δ) − log cosh(δ)  ≈ δ²/2  for small δ
```

The Fisher-information bound for any per-token i.i.d. perturbation gives `SNR² ≤ 2n·KL/token` to first order; CKAS achieves `SNR² ≈ n·δ²(1−e^{−H₂})² ≈ 2n·KL·(1−e^{−H₂})²`, i.e. it is **Fisher-tight up to the entropy factor** — no i.i.d. scheme can beat it by more than that factor at small δ. The honest implication: the baseline *signal* is also Fisher-tight when paired with an ideal matched filter, so CKAS's measured R2 advantage comes from (a) DTW being far from the LRT, (b) the exact null versus a Gaussian fit to a 1,000-sample MC tail, and (c) robustness, not from a free-lunch improvement in the information channel. The minimal-KL operating point for a target z* is `δ = atanh(z*/√n)` — e.g. for n=200 and z*=3.1 (1% one-sided + margin), δ ≈ 0.22, KL ≈ 0.024 nats/token.

### C — Implementation

Complete implementation in **`proposal_1_ckas.py`** (runnable; `python3 proposal_1_ckas.py` executes the self-test). Key engineering points, all commented inline in the script:

- Hashing is pure tensorized int64 arithmetic (splitmix64 finalizer with role-separated odd multipliers); PyTorch int64 overflow wraps mod 2^64, which is exactly the semantics splitmix64 needs. No Python loops over the vocabulary.
- During generation the context at masked positions is taken from `where(mask_index, x0, current_text)` — the model's current best estimate `x0` stands in for not-yet-committed neighbours; the consequence of estimate churn is assessed honestly in Section E.
- The `(positions, V)` score matrix is built in chunks of 128 positions to bound peak memory at `O(chunk·V)`.
- Edge cases: positions `t < w` receive no bias and are excluded from detection; OOV tokens are clamped; sequences with fewer than 20 scoreable distinct pairs return `is_watermarked=False, p=1.0`.

### D — Robustness Profile

Measured TPR@1%FPR at matched per-token KL = 0.074 nats (baseline δ=0.98 ↔ CKAS δ=0.4), 200-token sequences:

| Condition | Baseline (DTW) | CKAS w=1 | CKAS w=2 | CKAS w=3 |
|---|---|---|---|---|
| Clean | 0.95 | 1.00 | 1.00 | 1.00 |
| 20% random substitution | 0.78 | 0.93 | 0.60 | 0.40 |
| 10% random deletion | 0.56 | 0.99 | 0.99 | 0.94 |
| 15% random deletion | 0.71 | — | ~0.90 | 0.76 |

1. **Random substitution (10/20/40%).** A substituted token at position `t` destroys its own score and corrupts the context hash of the next `w` positions: per-channel signal retention is `(1−ε)^{w+1}`. At 20% this is 0.51 for w=1 but 0.26 for w=3 — context keying *amplifies* substitution damage, which is why w is the central tradeoff dial. At 40% substitution, single-window CKAS retains `0.6² = 36%` of its mean shift; detection survives only on long texts or higher δ.
2. **Targeted green-token substitution.** To identify the green half at position `t` the adversary needs the key or an oracle for `PRF(h_t, ·)`. With `w ≥ 1` and a 64-bit keyed hash, observing API outputs reveals one sample per context; contexts virtually never repeat across queries, so list-reconstruction attacks that defeat static partitions collect zero reusable information. Residual vulnerability: an adversary with *unlimited* same-prefix resampling can estimate the green set for one specific context — expensive and non-transferable.
3. **Paraphrase (~50% token change).** Expected retention `(0.5)^{w+1}` of distinct pairs — w=1 retains 25% of signal mean, needing roughly 16× the clean sequence length at the same δ for equal power. Honest assessment: CKAS, like every token-level scheme here, is not paraphrase-robust by design; it merely degrades smoothly rather than collapsing.
4. **Insertion/deletion (5–15%).** This is where CKAS dominates the baseline: scores depend only on the local w-gram, so an edit perturbs exactly `w+1` positions and leaves the rest of the sequence's scores byte-identical. No alignment, no DTW, no resynchronization needed. Measured: TPR 0.99 vs baseline 0.56 at 10% deletion.
5. **Carrier/frequency attack.** There is no spectral carrier to notch out. Knowledge of the structure ("keyed w-gram partition") without the key offers only the generic attack — randomize tokens — which is attack 1.

### E — Honest Tradeoffs

- **Cost.** Generation: one `(positions × V)` int64 hash per denoising step — measured ~2–4× the baseline outer product's wall time on CPU, fully parallel on GPU; memory bounded by chunking. Detection is O(n) and faster than DTW's O(n·band).
- **Unmodeled failure modes.** (i) Early in diffusion decoding, `x0` context estimates churn; positions decoded against a context that later changes carry misaligned scores — diluting realized signal below the i.i.d. analysis. (ii) The PRF-independence null assumes the tokenizer re-tokenizes generated text canonically; merge ambiguity creates score mismatches at detection. (iii) Low-entropy spans (code, lists) push `g_t` toward 0/1 and contribute near-zero signal at full KL cost unless the entropy gate is enabled.
- **When worse than baseline.** Substitution-heavy threat models with w ≥ 2 (see table — baseline 0.78 vs w=3's 0.40 at 20% substitution); and extremely short texts (n < 30 distinct pairs) where the binomial test is underpowered at 1% FPR.
- **Falsifying experiment.** On a real masked-diffusion LM (LLaDA-8B or Dream-7B), generate 1,000 watermarked and 1,000 clean 200-token samples at five matched KL budgets for {baseline, CKAS w∈{1,2,3}}; if CKAS w=2 does not exceed baseline TPR@1%FPR on clean *and* 10% deletion at every budget, the central claim fails. The same grid under 20% substitution falsifies the recommended-default choice of w.

---
## Proposal 2: ARC — Anchor-Resynchronized Carrier with a Matched-Filter Detector

### A — Conceptual Design

Keep the baseline's sinusoidal-carrier identity but fix its two fatal couplings. First, the carrier's clock: instead of absolute position `t` (which a single insertion shifts globally, destroying DTW alignment beyond the band), ARC drives the sinusoid with a **content-derived clock** — `τ_t` = tokens elapsed since the last *anchor*, where anchors are bigrams whose keyed hash lands in a 1/8 fraction of hash space. Anchors are recomputable from the text alone, so any insertion or deletion corrupts only the segment containing the edit; the clock resynchronizes at the next anchor. Second, the partition: each inter-anchor segment draws its own keyed ±1 token partition and its own keyed phase offset from the segment's anchor hash, so no static vocabulary split exists (R1) and the spectral pattern cannot be estimated by averaging across documents. Detection abandons DTW for the matched filter — the LRT for a known-carrier signal — with an exact conditional variance, eliminating Monte Carlo (R4).

### B — Mathematical Derivation

**1. Signal model.** Anchor predicate `A(x_{t−1}, x_t) = 1{hash_key(bigram) mod 2^b = 0}` with `b = 3`, giving anchor probability 1/8 and mean segment length 8 ≈ one carrier period at `f = 0.125`. For position `t` in segment `σ` with intra-segment clock `τ_t`:

```
bias_t(v) = δ · sin(2πf·τ_t + φ_σ) · s_σ(v),
φ_σ  = keyed phase from anchor hash (4096 levels),
s_σ(v) ∈ {−1,+1}  = keyed per-segment partition.
```

The product `c_t = sin(2πf·τ_t + φ_σ)` is the carrier; the watermarked conditional is `P1(v) ∝ P0(v)·exp(δ·c_t·s_σ(v))`.

**2. Detection statistic from the likelihood ratio.** With recomputed carrier `c_t` and scores `s_t = s_σ(x_t)`:

```
log P1/P0 = Σ_t [ δ·c_t·s_t − log Z_t ],  Z_t independent of x_t given context
⇒ LRT statistic  T = Σ_t c_t · s_t    (the matched filter)
```

Duplicate `(segment-hash, token)` pairs share one PRF coin, so the detector groups them: with group weights `g_j = Σ_{t∈group j} c_t`, the statistic is `T = Σ_j g_j·ε_j` with independent Rademacher `ε_j`.

**3. Null distribution.** Exactly, `E[T|H0] = 0` and `Var[T|H0] = Σ_j g_j²` — computable from the observed text alone, no simulation. `z = T/√(Σ g_j²)` with a Gaussian p-value (Berry–Esseen error O(Σ|g_j|³/(Σg_j²)^{3/2}), small because |g_j| ≤ segment length × 1). The script also implements the assumption-free Hoeffding bound `p ≤ exp(−z²/2)`, exact-level at the cost of conservatism. Measured null FPR at nominal 1%: 0.0125 with n=400 trials — within binomial noise of nominal (95% CI ≈ [0.005, 0.022]).

**4. SNR scaling.** Under H1, `E[c_t·s_t] ≈ tanh(δ·c_t)·c_t ≈ δ·c_t²` for small δ, so

```
E[T|H1] ≈ δ·Σ c_t² ,   SNR ≈ δ·√(Σ c_t²) ≈ δ·√(n/2)
```

since `E[c²] = 1/2` over a period. The √2 penalty relative to CKAS's constant carrier is the price of keeping a sinusoid at all: half the KL budget is spent at low-|c| positions that contribute quadratically less signal.

**5. KL / bias efficiency and operating point.** Per-position KL ≈ `c_t²·δ²/2`; total KL ≈ `n·δ²/4`. Hence `SNR² ≈ 2·KL_total · (Σc²)²/(n·Σc²) = 2·KL_total · E[c²]·n/n` — ARC is also Fisher-tight *given* its carrier, but the carrier itself discards the √2 factor. Optimal operating point mirrors CKAS with δ_eff = δ/√2: for target z* choose `δ ≈ √2·z*/√n`. Defaults: δ=2.5, f=0.125 (period = mean segment length, maximizing within-segment phase coverage), b=3.

### C — Implementation

Complete implementation in **`proposal_2_arc.py`** (self-test included). Notable engineering, commented inline:

- Segment IDs via a vectorized `cummax` over anchor positions — no Python loop over the sequence; `τ_t` is `t − last_anchor_pos`.
- Per-segment partitions are realized lazily by hashing `(segment_anchor_hash, v)` with the same splitmix64 machinery as CKAS — never materializing per-segment vocab vectors.
- During generation, anchors are computed on `where(mask_index, x0, current_text)`; segments whose anchor bigram is still masked inherit the previous segment's clock (graceful, slightly diluted).
- Edge cases: text before the first anchor uses a key-derived global phase; segments shorter than 2 tokens contribute weight but tiny `Σc²`; detection requires ≥ 25 effective groups.

### D — Robustness Profile

Measured at matched KL where comparable; structural arguments otherwise:

1. **Random substitution (10/20/40%).** Two damage channels: a substituted token loses its own score (as in CKAS), and — worse — a substitution that creates or destroys an *anchor bigram* re-clocks and re-partitions the entire remainder of its segment plus the next segment boundary. Expected anchor-set damage at rate ε is ≈ 2ε·(anchored fraction). Measured: TPR 0.85 at 20% substitution (vs CKAS w=1 0.93, baseline 0.78). At 40% the anchor structure largely dissolves; detection fails on 200-token texts.
2. **Targeted green substitution.** Requires recovering per-segment partitions; each segment exposes ~8 tokens of one partition — far too few samples to estimate a 50/50 split over V=50k. Effectively key-secure.
3. **Paraphrase (~50%).** Anchors are bigrams of surface tokens; paraphrase rewrites most bigrams, so segments do not survive. ARC fails under paraphrase, full stop — stated as a design boundary, not minimized.
4. **Insertion/deletion (5–15%).** The headline repair: an edit inside a segment shifts `τ` only until the next anchor (mean 4 positions), so damage is O(edits·meanSegLen/2) positions instead of the baseline's O(n). Measured: TPR 0.98 at 10% deletion vs baseline 0.56. Deletions that remove an anchor merge two segments — bounded local damage.
5. **Carrier/frequency attack.** The per-segment keyed phase φ_σ is the defense: an adversary who knows f cannot subtract or notch the carrier without segment phases, and averaging across documents finds no stable spectral line because phases are i.i.d. per segment. A within-document periodogram on |s_t| reveals energy at f only if the adversary already has the partitions — circular.

### E — Honest Tradeoffs

- **Cost.** Comparable to baseline generation (one outer-product-like bias per step plus anchor hashing); detection O(n).
- **Unmodeled failure modes.** (i) Anchor density is text-dependent: anchor-free stretches (probability `(7/8)^L`) run the clock long past one period with no resync protection; (ii) x0 churn moves anchors between denoising steps, smearing phase early in decoding; (iii) the Gaussian p-value is approximate for short texts — the Hoeffding fallback is exact but conservative (~30% power loss at the 1% tail).
- **When worse than alternatives.** Everywhere CKAS is available, frankly: ARC pays √2 in SNR for the sinusoid and adds anchor fragility under substitution. Its justification is narrower — deployments that *require* a spectral/forensic carrier structure (e.g., compatibility with existing DWT-style spectral tooling, or layered watermarks where a frequency-domain second channel is wanted).
- **Falsifying experiment.** Generate matched-KL samples and apply a *combined* 10% substitution + 10% deletion attack; if ARC's TPR@1%FPR does not exceed the baseline's by ≥ 15 points (the resynchronization claim), the design premise — that anchor re-clocking is worth its substitution fragility — is falsified.

---

## Proposal 3: KLB — KL-Budgeted Exponential Tilting (Per-Position Distortion Control)

### A — Conceptual Design

CKAS and the baseline apply one global δ regardless of what each position's distribution looks like — spending heavy KL on confident (low-entropy) positions where the tilt either barely moves the distribution (g→0/1, wasted bias) or noticeably degrades quality. KLB ports ZurichWatermark's central mechanism into the diffusion `apply_watermark` interface: at every masked position, given the *actual current logits*, solve for the largest per-position tilt `δ_t ≤ δ_max` whose **exact** KL to the unwatermarked distribution stays within a hard budget `κ` nats (bisection, 40 iterations, closed-form KL — no sampling). The watermark thus carries a per-position quality guarantee: no single token's distribution is ever distorted by more than κ nats, while high-entropy positions receive the full detectable tilt. Scores are the same context-keyed PRF as CKAS (R1, R3, R4 inherited).

### B — Mathematical Derivation

**1. Signal model.** `P1(v) ∝ P0(v)·exp(δ_t·s(v;h_t))` with position-specific δ_t. For green mass `g_t` under P0, the partition function is `Z(δ) = g_t·e^δ + (1−g_t)·e^{−δ}` and the KL is available in closed form:

```
KL(δ) = δ·E_q[s] − log Z(δ),   E_q[s] = (g·e^δ − (1−g)e^{−δ})/Z(δ)
```

`KL(δ)` is continuous, strictly increasing on δ ≥ 0 (dKL/dδ = δ·Var_q[s] ≥ 0), and KL(0)=0 — so bisection on `[0, δ_max]` for `KL(δ) = κ` converges geometrically; 40 iterations give |KL−κ| < 10⁻⁹. Verified in the self-test against brute-force KL over the full distribution (agreement to 10⁻¹², and the realized average KL/token equals the budget to 4 decimal places: 0.1000 at κ=0.1).

**2. Detection statistic.** δ_t is a function of generation-time logits the detector never sees, so the weighted LRT `Σδ_t·s_t` is unobservable. KLB therefore uses the same unweighted exact binomial as CKAS — a deliberate, quantified mismatch: by Cauchy–Schwarz the unweighted statistic loses at most a factor `1/(1+CV²(δ))` in SNR² versus the oracle-weighted LRT, where CV is the coefficient of variation of δ_t across positions. With a budget-binding allocation, δ_t concentrates near the cap and CV is small (measured CV² ≈ 0.04 on the Zipfian test LM → ≤ 4% SNR² loss).

**3. Null distribution.** Identical to CKAS: deduplicated `K ~ Binomial(n, 1/2)` exactly under H0 — δ_t affects only H1. Measured null FPR at 1%: 0.0033–0.0067.

**4. SNR scaling.** `E[s_t] ≈ tanh(δ_t)` at g≈1/2; with the budget binding at δ̄ = KL⁻¹(κ), `SNR ≈ √n·tanh(δ̄)`. At κ=0.1 nats, δ̄(g=0.5) = 0.472 → measured mean z = 6.05 on 200-token sequences, TPR@1%FPR = 1.00.

**5. KL/bias efficiency — the honest part.** A small-δ Lagrange analysis of "maximize Σ E[s_t] subject to Σ KL_t ≤ B" shows the optimum equalizes `dE[s]/dKL ≈ 1/δ_t · const` across positions — and to first order this is achieved by **uniform δ**. Adaptive allocation therefore buys *no first-order detection gain*; its measured efficiency (SNR²/KL = 1.8 vs the Fisher bound of 2) matches CKAS. KLB's real contributions are second-order and operational: (i) a hard worst-case per-position distortion guarantee (uniform-δ schemes can spend 10× κ on a near-deterministic position for ~zero signal), (ii) the cap prevents budget overspend on low-entropy text, and (iii) O(δ⁴) efficiency gains in the strongly heterogeneous-entropy regime. The claim is quality-controlled watermarking at equal detection power, not more power.

### C — Implementation

Complete implementation in **`proposal_3_klb.py`** (self-test verifies the closed-form KL, bisection accuracy, budget adherence, and detection power on a Zipfian base LM). Engineering notes inline:

- Bisection is fully vectorized across all masked positions simultaneously (tensors of lo/hi bounds; no per-position Python loop); `log Z` via `logaddexp` for stability at large δ.
- Returns `logits + temperature·δ_t·s` so downstream softmax temperature does not silently rescale the calibrated KL.
- Edge cases: positions with g ∈ {0,1} (entire support one color) get δ_t = 0 — zero KL, zero signal, correctly skipped; δ_max=4 caps the tilt when κ is unreachable (extremely peaked distributions).

### D — Robustness Profile

Inherits CKAS's profile (same scores, same detector, context window 2 by default) for substitution, targeted substitution, paraphrase, and insertion/deletion — see Proposal 1's table. Differences:

1. **Effective robustness margin varies with text entropy.** On low-entropy spans KLB deliberately injects less signal (quality guarantee), so post-attack power degrades sooner there than fixed-δ CKAS — the price of the distortion cap, visible only under attack.
2. **Carrier/frequency attack.** None applicable (no carrier).
3. **Adaptive-adversary note.** δ_t depends on P0's logits; an adversary cannot exploit this without the base model *and* the key.

### E — Honest Tradeoffs

- **Cost.** The bisection adds ~40 vectorized passes over `(positions × 2)` scalars per denoising step — negligible next to the `(positions × V)` hashing it shares with CKAS. Requires logits (already in the signature).
- **Unmodeled failure modes.** (i) g_t is computed from the current denoising step's logits; the *final* committed distribution differs, so the realized KL drifts from κ — the guarantee is per-step, not per-final-sample; (ii) same x0/tokenizer caveats as CKAS.
- **When worse.** Detection-power-per-engineering-effort: it is strictly more machinery than CKAS for equal first-order power. Choose it only when the per-position distortion guarantee matters (production quality bars, human-facing text) or entropy is highly heterogeneous (code + prose mixtures).
- **Falsifying experiment.** Human/LLM-judge quality evaluation at matched *average* KL: KLB's premise is that capping per-position KL improves perceived quality. If pairwise preference between KLB and fixed-δ CKAS outputs at equal average KL and equal TPR shows no quality difference (e.g., < 52% preference, n=1,000 pairs), KLB's added complexity is unjustified.

---

## Proposal 4: MWE — Multi-Window Ensemble with a Chi-Bar-Squared Combiner

### A — Conceptual Design

Proposal 1 exposed a hard dial: small context windows resist substitution, large windows resist partition-estimation. MWE refuses to pick — it embeds `k=4` *independent* CKAS channels with windows w ∈ {1,2,3,4} and per-channel salts, splitting the KL budget so each channel runs at δ/√k (equal total KL). Detection computes each channel's exact binomial z and combines them with the one-sided chi-bar-squared statistic `X = Σ_j max(z_j, 0)²`, whose null distribution is an **exact finite mixture of chi-squares** — no Monte Carlo (R4). The ensemble degrades gracefully: whichever channels survive a given attack carry the detection, and the combiner's null is exact regardless of which channels died.

### B — Mathematical Derivation

**1. Signal model.** Channel j tilts by `(δ/√k)·s_j(v; h_t^{(j)})` where `h^{(j)}` hashes the last `w_j` tokens with salt_j. Biases add in logit space; cross-channel scores are independent coins (distinct salted PRFs), so total per-position KL ≈ `Σ_j (δ/√k)²/2 = δ²/2` — identical to single-channel CKAS at δ.

**2. Detection statistic.** Per channel: deduplicated count → exact binomial → continuity-corrected `z_j`. Channels are independent under H0 (distinct salts ⇒ distinct PRF queries; cross-terms vanish — verified empirically: null channel correlations < 0.03). The alternative is one-sided in every channel (any surviving channel has positive drift), for which the LRT against `μ ≥ 0, μ ≠ 0` yields the chi-bar-squared statistic:

```
X = Σ_{j=1}^{k} max(z_j, 0)²
```

**3. Null distribution — exact.** Under H0 each z_j is (asymptotically, with continuity correction) standard normal, independent, so

```
P[X > x | H0] = Σ_{m=1}^{k} C(k,m)·2^{−k} · P[χ²_m > x]
```

(the m channels with positive z contribute χ² mass; the binomial weights count sign patterns). Verified against 200,000-sample MC to three decimals. Measured null FPR at nominal 1%: 0.0033.

**4. SNR scaling.** Clean text: every channel carries `z_j ≈ √n·δ/√k·(1−e^{−H₂})`, so `E[X] ≈ n·δ²(1−e^{−H₂})²/1 + k` — same leading-order power as single-channel CKAS, minus a small omnibus penalty (testing k directions when all are alive). Under attack at substitution rate ε, channel j retains factor `(1−ε)^{w_j+1}`; the combiner automatically reweights toward survivors because dead channels contribute only their null χ²₁/2 share.

**5. KL/bias efficiency and operating point.** Fisher-tight in aggregate on clean text (same argument as CKAS, budget simply split). The omnibus penalty is the explicit price: at low total budget (δ=0.4) measured clean TPR is 1.00 but 20%-substitution TPR drops to 0.49 versus single-window w=1's 0.93 — at low SNR, splitting the budget four ways hurts more than diversification helps. MWE's operating region is **moderate-to-high budget under adversarial pressure**: at δ=2.4 it holds TPR 0.98 at 40% substitution, where every single-window design has collapsed. Optimal k: choose k ≈ number of distinct attack regimes you must survive; k=4 covers ε ∈ [0, 0.4].

### C — Implementation

Complete implementation in **`proposal_4_mwe.py`** (self-test includes the chi-bar-squared null verification and a per-channel degradation check). Engineering notes inline:

- Channel salts derived by SHA-256 of (key, j) — documented domain separation; channels reuse CKAS's chunked hashing with the salt folded into the role constants.
- The combiner's mixture tail is three lines of `scipy.stats.chi2.sf` — exact, fast, no caching.
- Edge cases: channels with < 20 distinct pairs are dropped and k reduced *before* computing the mixture null (the null adapts to the surviving channel count); per-channel dedup is per-salt (a pair duplicated in channel 1 may be fresh in channel 2).

### D — Robustness Profile

1. **Random substitution — the showcase.** Predicted per-channel mean z at 40% substitution from `(1−ε)^{w+1}`: [4.25, 2.55, 1.53, 0.92]; **measured: [4.14, 2.31, 1.29, 0.83]** — the retention law holds to within sampling error, and aggregate TPR@1%FPR = 0.98 at an attack level that destroys the baseline, CKAS w≥2, and ARC outright.
2. **Targeted green substitution.** Strictly harder than attacking CKAS: the adversary must estimate four independent partitions; flipping a token green→red in one channel is a coin flip in the other three.
3. **Paraphrase (~50%).** Channel 1 (w=1) retains 25% of its mean — the ensemble's paraphrase behaviour is essentially channel 1 alone at quarter budget. Weak, like everything here; slightly better than single-channel w=2+.
4. **Insertion/deletion (5–15%).** All channels are content-keyed (no positional carrier); measured TPR 0.88 at 10% deletion at the *low* δ=0.4 budget (omnibus penalty), → 1.00 at moderate budgets.
5. **Carrier/frequency attack.** Not applicable.

### E — Honest Tradeoffs

- **Cost.** Generation ≈ k× CKAS hashing (4 chunked passes) — the most expensive proposal; still GPU-parallel. Detection O(k·n).
- **Unmodeled failure modes.** (i) The Gaussian-z assumption inside the mixture null inherits binomial-discreteness error at small n per channel — continuity correction keeps measured FPR at/below nominal, but the exactness claim is per-channel-binomial + asymptotic combiner, slightly weaker than CKAS's fully exact null; (ii) channel independence relies on the PRF idealization across salts; (iii) same x0/tokenizer caveats.
- **When worse.** Low KL budgets and benign settings: the omnibus penalty and budget split make it strictly inferior to CKAS w=1 there (0.49 vs 0.93 at 20% substitution, δ=0.4). MWE is insurance — it costs premium when no disaster occurs.
- **Falsifying experiment.** The retention law is the load-bearing claim: run the per-channel z grid across ε ∈ {0.1, 0.2, 0.3, 0.4} on a real masked-diffusion LM; if measured per-channel means deviate from `(1−ε)^{w_j+1}` predictions by > 20% systematically (e.g., because real-text contexts repeat and dedup interacts with windows non-uniformly), the budget-split optimization and the choice k=4 lose their basis.

---

## Comparative Summary

**Proposal 1 — CKAS** is the recommended default and the strongest overall. It is the only design that simultaneously discharges all four weaknesses with the *minimum* mechanism: exact UMP detection (binomial LRT), exact null (R4), formal zero-MI context keying (R1), and edit locality (R3), while matching the Fisher efficiency bound so its R2 advantage over the baseline (1.00 vs 0.95 clean, 0.99 vs 0.56 at 10% deletion, matched KL) is realized rather than theoretical. Its one genuine dial — window size — is exposed, benchmarked, and defaulted to the w=2 compromise. Expected real-world rank: **1st**.

**Proposal 4 — MWE** is the choice when an adversary is assumed. It is the only design still detecting at 40% substitution (TPR 0.98), its channel-degradation predictions matched measurement almost exactly, and its combiner null is analytic. It pays for this with k× generation cost and a real omnibus penalty that makes it *worse* than CKAS at low budgets — deploy it at moderate-to-high KL in adversarial settings, not as a default. Rank: **2nd**.

**Proposal 3 — KLB** is honest about being a quality feature, not a power feature: uniform δ is already first-order KL-optimal, so its adaptive allocation buys a hard per-position distortion guarantee and low-entropy overspend protection rather than detection gains. It composes cleanly with CKAS (same scores/detector) and is the right add-on for production systems with strict output-quality bars. As a standalone power improvement it ranks below the two above. Rank: **3rd**.

**Proposal 2 — ARC** repairs the baseline's positional carrier as elegantly as a sinusoid can be repaired — content-clocked, per-segment keyed, matched-filter detected with exact variance — and decisively beats the baseline under insertion/deletion (0.98 vs 0.56). But it pays √2 SNR for keeping the carrier, adds anchor fragility under substitution (0.85 at 20%), and dies under paraphrase. It is the right choice only where spectral/forensic carrier structure is itself a requirement. Rank: **4th** on expected real-world performance, while remaining the most faithful evolution of the original DWT design.

Shared limits, stated once: all four schemes dilute when early-step x0 estimates churn during diffusion decoding, all degrade on low-entropy text in proportion to `1 − e^{−H₂}`, none survives strong paraphrase, and all assume canonical re-tokenization at detection.
