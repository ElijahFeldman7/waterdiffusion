import torch
from transformers import AutoTokenizer
from .watermark_interface import Watermark
from .random_utils import OnTheFlyGreenlist
from ..utils import (
    batched_multi_fft_convolution_idx,
    compute_prob_of_min,
    offset_unfold,
)
import torch.nn.functional as F
import scipy.stats as stats
import numpy as np

class OurWatermark(Watermark):
    def __init__(
        self,
        delta: float = 2.0,
        enforce_kl: bool = True,
        convolution_kernel: list[int] = [-1],
        greenlist_type: str = "bernoulli",
        greenlist_params: dict = {"gamma": 0.25},
        booster_only: bool = False,
        greenify_only: bool = False,
        tokenizer: AutoTokenizer = None,
        topk: int = 100,
        n_iter: int = 1,
        seeding_scheme: str = "sumhash",
        device: str = "cuda",
    ):
        
        assert not(booster_only and greenify_only), "You can not use only expectation boost and only predictive bias at the same time."
        
        self.seeding_scheme = seeding_scheme
        self.device = device
        self.salt_key = 42
        self.greenlist_type = greenlist_type
        self.greenlist_params = greenlist_params
        self.n_iter = n_iter
        self.delta = delta
        self.enforce_kl = enforce_kl
        self.booster_only = booster_only
        self.greenify_only = greenify_only

        vocab_size = len(tokenizer.get_vocab())
        self.vocab_size = vocab_size

        self.convolution_kernel = torch.tensor(
            convolution_kernel, device=self.device, dtype=torch.long
        )
        self.context_size = len(self.convolution_kernel)
        self.topk = topk

        # Greenlist
        self._init_greenlist_masks()

        self.temperature = None
        self.mask_token_id = None

        # For minhash, for generality, we allow any permutation of the vocabulary
        if self.seeding_scheme == "minhash":
            with torch.random.fork_rng(): # This ensures the permutation is always the same
                seed = 0
                torch.manual_seed(seed)
                self.permutation = torch.randperm(
                    self.vocab_size, device=self.device, dtype=torch.long
                ) 
                self.inv_permutation = torch.argsort(self.permutation) 

    def get_key_params(self):
        out = {
            "delta": self.delta,
            "convolution_kernel": self.convolution_kernel.tolist(),
            "topk": self.topk,
            "watermark_type": "DiffusionKGW_Optimal_Gaussian",
            "seeding_scheme": self.seeding_scheme,
            "greenlist_type": self.greenlist_type,
            "greenlist_params": self.greenlist_params,
            "enforce_kl": self.enforce_kl,
            "n_iter": self.n_iter,
        }
        return out

    def update_conv_kernel(self, new_kernel: list[int]):
        
        self.convolution_kernel = torch.tensor(
            new_kernel, device=self.device, dtype=torch.long
        )
        context_size = len(self.convolution_kernel)
        if context_size != self.context_size:
            self.context_size = context_size
            self._init_greenlist_masks()

    def _init_greenlist_masks(self):

        # Use a custom random implementation that is stateless so we can batch computations
        self.greenlist = OnTheFlyGreenlist(
            hash_size=self.context_size * self.vocab_size,
            vocab_size=self.vocab_size,
            mode=self.greenlist_type,
            distrib_params=self.greenlist_params,
            seed=42,
            device=self.device,
            dtype=torch.float32,
        )

    def get_boundaries(self, L: int) -> tuple[int, int]:
        """Returns the boundaries due to the hashing scheme."""
        start_at = max(-int(self.convolution_kernel.min()), 0)
        end_at = L - max(int(self.convolution_kernel.max()), 0)
        return start_at, end_at

    def get_hashes_sequences(
        self,
        input_ids: torch.LongTensor,  # (B, S)
    ) -> torch.LongTensor:  # (B, S, H)
        """Get the hashes sequences for the given input_ids and convolution kernel. Beware of the offset."""

        contexts = offset_unfold(
            input_ids,
            self.convolution_kernel,
        )  # (batch, seq_len, context_size)
        # Process the hash
        if self.seeding_scheme == "minhash":
            permuted_contexts = self.permutation[contexts]  # (B, S, context_size)
            hashes = permuted_contexts.min(dim=-1).values
        else:
            hashes = contexts.sum(dim=-1)

        return hashes

    def get_hashes_prob(
        self,
        probs: torch.FloatTensor,  # (B, S, V)
        convolution_kernel: torch.LongTensor,  # (context_size,)
        start_at: int = None,
        end_at: int = None,
    ) -> torch.FloatTensor:  # (B, S, H)
        """Get the hashes probabilities for the given probabilities and convolution kernel.

        Args:
            probs (torch.FloatTensor): The probabilities tensor of shape (B, S, V).
            convolution_kernel (torch.LongTensor): The convolution kernel tensor.
            start_at (int, optional): The start index for the valid range. Defaults to the minimum range allowed by the convolution kernel.
            end_at (int, optional): The end index for the valid range. Defaults to the maximum range allowed by the convolution kernel.

        """

        B, S, V = probs.shape

        if start_at is None:
            start_at = max(-int(convolution_kernel.min()), 0)
        if end_at is None:
            end_at = S - max(int(convolution_kernel.max()), 0)

        if self.seeding_scheme == "sumhash":
            hashes_prob = batched_multi_fft_convolution_idx(
                probs,
                convolution_idx=convolution_kernel,
                start_at=start_at,
                end_at=end_at,
            )
        elif self.seeding_scheme == "minhash":
            permuted_probs = probs[:, :, self.inv_permutation]
            hashes_prob = compute_prob_of_min(
                permuted_probs,
                convolution_kernel=convolution_kernel,
            )


        else:
            raise ValueError(f"Unknown seeding scheme: {self.seeding_scheme}")

        return hashes_prob

    def watermark_logits(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Watermark the logits. Returns a sampling logits and a remasking logits."""

        mask_index = input_ids == self.mask_token_id
        inv_mask = ~mask_index
        masked_logits = logits.clone()
        masked_logits[inv_mask] = -1e9  # Mask out the logits for non-masked tokens
        masked_logits[inv_mask, input_ids[inv_mask]] = 0

        with torch.enable_grad():
            logit_booster = self.compute_logit_booster(masked_logits, mask_index)

        logit_booster = logit_booster.to(logits.dtype)  # Ensure same dtype as logits

        watermarked_logits = logits + logit_booster

        return watermarked_logits, watermarked_logits
    
    def _compute_energy_topk(
        self,
        probs: torch.Tensor,  # (B, L, V)
        hashes_prob: torch.Tensor,  # (B, L, H)
        top_k_v: int = 32,  # tokens to keep  (k_v)
        top_k_h: int = 32,  # hashes to keep  (k_h)
    ) -> torch.Tensor:  # (B,)
        """
        Compute the energy function $J$ using top-k approximation.
        """
        B, L, V = probs.shape
        _, _, H = hashes_prob.shape
        dtype = probs.dtype
        
        if self.booster_only: # Expectation boost means no gradients through hashes_prob
            hashes_prob = hashes_prob.detach().requires_grad_(False)
        if self.greenify_only: # Predictive bias means no gradients through probs
            probs = probs.detach().requires_grad_(False)

        vals_v, idx_v = torch.topk(probs, k=top_k_v, dim=-1)  # (B,L,k_v)
        vals_h, idx_h = torch.topk(hashes_prob, k=top_k_h, dim=-1)  # (B,L,k_h)

        # reshape (B,L) → (B·L) so advanced indexing is vectorised
        BL = B * L
        idx_v_f = idx_v.reshape(BL, top_k_v)  # (BL,k_v)
        idx_h_f = idx_h.reshape(BL, top_k_h)  # (BL,k_h)

        # broadcast indices to shape (BL, k_h, k_v)
        h_inds = idx_h_f.unsqueeze(-1)  # (BL,k_h,1)
        v_inds = idx_v_f.unsqueeze(1)  # (BL,1,k_v)
        mask = self.greenlist.lookup(h_inds, v_inds)
        mask = mask.reshape(B, L, top_k_h, top_k_v)  # (B,L,k_h,k_v)

        contrib = (
            vals_h.unsqueeze(-1)  # (B,L,k_h,1)
            * vals_v.unsqueeze(-2)  # (B,L,1,k_v)
            * mask.to(dtype) 
        ).sum(dim=(1, 2, 3))  # (B,)

        return contrib

    def find_delta(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        alpha: torch.Tensor,
        var_alpha: torch.Tensor,
        mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """Solve the bisection for KL(p || q) = delta."""

        if mask is None:
            mask = torch.ones_like(var_alpha, dtype=torch.bool)  # (B,S,1)
        else:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            mask = mask.bool()

        lo = torch.zeros_like(var_alpha)  # (B,S,1)
        delta0 = torch.sqrt(2 * self.delta / (var_alpha + 1e-8))
        hi = delta0.mul_(2).clamp_max_(1e4)
        hi = torch.where(mask, hi, torch.zeros_like(hi))  # disable masked-out pos

        for _ in range(16):
            mid = 0.5 * (lo + hi)
            kl = self.kl_from_delta(mid, logits, probs, alpha)  # (B,S,1)

            too_high = (kl > self.delta) & mask  # update only if active
            hi = torch.where(too_high, mid, hi)
            lo = torch.where(too_high, lo, mid)

        delta = torch.where(mask, lo, torch.zeros_like(lo))

        return delta

    def kl_from_delta(self, d: torch.Tensor, logits, probs, alpha) -> torch.Tensor:
        B, S, V_logits = logits.shape
        V = self.vocab_size
        logit_booster = torch.zeros((B, S, V_logits), device=logits.device, dtype=logits.dtype)
        logit_booster[:,:,:V] = d * alpha

        q = F.softmax(
            (logits + logit_booster) / self.temperature, dim=-1
        )  # (B,S,V)

        kl = torch.sum(
            q * torch.log(q / probs), dim=-1, keepdim=True
        )

        kl[kl.isnan()] = float("inf")  # Avoid NaNs in the KL divergence
        kl = torch.clamp(kl, min=0.0)

        return kl

    def compute_logit_booster(
        self, logits: torch.FloatTensor, masked_inputs: torch.BoolTensor = None
    ) -> torch.FloatTensor:
        """Computes the logit booster for the given logits. For now we apply the watermark to all tokens. 
        This could be *significantly* optimized for production to apply only to masked tokens that are going to be unmasked.

        Args:
            logits (torch.FloatTensor): The logits tensor of shape (B, S, V).
            masked_inputs (torch.BoolTensor, optional): A boolean mask indicating which inputs are masked. Defaults to None.

        Returns:
            torch.FloatTensor: The logit booster tensor of shape (B, S, V).
        """

        B, S, _ = logits.shape
        V = self.vocab_size
        if masked_inputs is None:
            masked_inputs = torch.ones((B, S), dtype=torch.bool, device=logits.device)
            
        # Small optimization: skip the prompt
        first_mask_idx = masked_inputs[:, 0].nonzero(as_tuple=True)[0]
        min_mask_idx = first_mask_idx.min().item() if first_mask_idx.numel() > 0 else 0
        start_idx = min_mask_idx + self.convolution_kernel.min().item()
        start_idx = max(start_idx, 0)

        slice_logits = logits[:, start_idx:, :V]  # (B, S, V)
        probs = (
            torch.softmax(slice_logits / self.temperature, dim=-1)
            .detach()
            .requires_grad_(True)
        )  # (B, S, V)
        og_probs = probs.clone()
        
        B, S, _ = probs.shape

        for _ in range(self.n_iter):
            start_at, end_at = self.get_boundaries(S)
            hashes_prob = self.get_hashes_prob(
                probs=probs,
                convolution_kernel=self.convolution_kernel,
                start_at=start_at,
                end_at=end_at,
            )
            sliced_probs = probs[:, start_at:end_at, :]  # Align probs and hashes_prob

            J = self._compute_energy_topk(
                sliced_probs, hashes_prob, top_k_v=self.topk, top_k_h=self.topk
            )  # (B,)
            J.sum().backward()
            alpha = probs.grad

            if self.enforce_kl:
                mu_alpha = (og_probs * alpha).sum(dim=-1, keepdim=True)  # (B,S,1)
                var_alpha = (og_probs * (alpha - mu_alpha).square()).sum(
                    dim=-1, keepdim=True
                )  # (B,S,1)
                delta = self.find_delta(
                    slice_logits, og_probs, alpha, var_alpha, mask=masked_inputs
                )  # (B,S,1)

            else:
                delta = (
                    torch.ones((B, S, 1), device=logits.device, dtype=logits.dtype)
                    * self.delta
                    / (len(self.convolution_kernel) + 1)
                ) 

            probs = torch.softmax(
                (slice_logits + delta * alpha) / self.temperature,
                dim=-1,
            )  # (B, S, V)
            probs = probs.detach().requires_grad_(
                True
            ) 
            probs.grad = None  # Reset gradients for the next iteration

        logit_booster = torch.zeros_like(logits)
        logit_booster[:, start_idx:, :V] = alpha * delta  # (B, S, V)

        return logit_booster

    def detect(
        self,
        input_ids: torch.LongTensor,  # (seq_len)
        attention_mask: torch.FloatTensor = None,  # (seq_len)
    ) -> torch.FloatTensor:
        """Returns z-scores."""

        input_ids = input_ids.view(1, -1)
        attention_mask = (
            attention_mask.view(1, -1) if attention_mask is not None else None
        )

        if input_ids.shape[1] <= self.context_size:
            print("Input sequence is too short for detection.")
            out = {}
            return out

        input_ids = input_ids.to(self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).bool()
        attention_mask = attention_mask.to(self.device)

        _, S = input_ids.shape

        start_at, end_at = self.get_boundaries(S)

        hashes = self.get_hashes_sequences(
            input_ids
        )

        token_scores = self.greenlist.lookup(
            hashes, input_ids[:, start_at:end_at]
        )  # (batch, valid_length)

        # Mask repetitions of context + current token
        offsets = self.convolution_kernel
        if 0 not in offsets:
            offsets = torch.cat(
                [offsets, torch.tensor([0], device=self.device, dtype=torch.long)]
            )
        contexts = offset_unfold(
            input_ids,
            offsets,
        )  # (batch, seq_len, context_size)
        repetition_mask = self.mask_repetitions(contexts)  # (B, valid_length)

        # We compute the statistics using scipy for convenience
        # For now we enforce batch size of 1
        # This could be vectorized if needed
        for batch_idx in range(token_scores.shape[0]):
            token_scores = token_scores[batch_idx].detach().cpu().numpy()
            repetition_mask = repetition_mask[batch_idx].detach().cpu().numpy()

            token_scores_masked = token_scores[repetition_mask]

            if self.greenlist_type == "bernoulli":
                # We use a binomial test
                n_sucess = int(token_scores_masked.sum())
                n_trials = token_scores_masked.shape[0]
                results = stats.binomtest(
                    n_sucess,
                    n_trials,
                    self.greenlist_params["gamma"],
                    alternative="greater",
                )
                statistic, p_value = results.statistic, results.pvalue

            elif self.greenlist_type == "gaussian":
                # We use a z-test
                n_trials = token_scores_masked.shape[0]
                mean = token_scores_masked.mean()
                sigma = np.sqrt(1 / n_trials)
                statistic = mean / sigma
                p_value = stats.norm.sf(statistic)  # One-tailed p-value
                
            elif self.greenlist_type == "lognormal":
                # We use a Fenton–Wilkinson log-normal approximation
                n_trials = token_scores_masked.shape[0]
                s_obs = token_scores_masked.sum()
                statistic = token_scores_masked.mean()
                m  = np.exp(0.5)
                v  = (np.exp(1) - 1)*np.exp(1)
                
                sigma_S2 = np.log(1 + (n_trials*v)/(n_trials*m)**2)
                mu_S  = np.log(n_trials*m) - 0.5*sigma_S2
                
                p_value = 1 - stats.lognorm.cdf(s_obs, s=np.sqrt(sigma_S2),
                                        scale=np.exp(mu_S))

        out = {
            "z_score": float(statistic),
            "p_value": float(p_value),
            "token_color": token_scores.tolist(),
            "mask": repetition_mask.tolist(),
        }

        return out

    def mask_repetitions(self, contexts: torch.LongTensor) -> torch.BoolTensor:
        """
        contexts: (B, L, k)
        returns: (B, L) bool mask with True at first occurrence of each context along L
        """
        B, L, k = contexts.shape
        # pairwise comparison → (B, L, L)
        match = (contexts.unsqueeze(2) == contexts.unsqueeze(1)).all(-1)
        # mask out j >= i
        tril = torch.tril(
            torch.ones(L, L, dtype=torch.bool, device=contexts.device), diagonal=-1
        )  # (L, L)
        prev_match = match & tril.unsqueeze(0)  # (B, L, L)
        # first occurrence where there is no prior match
        return ~prev_match.any(dim=-1)
