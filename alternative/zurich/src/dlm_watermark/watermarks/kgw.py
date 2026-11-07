import torch
from torch.distributions import Normal
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from .watermark_interface import Watermark
from ..utils import offset_unfold
from typing import List


def additive_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    return salt_key * input_ids.sum().item()


def minimum_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    return salt_key * input_ids.min().item()


class KGWWatermarkAR(Watermark):
    """KGW Watermark for autoregressive models (adapted for diffusion)."""

    def __init__(
        self,
        gamma: float = 0.5,
        delta: float = 2.0,
        conv_kernel: List[int] = [-2, -1],
        tokenizer: AutoTokenizer = None,
        seeding_scheme: str = "sumhash",
    ):

        self.seeding_scheme = seeding_scheme

        device = "cuda"
        self.type = "kgw"
        self.device = device

        self.kgw_device = device

        self.salt_key = 42

        vocab_size = len(tokenizer.get_vocab())
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.convolution_kernel = torch.tensor(
            conv_kernel, device=self.device
        )
        self.k = len(conv_kernel)
        self.greenlist_size = int(self.vocab_size * self.gamma)

        # Greenlist
        self.rng = torch.Generator(device=self.device)
        self._init_greenlist_masks()

        self.temperature = None
        self.mask_token_id = None

    def get_key_params(self):
        out = {
            "gamma": self.gamma,
            "delta": self.delta,
            "convolution_kernel": self.convolution_kernel.cpu().tolist(),
            "watermark_type": "KGW",
        }
        return out

    def update_conv_kernel(self, new_kernel: list[int]):
        
        self.convolution_kernel = torch.tensor(
            new_kernel, device=self.device, dtype=torch.long
        )
        context_size = len(self.convolution_kernel)
        if context_size != self.context_size:
            self.context_size = context_size
            self.greelist_mask = None
            torch.cuda.empty_cache()  # Clear GPU memory
            self._init_greenlist_masks()

    def _init_greenlist_masks(self):

        assert self.vocab_size < 2**32, (
            "vocab_size must be less than 2^32 for the mask to fit in a tensor."
        )

        if self.seeding_scheme == "sumhash":
            hash_domain_size = self.k * self.vocab_size
        elif self.seeding_scheme == "minhash":
            hash_domain_size = self.vocab_size
        else:
            raise ValueError(f"Unknown seeding scheme: {self.seeding_scheme}")

        self.greenlist_mask = torch.full(
            (hash_domain_size, self.vocab_size),
            fill_value=False,
            dtype=torch.bool,
            device=self.kgw_device,
        )

        for i in tqdm(range(hash_domain_size), desc="Initializing greenlist masks"):
            greenlist_ids = self._get_greenlist_ids(
                torch.tensor(
                    [0] * (self.k - 1) + [i], dtype=torch.int32, device=self.device
                )  # Still seed on the main device
            )
            self.greenlist_mask[i, greenlist_ids] = True

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed RNG from local context. Not batched, because the generators we use (like cuda.random) are not batched."""
        # Need to have enough context for seed generation
        if input_ids.shape[-1] < self.k:
            raise ValueError(
                f"seeding_scheme requires at least a {self.k} token prefix to seed the RNG."
            )

        if self.seeding_scheme == "sumhash":
            prf_key = additive_prf(input_ids[-self.k :], salt_key=self.salt_key)
        elif self.seeding_scheme == "minhash":
            prf_key = minimum_prf(input_ids[-self.k :], salt_key=self.salt_key)
        else:
            raise ValueError(f"Unknown seeding scheme: {self.seeding_scheme}")
        # enable for long, interesting streams of pseudorandom numbers: print(prf_key)
        self.rng.manual_seed(
            prf_key % (2**64 - 1)
        )  # safeguard against overflow from long

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        self._seed_rng(input_ids.to(self.rng.device))

        greenlist_size = self.greenlist_size
        vocab_permutation = torch.randperm(
            self.vocab_size,
            device=input_ids.device,
            generator=self.rng,
            dtype=torch.long,
        )
        greenlist_ids = vocab_permutation[:greenlist_size]

        return greenlist_ids

    def hash(self, tokens: torch.LongTensor) -> torch.LongTensor:
        """Hash the given tokens along the last dimension to get the greenlist indices."""

        if self.seeding_scheme == "sumhash":
            hashes = tokens.sum(dim=-1)
        elif self.seeding_scheme == "minhash":
            hashes = tokens.min(dim=-1).values
        else:
            raise ValueError(f"Unknown seeding scheme: {self.seeding_scheme}")

        return hashes

    def get_greenlist(self, hashes: torch.LongTensor) -> torch.LongTensor:
        """Fetch the green tensors for the given hashes."""
        hashes = hashes.to(self.kgw_device)  #
        green_masks = self.greenlist_mask[hashes]  # (len(hashes), self.vocab_size)
        _, cols = green_masks.nonzero(as_tuple=True)
        greenlists = cols.view(
            -1, self.greenlist_size
        )  # (len(hashes), self.vocab_size)
        greenlists = greenlists.to(
            self.device
        )  # Move back to the main device if needed
        return greenlists

    def watermark_logits(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Watermark the logits. Returns a sampling logits and a remasking logits."""

        if self.mask_token_id is None:
            raise ValueError("Mask token ID must be set before watermarking logits.")

        # Compute the logit booster
        logit_booster = self.compute_logit_booster(input_ids, logits)
        watermarked_logits = logits + logit_booster
        
        # Apply the logit booster to the logits
        return watermarked_logits, watermarked_logits
    
    def compute_logit_booster(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        B, L = input_ids.shape
        logit_booster = torch.zeros_like(
            logits, dtype=torch.float32, device=logits.device
        )

        # 1) Compute k-gram windows of shape (B, valid_length, k)
        offsets = self.convolution_kernel   
        start_at = max(-min(offsets), 0)
        end_at = L - max(max(offsets), 0)
        valid_length = end_at - start_at
        windows = offset_unfold(input_ids, offsets)

        # 2) Which windows have *any* mask token?
        window_has_mask = (windows == self.mask_token_id).any(dim=-1)
        #    shape: (B, valid_length)

        # 3) Which positions do we want to boost?
        no_other_mask = ~window_has_mask
        #    final mask: last is mask AND no other mask in window
        boost_mask = no_other_mask  # (B, valid_length)

        hashes = self.hash(windows)  # (B, valid_length)

        # 5) Flatten to lookup green lists
        flat_hashes = hashes.reshape(-1)  # (B*(valid_length),)
        G = self.greenlist_size
        flat_green = self.get_greenlist(flat_hashes)  # (B*valid_length, G)
        green = flat_green.view(B, valid_length, G)  # (B, valid_length, G)

        # 6) Find which (batch, window) indices to boost
        coords = boost_mask.nonzero(
            as_tuple=False
        )  # (N, 2) pairs of (b, pos_in_hashes)
        if coords.numel() == 0:
            return logits  

        batch_idxs = coords[:, 0]  # (N,)
        win_pos = coords[:, 1]  # (N,)

        # map window-end positions back to original seq positions
        seq_pos = win_pos + start_at  # (N,)
        vocab_idxs = green[batch_idxs, win_pos]  # (N, G)

        valid = seq_pos < L  # boolean mask, (N,)
        batch_idxs = batch_idxs[valid]
        win_pos = win_pos[valid]
        seq_pos = seq_pos[valid]
        vocab_idxs = vocab_idxs[valid]

        # Expand batch & seq dims to match G (if not already done)
        batch_idxs = batch_idxs.unsqueeze(1).expand(-1, G)
        seq_pos = seq_pos.unsqueeze(1).expand(-1, G)

        # Finally add delta
        logit_booster[batch_idxs, seq_pos, vocab_idxs] += self.delta

        return logit_booster

    def detect(
        self,
        input_ids: torch.LongTensor,  # (seq_len)
        attention_mask: torch.FloatTensor = None,  # (seq_len)
    ) -> torch.FloatTensor:

        input_ids = input_ids.view(1, -1)
        attention_mask = (
            attention_mask.view(1, -1) if attention_mask is not None else None
        )

        if input_ids.shape[1] <= self.k:
            print("Input sequence is too short for detection.")
            out = {}
            return out

        input_ids_device = input_ids.device

        input_ids = input_ids.to(self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).bool()
        attention_mask = attention_mask.to(self.device)

        zscores, token_color, ignored_token_mask = self._detect(
            input_ids, attention_mask
        )
        cdf = Normal(0, 1).cdf(zscores)
        p_values = 1 - cdf

        zscore = zscores.to(input_ids_device).item()  # Convert to scalar
        p_value = p_values.to(input_ids_device).item()

        out = {
            "z_score": zscore,
            "p_value": p_value,
            "token_color": token_color[0].detach().cpu().tolist(),
            "mask": ignored_token_mask[0].detach().cpu().tolist(),
        }

        return out

    def get_token_color(self, input_ids: torch.LongTensor) -> torch.BoolTensor:
        """
        Returns a boolean mask indicating which tokens are in the greenlist.

        Args:
            input_ids (torch.LongTensor): Tensor of shape (batch, seq_len).

        Returns:
            torch.BoolTensor: A boolean tensor of shape (batch, seq_len),
                              where True means the token is in the greenlist.
        """
        input_ids = input_ids.to(self.device)

        token_color = torch.zeros_like(input_ids, dtype=torch.bool)

        offsets = self.convolution_kernel

        S = input_ids.shape[1]
        start_at = max(-min(offsets), 0)
        end_at = S - max(max(offsets), 0)

        for batch_idx in range(input_ids.shape[0]):
            for seq_idx in range(start_at, end_at):
                positions = seq_idx + offsets
                vals = input_ids[batch_idx].gather(dim=0, index=positions)
                hashes = self.hash(vals)
                hashes = torch.tensor([hashes], device=self.device)
                token_color[batch_idx, seq_idx] = input_ids[
                    batch_idx, seq_idx
                ] in self.get_greenlist(hashes)
        return token_color[:, start_at:end_at]

    def mask_repetitions(self, input_ids: torch.LongTensor) -> torch.BoolTensor:
        """
        Masks the repetition of "contexts" in the input_ids tensor.

        Args:
            input_ids (torch.LongTensor): Tensor of shape (batch, seq_len).

        Returns:
            torch.BoolTensor: A boolean tensor of shape (batch, seq_len),
                              where True means the token has no repeated "context"
        """

        convolution_idx = self.convolution_kernel
        offsets = convolution_idx

        S = input_ids.shape[1]
        start_at = max(-min(convolution_idx), 0)
        end_at = S - max(max(convolution_idx), 0)

        repetition_mask = torch.ones_like(input_ids, dtype=torch.bool)

        for batch_idx in range(input_ids.shape[0]):
            seen = set()  # Each sequence within a batch is treated independently
            for seq_idx in range(start_at, end_at):
                positions = seq_idx + offsets
                vals = input_ids[batch_idx].gather(dim=0, index=positions)
                vals = tuple(vals.tolist())

                if vals in seen:
                    repetition_mask[batch_idx, seq_idx] = False
                else:
                    seen.add(vals)

        return repetition_mask[:, start_at:end_at]

    def _detect(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        attention_mask: torch.FloatTensor = None,  # (batch, seq_len)
    ) -> torch.FloatTensor:
        # Resize the attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).bool()
        convolution_idx = self.convolution_kernel
        S = input_ids.shape[1]
        start_at = max(-min(convolution_idx), 0)
        end_at = S - max(max(convolution_idx), 0)
        attention_mask = attention_mask[:, start_at:end_at]  # (batch, valid_length)

        # Get the token color mask
        token_color = self.get_token_color(input_ids)  # (batch, valid_length)
        repetition_mask = self.mask_repetitions(input_ids)  # (batch, valid_length)

        # Mask of all ignored tokens
        ignored_tokens_mask = attention_mask * repetition_mask  # (batch, valid_length)
        token_color_masked = token_color * ignored_tokens_mask
        T = ignored_tokens_mask.sum(dim=1)  # (batch_size,)

        # Sum over the time dimension to get z-scores for each batch
        zscore = token_color_masked.sum(dim=1)  # (batch_size,)

        zscore = (zscore - self.gamma * T) / torch.sqrt(
            self.gamma * T * (1 - self.gamma)
        )

        return zscore, token_color, ignored_tokens_mask
