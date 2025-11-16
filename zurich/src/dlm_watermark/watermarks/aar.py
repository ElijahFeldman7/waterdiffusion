from typing import Optional, Dict, Any, List
import scipy.stats
import torch
from .watermark_interface import Watermark
from .random_utils import OnTheFlyGreenlist
from ..utils import offset_unfold


DEFAULT_SEED = 42


class AARWatermark(Watermark):
    def __init__(
        self,
        vocab_size: int,
        conv_kernel: List[int] = [-2, -1],
        seed: int = DEFAULT_SEED,
        eps: float = 1e-20,
        device: Optional[str] = None,
    ):

        vocab_size += 10 # For including special tokens

        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # clamp to avoid NaNs
        k = len(conv_kernel)
        self.uniform = OnTheFlyGreenlist(
            hash_size = vocab_size * k,
            vocab_size = vocab_size,
            mode = "uniform",
            device = device 
        )

        self.convolution_kernel = torch.tensor(
            conv_kernel, device=self.device
        )
        self.vocab_size = vocab_size
        self.seed = seed
        self.eps = eps
        
    def get_key_params(self):
        out = {
            "convolution_kernel": self.convolution_kernel.tolist(),
        }
        return out

    def hash(self, tokens: torch.LongTensor) -> torch.LongTensor:
        """Hash the given tokens along the last dimension to get the greenlist indices."""
        # We only do sumhash for AAR
        hashes = tokens.sum(dim=-1)
        return hashes

    def get_scores(self, hashes: torch.LongTensor) -> torch.LongTensor:
        """Fetch the green tensors for the given hashes."""
        hashes = hashes
        vocab = torch.arange(self.vocab_size, device=self.device, dtype=torch.long)

        hashes = hashes.unsqueeze(-1).unsqueeze(-1) # (B, 1, 1)
        vocab = vocab.expand(hashes.shape[0], 1, self.vocab_size) # (B, 1, V)

        uniform_scores = self.uniform.lookup(hashes, vocab)  # (B, 1, V)
        uniform_scores = uniform_scores.squeeze(1)  # (B, V)

        gumbel_scores = (-torch.log(torch.clamp(-torch.log(uniform_scores), min=self.eps)))  # (B, V)
        return gumbel_scores

    def watermark_logits(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Watermark the logits. Returns a sampling logits and a remasking logits."""

        if self.mask_token_id is None:
            raise ValueError("Mask token ID must be set before watermarking logits.")

        watermarked_logits = self.watermark_logits_argmax(input_ids, logits.clone())

        # Apply the logit booster to the logits
        return watermarked_logits, logits
        
    def watermark_logits_argmax(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        logits: torch.FloatTensor,  # (batch, seq_len, vocab_size)
    ) -> torch.LongTensor:
        """Finds argmax token for watermark, returns token indexes to be used for cross-entropy loss.
        
        Returns tensor of shape (batch, seq_len), where each element is a token index.
        """
        B, L = input_ids.shape
        V = self.vocab_size
        V = min(self.vocab_size, logits.shape[-1])  # Ensure we don't exceed logits size

        # 1) Compute k-gram windows of shape (B, valid_length, k)
        offsets = self.convolution_kernel   
        start_at = max(-min(offsets), 0)
        end_at = L - max(max(offsets), 0)
        valid_length = end_at - start_at
        windows = offset_unfold(input_ids, offsets)

        # 2) Which windows have *any* mask token?
        window_has_mask = (windows == self.mask_token_id).any(dim=-1)

         # 3) Which positions do we want to boost?
        no_other_mask = ~window_has_mask
        currently_mask = (input_ids == self.mask_token_id)[:,start_at:end_at]
        boost_mask = no_other_mask * currently_mask  # (B, valid_length)
    
        # 4) Hash each window
        hashes = self.hash(windows)  # (B, valid_length)

        # 5) Flatten to lookup green lists
        flat_hashes = hashes.reshape(-1)  # (B*(valid_length),)
        gumbel_scores = self.get_scores(flat_hashes)  # (B*valid_length, V)
        gumbel_scores = gumbel_scores.view(B, valid_length, V)  # (B, valid_length, V)

        # 6) Find which (batch, window) indices to boost
        coords = boost_mask.nonzero(
            as_tuple=False
        )  # (N, 2) pairs of (b, pos_in_hashes)
        if coords.numel() == 0:
            return logits  # nothing to do

        batch_idxs = coords[:, 0]  # (N,)
        win_pos = coords[:, 1]  # (N,)
        seq_pos = win_pos + start_at # (N,)
        valid = seq_pos < L  # boolean mask, (N,)
        win_pos = win_pos[valid]
        seq_pos = seq_pos[valid]
        
        # Important, temperature needs to be taken into account
        gumbel_logits = logits[batch_idxs, seq_pos, :V] + self.temperature * gumbel_scores[batch_idxs, win_pos, :V]
        argmax_logits = gumbel_logits.argmax(dim=-1)  # (N,)


        sampling_logits = torch.zeros_like(gumbel_logits) - float("inf")
        sampling_logits.scatter_(-1, argmax_logits.unsqueeze(-1), 0.0)



        logits[batch_idxs, seq_pos,:V] = sampling_logits.to(logits.dtype)
        logits[batch_idxs, seq_pos,V:] = -float("inf")  # Set the rest to -inf

        return logits
    
    def detect(
        self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor = None
    ) -> Dict[str, Any]:
        """
        Returns p-value, where null hypothesis is that the text is not watermarked.
        
        Under null hypothesis, each u is Uniform(0, 1), so each score (-log(1 -u )) is Exp(1).
        So the sum of scores is distributed as Gamma(n_tokens, 1).
        """

        input_ids = input_ids.view(1,-1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).bool()

        L = input_ids.shape[1]

        offsets = self.convolution_kernel   
        start_at = max(-min(offsets), 0)
        end_at = L - max(max(offsets), 0)

        hashes = offset_unfold(input_ids, self.convolution_kernel).sum(dim=-1).view(-1,1,1) # (valid_length,1,1)
        tokens = input_ids[0,start_at:end_at].view(-1,1,1) # (valid_length,1,1)

        scores = self.uniform.lookup(hashes, tokens).view(-1)  # (valid_length,)

        repetition_mask = self.mask_repetitions(input_ids)[0] # (valid_length)
        attention_mask = attention_mask[0,start_at:end_at]
        mask = repetition_mask* attention_mask
        n_tokens = torch.sum(mask).item()
        scores_masked = scores[mask]
        scores_masked = torch.clamp(scores_masked, max=1-1e-9)  # Avoid log(0)
        score = torch.sum(-torch.log(1-scores_masked)).item()
        
        p_value = scipy.stats.gamma.sf(score, n_tokens, loc=0, scale=1).item()

        out = {
            "z_score": score,
            "p_value": p_value,
            "token_color": scores.detach().cpu().tolist(),
            "mask": mask.detach().cpu().tolist(),
        }

        return out

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
