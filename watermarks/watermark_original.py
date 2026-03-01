from typing import Any, Dict, List, Optional, Union
import math
import torch
import torch.nn.functional as F
import numpy as np
import hashlib
from scipy.stats import norm

class SimpleRemaskWatermark:

    def __init__(
        self,
        key: str = "",
        delta: float = 4.0,
        num_neighbors: int = 4,
        context_range: int = 30,
        adaptive_bias_threshold: float = 0.1,
        confidence_gate_threshold: float = 0.5,
    ):
        """
        Initializes the advanced watermarking algorithm.
        - delta: The strength of the watermark bias.
        - num_neighbors: The number of neighbors to check for each token (k).
        - context_range: The max distance of neighbors.
        - adaptive_bias_threshold: Small margin for the adaptive bias (epsilon).
        - confidence_gate_threshold: Probability threshold to activate the watermark.
        """
        self.key = key
        self.delta = delta
        self.num_neighbors = num_neighbors
        self.context_range = context_range
        self.adaptive_bias_threshold = adaptive_bias_threshold
        self.confidence_gate_threshold = confidence_gate_threshold

    @staticmethod
    def get_pair_color(key: str, token_a: int, token_b: int) -> int:
        """
        Deterministically returns a color (1 for green, -1 for red) for a token pair.
        """
        a = int(token_a)
        b = int(token_b)
        lo, hi = (a, b) if a < b else (b, a)
        h = hashlib.sha256()
        h.update(key.encode('utf-8'))
        h.update(b'|')
        if a == b:
            h.update(str(a).encode('utf-8'))
        else:
            h.update(str(lo).encode('utf-8'))
            h.update(b'|')
            h.update(str(hi).encode('utf-8'))
        v = int.from_bytes(h.digest()[:2], 'big')
        bit = v & 1
        
        color = 1 if bit == 1 else -1
        
        if a == b:
            return color
        
        # The order-dependent rule flips the color based on token order
        if a < b:
            return color
        else: # a > b
            return -color

    def _get_context_window(self, i: int) -> List[int]:
        """Generates a pseudo-random, position-dependent context window."""
        h = hashlib.sha256(f"{self.key}|{i}".encode('utf-8'))
        seed = h.digest()
        
        offsets = set()
        counter = 0
        while len(offsets) < self.num_neighbors:
            h = hashlib.sha256(seed)
            h.update(str(counter).encode('utf-8'))
            digest = h.digest()
            counter += 1
            for k in range(0, len(digest) - 1, 2):
                if len(offsets) >= self.num_neighbors:
                    break
                offset = int.from_bytes(digest[k:k+2], 'big') % (2 * self.context_range) - self.context_range
                if offset != 0:
                    offsets.add(offset)
        return list(offsets)

    @torch.no_grad()
    def bias_logits(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Applies the final advanced watermark: two-sided, score-averaging, adaptive, and gated.
        """
        x0_context = kwargs.get("x0_context")
        if x0_context is None:
            x0_context = torch.argmax(logits, dim=-1)

        top_k_candidates = 5

        batch_size, seq_len, vocab_size = logits.shape
        for b in range(batch_size):
            for i in range(seq_len):
                logits_slice = logits[b, i]
                
                # Confidence Gating: Check model's confidence before doing anything
                probs = F.softmax(logits_slice, dim=-1)
                if probs.max() > self.confidence_gate_threshold:
                    continue # Skip watermarking if model is already confident

                top_logits, top_indices = torch.topk(logits_slice, top_k_candidates)
                
                scores = []
                context_window = self._get_context_window(i)
                for candidate_token in top_indices:
                    candidate_token = candidate_token.item()
                    score_sum = 0.0
                    for j in context_window:
                        neighbor_index = i + j
                        if 0 <= neighbor_index < seq_len:
                            other_token = int(x0_context[b, neighbor_index].item())
                            color = self.get_pair_color(self.key, candidate_token, other_token) if j > 0 else self.get_pair_color(self.key, other_token, candidate_token)
                            score_sum += color
                    
                    # Implement Score Averaging
                    avg_score = score_sum / self.num_neighbors if self.num_neighbors > 0 else 0.0
                    scores.append(avg_score)

                t_orig_idx = 0
                t_green_idx = max(range(top_k_candidates), key=lambda k: scores[k])
                c_green = scores[t_green_idx]

                if c_green <= 0 or t_orig_idx == t_green_idx:
                    continue

                t_orig_logit = top_logits[t_orig_idx]
                t_green_logit = top_logits[t_green_idx]
                t_green_token = top_indices[t_green_idx]

                bias_needed = t_orig_logit - t_green_logit + self.adaptive_bias_threshold
                standard_bias = c_green * self.delta
                
                final_bias = min(standard_bias, bias_needed)
                
                if final_bias > 0:
                    logits[b, i, t_green_token] += final_bias
        
        return logits
    
    def detect(self, text: str, tokenizer: Any, **kwargs) -> Dict[str, Any]:
        """
        Detects the watermark using a two-sided z-score test.
        """
        z_threshold = kwargs.get("z_threshold", 4.0)
        tokens_tensor = tokenizer.encode(text, return_tensors="pt")[0]
        text_tokens = tokens_tensor.cpu().tolist() if isinstance(tokens_tensor, torch.Tensor) else tokens_tensor

        score_sum = 0
        checked_pairs = set()
        seq_len = len(text_tokens)

        for i in range(seq_len):
            context_window = self._get_context_window(i)
            token_i = text_tokens[i]

            for j in context_window:
                neighbor_index = i + j
                if 0 <= neighbor_index < seq_len:
                    pair = tuple(sorted((i, neighbor_index)))
                    if pair not in checked_pairs:
                        checked_pairs.add(pair)
                        token_j = text_tokens[neighbor_index]
                        
                        if i < neighbor_index:
                            color = self.get_pair_color(self.key, token_i, token_j)
                        else:
                            color = self.get_pair_color(self.key, token_j, token_i)
                        score_sum += color
        
        num_pairs_checked = len(checked_pairs)
        if num_pairs_checked == 0:
            return {"is_watermarked": False, "z_score": 0.0, "p_value": 1.0, "num_pairs": 0}

        z_score = score_sum / math.sqrt(num_pairs_checked) if num_pairs_checked > 0 else 0.0
        p_value = 1.0 - norm.cdf(z_score)
        
        return {
            "is_watermarked": z_score > z_threshold,
            "z_score": z_score,
            "p_value": p_value,
            "num_pairs": num_pairs_checked,
            "raw_score": score_sum,
            "z_threshold": z_threshold,
        }
