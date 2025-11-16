import torch
from .watermark_interface import Watermark
from transformers import AutoTokenizer
import numpy as np
from scipy.stats import hypergeom

class UnigramWatermark(Watermark):
    def __init__(
        self,
        delta: float,
        gamma: float,
        tokenizer: AutoTokenizer,
        seed: int = 42,
        device: str = "cuda",
    ):
        self.seed = seed
        self.delta = delta
        self.gamma = gamma
        self.vocab_size = len(tokenizer.get_vocab())
        self.device = device    
       
        self._init_greenlist()

    def _init_greenlist(self):

        partition_size = int(self.vocab_size * self.gamma)

        # Fork RNG
        with torch.random.fork_rng(devices=[self.device]):
            torch.manual_seed(self.seed)
            vocab_permutation = torch.randperm(self.vocab_size, device=self.device)
            self.greenlist = vocab_permutation[:partition_size]

    def get_key_params(self):
        return {
            "seed": self.seed,
            "delta": self.delta,
            "gamma": self.gamma,
        }
    
    def watermark_logits(self, input_ids: torch.LongTensor, logits: torch.FloatTensor):

        # Increase the logits of the greenlist tokens
        logits[:, :, self.greenlist] += self.delta

        return logits, logits

    def _get_token_color(self, token_id: int) -> int:
        if token_id in self.greenlist:
            return 1
        return 0

    def detect(self, input_ids, attention_mask=None, alpha: float = 0.05):
        ids = input_ids.view(-1).detach().cpu().tolist()
        # keep only valid vocab ids, sample without replacement
        seq = list({t for t in ids if t < self.vocab_size})
        n = len(seq)
        if n == 0:
            return {"z_score": np.nan, "p_value": 1.0, "token_color": []}

        green_set = set(self.greenlist.tolist())
        color_sequence = [1 if t in green_set else 0 for t in seq]
        num_green = int(sum(color_sequence))

        gamma = self.gamma
        z_score = (num_green - gamma * n) / np.sqrt(gamma * (1 - gamma) * n)

        # Hypergeometric test
        N = int(self.vocab_size)
        K = int(len(self.greenlist))  
        p_value = float(hypergeom.sf(num_green - 1, N, K, n))

        return {
            "z_score": float(z_score),
            "p_value": p_value,
            "num_green": num_green,
            "n_unique": n,
            "token_color": color_sequence,
        }
