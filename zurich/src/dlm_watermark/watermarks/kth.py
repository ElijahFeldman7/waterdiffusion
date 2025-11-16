from typing import Optional, Dict, Any
from levenshtein_rust import permutation_test_parallel, detect
import torch
from .watermark_interface import Watermark

DEFAULT_SEED = 42

def permutation_test(tokens, key, n, k, vocab_size, n_runs=1000):
    generator = torch.Generator()  # generator is always cpu for reproducibility
    generator.manual_seed(key)

    xi = torch.rand((n, vocab_size), generator=generator, dtype=torch.float32)
    xi = xi.numpy()

    test_result = detect(tokens, n, k, xi, 0.0)
    # We use the rust implementation for speed. See in additional/ to compile it locally
    p_val = permutation_test_parallel(tokens, n, k, vocab_size, test_result, n_runs)

    return p_val

class KTHWatermark(Watermark):
    
    def __init__(
        self,
        vocab_size: int,
        key_len: int,
        seed: int = DEFAULT_SEED,
        device: Optional[str] = None,
        eps: float = 1e-20,
        num_shifts: int = 1,
    ):

        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        generator = torch.Generator()  # generator is always cpu for reproducibility
        generator.manual_seed(seed)

        uniform = torch.clamp(torch.rand((key_len, vocab_size), generator=generator, dtype=torch.float32), min=eps)
        self.gumbel = (-torch.log(torch.clamp(-torch.log(uniform), min=eps))).to(device)

        self.seed = seed
        self.eps = eps
        self.vocab_size = vocab_size
        self.device = device
        self.key_len = key_len
        self.cur_shift = 0
        self.num_shifts = num_shifts

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        index = (input_ids.shape[1] + self.cur_shift) % self.key_len
        gumbel = self.gumbel[index]  # (batch_size, vocab_size)
        return scores[..., :gumbel.shape[-1]] + gumbel
    
    def get_key_params(self):
        params = {
            "seed": self.seed,
            "key_len": self.key_len,
        }
        return params

    def watermark_logits(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        """Watermark the logits. Returns a sampling logits and a remasking logits."""
    
        logits_clone = logits.clone()
    
        index = (torch.arange(input_ids.shape[1], device=input_ids.device)) % self.key_len  # (seq_len,)
        gumbel = self.gumbel[index]  # (seq_len, vocab_size)
        # tokenizer vocab size and model outputs vocab size may be different
        logits_clone[..., :gumbel.shape[-1]] += self.temperature * gumbel  # (batch, seq_len, vocab_size)
        tokens = torch.argmax(logits_clone, dim=-1)  # (batch, seq_len)
        del logits_clone
        
        # Set to 0 only the selected tokens, others are set to -inf
        sampling_logits = torch.zeros_like(logits, device=logits.device) - float("inf")
        sampling_logits.scatter_(-1, tokens.unsqueeze(-1), 0.0)  
    
        return sampling_logits, logits
    
    def detect(
        self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor = None
    ) -> Dict[str, Any]:
        
        # Flatten the input_ids to a 1D array for the detection function
        input_ids = input_ids.view(-1)
        
        input_ids = input_ids.cpu().numpy()  # Convert to numpy array for the detection function
        
        pvalue = permutation_test(input_ids, self.seed, self.key_len, len(input_ids), self.vocab_size)
        
        out = {"p_value": pvalue}
            
        return out
    
