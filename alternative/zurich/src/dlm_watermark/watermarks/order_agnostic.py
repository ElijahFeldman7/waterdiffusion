import torch
from .watermark_interface import Watermark
from transformers import AutoTokenizer
from typing import List

class OrderAgnosticWatermark(Watermark):
    def __init__(
        self,
        l: int,
        transition_matrix: List[List[float]],
        initial_state: List[float],
        delta: float,
        tokenizer: AutoTokenizer,
        patterns: List[List[int]],
        pattern_length: int,
        device: str = "cuda",
    ):
        self.l = l
        self.transition_matrix = torch.tensor(transition_matrix, device=device)
        self.initial_state = torch.tensor(initial_state, device=device)
        self.delta = delta
        self.vocab_size = len(tokenizer.get_vocab())
        self.pattern_length = pattern_length
        self.patterns = patterns
        self._check_patterns()
        self.device = device
        self.key_space = self._get_key_space()
        self._init_greenlist()

    def _check_patterns(self):
        """Ensure all patterns are of the correct length and only contain valid colors."""

        for pattern in self.patterns:
            if len(pattern) != self.pattern_length:
                raise ValueError(
                    f"All patterns must be of length {self.pattern_length}, but got pattern of length {len(pattern)}"
                )
            for color in pattern:
                if color < 0 or color >= self.l:
                    raise ValueError(
                        f"Pattern contains invalid color {color}. Colors must be in range [0, {self.l - 1}]"
                    )

    def _init_greenlist(self):
        self.greenlists = []

        partition_size = self.vocab_size // self.l

        # Fork RNG
        with torch.random.fork_rng(devices=["cuda"]):
            torch.manual_seed(42)
            vocab_permutation = torch.randperm(self.vocab_size, device="cuda")

            for i in range(self.l):
                start = i * partition_size
                end = (i + 1) * partition_size
                if i == self.l - 1:  # last partition takes the remainder
                    end = self.vocab_size
                self.greenlists.append(vocab_permutation[start:end])

            self.current_state = self.sample_key(None)

    def set_temperature(self, temperature):
        """Hook to also reset the remasking seed when temperature is set. We always set the temperature before each generation."""
        self.key_sequence = None
        return super().set_temperature(temperature)

    def sample_key(self, prev_key: int) -> int:
        if prev_key is None:
            return torch.multinomial(self.initial_state, num_samples=1).item()

        return torch.multinomial(self.transition_matrix[prev_key], num_samples=1).item()

    def _get_key_space(self):
        key_space = torch.arange(0, self.l).long().cuda()
        return key_space

    def get_key_params(self):
        return {
            "l": self.l,
            "transition_matrix": self.transition_matrix.detach().cpu().numpy().tolist(),
            "initial_state": self.initial_state.detach().cpu().numpy().tolist(),
            "delta": self.delta,
        }


    def watermark_logits(self, input_ids: torch.LongTensor, logits: torch.FloatTensor):

        if self.key_sequence is None:
            #print("Sampling new key sequence")
            self.key_sequence = []
            for _ in range(input_ids.shape[1]):
                self.current_state = self.sample_key(self.current_state)
                self.key_sequence.append(self.current_state)

        # Increase the logits of the greenlist tokens
        for i, key in enumerate(self.key_sequence):
            logits[:, i, self.greenlists[key]] += self.delta

        return logits, logits

    def _get_token_color(self, token_id: int) -> int:
        for i, greenlist in enumerate(self.greenlists):
            if token_id in greenlist:
                return i
        return -1  # Token not found in any greenlist

    def _decompose_integer_base(self, number: int, base: int, length: int) -> List[int]:
        """Decompose an integer into a list of digits in the given base."""
        digits = []
        for _ in range(length):
            digits.append(number % base)
            number //= base
        return digits[::-1]  # Reverse to get the correct order
    
    def _get_pattern_occurence_probability(self, sequence_length: int) -> float:

        #Dynamic programming table:
        dp = torch.zeros((sequence_length, sequence_length - self.pattern_length + 1, self.l**(self.pattern_length -1)), dtype=torch.float64)
        dp[self.pattern_length -1, 0, :] = (1/self.l)**(self.pattern_length-1)

        for i in range(self.pattern_length, sequence_length):
            for s in range(self.l ** (self.pattern_length - 1)):
                tail = self._decompose_integer_base(s, self.l, self.pattern_length - 1)  
                for key in range(self.l):
                    prev_tail = [key] + tail[:-1]                         
                    s_prev = sum(d * (self.l ** idx) for idx, d in enumerate(reversed(prev_tail)))
                    hits_pattern = ([key] + tail) in self.patterns        
                    for c in range(i - self.pattern_length + 1):          
                        if hits_pattern:
                            dp[i, c + 1, s] += (1 / self.l) * dp[i - 1, c, s_prev]
                        else:
                            dp[i, c, s]     += (1 / self.l) * dp[i - 1, c, s_prev]

        prob_pattern = dp[sequence_length -1, :, :].sum(dim=(-1))

        return prob_pattern


    def detect(self, input_ids, attention_mask=None):
        input_ids = input_ids.view(-1).detach().cpu().tolist()

        sequence_length = len(input_ids)

        pattern_occurence = 0
        key_sequence = [
            self._get_token_color(token_id)
            for token_id in input_ids
            if token_id < self.vocab_size
        ]

        for i in range(self.pattern_length, sequence_length):

            if key_sequence[i - self.pattern_length:i] in self.patterns:
                pattern_occurence += 1

        prob_pattern = self._get_pattern_occurence_probability(sequence_length)
        fpr = prob_pattern[pattern_occurence:].sum().item()

        out = {
            "z_score": pattern_occurence,
            "p_value": fpr,
            "token_color": key_sequence,
        }

        return out
