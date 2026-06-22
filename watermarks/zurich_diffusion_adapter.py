import sys
from pathlib import Path
from typing import Dict, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZURICH_SRC = PROJECT_ROOT / "zurich" / "src"
if str(ZURICH_SRC) not in sys.path:
    sys.path.insert(0, str(ZURICH_SRC))

from dlm_watermark.watermarks.diffusion_watermark import OurWatermark as ZurichDiffusionCore


class _VocabShim:
    def __init__(self, vocab_size: int):
        self._vocab_size = int(vocab_size)

    def get_vocab(self):
        return {str(i): i for i in range(self._vocab_size)}


class ZurichCompatWatermark:
    def __init__(
        self,
        key: str = "xooks",
        delta: float = 2.0,
        vocab_size: int = 0,
        device: str = "cuda:0",
    ):
        self.key = key
        self.delta = float(delta)
        self.device = device

        self.core = ZurichDiffusionCore(
            delta=self.delta,
            enforce_kl=False,
            convolution_kernel=[-1],
            greenlist_type="bernoulli",
            greenlist_params={"gamma": 0.25},
            topk=50,
            n_iter=1,
            seeding_scheme="sumhash",
            tokenizer=_VocabShim(vocab_size),
            device=device,
        )

    def set_temperature(self, temperature: float):
        self.core.set_temperature(temperature)

    def set_mask_token(self, mask_token_id: int):
        self.core.set_mask_token(mask_token_id)

    def watermark_logits(self, input_ids: torch.LongTensor, logits: torch.FloatTensor):
        return self.core.watermark_logits(input_ids, logits)

    def detect(
        self,
        text: str,
        tokenizer,
        p_value_threshold: float = 0.01,
    ) -> Dict[str, object]:
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask: Optional[torch.Tensor] = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        out = self.core.detect(input_ids=input_ids, attention_mask=attention_mask)
        z = float(out.get("z_score", 0.0))
        p = float(out.get("p_value", 1.0))

        return {
            "is_watermarked": bool(p < p_value_threshold),
            "score": z,
            "z_score": z,
            "p_value": p,
            "token_color": out.get("token_color", []),
            "mask": out.get("mask", []),
        }
