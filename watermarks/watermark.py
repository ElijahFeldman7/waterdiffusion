from typing import Any, Dict

import torch

class BaseDiffusionWatermark():
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        pass
    
    def apply_watermark(
        self,
        logits: torch.Tensor,
        current_text: torch.Tensor,
        mask_index: torch.Tensor,
        step: int,
        confidence: torch.Tensor,
        x0: torch.Tensor,
        key: str,
        **kwargs
    ) -> torch.Tensor:
        pass
    
    def detect(
        self,
        text: str,
        tokenizer: Any,
        **kwargs
    ) -> Dict[str, Any]:
        pass
    
    def compute_score(
        self,
        tokens: torch.Tensor,
        **kwargs
    ) -> float:
        pass
    
    def get_detection_threshold(self, false_positive_rate: float = 0.01) -> float:
        raise NotImplementedError("Subclass must implement get_detection_threshold")
    
    def compute_p_value(self, score: float) -> float:
        raise NotImplementedError("Subclass must implement compute_p_value")
    
    def reset_state(self) -> None:
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"

