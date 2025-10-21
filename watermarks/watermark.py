from typing import Any, Dict, Optional, Tuple

import torch

class BaseDiffusionWatermark():
    """
    Abstract base class defining the interface for diffusion language model watermarks.
    
    All watermark implementations should inherit from this class and implement the
    required abstract methods for watermark application and detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the watermark with configuration parameters.
        
        Args:
            config: Dictionary containing watermark-specific configuration parameters
        """
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """
        Validate that the configuration contains all required parameters.
        
        Raises:
            ValueError: If required configuration parameters are missing or invalid
        """
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
        """
        Apply watermark by modifying the confidence scores used for remasking.
        
        This method receives the confidence scores (which determine which tokens
        get unmasked) and can modify them to implement watermarking through
        controlled token selection.
        
        Args:
            logits: The model's output logits [batch_size, seq_len, vocab_size]
            current_text: Current full token sequence [batch_size, seq_len]
            mask_index: Boolean tensor indicating which positions are masked [batch_size, seq_len]
            step: The current diffusion step
            confidence: Confidence scores for each position [batch_size, seq_len]
            x0: Predicted tokens at current step [batch_size, seq_len]
            key: Secret key for watermarking
            **kwargs: Additional model-specific parameters
            
        Returns:
            Modified confidence scores [batch_size, seq_len]
        """
        pass
    
    def detect(
        self,
        text: str,
        tokenizer: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Detect whether a given text contains the watermark.
        
        Args:
            text: The text to analyze for watermark presence
            tokenizer: Tokenizer for converting text to tokens
            **kwargs: Additional detection parameters
            
        Returns:
            Dictionary containing:
                - 'is_watermarked': Boolean indicating if watermark was detected
                - 'score': Numerical watermark score/statistic
                - 'threshold': Detection threshold used
                - Additional implementation-specific metrics
        """
        pass
    
    def compute_score(
        self,
        tokens: torch.Tensor,
        **kwargs
    ) -> float:
        """
        Compute a numerical score representing watermark strength.
        
        Args:
            tokens: Token sequence to analyze [seq_len] or [batch_size, seq_len]
            **kwargs: Additional scoring parameters
            
        Returns:
            Watermark score as a float
        """
        pass
    
    def get_detection_threshold(self, false_positive_rate: float = 0.01) -> float:
        """
        Calculate the detection threshold for a given false positive rate.
        
        Args:
            false_positive_rate: Desired false positive rate (default: 0.01)
            
        Returns:
            Detection threshold value
        """
        raise NotImplementedError("Subclass must implement get_detection_threshold")
    
    def compute_p_value(self, score: float) -> float:
        """
        Compute the p-value for a given watermark score.
        
        Args:
            score: The watermark score to analyze
            
        Returns:
            P-value indicating statistical significance
        """
        raise NotImplementedError("Subclass must implement compute_p_value")
    
    def reset_state(self) -> None:
        """
        Reset any internal state maintained by the watermark.
        
        This method is called between generation requests to ensure
        each generation starts with a clean state.
        """
        pass
    
    def __repr__(self) -> str:
        """
        String representation of the watermark instance.
        """
        return f"{self.__class__.__name__}(config={self.config})"

