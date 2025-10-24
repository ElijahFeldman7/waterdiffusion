from typing import Any, Dict

import torch
from scipy import stats

from watermarks.watermark import BaseDiffusionWatermark
from watermarks.utils import get_green_positions, is_position_green


class SimpleRemaskWatermark(BaseDiffusionWatermark):
    """
    Simple remask watermark implementation that passes through logits unchanged.
    
    This serves as a baseline/template for implementing custom watermarking strategies.
    """
    
    def __init__(self, config: Dict[str, Any], key: str = "", delta: float = 2.0):
        super().__init__(config)
        self.key = key
        self.delta = delta
    
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
        Apply watermark by modifying confidence scores for remasking.
        
        This baseline implementation returns confidence scores unchanged.
        Custom watermarking logic should modify the confidence scores to
        control which tokens get selected/unmasked at each diffusion step.
        
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            current_text: Current full token sequence [batch_size, seq_len]
            mask_index: Boolean tensor indicating masked positions [batch_size, seq_len]
            step: Current diffusion step
            confidence: Confidence scores for remasking [batch_size, seq_len]
            x0: Predicted tokens at current step [batch_size, seq_len]
            key: Secret key for watermarking
            **kwargs: Additional parameters
            
        Returns:
            Modified confidence scores [batch_size, seq_len] (currently unchanged)
        """
        modified_confidence = confidence.clone()
        
        # Iterate through sequence positions (0-indexed)
        for pos in range(confidence.shape[1]):
            if is_position_green(self.key, pos, green_percentage=0.5):
                modified_confidence[:, pos] += self.delta
        
        return modified_confidence
    
    def detect(
        self,
        text: str,  
        tokenizer: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Detect watermark in text using Welch's t-test on confidence scores.
        
        Separates positions into green and red lists based on the secret key,
        then compares the mean confidence scores between the two groups using
        a Welch's t-test.
        
        Args:
            text: Text to analyze
            tokenizer: Tokenizer instance
            **kwargs: Additional parameters
                - confidence_scores: Optional tensor of confidence scores [seq_len]
                - model: Optional model to compute confidence scores
                - p_value_threshold: Threshold for detection (default: 0.01)
            
        Returns:
            Detection result dictionary with:
                - is_watermarked: True if p-value < threshold
                - score: p-value from Welch's t-test
                - threshold: Detection threshold used
                - green_mean: Mean confidence of green positions
                - red_mean: Mean confidence of red positions
                - green_count: Number of green positions
                - red_count: Number of red positions
        """
        tokens = tokenizer.encode(text, return_tensors="pt")[0]
        seq_len = len(tokens)
        p_value_threshold = kwargs.get('p_value_threshold', 0.01)
        
        confidence_scores = kwargs.get('confidence_scores', None)
        
        if confidence_scores is None:
            model = kwargs.get('model', None)
            if model is not None:
                with torch.no_grad():
                    logits = model(tokens.unsqueeze(0)).logits[0]
                    probs = torch.softmax(logits, dim=-1)
                    confidence_scores = torch.gather(probs, 1, tokens.unsqueeze(-1)).squeeze(-1)
            else:
                confidence_scores = torch.ones(seq_len)
        
        if isinstance(confidence_scores, torch.Tensor):
            confidence_scores = confidence_scores.cpu().numpy()
        
        green_confidences = []
        red_confidences = []
        
        for pos in range(seq_len):
            confidence = float(confidence_scores[pos])
            
            if is_position_green(self.key, pos, green_percentage=0.5):
                green_confidences.append(confidence)
            else:
                red_confidences.append(confidence)
        
        if len(green_confidences) < 2 or len(red_confidences) < 2:
            return {
                "is_watermarked": False,
                "score": 1.0,
                "threshold": p_value_threshold,
                "green_mean": None,
                "red_mean": None,
                "green_count": len(green_confidences),
                "red_count": len(red_confidences),
                "error": "Insufficient data for t-test"
            }
        
        t_statistic, p_value = stats.ttest_ind(
            green_confidences, 
            red_confidences, 
            equal_var=False
        )
        
        is_watermarked = p_value < p_value_threshold
        
        return {
            "is_watermarked": is_watermarked,
            "score": float(p_value),
            "threshold": p_value_threshold,
            "t_statistic": float(t_statistic),
            "green_mean": float(sum(green_confidences) / len(green_confidences)),
            "red_mean": float(sum(red_confidences) / len(red_confidences)),
            "green_count": len(green_confidences),
            "red_count": len(red_confidences)
        }
    
    def compute_score(
        self,
        tokens: torch.Tensor,
        **kwargs
    ) -> float:
        """
        Compute watermark score. Returns 0.0 for baseline.
        
        Args:
            tokens: Token sequence to analyze
            **kwargs: Additional parameters
            
        Returns:
            Watermark score (0.0 for baseline)
        """
        return 0.0
