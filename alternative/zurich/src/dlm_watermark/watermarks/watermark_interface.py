from typing import Dict, Any, Tuple
import torch

class Watermark():
    """
    Base class for watermarks.
    This class is used to define the interface for watermarks.
    """

    def __init__(self, *args, **kwargs):
        pass
    
    def get_key_params(self) -> Dict[str, Any]:
        """
        Get the key parameters of the watermark.
        Returns a dictionary with the key parameters.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def set_temperature(self, temperature):
        """Set the temperature for the watermarking."""
        self.temperature = temperature
    
    def set_mask_token(self, mask_token_id: int):
        """Set the mask token ID for the watermarking."""
        self.mask_token_id = mask_token_id

    def watermark_logits(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Watermark the logits. Returns a sampling logits and a remasking logits. 
        The sampling is used to sample the next sequence. The remasking is used for the remasking step.

        This is useful for distortion-free watermarks where the sampling is turned into a dirac-distribution.
        """
        sample_logits, remask_logits = logits, logits
        return sample_logits, remask_logits

    def detect(
        self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor = None
    ) -> Dict[str, Any]:
        """
        Detect the watermark in the input_ids.
        Returns a dictionary with the detection results.
        """
        raise NotImplementedError("This method should be implemented in subclasses."
    )