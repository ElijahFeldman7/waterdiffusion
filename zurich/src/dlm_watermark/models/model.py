import torch
from ..configs import ModelConfiguration
from dataclasses import dataclass
 
@dataclass
class DiffusionOutput:
    sequences: torch.LongTensor
    output_sequences: torch.LongTensor
    ages: torch.LongTensor
    
class DiffusionLM():
    
    def __init__(self, config: ModelConfiguration, hf_model):
        self.config = config
        if hf_model is not None:
            hf_model.eval()
            self.device = hf_model.device
            self.model = hf_model
            
    def __call__(self, *args, **kwargs):
        """
        Call the model with given arguments.
        
        Args:
            *args: Positional arguments for the model.
            **kwargs: Keyword arguments for the model.
        
        Returns:
            Output of the model.
        """
        return self.model(*args, **kwargs)
        
    def load_model(self, tokenizer_only: bool = False):
        """
        Load the model and tokenizer.
        
        Args:
            tokenizer_only (bool): If True, only load the tokenizer.
        
        Returns:
            model: Loaded model or None if tokenizer_only is True.
            tokenizer: Loaded tokenizer.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def generate_diffusion(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor = None, watermark = None, **kwargs) -> 'DiffusionOutput':
        """
        Generate text using the diffusion model.
        
        Args:
            prompt (torch.Tensor): Input prompt tensor.
            attention_mask (torch.Tensor, optional): Attention mask tensor.
            watermark (optional): Watermarking object.
            **kwargs: Additional keyword arguments for generation.
        
        Returns:
            torch.Tensor: Generated text tensor.
        """

        if watermark is not None:
            watermark.set_temperature(self.config.temperature)
            watermark.set_mask_token(mask_token_id=self.config.model_specific_arguments.get('mask_id', 151666))
    
    def eval(self):
        """
        Set the model to evaluation mode.
        """
        self.model.eval()
        
    def config_dict(self):
        """
        Get the model configuration as a dictionary.
        Returns:
            dict: Model configuration dictionary.
        """
        
        common_config = {
            "model_name": self.config.model_name,
            "steps": self.config.steps,
            "gen_length": self.config.gen_length,
            "temperature": self.config.temperature,
            "remasking": self.config.remasking,
        }
        
        specifc_config = self.config.model_specific_arguments.copy()
        
        output_config = {**common_config, **specifc_config}
        return output_config
