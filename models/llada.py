from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModel, AutoTokenizer

MASK_TOKEN_ID = 126336


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_torch_dtype(dtype_str: str):
    """
    Convert dtype string to torch dtype.
    
    Args:
        dtype_str: String representation of dtype (e.g., 'bfloat16', 'float32')
        
    Returns:
        Corresponding torch dtype
    """
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Add Gumbel noise to logits for sampling during generation.
    
    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        temperature: Temperature parameter for controlling randomness (0 = deterministic)
        
    Returns:
        Logits with Gumbel noise applied
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Calculate the number of tokens to unmask at each diffusion step.
    
    Args:
        mask_index: Boolean tensor indicating which tokens are masked [batch_size, seq_len]
        steps: Total number of diffusion steps
        
    Returns:
        Tensor of shape [batch_size, steps] indicating tokens to unmask per step
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate(
    model, 
    prompt: torch.Tensor, 
    steps: int = 128, 
    gen_length: int = 128,
    block_length: int = 128, 
    temperature: float = 0., 
    cfg_scale: float = 0.,
    remasking: str = 'low_confidence', 
    mask_id: int = MASK_TOKEN_ID,
    watermark = None
) -> torch.Tensor:
    """
    Generate text using LLaDA's masked diffusion process (reverse diffusion).
    
    This implements the core generation algorithm where tokens are iteratively
    unmasked over multiple diffusion steps. The generation proceeds in blocks,
    with each block going through a full diffusion process.
    
    Args:
        model: LLaDA model instance
        prompt: Input prompt tokens [1, prompt_len]
        steps: Total number of diffusion steps (should equal gen_length for best results)
        gen_length: Total length of text to generate
        block_length: Length of each generation block (gen_length must be divisible by this)
        temperature: Sampling temperature (0 = greedy, >0 = stochastic)
        cfg_scale: Classifier-free guidance scale (0 = no guidance)
        remasking: Strategy for selecting tokens to remask ('low_confidence' or 'random')
        mask_id: Token ID used for masking
        watermark: Optional watermark instance with apply_watermark method
        
    Returns:
        Generated sequence [1, prompt_len + gen_length]
    """
    device = prompt.device
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits
            
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")
            
            x0_p[:, block_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            if watermark is not None:
                confidence = watermark.apply_watermark(
                    logits=logits,
                    current_text=x,
                    mask_index=mask_index,
                    step=num_block * steps_per_block + i,
                    confidence=confidence,
                    x0=x0,
                    key=getattr(watermark, 'key', '')
                )
            
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            
            x[transfer_index] = x0[transfer_index]
            
    return x


class LLaDAGenerator:
    """
    Wrapper class for LLaDA model generation with convenient interface.
    
    Handles model loading, tokenization, and text generation using the
    masked diffusion process. Can be initialized from a config file or
    with explicit parameters.
    """
    
    def __init__(
        self,
        config: Optional[Union[str, Dict[str, Any]]] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype = None,
        watermark = None
    ):
        """
        Initialize LLaDA generator from config file or explicit parameters.
        
        Args:
            config: Path to YAML config file or config dictionary
            model_path: HuggingFace model path (overrides config if provided)
            device: Device to run model on (overrides config if provided)
            torch_dtype: Data type for model weights (overrides config if provided)
            watermark: Optional watermark instance for watermarking generation
        """
        self.watermark = watermark
        if config is not None:
            if isinstance(config, str):
                self.config = load_config(config)
            else:
                self.config = config
            
            model_config = self.config.get('model', {})
            gen_config = self.config.get('generation', {})
            tokenizer_config = self.config.get('tokenizer', {})
            
            self.model_path = model_path or model_config.get('model_path', 'GSAI-ML/LLaDA-8B-Instruct')
            self.device = device or model_config.get('device', 'cuda')
            dtype_str = model_config.get('dtype', 'bfloat16')
            self.torch_dtype = torch_dtype or get_torch_dtype(dtype_str)
            self.trust_remote_code = model_config.get('trust_remote_code', True)
            self.device_map = model_config.get('device_map', 'auto')
            
            self.default_gen_length = gen_config.get('max_length', 128)
            self.default_steps = gen_config.get('num_diffusion_steps', 128)
            self.default_temperature = gen_config.get('temperature', 0.0)
            
            self.tokenizer_path = tokenizer_config.get('tokenizer_path', self.model_path)
            self.use_chat_template = tokenizer_config.get('use_chat_template', True)
        else:
            self.model_path = model_path or "GSAI-ML/LLaDA-8B-Instruct"
            self.device = device or "cuda"
            self.torch_dtype = torch_dtype or torch.bfloat16
            self.trust_remote_code = True
            self.device_map = "auto"
            
            self.default_gen_length = 128
            self.default_steps = 128
            self.default_temperature = 0.0
            
            self.tokenizer_path = self.model_path
            self.use_chat_template = True
            self.config = None
        
        if not torch.cuda.is_available() and self.device == "cuda":
            print("Warning: CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, 
            trust_remote_code=self.trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            self.model_path, 
            trust_remote_code=self.trust_remote_code, 
            torch_dtype=self.torch_dtype,
            device_map=self.device_map
        ).eval()
        
        if self.device_map != "auto":
            self.model = self.model.to(self.device)
        else:
            self.device = next(self.model.parameters()).device
        
        self.mask_token_id = (
            self.tokenizer.mask_token_id 
            if hasattr(self.tokenizer, 'mask_token_id') and self.tokenizer.mask_token_id is not None 
            else MASK_TOKEN_ID
        )
    
    def generate_text(
        self,
        prompt: str,
        steps: Optional[int] = None,
        gen_length: Optional[int] = None,
        block_length: int = 32,
        temperature: Optional[float] = None,
        cfg_scale: float = 0.,
        remasking: str = 'low_confidence',
        use_chat_template: Optional[bool] = None,
        return_full_sequence: bool = False
    ) -> Union[str, Tuple[str, torch.Tensor]]:
        """
        Generate text from a prompt string. Uses config defaults if parameters not provided.
        
        Args:
            prompt: Input text prompt
            steps: Number of diffusion steps (defaults to config value)
            gen_length: Length of text to generate in tokens (defaults to config max_length)
            block_length: Block size for generation
            temperature: Sampling temperature (defaults to config value)
            cfg_scale: Classifier-free guidance scale
            remasking: Remasking strategy
            use_chat_template: Whether to apply chat template formatting (defaults to config value)
            return_full_sequence: If True, return (text, full_token_tensor) tuple
            
        Returns:
            Generated text string, or (text, tokens) tuple if return_full_sequence=True
        """
        steps = steps if steps is not None else self.default_steps
        gen_length = gen_length if gen_length is not None else self.default_gen_length
        temperature = temperature if temperature is not None else self.default_temperature
        use_chat_template = use_chat_template if use_chat_template is not None else self.use_chat_template
        
        if use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            chat_input = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
            input_ids = self.tokenizer(chat_input, return_tensors="pt")['input_ids'].to(self.device)
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids'].to(self.device)
        
        prompt_len = input_ids.shape[1]
        
        generated_tensor = generate(
            self.model,
            input_ids,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=self.mask_token_id,
            watermark=self.watermark
        )
        
        response_tokens = generated_tensor[0, prompt_len:]
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        if return_full_sequence:
            return response_text, generated_tensor
        return response_text
    
    def generate_tokens(
        self,
        prompt_tokens: torch.Tensor,
        steps: Optional[int] = None,
        gen_length: Optional[int] = None,
        block_length: int = 32,
        temperature: Optional[float] = None,
        cfg_scale: float = 0.,
        remasking: str = 'low_confidence'
    ) -> torch.Tensor:
        """
        Generate tokens from prompt tokens (lower-level interface). Uses config defaults if not provided.
        
        Args:
            prompt_tokens: Input prompt as token tensor [1, prompt_len]
            steps: Number of diffusion steps (defaults to config value)
            gen_length: Length of text to generate (defaults to config max_length)
            block_length: Block size for generation
            temperature: Sampling temperature (defaults to config value)
            cfg_scale: Classifier-free guidance scale
            remasking: Remasking strategy
            
        Returns:
            Generated token tensor [1, prompt_len + gen_length]
        """
        steps = steps if steps is not None else self.default_steps
        gen_length = gen_length if gen_length is not None else self.default_gen_length
        temperature = temperature if temperature is not None else self.default_temperature
        
        return generate(
            self.model,
            prompt_tokens,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=self.mask_token_id,
            watermark=self.watermark
        )


if __name__ == "__main__":
    print("Example 1: Loading from config file")
    generator_from_config = LLaDAGenerator(
        config="configs/llada_8b_instruct.yaml"
    )
    
    prompt = "What is the capital of France?"
    response = generator_from_config.generate_text(prompt)
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Config used: steps={generator_from_config.default_steps}, "
          f"max_length={generator_from_config.default_gen_length}, "
          f"temperature={generator_from_config.default_temperature}")
    
    print("\n" + "="*50 + "\n")
    
    print("Example 2: Loading with explicit parameters")
    generator_explicit = LLaDAGenerator(
        model_path="GSAI-ML/LLaDA-8B-Instruct",
        device="cuda"
    )
    
    response2 = generator_explicit.generate_text(
        prompt,
        steps=128,
        gen_length=128,
        block_length=32,
        temperature=0.0
    )
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response2}")
    
    print("\n" + "="*50 + "\n")
    
    print("Example 3: Using with watermark")
    from watermarks.simple_remask_watermark import SimpleRemaskWatermark
    
    watermark = SimpleRemaskWatermark(config={}, key="my_secret_key", delta=2.0)
    generator_with_watermark = LLaDAGenerator(
        model_path="GSAI-ML/LLaDA-8B-Instruct",
        device="cuda",
        watermark=watermark
    )
    
    response3 = generator_with_watermark.generate_text(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response with watermark: {response3}")

