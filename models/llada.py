from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModel, AutoTokenizer

MASK_TOKEN_ID = 126336


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_torch_dtype(dtype_str: str):
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)

def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, torch_dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, torch_dtype=torch.int64) + base
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
    device = prompt.device
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, torch_dtype=torch.long, device=device)
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
            
            transfer_index = torch.zeros_like(x0, torch_dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            
            x[transfer_index] = x0[transfer_index]
            
    return x


class LLaDAGenerator:
    def __init__(
        self,
        config: Optional[Union[str, Dict[str, Any]]] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
        watermark = None,
        local_files_only: bool = False,
        device_map: Optional[Union[str, Dict[str, int]]] = None
    ):
        self.watermark = watermark
        self.local_files_only = local_files_only
        if config is not None:
            if isinstance(config, str):
                config_path_candidate = Path(config)
                if config_path_candidate.is_file():
                    print(f"Loading config file: {config_path_candidate}")
                    self.config = load_config(str(config_path_candidate))
                else:
                    print(f"Config is not a file, treating as model id: {config}")
                    self.config = None
                    model_path = model_path or config
            else:
                self.config = config
            model_config = self.config.get('model', {}) if self.config else {}
            gen_config = self.config.get('generation', {}) if self.config else {}
            tokenizer_config = self.config.get('tokenizer', {}) if self.config else {}
            
            self.model_path = model_path or model_config.get('model_path', 'GSAI-ML/LLaDA-8B-Instruct')
            self.device = device or model_config.get('device', 'cuda')
            dtype_str = model_config.get('dtype', 'bfloat16')
            if dtype is not None:
                self.torch_dtype = get_torch_dtype(dtype) if isinstance(dtype, str) else dtype
            else:
                self.torch_dtype = get_torch_dtype(dtype_str)
            self.trust_remote_code = model_config.get('trust_remote_code', True)
            self.device_map = device_map if device_map is not None else model_config.get('device_map', 'auto')
            
            self.default_gen_length = gen_config.get('max_length', 128)
            self.default_steps = gen_config.get('num_diffusion_steps', 128)
            self.default_temperature = gen_config.get('temperature', 0.0)
            
            self.tokenizer_path = tokenizer_config.get('tokenizer_path', self.model_path)
            self.use_chat_template = tokenizer_config.get('use_chat_template', True)
        else:
            self.model_path = model_path or "GSAI-ML/LLaDA-8B-Instruct"
            self.device = device or "cuda"
            if dtype is not None:
                self.torch_dtype = get_torch_dtype(dtype) if isinstance(dtype, str) else dtype
            else:
                self.torch_dtype = torch.bfloat16
            self.trust_remote_code = True
            self.device_map = device_map if device_map is not None else "auto"
            
            self.default_gen_length = 128
            self.default_steps = 128
            self.default_temperature = 0.0
            
            self.tokenizer_path = self.model_path
            self.use_chat_template = True
            self.config = None
        if not torch.cuda.is_available() and self.device == "cuda":
            print("Warning: CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        print(f"Loading tokenizer from {self.tokenizer_path} (local_files_only={self.local_files_only})")
        print(f"DEBUG: self.model_path = {self.model_path}")
        print(f"DEBUG: self.tokenizer_path = {self.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only
        )
        print(f"Loading model from {self.model_path} (local_files_only={self.local_files_only})")
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            local_files_only=self.local_files_only
        )
        if hasattr(self.model, 'tie_weights'):
            self.model.tie_weights()
        self.model = self.model.eval()
        
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
    # Minimal smoke test
    gen = LLaDAGenerator(model_path="GSAI-ML/LLaDA-8B-Instruct", device="cuda")
    print(gen.generate_text("What is the capital of France?", steps=8, gen_length=16))

