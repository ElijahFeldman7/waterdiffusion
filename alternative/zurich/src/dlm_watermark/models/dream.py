import torch
import torch.nn.functional as F
import numpy as np
from functools import partial
from .model import DiffusionLM, DiffusionOutput
from ..configs import ModelConfiguration
from .dream_generation_utils import (
    DreamGenerationConfig,
    DreamModelOutput,
    sample_tokens,
)
from typing import Optional, Union

def compute_watermark_hook(step, x, logits, watermark):
    
    with torch.no_grad():
        sampling_logits, logits = watermark.watermark_logits(x, logits)
    
    return sampling_logits, logits

def _sample(
    self,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor],
    generation_config: DreamGenerationConfig,
    generation_tokens_hook_func,
    generation_logits_hook_func,
    watermark,
) -> Union[DreamModelOutput, torch.LongTensor]:
    
    # init values
    output_history = generation_config.output_history
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    mask_token_id = generation_config.mask_token_id
    steps = generation_config.steps
    eps = generation_config.eps
    alg = generation_config.alg
    alg_temp = generation_config.alg_temp
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k

    histories = [] if (return_dict_in_generate and output_history) else None

    # pad input_ids to max_length
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
    ages = torch.zeros_like(x)
    age = 1

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        # we do not mask the [MASK] tokens so value = 1.0
        attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        # attention_mask is of shape [B, N]
        # broadcast to [B, 1, N, N]
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
    else:
        tok_idx = None
        attention_mask = "full"

    timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

    # this allows user-defined token control of the intermediate steps
    x = generation_tokens_hook_func(None, x, None)
    for i in range(steps):
        mask_index = (x == mask_token_id)
        logits = self(x, attention_mask, tok_idx).logits
        logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
        
        # this allows user-defined logits control of the intermediate steps
        out = generation_logits_hook_func(i, x, logits)
        if isinstance(out, tuple):
            sampling_logits, logits = out
        else:
            sampling_logits = out
            logits = out
            
        mask_sampling_logits = sampling_logits[mask_index]
        mask_logits = logits[mask_index]
        t = timesteps[i]
        s = timesteps[i + 1]
    
        if alg == 'origin':
            p_transfer = 1 - s / t if i < steps - 1 else np.inf
            
            _, x0 = sample_tokens(mask_sampling_logits, mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
            
            confidence = torch.rand(*x0.shape, device=self.device)
            confidence = 1 - confidence
            
            
            transfer_index_t_s = confidence < p_transfer
            
            x0[~transfer_index_t_s] = mask_token_id
            
            x[mask_index] = x0.clone()
            ages[mask_index] = ages[mask_index] + age * torch.ones_like(ages[mask_index], device=self.device, dtype=torch.long) * transfer_index_t_s
            age += 1
        else:
            if alg == 'maskgit_plus':
                confidence, x0 = sample_tokens(mask_sampling_logits, mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
            elif alg == 'topk_margin':
                confidence, x0 = sample_tokens(mask_sampling_logits, mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
            elif alg == 'entropy':
                confidence, x0 = sample_tokens(mask_sampling_logits, mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
            elif alg == "ar":
                confidence, x0 = sample_tokens(mask_sampling_logits, mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, autoregressive=True)
            else:
                raise RuntimeError(f"Unknown alg: {alg}")
                        
            num_mask_token = mask_index.sum() / mask_index.shape[0]
            number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
            full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
            full_confidence[mask_index] = confidence
            if number_transfer_tokens > 0:
                if alg_temp is None or alg_temp == 0:
                    _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                else:
                    full_confidence = full_confidence / alg_temp
                    full_confidence = F.softmax(full_confidence, dim=-1)
                    transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                x_[mask_index] = x0.clone()
                row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                x[row_indices,transfer_index] = x_[row_indices,transfer_index]
                ages[row_indices, transfer_index] = age
                age += 1

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(i, x, logits)

        if histories is not None:
            histories.append(x.clone())
    
    return DiffusionOutput(
        sequences=x,
        ages=ages,
        output_sequences=x[:, input_ids.shape[1]:]
    )


class Dream(DiffusionLM):
    """
    Dream model for diffusion-based language modeling.
    Inherits from DiffusionLM and implements the generation method.
    Does not support ages yet.
    """

    def __init__(self, config: ModelConfiguration, hf_model):
        
        # Override the _sample method for KTH watermarking support
        if hf_model is not None:
            funcType = type(hf_model._sample)
            hf_model._sample = funcType(_sample, hf_model)
        
        super().__init__(config, hf_model)
        
        # Populate the model-specific arguments
        self.output_history = config.model_specific_arguments.get('output_history', False)
        self.alg_temp = config.model_specific_arguments.get('alg_temp', 0.0)

        

    def generate_diffusion(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor, watermark=None, **kwargs):
        """
        Generate text using the LLaDA model.
        
        Args:
            prompt (torch.Tensor): Input prompt tensor.
            watermarker (optional): Watermarking object.
        
        Returns:
            torch.Tensor: Generated text tensor.
        """
        
        super().generate_diffusion(input_ids, watermark=watermark, **kwargs)
        
        
        
                        
        if watermark is None:
            watermark_hook = lambda step, x, logits: logits  # No watermarking hook
            
            funcType = type(self.model._sample)
            watermark_sample = partial(
                _sample,
                watermark=None
            )
            self.model._sample = funcType(watermark_sample, self.model)
            
        else:
            watermark_hook = partial(compute_watermark_hook, watermark=watermark)
            
            
            watermark_sample = partial(
                _sample,
                watermark=watermark
            )
            funcType = type(watermark_sample)
            self.model._sample = funcType(watermark_sample, self.model)
            
        
        output = self.model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.gen_length,
                    output_history=self.output_history,
                    return_dict_in_generate=True,
                    steps=self.config.steps,
                    temperature=self.config.temperature,
                    alg=self.config.remasking,
                    alg_temp=self.alg_temp,
                    generation_logits_hook_func=watermark_hook,
                )
        return output
