import torch
import torch.nn.functional as F
import numpy as np
from functools import partial
from .model import DiffusionLM, DiffusionOutput
from ..configs import ModelConfiguration
from .dreamon_generation_utils import (
    DreamGenerationConfig,
    DreamModelOutput,
    sample_tokens,
)
from typing import Optional, Union
from transformers.utils import (
    is_torchdynamo_compiling,
)
import warnings


def compute_watermark_hook(step, x, logits, watermark):
    
    with torch.no_grad():
        sampling_logits, logits = watermark.watermark_logits(x, logits)
    
    return logits

@torch.no_grad()
def diffusion_generate(
    self,
    inputs: Optional[torch.Tensor] = None,  # prefix, mask, suffix
    generation_config: Optional[DreamGenerationConfig] = None,
    **kwargs,
) -> Union[DreamModelOutput, torch.LongTensor]:
    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
    generation_config = self._prepare_generation_config(generation_config, **kwargs)
    generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
    generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

    # 2. Define model inputs
    assert inputs is not None
    input_ids = inputs
    device = input_ids.device
    attention_mask = kwargs.pop("attention_mask", None)
    assert attention_mask is None, 'We currently do not support attention_mask for DreamOn since we recompute attention mask after each denoising step.'
    self._prepare_special_tokens(generation_config, device=device)

    # 3. Prepare `max_length`.
    input_ids_length = input_ids.shape[-1]
    ## get number of mask tokens as start_gen_len
    mask_token_id = generation_config.mask_token_id
    mask_token_indices = torch.where(input_ids == mask_token_id)[1]
    
    if mask_token_indices.numel() == 0:
        raise ValueError("No mask tokens found in the input_ids.")
    
    num_mask_tokens = mask_token_indices.numel()
    ## get the first index of mask 
    mask_token_index = mask_token_indices[0]
    ## assign it as prefix_len
    prefix_len = mask_token_index
    start_gen_len = num_mask_tokens  # Ensure start_gen_len is defined

    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    generation_config = self._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        input_ids_length=input_ids_length,
        num_mask_tokens=num_mask_tokens,
    )

    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 4. Check input_ids
    if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    result = self._sample(
        input_ids,
        prefix_len,
        start_gen_len,
        generation_config=generation_config,
        generation_tokens_hook_func=generation_tokens_hook_func,
        generation_logits_hook_func=generation_logits_hook_func
    )

    return result

def _sample(
        self,
        input_ids: torch.LongTensor,
        prefix_len: int,
        start_gen_len: int,
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        watermark
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate

        pad_delete_to_right = generation_config.pad_delete_to_right
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        device = input_ids.device  

        num_generation_tokens = start_gen_len
        max_length = generation_config.max_length
        max_gen_len = max_length - prefix_len
        expand_budget = generation_config.expand_budget
        if expand_budget is None:
            expand_budget = max_gen_len * 2

        number_transfer_tokens = generation_config.number_transfer_tokens
        eos_token_id = generation_config.eos_token_id 
        delete_token_id = generation_config.delete_token_id  
        expand_token_id = generation_config.expand_token_id
        mask_token_id = generation_config.mask_token_id

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=eos_token_id)

        for i in range(2 * max_gen_len + 2 * expand_budget):
            #### 1. --- Prepare Input ---
            current_window_length = input_ids.shape[1] - start_gen_len + num_generation_tokens
            attention_mask = torch.ones([input_ids.shape[0], current_window_length], dtype=torch.int16).to(device)
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=0)

            mask_index = (x == mask_token_id) & (attention_mask == 1)
            if torch.all(~mask_index[:, :current_window_length]):
                break  # exit if all mask tokens are denoised

            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)

            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )

            output = self(x, attention_mask, tok_idx)
            logits = output.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            logits = generation_logits_hook_func(i, x, logits)

            logits = logits[mask_index]

            # block the logit for expansion when token budget is all used
            if current_window_length == max_length or expand_budget == 0:
                logits[:, expand_token_id] -= 1e9

            ### 2. ----sample tokens
            if alg == 'maskgit_plus':
                confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
            elif alg == 'topk_margin':
                confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
            elif alg == 'entropy':
                confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
            else:
                raise RuntimeError(f"Unknown alg: {alg}")

            if alg_temp is None or alg_temp == 0:
                _, transfer_index = torch.topk(confidence, number_transfer_tokens)
            else:
                confidence = confidence / alg_temp
                confidence = F.softmax(confidence, dim=-1)
                transfer_index = torch.multinomial(confidence, num_samples=number_transfer_tokens)
            x0_ = torch.zeros_like(x0, device=device, dtype=torch.long) + mask_token_id
            x0_[transfer_index] = x0[transfer_index].clone()
            x[mask_index] = x0_

            if histories is not None:
                histories.append(x[0,:current_window_length].clone())

            ### 3. ---- delete -------
            # pad delete to right if needed
            if pad_delete_to_right:
                x_seq = x[0]  # Flatten to 1D: shape [seq_len]

                # Find indices where EOS occurs
                delete_indices = (x_seq == delete_token_id).nonzero(as_tuple=True)

                if len(delete_indices[0]) > 0:
                    # Get the first occurrence of delete
                    first_delete_idx = delete_indices[0][0].item()
                    position_mask = torch.arange(x_seq.size(0), device=device) >= first_delete_idx
                    replace_mask = position_mask & mask_index[0]
                    # Set all tokens after EOS to eos_id
                    x_seq.masked_fill_(replace_mask, delete_token_id)
                    x = x_seq.unsqueeze(0)

            # delete
            delete_indices = ((x[0] == delete_token_id) & (mask_index[0] == 1)).nonzero(as_tuple=False).squeeze(1)
            if delete_indices.numel() > 0:
                for idx in sorted(delete_indices.tolist(), reverse=True):
                    x = torch.cat((
                        x[:, :idx],
                        x[:, idx + 1:],
                        torch.tensor([[mask_token_id]], device=device)
                    ), dim=1)
                    num_generation_tokens -= 1
                if histories is not None:
                    current_window_length = input_ids.shape[1] - start_gen_len + num_generation_tokens
                    histories.append(x[0,:current_window_length].clone())
            ### 4. ---- expand --------
            expand_indices = (x[0] == expand_token_id).nonzero(as_tuple=False).squeeze(1)
            if expand_indices.numel() > 0:
                # Process from right to left to prevent shifting issues
                for idx in sorted(expand_indices.tolist(), reverse=True):
                    x = torch.cat((
                        x[:, :idx],
                        torch.tensor([[mask_token_id, mask_token_id]], device=device),
                        x[:, idx + 1:]
                    ), dim=1)
                    num_generation_tokens += 1
                    expand_budget -= 1
                    # Truncate back to max_tokens if needed
                    if x.shape[1] > max_length:
                        x = x[:, :max_length]
                
                if histories is not None:
                    current_window_length = input_ids.shape[1] - start_gen_len + num_generation_tokens
                    histories.append(x[0,:current_window_length].clone())

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

        return DiffusionOutput(
            sequences=x,
            ages=torch.zeros_like(x),
            output_sequences=x[:, prefix_len:prefix_len + num_generation_tokens]
        )


class DreamOn(DiffusionLM):
    """
    DreamOn model for diffusion-based language modeling.
    Inherits from DiffusionLM and implements the generation method.
    """

    def __init__(self, config: ModelConfiguration, hf_model):
        
        # Override the _sample method and diffusion_generate method for watermarking support
        if hf_model is not None:
            funcType = type(hf_model._sample)
            hf_model._sample = funcType(_sample, hf_model)

            funcType = type(hf_model.diffusion_generate)
            hf_model.diffusion_generate = funcType(diffusion_generate, hf_model)

        super().__init__(config, hf_model)
        
        # Populate the model-specific arguments
        self.output_history = config.model_specific_arguments.get('output_history', False)
        self.alg_temp = config.model_specific_arguments.get('alg_temp', 0.0)
        self.expand_budget = config.model_specific_arguments.get('expand_budget', 100)

        

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
                    max_new_tokens=self.config.gen_length,
                    output_history=self.output_history,
                    return_dict_in_generate=True,
                    temperature=self.config.temperature,
                    alg=self.config.remasking,
                    alg_temp=self.alg_temp,
                    generation_logits_hook_func=watermark_hook,
                    expand_buget=self.expand_budget
                )
        return output
