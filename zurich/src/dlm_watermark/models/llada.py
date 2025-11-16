"""
Forked from: https://github.com/ML-GSAI/LLaDA
"""

import torch
import torch.nn.functional as F
import numpy as np
from .model import DiffusionLM, DiffusionOutput
from ..configs import ModelConfiguration


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    return logits - torch.log(-torch.log(noise)) * temperature


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.\

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(
    model,
    prompt,
    watermarker=None,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    """

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    ages = torch.zeros_like(x)  # Store the age of each token in the generation process.
    age = 1

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1] + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length :,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
      
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            if watermarker is not None:
                sampling_logits, logits = watermarker.watermark_logits(x, logits)
            else:
                sampling_logits = logits

            logits_with_noise = add_gumbel_noise(
                sampling_logits, temperature=temperature
            )
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )  # b, l
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif (
                remasking == "ar"
            ):  # Autoregressive remasking, we unmask the first mask
                x0_p = (
                    -torch.arange(x0.shape[1], device=x0.device)
                    .unsqueeze(0)
                    .repeat(x0.shape[0], 1)
                    / x0.shape[1]
                )
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            ages[transfer_index] = age
            age += 1

    output = DiffusionOutput(sequences=x, ages=ages, output_sequences=x[:, prompt.shape[1]:])

    return output


class LLaDA(DiffusionLM):
    """
    LLaDA model for diffusion-based language modeling.
    Inherits from DiffusionLM and implements the generation method.
    """

    def __init__(self, config: ModelConfiguration, hf_model):
        super().__init__(config, hf_model)

        # Populate the model-specific arguments
        self.cfg_scale = config.model_specific_arguments.get("cfg_scale", 0.0)
        self.block_length = config.model_specific_arguments.get("block_length", 128)
        self.mask_id = config.model_specific_arguments.get("mask_id", 126336)

    def generate_diffusion(self, input_ids: torch.LongTensor, watermark=None, **kwargs):
        """
        Generate text using the LLaDA model.

        Args:
            prompt (torch.Tensor): Input prompt tensor.
            watermarker (optional): Watermarking object.

        Returns:
            torch.Tensor: Generated text tensor.
        """

        assert input_ids.shape[0] == 1, "LLaDA only supports batch size of 1."

        super().generate_diffusion(input_ids, watermark=watermark, **kwargs)

        output = generate(
            model=self.model,
            prompt=input_ids,
            watermarker=watermark,
            steps=self.config.steps,
            gen_length=self.config.gen_length,
            block_length=self.block_length,
            temperature=self.config.temperature,
            cfg_scale=self.cfg_scale,
            remasking=self.config.remasking,
            mask_id=self.mask_id,
        )

        return output
