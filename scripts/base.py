import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union
import math
import gc

# Constants
MASK_TOKEN_ID = 126336  # The token ID of [MASK] in LLaDA

print("Ready to run LLaDA algorithm!")

def forward_process(batch: torch.Tensor, prompt_index: torch.Tensor, mask_id: int = MASK_TOKEN_ID) -> Tuple[torch.Tensor, torch.Tensor]:
    b, l = batch.shape
    # Calculate target length (non-prompt tokens)
    target_len = (l - prompt_index.sum()).item()
    k = torch.randint(1, target_len + 1, (), device=batch.device)
    # Create linear noise schedule
    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len
    # Create mask indices for target tokens
    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    # Randomly permute mask positions
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]
    # Combine prompt and target masks
    is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
    noisy_batch = torch.where(is_mask, mask_id, batch)
    # Return mask probability
    return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)
def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    gumbel noise is intentionally not gumbel max as they think it reduces quality
    """
    if temperature == 0:
        return logits

    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Calculate the number of tokens to transfer (unmask) at each step.

    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    Args:
        mask_index: Boolean tensor indicating masked positions
        steps: Number of denoising steps

    Returns:
        Number of tokens to transfer at each step
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens
def get_logits_with_cfg(model, batch: torch.Tensor, prompt_index: torch.Tensor, cfg_scale: float, mask_id: int = MASK_TOKEN_ID) -> torch.Tensor:
    """
    Get model logits with optional classifier-free guidance.

    Args:
        model: LLaDA model
        batch: Input batch
        prompt_index: Boolean mask for prompt positions
        cfg_scale: Classifier-free guidance scale
        mask_id: Mask token ID

    Returns:
        Guided logits
    """
    if cfg_scale > 0.:
        assert len(prompt_index) == batch.shape[1]
        prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
        un_batch = batch.clone()
        un_batch[prompt_index] = mask_id
        batch = torch.cat([batch, un_batch])

    logits = model(batch).logits

    if cfg_scale > 0.:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    return logits
@torch.no_grad()
def generate(model, prompt: torch.Tensor, steps: int = 128, gen_length: int = 128,
             block_length: int = 128, temperature: float = 0., cfg_scale: float = 0.,
             remasking: str = 'low_confidence', mask_id: int = MASK_TOKEN_ID) -> torch.Tensor:
    """
    Generate text using LLaDA diffusion model.

    Args:
        model: LLaDA model (mask predictor)
        prompt: Input prompt tensor of shape (1, L)
        steps: Number of sampling steps (≤ gen_length)
        gen_length: Generated answer length
        block_length: Block length for semi-autoregressive generation (≤ gen_length)
        temperature: Categorical distribution sampling temperature
        cfg_scale: Unsupervised classifier-free guidance scale
        remasking: Remasking strategy ('low_confidence' or 'random')
        mask_id: Token ID of [MASK]

    Returns:
        Generated sequence tensor
    """
    # Initialize sequence with masks
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    # Validate block configuration
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # Process each block
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length

        # Get mask indices for current block
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # Process each step in the block
        for i in range(steps_per_block):
            mask_index = (x == mask_id)

            # Get model predictions with optional CFG
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # Apply Gumbel noise for sampling
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # Calculate confidence scores for remasking
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")

            # Don't consider positions beyond current block
            x0_p[:, block_end:] = -np.inf

            # Apply predictions where we have masks
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # Select tokens to unmask based on confidence
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True

            # Apply the selected tokens
            x[transfer_index] = x0[transfer_index]

    return x
@torch.no_grad()
def get_log_likelihood(model, prompt: torch.Tensor, answer: torch.Tensor,
                      mc_num: int = 128, batch_size: int = 16, cfg_scale: float = 0.,
                      mask_id: int = MASK_TOKEN_ID) -> float:
    """
    Estimate log likelihood using Monte Carlo sampling.

    Args:
        model: LLaDA model
        prompt: Prompt tensor of shape (l1)
        answer: Answer tensor of shape (l2)
        mc_num: Monte Carlo estimation times
        batch_size: Mini batch size
        cfg_scale: Unsupervised classifier-free guidance scale
        mask_id: Token ID of [MASK]

    Returns:
        Estimated log likelihood
    """
    seq = torch.concatenate([prompt, answer])[None, :]
    seq = seq.repeat((batch_size, 1)).to(model.device)
    prompt_index = torch.arange(seq.shape[1], device=model.device) < len(prompt)

    loss_ = []
    for _ in range(mc_num // batch_size):
        perturbed_seq, p_mask = forward_process(seq, prompt_index, mask_id)
        mask_index = perturbed_seq == mask_id

        logits = get_logits_with_cfg(model, perturbed_seq, prompt_index, cfg_scale, mask_id)

        loss = F.cross_entropy(logits[mask_index], seq[mask_index], reduction='none') / p_mask[mask_index]
        loss = loss.sum() / batch_size

        loss_.append(loss.item())

    return - sum(loss_) / len(loss_)
def compute_training_loss(model, input_ids: torch.Tensor, prompt_lengths: Optional[torch.Tensor] = None,
                         eps: float = 1e-3, mask_id: int = MASK_TOKEN_ID) -> torch.Tensor:
    """
    Compute training loss for LLaDA model.

    This function implements the core training objective from the paper:
    - Forward process: Add noise by masking tokens
    - Loss computation: Cross-entropy on masked positions weighted by mask probability

    Args:
        model: LLaDA model
        input_ids: Input token sequence of shape (batch_size, seq_len)
        prompt_lengths: Optional prompt lengths for SFT (if None, treats as pre-training)
        eps: Minimum mask probability
        mask_id: Token ID for mask token

    Returns:
        Training loss tensor
    """
    b, l = input_ids.shape

    # Forward process: add noise by masking
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, mask_id, input_ids)

    # For SFT: don't mask prompt tokens
    if prompt_lengths is not None:
        token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
        prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
        noisy_batch[prompt_mask] = input_ids[prompt_mask]
        masked_indices = (noisy_batch == mask_id)

    # Get model predictions
    logits = model(input_ids=noisy_batch).logits

    # Compute weighted cross-entropy loss
    token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]

    if prompt_lengths is not None:
        # For SFT: normalize by answer length
        prompt_mask = prompt_mask.to(torch.int64)
        answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])
        loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
    else:
        # For pre-training: normalize by sequence length
        loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

    return loss
# Example usage with pre-trained LLaDA model
def example_usage():
    """
    Example of how to use the LLaDA algorithm with a pre-trained model.
    Note: This requires the transformers library and a pre-trained LLaDA model.
    """
    try:
        from transformers import AutoTokenizer, AutoModel

        # Load pre-trained model and tokenizer
        model_path = 'GSAI-ML/LLaDA-8B-Instruct'  # or 'GSAI-ML/LLaDA-8B-Base'

        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

        # Example prompt
        prompt_text = "What is the capital of France?"

        # For Instruct model, use chat template
        messages = [{"role": "user", "content": prompt_text}]
        chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(chat_input)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        print(f"Prompt: {prompt_text}")
        print(f"Input IDs shape: {input_ids.shape}")

        # Generate response
        print("Generating response...")
        generated = generate(
            model,
            input_ids,
            steps=128,
            gen_length=128,
            block_length=32,
            temperature=0.,
            cfg_scale=0.,
            remasking='low_confidence'
        )

        # Decode response
        response_tokens = generated[0, input_ids.shape[1]:]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

        print(f"Generated response: {response_text}")

        return response_text

    except ImportError:
        print("transformers library not available. Please install it to run this example.")
        print("pip install transformers")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have access to the LLaDA model and sufficient GPU memory.")
        return None

example_usage()
