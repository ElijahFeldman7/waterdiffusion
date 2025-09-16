import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union, Dict
import math
import gc
MASK_TOKEN_ID = 126336  
VOCAB_SIZE = 128256

class TokenIntervalMapper:
    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.interval_width = 1.0 / vocab_size
        self.token_to_interval = {}
        for token_id in range(vocab_size):
            start = token_id * self.interval_width
            end = (token_id + 1) * self.interval_width
            self.token_to_interval[token_id] = (start, end)

    def token_to_interval(self, token_id: int) -> Tuple[float, float]:
        return self.token_to_interval[token_id]

    def token_to_center(self, token_id: int) -> float:
        start, end = self.token_to_interval[token_id]
        return (start + end) / 2

    def interval_to_token(self, value: float) -> int:
        if value < 0 or value >= 1:
            raise ValueError(f"Value {value} must be in [0, 1)")
        return int(value * self.vocab_size)

    def token_to_tensor(self, token_id: int) -> torch.Tensor:
        center = self.token_to_center(token_id)
        return torch.tensor(center, dtype=torch.float32)

    def tokens_to_tensor(self, token_ids: torch.Tensor) -> torch.Tensor:
        centers = torch.zeros_like(token_ids, dtype=torch.float32)
        for i in range(token_ids.shape[0]):
            for j in range(token_ids.shape[1]):
                centers[i, j] = self.token_to_center(token_ids[i, j].item())
        return centers

token_mapper = TokenIntervalMapper(VOCAB_SIZE)

print(f"Device: {device}")
print(f"Token mapper initialized with vocab size: {VOCAB_SIZE}")
print("Ready to run modified LLaDA algorithm with continuous token intervals!")


def forward_process_continuous(batch: torch.Tensor, prompt_index: torch.Tensor,
                               token_mapper: TokenIntervalMapper,
                               mask_value: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    b, l = batch.shape

    continuous_batch = token_mapper.tokens_to_tensor(batch).to(batch.device)

    target_len = (l - prompt_index.sum()).item()
    k = torch.randint(1, target_len + 1, (), device=batch.device)

    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_noise = indices < x.unsqueeze(1)

    for i in range(b):
        is_noise[i] = is_noise[i][torch.randperm(target_len)]

    is_noise = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_noise), dim=1)

    noise_mask = torch.rand_like(continuous_batch) * 0.1  # Small random noise
    noisy_batch = torch.where(is_noise, mask_value + noise_mask, continuous_batch)

    return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

def forward_process(batch: torch.Tensor, prompt_index: torch.Tensor, mask_id: int = MASK_TOKEN_ID) -> Tuple[torch.Tensor, torch.Tensor]:

    b, l = batch.shape

    target_len = (l - prompt_index.sum()).item()
    k = torch.randint(1, target_len + 1, (), device=batch.device)

    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)

    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]

    is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
    noisy_batch = torch.where(is_mask, mask_id, batch)

    return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits

    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def get_logits_with_cfg(model, batch: torch.Tensor, prompt_index: torch.Tensor, cfg_scale: float, mask_id: int = MASK_TOKEN_ID) -> torch.Tensor:

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
def generate_continuous(model, prompt: torch.Tensor, token_mapper: TokenIntervalMapper,
                       steps: int = 128, gen_length: int = 128, block_length: int = 128,
                       temperature: float = 0., cfg_scale: float = 0.,
                       remasking: str = 'low_confidence', mask_value: float = 0.5) -> torch.Tensor:

    x = torch.full((1, prompt.shape[1] + gen_length), mask_value, dtype=torch.float32).to(model.device)

    prompt_continuous = token_mapper.tokens_to_tensor(prompt).to(model.device)
    x[:, :prompt.shape[1]] = prompt_continuous.clone()

    prompt_index = torch.arange(prompt.shape[1], device=model.device)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length

        block_mask_index = torch.abs(x[:, block_start:block_end] - mask_value) < 1e-6
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = torch.abs(x - mask_value) < 1e-6

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_value
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            if temperature > 0:
                logits = logits / temperature

            x0 = torch.tanh(logits)
            x0 = (x0 + 1) / 2

            if remasking == 'low_confidence':
                confidence = -torch.abs(logits)
            elif remasking == 'random':
                confidence = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")

            confidence[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, confidence, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index]

    return x

@torch.no_grad()
def generate(model, prompt: torch.Tensor, steps: int = 128, gen_length: int = 128,
             block_length: int = 128, temperature: float = 0., cfg_scale: float = 0.,
             remasking: str = 'low_confidence', mask_id: int = MASK_TOKEN_ID) -> torch.Tensor:
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
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
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index]

    return x

@torch.no_grad()
def get_log_likelihood(model, prompt: torch.Tensor, answer: torch.Tensor,
                      mc_num: int = 128, batch_size: int = 16, cfg_scale: float = 0.,
                      mask_id: int = MASK_TOKEN_ID) -> float:
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

def compute_training_loss_continuous(model, input_ids: torch.Tensor, token_mapper: TokenIntervalMapper,
                                    prompt_lengths: Optional[torch.Tensor] = None,
                                    eps: float = 1e-3, mask_value: float = 0.5) -> torch.Tensor:
    
    b, l = input_ids.shape

    continuous_input = token_mapper.tokens_to_tensor(input_ids).to(input_ids.device)

    t = torch.rand(b, device=input_ids.device)
    p_noise = (1 - eps) * t + eps
    p_noise = p_noise[:, None].repeat(1, l)

    noise = torch.randn_like(continuous_input) * 0.1 
    noise_mask = torch.rand((b, l), device=input_ids.device) < p_noise

    noisy_batch = torch.where(noise_mask, mask_value + noise, continuous_input)

    if prompt_lengths is not None:
        token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
        prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
        noisy_batch[prompt_mask] = continuous_input[prompt_mask]
        noise_mask = torch.abs(noisy_batch - continuous_input) > 1e-6

    logits = model(input_ids=noisy_batch).logits

    if logits.min() < 0 or logits.max() > 1:
        logits = torch.sigmoid(logits)

    token_loss = F.mse_loss(logits[noise_mask], continuous_input[noise_mask], reduction='none') / p_noise[noise_mask]

    if prompt_lengths is not None:
        prompt_mask = prompt_mask.to(torch.int64)
        answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])
        loss = torch.sum(token_loss / answer_lengths[noise_mask]) / input_ids.shape[0]
    else:
        loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

    return loss

def compute_training_loss(model, input_ids: torch.Tensor, prompt_lengths: Optional[torch.Tensor] = None,
                         eps: float = 1e-3, mask_id: int = MASK_TOKEN_ID) -> torch.Tensor:
    b, l = input_ids.shape

    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, mask_id, input_ids)

    if prompt_lengths is not None:
        token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
        prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
        noisy_batch[prompt_mask] = input_ids[prompt_mask]
        masked_indices = (noisy_batch == mask_id)

    logits = model(input_ids=noisy_batch).logits

    token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]

    if prompt_lengths is not None:
        prompt_mask = prompt_mask.to(torch.int64)
        answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])
        loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
    else:
        loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

    return loss

def test_continuous_algorithm():
    print("\n1. Testing Token-to-Interval Mapping:")
    test_tokens = torch.tensor([1, 100, 1000, VOCAB_SIZE-1])
    print(f"Test tokens: {test_tokens.tolist()}")

    for token_id in test_tokens:
        interval = token_mapper.token_to_interval[token_id.item()]
        center = token_mapper.token_to_center(token_id.item())
        print(f"Token {token_id.item()}: interval [{interval[0]:.6f}, {interval[1]:.6f}), center: {center:.6f}")

    print("\n2. Testing Continuous Forward Process:")
    batch = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    prompt_index = torch.tensor([True, True, False, False, False], device=device)

    print(f"Original batch: {batch}")
    print(f"Prompt index: {prompt_index}")

    noisy_batch, p_noise = forward_process_continuous(batch, prompt_index, token_mapper)
    print(f"Noisy batch (continuous): {noisy_batch}")
    print(f"Noise probabilities: {p_noise}")

    print("\n3. Testing Continuous-to-Token Conversion:")
    continuous_values = torch.tensor([0.1, 0.5, 0.9])
    for val in continuous_values:
        token_id = token_mapper.interval_to_token(val.item())
        print(f"Continuous value {val.item():.3f} -> Token {token_id}")

    print("\n4. Testing Continuous Training Loss:")
    class MockModel:
        def __init__(self):
            self.device = device

        def __call__(self, input_ids):
            class MockOutput:
                def __init__(self, logits):
                    self.logits = logits
            return MockOutput(torch.randn_like(input_ids))

    mock_model = MockModel()
    test_input = torch.tensor([[1, 2, 3, 4, 5]], device=device)

    try:
        loss = compute_training_loss_continuous(mock_model, test_input, token_mapper)
        print(f"Training loss: {loss.item():.6f}")
    except Exception as e:
        print(f"Error in training loss computation: {e}")

    print("\n✅ Continuous LLaDA algorithm test completed!")
    print("\nKey Differences from Original:")
    print("- Tokens mapped to continuous intervals [0, 1)")
    print("- Forward process adds continuous noise instead of discrete masking")
    print("- Training uses MSE loss instead of cross-entropy")
    print("- Generation works with continuous values")
def example_usage():
    try:
        from transformers import AutoTokenizer, AutoModel

        model_path = 'GSAI-ML/LLaDA-8B-Instruct'

        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

        prompt_text = "What is the capital of France?"

        messages = [{"role": "user", "content": prompt_text}]
        chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(chat_input)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        mp = TokenIntervalMapper()
        print(f"Prompt: {prompt_text}")
        print(f"Input IDs shape: {input_ids.shape}")

        print("Generating response with original discrete method...")
        generated = generate_continuous(
            model,
            input_ids,
            mp,
            steps=128,
            gen_length=128,
            block_length=32,
            temperature=0.,
            cfg_scale=0.,
            remasking='low_confidence'
        )

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