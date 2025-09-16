import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union
import math
import gc
import hashlib
import random
from scipy.stats import norm
from typing import List, Set, Optional


MASK_TOKEN_ID = 126336 

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



class LLMWatermark:
    def __init__(self, vocab_size: int = 50000, delta: float = 2.0, gamma: float = 0.5):
        self.vocab_size = vocab_size
        self.delta = delta
        self.gamma = gamma

    def _get_green_tokens(self, previous_tokens: List[int], position: int) -> Set[int]:
 
        hash_input = str(previous_tokens + [position]).encode('utf-8')
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        rng = random.Random(hash_value)
        num_green = int(self.vocab_size * self.gamma)
        green_tokens = set()
        available_tokens = list(range(self.vocab_size))
        for _ in range(num_green):
            if not available_tokens:
                break
            idx = rng.randint(0, len(available_tokens) - 1)
            green_tokens.add(available_tokens.pop(idx))
        return green_tokens

    def apply_watermark(self, logits: torch.Tensor, previous_tokens: List[int],
                       current_position: int) -> torch.Tensor:
        watermarked_logits = logits.clone()
        green_tokens = self._get_green_tokens(previous_tokens, current_position)
        for batch_idx in range(logits.shape[0]):
            for token_id in green_tokens:
                if token_id < logits.shape[1]:
                    watermarked_logits[batch_idx, token_id] += self.delta
        return watermarked_logits

    def detect_watermark(self, tokens: List[int], p_value_threshold: float = 0.01) -> dict:
        if len(tokens) < 2:
            return {"is_watermarked": False, "p_value": 1.0, "green_ratio": 0.0}
        green_counts = []
        total_tokens = len(tokens) - 1
        for i in range(1, len(tokens)):
            previous_tokens = tokens[:i]
            green_tokens = self._get_green_tokens(previous_tokens, i)
            if tokens[i] in green_tokens:
                green_counts.append(1)
            else:
                green_counts.append(0)
        if not green_counts:
            return {"is_watermarked": False, "p_value": 1.0, "green_ratio": 0.0}
        green_ratio = sum(green_counts) / len(green_counts)
        expected_ratio = self.gamma
        deviation = abs(green_ratio - expected_ratio)
        z_score = (sum(green_counts) - self.gamma * len(green_counts)) / (self.gamma * (1 - self.gamma) / len(green_counts) ** 0.5)
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        is_watermarked = p_value < p_value_threshold
        return {
            "is_watermarked": is_watermarked,
            "p_value": p_value,
            "green_ratio": green_ratio,
            "expected_ratio": expected_ratio,
            "deviation": deviation,
            "total_tokens": total_tokens
        }

def test_watermark():
    print("🧪 Testing LLM Watermarking System")
    print("=" * 50)
    watermark = LLMWatermark(vocab_size=1000, delta=2.0, gamma=0.5)
    print("\n1. Generating watermarked sequence:")
    torch.manual_seed(42)
    sequence_length = 20
    generated_tokens = []
    generated_tokens.append(random.randint(0, 999))
    for i in range(1, sequence_length):
        logits = torch.randn(1, 1000)
        watermarked_logits = watermark.apply_watermark(
            logits, generated_tokens, i
        )
        next_token = torch.argmax(watermarked_logits[0]).item()
        generated_tokens.append(next_token)
    print(f"Generated tokens: {generated_tokens[:10]}...")
    print("\n2. Detecting watermark:")
    detection_result = watermark.detect_watermark(generated_tokens)
    print(f"Is watermarked: {detection_result['is_watermarked']}")
    print(f"P-value: {detection_result['p_value']:.4f}")
    print(f"Green ratio: {detection_result['green_ratio']:.4f}")
    print(f"Expected ratio: {detection_result['expected_ratio']:.4f}")
    print(f"Deviation: {detection_result['deviation']:.4f}")
    print("\n3. Generating non-watermarked sequence for comparison:")
    torch.manual_seed(123)
    non_watermarked_tokens = []
    non_watermarked_tokens.append(random.randint(0, 999))
    for i in range(1, sequence_length):
        logits = torch.randn(1, 1000)
        next_token = torch.argmax(logits[0]).item()
        non_watermarked_tokens.append(next_token)
    print(f"Non-watermarked tokens: {non_watermarked_tokens[:10]}...")
    detection_result_2 = watermark.detect_watermark(non_watermarked_tokens)
    print(f"Is watermarked: {detection_result_2['is_watermarked']}")
    print(f"P-value: {detection_result_2['p_value']:.4f}")
    print(f"Green ratio: {detection_result_2['green_ratio']:.4f}")
    print("\n✅ Watermarking test completed!")
    return generated_tokens, non_watermarked_tokens

watermarked_tokens, non_watermarked_tokens = test_watermark()

@torch.no_grad()
def generate_with_watermark(model, prompt: torch.Tensor, watermark: LLMWatermark,
                           steps: int = 128, gen_length: int = 128,
                           block_length: int = 128, temperature: float = 0.,
                           cfg_scale: float = 0., remasking: str = 'low_confidence',
                           mask_id: int = MASK_TOKEN_ID) -> torch.Tensor:
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    generated_tokens = prompt[0].tolist()
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
            watermarked_logits = logits.clone()
            for pos in range(logits.shape[1]):
                if mask_index[0, pos]:
                    watermarked_logits[:, pos, :] = watermark.apply_watermark(
                        logits[:, pos, :], generated_tokens, pos
                    )
            logits_with_noise = add_gumbel_noise(watermarked_logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if remasking == 'low_confidence':
                p = F.softmax(watermarked_logits, dim=-1)
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
            for pos in range(x.shape[1]):
                if transfer_index[0, pos] and pos >= prompt.shape[1]:
                    token_id = x[0, pos].item()
                    if len(generated_tokens) <= pos:
                        generated_tokens.extend([mask_id] * (pos - len(generated_tokens) + 1))
                    generated_tokens[pos] = token_id
    return x

def test_llada_watermark():
    print("\n🔬 Testing LLaDA + Watermarking Integration")
    print("=" * 50)
    class MockLLaDAModel:
        def __init__(self):
            self.device = device
        def __call__(self, input_ids):
            batch_size, seq_len = input_ids.shape
            vocab_size = 1000
            logits = torch.randn(batch_size, seq_len, vocab_size).to(self.device)
            class MockOutput:
                def __init__(self, logits):
                    self.logits = logits
            return MockOutput(logits)
    mock_model = MockLLaDAModel()
    watermark = LLMWatermark(vocab_size=1000, delta=2.0, gamma=0.5)
    prompt = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    print(f"Prompt: {prompt}")
    print(f"Prompt shape: {prompt.shape}")
    print("\nGenerating with watermark...")
    torch.manual_seed(42)
    generated = generate_with_watermark(
        mock_model,
        prompt,
        watermark,
        steps=32,
        gen_length=16,
        block_length=8,
        temperature=0.1,
        cfg_scale=0.,
        remasking='low_confidence'
    )
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")
    generated_tokens = generated[0, prompt.shape[1]:].tolist()
    print(f"New tokens: {generated_tokens}")
    print("\nDetecting watermark...")
    full_sequence = generated[0].tolist()
    detection_result = watermark.detect_watermark(full_sequence)
    print(f"Watermark detection results:")
    print(f"  Is watermarked: {detection_result['is_watermarked']}")
    print(f"  P-value: {detection_result['p_value']:.4f}")
    print(f"  Green ratio: {detection_result['green_ratio']:.4f}")
    print(f"  Expected ratio: {detection_result['expected_ratio']:.4f}")
    print(f"  Deviation: {detection_result['deviation']:.4f}")
    print("\n✅ LLaDA + Watermarking test completed!")
    return generated, detection_result

generated_with_watermark, detection_result = test_llada_watermark()

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

        print(f"Prompt: {prompt_text}")
        print(f"Input IDs shape: {input_ids.shape}")

        print("Generating response...")
        generated = generate_with_watermark(
            model,
            input_ids,
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
