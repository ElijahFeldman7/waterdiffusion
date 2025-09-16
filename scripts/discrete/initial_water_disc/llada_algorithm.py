import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union, List, Set
import math
import gc
import hashlib
import random
from scipy.stats import norm
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import os
from transformers import AutoTokenizer, AutoModel

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
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
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

class LLMWatermark:
    def __init__(self, 
                 vocab_size: int, 
                 delta: float = 2.0, 
                 gamma: float = 0.5, 
                 secret_key: str = "my_secure_default_key"):
        self.vocab_size = vocab_size
        self.delta = delta
        self.gamma = gamma
        self.secret_key = secret_key

    def _get_rng(self, previous_tokens: List[int], position: int, domain: str) -> random.Random:
        hash_input = f"{self.secret_key}:{domain}:{str(previous_tokens + [position])}".encode('utf-8')
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        rng = random.Random(hash_value)
        return rng

    def _get_green_tokens(self, previous_tokens: List[int], position: int) -> Set[int]:
        rng = self._get_rng(previous_tokens, position, domain="green_list")
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
        delta_rng = self._get_rng(previous_tokens, current_position, domain="delta_value")
        dynamic_delta = delta_rng.uniform(self.delta * 0.5, self.delta * 1.5)
        green_list_tensor = torch.tensor(list(green_tokens), device=logits.device, dtype=torch.long)
        if len(green_list_tensor) > 0:
            for batch_idx in range(logits.shape[0]):
                valid_green_tokens = green_list_tensor[green_list_tensor < logits.shape[1]]
                watermarked_logits[batch_idx].scatter_add_(0, valid_green_tokens, dynamic_delta)
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
        try:
            z_score = (sum(green_counts) - self.gamma * len(green_counts)) / ((self.gamma * (1 - self.gamma) * len(green_counts)) ** 0.5)
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
        except ZeroDivisionError:
             return {"is_watermarked": False, "p_value": 1.0, "green_ratio": 0.0}
        is_watermarked = p_value < p_value_threshold
        return {
            "is_watermarked": is_watermarked,
            "p_value": p_value,
            "green_ratio": green_ratio,
            "expected_ratio": expected_ratio,
            "total_tokens": len(green_counts)
        }

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
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
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

def generate_rsa_keys():
    if os.path.exists('private_key.pem') and os.path.exists('public_key.pem'):
        print("RSA key pair already exists. Skipping generation.")
        return
    print("Generating RSA key pair...")
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    public_key = private_key.public_key()
    pem_priv = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    with open('private_key.pem', 'wb') as f: f.write(pem_priv)
    print("Saved private_key.pem")
    pem_pub = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    with open('public_key.pem', 'wb') as f: f.write(pem_pub)
    print("Saved public_key.pem")

def load_public_key(filename: str = "public_key.pem") -> rsa.RSAPublicKey:
    with open(filename, "rb") as key_file:
        return serialization.load_pem_public_key(key_file.read(), backend=default_backend())

def load_private_key(filename: str = "private_key.pem") -> rsa.RSAPrivateKey:
    with open(filename, "rb") as key_file:
        return serialization.load_pem_private_key(key_file.read(), password=None, backend=default_backend())

def encrypt_key(secret_key: str, public_key: rsa.RSAPublicKey) -> bytes:
    return public_key.encrypt(
        secret_key.encode('utf-8'),
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
    )

def decrypt_key(encrypted_key: bytes, private_key: rsa.RSAPrivateKey) -> str:
    return private_key.decrypt(
        encrypted_key,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
    ).decode('utf-8')

def generation_workflow(model, tokenizer, prompt_tokens):
    my_secret_key = "sUpEr_sEcReT_kEy_123!"
    public_key = load_public_key("public_key.pem")

    watermark = LLMWatermark(
        vocab_size=tokenizer.vocab_size,
        delta=4.0, 
        gamma=0.25, 
        secret_key=my_secret_key
    )

    mask_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None else MASK_TOKEN_ID
    if mask_id != MASK_TOKEN_ID:
        print(f"Using tokenizer's mask_token_id: {mask_id}")

    print("Generating watermarked text (This may take a while)...")
    generated_tensor = generate_with_watermark(
        model, 
        prompt_tokens, 
        watermark,
        steps=128,
        gen_length=128,
        block_length=32,
        temperature=0.,
        cfg_scale=0.,
        remasking='low_confidence',
        mask_id=mask_id
    )

    print("Encrypting secret key...")
    encrypted_key_bundle = encrypt_key(my_secret_key, public_key)

    print("Generation complete.")
    return generated_tensor, encrypted_key_bundle, tokenizer.vocab_size

def detection_workflow(generated_tensor: torch.Tensor, encrypted_key: bytes, vocab_size: int):
    private_key = load_private_key("private_key.pem")
    print("Decrypting secret key...")
    try:
        decrypted_secret_key = decrypt_key(encrypted_key, private_key)
        print(f"Successfully decrypted key: '{decrypted_secret_key}'")
    except Exception as e:
        print(f"DECRYPTION FAILED: {e}. This key is not for us.")
        return None, None

    detector = LLMWatermark(
        vocab_size=vocab_size,
        gamma=0.25, 
        secret_key=decrypted_secret_key
    )

    print("Running statistical detection...")
    token_list = generated_tensor[0].tolist()
    results = detector.detect_watermark(token_list)
    print(results)
    
    return detector, token_list

def get_watermark_details_for_position(detector: LLMWatermark, 
                                       all_tokens: List[int], 
                                       position: int) -> dict:
    if position < 1 or position >= len(all_tokens):
        return {"error": "Invalid position"}
    previous_tokens = all_tokens[:position]
    green_tokens = detector._get_green_tokens(previous_tokens, position)
    delta_rng = detector._get_rng(previous_tokens, position, domain="delta_value")
    dynamic_delta = delta_rng.uniform(detector.delta * 0.5, detector.delta * 1.5) 
    actual_token = all_tokens[position]
    was_on_green_list = actual_token in green_tokens
    return {
        "position": position,
        "actual_token": actual_token,
        "was_on_green_list": was_on_green_list,
        "dynamic_delta_applied": round(dynamic_delta, 2),
        "green_list_size": len(green_tokens),
    }

if __name__ == "__main__":
    generate_rsa_keys()

    model = None
    tokenizer = None
    
    try:
        model_path = 'GSAI-ML/LLaDA-8B-Instruct'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("Warning: Running LLaDA-8B on CPU will be extremely slow.")

        print("Loading LLaDA model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
        print("Model loaded successfully.")

        prompt_text = "What is the capital of France?"
        messages = [{"role": "user", "content": prompt_text}]
        chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(chat_input, return_tensors="pt")['input_ids'].to(device)
        prompt_len = input_ids.shape[1]

        print(f"Prompt: {prompt_text}")

        print("\n--- 1. GENERATION WORKFLOW ---")
        generated_tensor, encrypted_key, vocab_size = generation_workflow(model, tokenizer, input_ids)
        
        response_tokens = generated_tensor[0, prompt_len:]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        print(f"Generated response: {response_text}")

        print("\n--- 2. DETECTION WORKFLOW ---")
        detector, token_list = detection_workflow(generated_tensor, encrypted_key, vocab_size)

        print("\n--- 3. DEDUCE LOGIT MODIFICATIONS ---")
        if detector and token_list:
            pos_to_check = prompt_len + 10
            if pos_to_check < len(token_list):
                details = get_watermark_details_for_position(detector, token_list, pos_to_check)
                print(f"Details for token at position {pos_to_check}:")
                print(details)

    except ImportError:
        print("transformers library not available. Please install it to run this example.")
    except Exception as e:
        print(f"An error occurred: {e}")
