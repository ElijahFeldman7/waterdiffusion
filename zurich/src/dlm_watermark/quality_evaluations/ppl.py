"""
Forked from https://github.com/chenchenygu/watermark-learnability
"""


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm

def _load_ppl_model(ppl_model_name):
    """Load a perplexity model."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(ppl_model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ppl_model_name)
    
    return model, tokenizer

def compute_ppl(ppl_model_name, prompts, completions, batch_size):
    
    model, tokenizer = _load_ppl_model(ppl_model_name)
    ppls = _compute_ppl(model, tokenizer, prompts, completions, batch_size)
    
    return ppls
    
def _compute_ppl(model, tokenizer, prompts, completions, batch_size):
    """Compute perplexities under `ppl_model_name`."""
    
    device = model.device

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for i in tqdm(range(0, len(prompts), batch_size), desc="Computing PPL"):
        
        prompt_text, completion = prompts[i:i + batch_size], completions[i:i + batch_size]
        s = [f"{p}{c}" for p, c in zip(prompt_text, completion)]
        
        encodings = tokenizer(
            s,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_batch = encodings["input_ids"]
        attn_mask = encodings["attention_mask"]

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        prompt_encodings = tokenizer(
            prompt_text,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
        prompt_attn_mask = prompt_encodings["attention_mask"]

        # match shape of prompt_attn_mask and attn_mask by padding with 0
        padding = torch.zeros(
            (attn_mask.shape[0], attn_mask.shape[1] - prompt_attn_mask.shape[1]),
        ).to(device)
        padded_prompt_attn_mask = torch.cat([prompt_attn_mask, padding], dim=1)
        prompt_mask = (padded_prompt_attn_mask == 1)
        
        # don't score prompt tokens
        attn_mask[prompt_mask] = 0

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return ppls