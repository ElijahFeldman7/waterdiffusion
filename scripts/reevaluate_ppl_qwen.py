import json
import os
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss

def compute_ppl_batch(model, tokenizer, prompt_texts, completion_texts, device):
    """
    Exact implementation of Zurich's PPL logic.
    Scores ONLY the completion tokens, while providing the prompt as context.
    """
    # Combine prompt and completion for full sequence scoring
    full_sequences = [f"{p}{c}" for p, c in zip(prompt_texts, completion_texts)]
    
    encodings = tokenizer(
        full_sequences,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_batch = encodings["input_ids"]
    attn_mask = encodings["attention_mask"]

    with torch.no_grad():
        out_logits = model(encoded_batch, attention_mask=attn_mask).logits

    # Tokenize prompts separately to identify where completion begins
    prompt_encodings = tokenizer(
        prompt_texts,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)
    prompt_attn_mask = prompt_encodings["attention_mask"]

    # Create a mask for the prompt tokens within the full sequence
    # Pad prompt_attn_mask with zeros to match encoded_batch length
    padding = torch.zeros(
        (attn_mask.shape[0], attn_mask.shape[1] - prompt_attn_mask.shape[1]),
        dtype=torch.long,
        device=device
    )
    padded_prompt_attn_mask = torch.cat([prompt_attn_mask, padding], dim=1)
    prompt_token_mask = (padded_prompt_attn_mask == 1)
    
    # Clone attn_mask to avoid modifying original, then mask out prompt tokens
    # so they don't contribute to the PPL loss
    scoring_mask = attn_mask.clone()
    scoring_mask[prompt_token_mask] = 0

    # standard shift for causal LM
    shift_logits = out_logits[..., :-1, :].contiguous()
    shift_labels = encoded_batch[..., 1:].contiguous()
    shift_mask = scoring_mask[..., 1:].contiguous()

    loss_fct = CrossEntropyLoss(reduction="none")
    
    # Calculate loss per token
    elementwise_loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
    
    # Weighted average loss (only completion tokens)
    token_losses = (elementwise_loss * shift_mask).sum(dim=1)
    n_completion_tokens = shift_mask.sum(dim=1)
    
    # Avoid division by zero
    n_completion_tokens = torch.clamp(n_completion_tokens, min=1)
    avg_loss = token_losses / n_completion_tokens
    
    ppls = torch.exp(avg_loss)
    return ppls.tolist()

def main():
    parser = argparse.ArgumentParser(description="Re-evaluate PPL for JSONL result files using Qwen-32B.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file or directory.")
    parser.add_argument("--output-suffix", type=str, default="_qwen_ppl", help="Suffix for output file.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct", help="The Qwen model to use.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} (bfloat16)...")
    
    # Note: 32B model requires ~64GB VRAM in BF16. A100 80GB is recommended.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    input_path = Path(args.input)
    if input_path.is_dir():
        files = list(input_path.glob("**/results*.jsonl"))
    else:
        files = [input_path]

    for fpath in files:
        output_path = fpath.parent / f"{fpath.stem}{args.output_suffix}{fpath.suffix}"
        print(f"\nProcessing {fpath}")
        
        with open(fpath, "r") as f:
            rows = [json.loads(line) for line in f]

        # Determine which completions exist
        has_wm = "completion_wm" in rows[0] or "completion" in rows[0]
        has_no_wm = "completion_no_wm" in rows[0]

        for task in ["wm", "no_wm"]:
            if task == "wm" and not has_wm: continue
            if task == "no_wm" and not has_no_wm: continue
            
            comp_key = "completion_wm" if "completion_wm" in rows[0] else "completion"
            if task == "no_wm": comp_key = "completion_no_wm"
            
            out_key = f"ppl_{task}_qwen"
            print(f"Evaluating {task} completions...")
            
            for i in tqdm(range(0, len(rows), args.batch_size)):
                batch = rows[i:i + args.batch_size]
                prompts = [r.get("prompt", "") for r in batch]
                completions = [r.get(comp_key, "") for r in batch]
                
                ppls = compute_ppl_batch(model, tokenizer, prompts, completions, device)
                
                for j, ppl in enumerate(ppls):
                    rows[i + j][out_key] = ppl

        with open(output_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()
