import json
import os
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Standard LLaDA mask token ID
MASK_TOKEN_ID = 126336

class LLaDAPPLCalculator:
    def __init__(self, model_path="GSAI-ML/LLaDA-8B-Instruct", device="cuda"):
        self.device = device
        print(f"Loading LLaDA tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"Loading LLaDA model from {model_path}...")
        self.model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            device_map="auto" if device == "cuda" else None
        )
        if device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        
    @torch.no_grad()
    def compute_pseudo_ppl(self, prompt, completion):
        """
        Calculates Pseudo-Perplexity: exp(-1/N * sum(log P(x_i | x_prompt, x_{not i})))
        """
        # Combine prompt and completion
        full_text = f"{prompt}{completion}"
        enc = self.tokenizer(full_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.model.device)
        
        # Identify completion tokens
        prompt_enc = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_enc["input_ids"].shape[1]
        
        completion_ids = input_ids[0, prompt_len:]
        n_tokens = len(completion_ids)
        
        if n_tokens == 0:
            return float("nan")
            
        total_log_prob = 0.0
        
        # For each token in the completion, mask it and get the model's prediction
        # Optimization: We can batch this if memory allows, but for now 1-by-1
        for i in range(n_tokens):
            masked_input = input_ids.clone()
            target_idx = prompt_len + i
            target_token = masked_input[0, target_idx].item()
            
            # Mask the token
            masked_input[0, target_idx] = MASK_TOKEN_ID
            
            # Forward pass
            logits = self.model(masked_input).logits # (1, S, V)
            
            # Get log-prob of the correct token
            probs = F.log_softmax(logits[0, target_idx], dim=-1)
            token_log_prob = probs[target_token].item()
            
            total_log_prob += token_log_prob
            
        avg_log_prob = total_log_prob / n_tokens
        ppl = torch.exp(torch.tensor(-avg_log_prob)).item()
        return ppl

def main():
    parser = argparse.ArgumentParser(description="Calculate LLaDA Pseudo-Perplexity for JSONl files.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file.")
    parser.add_argument("--output", type=str, help="Output JSONL file (defaults to input_with_llada_ppl.jsonl).")
    parser.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc = LLaDAPPLCalculator(model_path=args.model, device=device)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_with_llada_ppl.jsonl"

    print(f"Processing {input_path}...")
    
    results = []
    with open(input_path, "r") as f:
        lines = f.readlines()
        if args.limit:
            lines = lines[:args.limit]

    with open(output_path, "w") as out_f:
        for line in tqdm(lines):
            try:
                row = json.loads(line)
            except:
                continue
            
            # Handle different JSONL formats
            prompt = row.get("prompt", "")
            
            # Check for WM completion
            if "completion_wm" in row:
                row["llada_ppl_wm"] = calc.compute_pseudo_ppl(prompt, row["completion_wm"])
            elif "completion" in row:
                row["llada_ppl_wm"] = calc.compute_pseudo_ppl(prompt, row["completion"])
                
            # Check for No WM completion
            if "completion_no_wm" in row:
                row["llada_ppl_no_wm"] = calc.compute_pseudo_ppl(prompt, row["completion_no_wm"])
                
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()

    print(f"Finished. Results saved to {output_path}")

if __name__ == "__main__":
    main()
