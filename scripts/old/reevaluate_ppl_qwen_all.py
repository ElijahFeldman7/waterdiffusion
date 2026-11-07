"""
Recalculate perplexity scores using Qwen-32B for all proposal runs.

Targets:
  - runs/prompt1_8519605  (Proposal 1 - CKAS)
  - runs/prompt2_8522189  (Proposal 2 - ARC)
  - runs/prompt3_8522188  (Proposal 3 - KLB)
  - runs/prompt4_8522187  (Proposal 4)
  - runs/zurich_eval_same_prompts_7006629  (Zurich baseline)

For each delta subfolder, reads the existing results JSONL file, scores
both watermarked and non-watermarked completions with Qwen-32B, and writes
a new file with `ppl_wm_qwen` and `ppl_no_wm_qwen` columns added.
"""

import json
import os
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss


# ── All runs to process ──────────────────────────────────────────────────────

RUN_DIRS = [
    "prompt1_8519605",
    "prompt2_8522189",
    "prompt3_8522188",
    "prompt4_8522187",
    "zurich_eval_same_prompts_7006629",
]


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
    padding = torch.zeros(
        (attn_mask.shape[0], attn_mask.shape[1] - prompt_attn_mask.shape[1]),
        dtype=torch.long,
        device=device
    )
    padded_prompt_attn_mask = torch.cat([prompt_attn_mask, padding], dim=1)
    prompt_token_mask = (padded_prompt_attn_mask == 1)

    # Clone attn_mask to avoid modifying original, then mask out prompt tokens
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
    n_completion_tokens = torch.clamp(n_completion_tokens, min=1)
    avg_loss = token_losses / n_completion_tokens

    ppls = torch.exp(avg_loss)
    return ppls.tolist()


def process_file(fpath, model, tokenizer, device, batch_size):
    """Process a single results JSONL file, adding Qwen PPL scores."""
    output_path = fpath.parent / f"{fpath.stem}_qwen_ppl{fpath.suffix}"

    # Skip if already processed
    if output_path.exists():
        print(f"  ⏭  Already exists: {output_path.name}, skipping.")
        return

    with open(fpath, "r") as f:
        rows = [json.loads(line) for line in f]

    if not rows:
        print(f"  ⚠  Empty file: {fpath}")
        return

    # Determine completion keys
    sample = rows[0]
    wm_key = "completion_wm" if "completion_wm" in sample else "completion"
    has_wm = wm_key in sample
    has_no_wm = "completion_no_wm" in sample

    # Score watermarked completions
    if has_wm:
        print(f"  Scoring watermarked completions ({wm_key})...")
        for i in tqdm(range(0, len(rows), batch_size), desc="  wm"):
            batch = rows[i:i + batch_size]
            prompts = [r.get("prompt", "") for r in batch]
            completions = [r.get(wm_key, "") for r in batch]
            ppls = compute_ppl_batch(model, tokenizer, prompts, completions, device)
            for j, ppl in enumerate(ppls):
                rows[i + j]["ppl_wm_qwen"] = ppl

    # Score non-watermarked completions
    if has_no_wm:
        print(f"  Scoring non-watermarked completions...")
        for i in tqdm(range(0, len(rows), batch_size), desc="  no_wm"):
            batch = rows[i:i + batch_size]
            prompts = [r.get("prompt", "") for r in batch]
            completions = [r.get("completion_no_wm", "") for r in batch]
            ppls = compute_ppl_batch(model, tokenizer, prompts, completions, device)
            for j, ppl in enumerate(ppls):
                rows[i + j]["ppl_no_wm_qwen"] = ppl

    with open(output_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  ✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate PPL with Qwen-32B for all proposal runs."
    )
    parser.add_argument(
        "--project-root", type=str, default=None,
        help="Project root directory. Auto-detected if not specified."
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for PPL evaluation."
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-32B-Instruct",
        help="The Qwen model to use for PPL evaluation."
    )
    parser.add_argument(
        "--runs", nargs="*", default=None,
        help="Specific run directory names to process. Defaults to all."
    )
    args = parser.parse_args()

    # Resolve project root
    if args.project_root:
        root = Path(args.project_root)
    else:
        root = Path(__file__).resolve().parent.parent
    runs_dir = root / "runs"

    run_dirs = args.runs if args.runs else RUN_DIRS

    # ── Load model ────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} (bfloat16)...")
    print("Note: Qwen2.5-32B requires ~64GB VRAM. A100 80GB recommended.\n")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # ── Process each run ──────────────────────────────────────────────────
    for run_name in run_dirs:
        run_path = runs_dir / run_name
        if not run_path.exists():
            print(f"\n⚠  Run directory not found: {run_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {run_name}")
        print(f"{'='*60}")

        # Find all delta subdirectories
        delta_dirs = sorted(
            run_path.glob("delta_*"),
            key=lambda p: float(p.name.split("_", 1)[1])
        )

        if not delta_dirs:
            print(f"  No delta_* subdirectories found in {run_name}")
            continue

        for delta_dir in delta_dirs:
            delta_val = delta_dir.name.split("_", 1)[1]
            print(f"\n  Delta {delta_val}:")

            # Look for results file (prefer results_merged, fall back to results_with_ppl)
            results_file = delta_dir / "results_merged.jsonl"
            if not results_file.exists():
                results_file = delta_dir / "results_with_ppl.jsonl"
            if not results_file.exists():
                # Also check for shard files
                shard_files = list(delta_dir.glob("shard_*.jsonl"))
                if shard_files:
                    results_file = shard_files[0]
                    print(f"  Using shard file: {results_file.name}")

            if not results_file.exists():
                print(f"  ⚠  No results file found in {delta_dir.name}")
                continue

            process_file(results_file, model, tokenizer, device, args.batch_size)

    print(f"\n{'='*60}")
    print("All done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
