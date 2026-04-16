#!/usr/bin/env python3
"""
Run Zurich watermark on LLADA and produce JSONL output with TPR@1% metrics.
Simplified version that directly generates and merges watermarked/non-watermarked completions.
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
from typing import List, Dict
from tqdm import tqdm

# Add zurich to path - support both local and cluster paths
zurich_paths = [
    "/scratch/efeldma5/waterdiffusion/waterdiffusion/zurich",  # Cluster path
    "/Users/eli/research/waterdiffusion/zurich",  # Local path
]
zurich_path = None
for path in zurich_paths:
    if os.path.exists(path):
        zurich_path = path
        break

if zurich_path is None:
    print("Error: Zurich directory not found in any of:")
    for p in zurich_paths:
        print(f"  - {p}")
    sys.exit(1)

sys.path.insert(0, zurich_path)

from dlm_watermark.models.model_factory import load_model
from dlm_watermark.watermarks.watermark_factory import load_watermark_from_config
from dlm_watermark.configs import (
    OurWatermarkConfiguration,
    WatermarkType,
    ModelConfiguration
)


def load_prompts(jsonl_path: str, num_samples: int = None) -> List[Dict]:
    """Load prompts from WaterBench JSONL format."""
    prompts = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if num_samples and i >= num_samples:
                break
            data = json.loads(line)
            prompts.append({
                "input": data.get("input", ""),
                "source_file": Path(jsonl_path).name
            })
    return prompts


def generate_completion(model, tokenizer, prompt: str, watermark=None) -> str:
    """Generate a single completion."""
    device = model.device
    
    # Format the prompt as expected by the model
    formatted_prompt = f"You are a helpful assistant, please answer the following question within 300 words:\n\n{prompt}"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    try:
        torch.cuda.empty_cache()
        generation_output = model.generate_diffusion(
            input_ids=input_ids,
            attention_mask=attention_mask,
            watermark=watermark
        )
        sequences = generation_output.output_sequences
        completion = tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
        return completion
    except Exception as e:
        print(f"Error in generation: {e}")
        return ""


def detect_watermark(watermark, tokenizer, completion: str, device: str) -> Dict:
    """Detect watermark in completion."""
    if watermark is None:
        return {"z_score": 0.0, "p_value": 1.0}
    
    if not completion or len(completion) < 10:
        return {"z_score": 0.0, "p_value": 1.0}
    
    try:
        inputs = tokenizer(completion, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        detection = watermark.detect(input_ids=input_ids, attention_mask=attention_mask)
        return {
            "z_score": float(detection.get("z_score", 0.0)),
            "p_value": float(detection.get("p_value", 1.0))
        }
    except Exception as e:
        print(f"Error in detection: {e}")
        return {"z_score": 0.0, "p_value": 1.0}


def calculate_tpr_at_fpr(z_scores_wm: np.ndarray, z_scores_no_wm: np.ndarray, fpr_threshold: float = 0.01) -> float:
    """Calculate TPR@FPR."""
    
    if len(z_scores_wm) == 0 or len(z_scores_no_wm) == 0:
        return 0.0
    
    all_scores = np.concatenate([z_scores_no_wm, z_scores_wm])
    all_labels = np.concatenate([np.zeros_like(z_scores_no_wm), np.ones_like(z_scores_wm)])
    
    # Sort by score in descending order (higher = more likely watermarked)
    sorted_indices = np.argsort(-all_scores)
    sorted_labels = all_labels[sorted_indices]
    sorted_scores = all_scores[sorted_indices]
    
    # Find threshold that gives us target FPR
    n_no_wm = len(z_scores_no_wm)
    target_fp_count = max(1, int(np.ceil(fpr_threshold * n_no_wm)))
    
    # Count FPs as we go down the sorted list
    fp_count = 0
    threshold = sorted_scores[-1]  # Default to minimum score
    
    for i, label in enumerate(sorted_labels):
        if label == 0:  # No watermark
            fp_count += 1
            if fp_count >= target_fp_count:
                threshold = sorted_scores[i]
                break
    
    # Compute TPR at this threshold
    tp = np.sum(z_scores_wm >= threshold)
    tpr = tp / len(z_scores_wm) if len(z_scores_wm) > 0 else 0.0
    
    return tpr


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Zurich watermark on LLADA")
    parser.add_argument("--dataset", type=str, 
                        default="/scratch/efeldma5/waterdiffusion/waterdiffusion/zurich/data/WaterBench/2-1_longform_qa.jsonl",
                        help="Path to dataset")
    parser.add_argument("--output", type=str, default="watermark_results_zurich.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--num_samples", type=int, default=300,
                        help="Number of samples to evaluate")
    parser.add_argument("--delta_values", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0],
                        help="Delta values to evaluate")
    
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.dataset}")
    
    # Check if dataset exists, try alternative paths
    if not os.path.exists(args.dataset):
        print(f"Warning: Dataset not found at {args.dataset}")
        # Try local path
        local_dataset = args.dataset.replace(
            "/scratch/efeldma5/waterdiffusion/waterdiffusion/zurich",
            "/Users/eli/research/waterdiffusion/zurich"
        )
        if os.path.exists(local_dataset):
            print(f"Using local dataset instead: {local_dataset}")
            args.dataset = local_dataset
        else:
            print(f"Error: Dataset not found at either location")
            sys.exit(1)
    
    prompts = load_prompts(args.dataset, num_samples=args.num_samples)
    print(f"Loaded {len(prompts)} prompts")
    
    if len(prompts) == 0:
        print("No prompts found!")
        return
    
    print("\nLoading model...")
    model_config = ModelConfiguration(
        model_name="GSAI-ML/LLaDA-8B-Instruct",
        tokenizer_name="GSAI-ML/LLaDA-8B-Instruct"
    )
    model, tokenizer = load_model(model_config)
    device = model.device
    
    print(f"✓ Model loaded on device: {device}")
    print(f"✓ Total prompts: {len(prompts)}")
    print(f"✓ Delta values: {args.delta_values}")
    print(f"✓ Total deltas to evaluate: {len(args.delta_values)}")
    
    all_results = []
    
    for delta in args.delta_values:
        print(f"\n{'='*70}")
    print(f"Evaluating with delta = {delta}")
    print(f"{'='*70}")
        
        # Create watermark
        watermark_config = OurWatermarkConfiguration(
            delta=delta,
            enforce_kl=False,
            convolution_kernel=[-1],
            topk=50,
            n_iter=1,
            seeding_scheme="sumhash",
            greenlist_type="bernoulli",
            greenlist_params={"gamma": 0.25}
        )
        
        watermark = load_watermark_from_config(
            config=watermark_config,
            tokenizer=tokenizer,
            watermark_type=WatermarkType.OURS
        )
        
        z_scores_wm = []
        z_scores_no_wm = []
        delta_results = []
        
        for i, prompt_data in enumerate(tqdm(prompts, desc=f"Delta {delta}")):
            prompt = prompt_data["input"]
            source_file = prompt_data["source_file"]
            
            if not prompt or len(prompt) < 10:
                continue
            
            # Generate watermarked completion
            completion_wm = generate_completion(model, tokenizer, prompt, watermark=watermark)
            
            # Generate non-watermarked completion
            completion_no_wm = generate_completion(model, tokenizer, prompt, watermark=None)
            
            # Detect on watermarked
            detect_wm = detect_watermark(watermark, tokenizer, completion_wm, device)
            z_wm = detect_wm.get("z_score", 0.0)
            p_wm = detect_wm.get("p_value", 1.0)
            z_scores_wm.append(z_wm)
            
            # Detect on non-watermarked
            detect_no_wm = detect_watermark(watermark, tokenizer, completion_no_wm, device)
            z_no_wm = detect_no_wm.get("z_score", 0.0)
            p_no_wm = detect_no_wm.get("p_value", 1.0)
            z_scores_no_wm.append(z_no_wm)
            
            # Store result (will add TPR later)
            result = {
                "prompt": prompt,
                "completion_wm": completion_wm,
                "completion_no_wm": completion_no_wm,
                "source_file": source_file,
                "delta": delta,
                "det_wm_z_score": z_wm,
                "det_wm_p_value": p_wm,
                "det_no_wm_z_score": z_no_wm,
                "det_no_wm_p_value": p_no_wm,
            }
            all_results.append(result)
            delta_results.append(result)
        
        # Calculate TPR@1%
        if len(z_scores_wm) > 0 and len(z_scores_no_wm) > 0:
            z_scores_wm_arr = np.array(z_scores_wm, dtype=float)
            z_scores_no_wm_arr = np.array(z_scores_no_wm, dtype=float)
            tpr_at_1pct = calculate_tpr_at_fpr(z_scores_wm_arr, z_scores_no_wm_arr, fpr_threshold=0.01)
            print(f"\nTPR@1% FPR: {tpr_at_1pct:.4f}")
            print(f"Samples processed: {len(z_scores_wm)}")
            
            # Add TPR to results for this delta
            for result in delta_results:
                result["tpr_at_1pct"] = tpr_at_1pct
        else:
            print("Warning: No valid scores for TPR calculation")
    
    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
    
    print(f"\n{'='*70}")
    print(f"✓ Results saved to {args.output}")
    print(f"✓ Total results: {len(all_results)}")
    print(f"✓ Deltas evaluated: {len(args.delta_values)}")
    print(f"✓ Samples per delta: ~{len(all_results) // len(args.delta_values) if args.delta_values else 0}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
