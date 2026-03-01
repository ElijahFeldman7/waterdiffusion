import json
import math
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt

# Correctly import from the main project structure
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.llada import LLaDAGenerator
from watermarks.watermark_original import SimpleRemaskWatermark
from scripts.text_quality import calculate_perplexity, evaluate_text_quality
from scripts.analysis import analyze_jsonl

# --- Configuration ---
SECRET_KEY = "xooks"
WATERMARK_DELTA = 4.0
NUM_SAMPLES = 100
OUTPUT_DIR = "runs"
RUN_NAME = "llada_8b_adv_wm_test"
SEED = 42
GEN_STEPS = 400
GEN_LENGTH = 400
Z_THRESHOLD = 4.0

# --- Model Initializers ---
# These are loaded on demand to avoid memory issues on login nodes
perplexity_model = None
perplexity_tokenizer = None
similarity_model = None

def init_eval_models():
    """Initializes models needed for evaluation."""
    global perplexity_model, perplexity_tokenizer, similarity_model
    if perplexity_model is None:
        print("Initializing evaluation models (GPT2, SentenceTransformer)...")
        perplexity_model = GPT2LMHeadModel.from_pretrained('gpt2')
        perplexity_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        perplexity_model.eval()
        similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Evaluation models loaded.")

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_run_desc(run_dir: Path, watermark_params: dict):
    config_data = {
        "TIMESTAMP": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "RUN_NAME": RUN_NAME,
        "SEED": SEED,
        "NUM_PROMPTS": NUM_SAMPLES,
        "GENERATION_LENGTH": GEN_LENGTH,
        "GENERATION_STEPS": GEN_STEPS,
        "DETECTION_Z_THRESHOLD": Z_THRESHOLD,
        **watermark_params
    }
    desc_path = run_dir / "desc.txt"
    with open(desc_path, 'w') as f:
        f.write("RUN CONFIG\n")
        for key, value in config_data.items():
            f.write(f"{key}: {value}\n")
    print(f"Run config saved to {desc_path}")

def main():
    base_output_dir = Path(OUTPUT_DIR)
    date_str = datetime.utcnow().strftime("%m_%d_%H_%M")
    run_dir = base_output_dir / f"{date_str}_{RUN_NAME}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_file_path = run_dir / "results.jsonl"
    report_txt_path = run_dir / "summary_report.txt"

    print(f"Results will be saved in: {run_dir}")

    try:
        set_global_seed(SEED)
        print(f"Seed set to {SEED}")

        watermark_params = {
            "key": SECRET_KEY,
            "delta": WATERMARK_DELTA,
            "num_neighbors": 4,
            "adaptive_bias_threshold": 0.1,
            "confidence_gate_threshold": 0.5
        }
        watermark = SimpleRemaskWatermark(**watermark_params)
        save_run_desc(run_dir, watermark_params)

        print("Loading LLaDA model...")
        generator = LLaDAGenerator(
            config=str(Path(__file__).parent.parent / "configs" / "llada_8b_instruct.yaml"),
            watermark=watermark,
            device_map="auto"
        )
        print(f"Model {generator.model_path} loaded on device: {generator.device}")
        
        init_eval_models()

        print("Loading eli5 dataset stream...")
        ds = load_dataset("sentence-transformers/eli5", split="train", streaming=True)

        print(f"Running generation and detection for {NUM_SAMPLES} samples...")
        count = 0
        with open(results_file_path, 'w') as f:
            for ex in ds:
                if count >= NUM_SAMPLES: break
                prompt = ex.get("question", "").strip()
                if not (50 <= len(prompt) <= 1024): continue

                try:
                    # Generate with watermark
                    generator.watermark = watermark
                    generated_text_wm = generator.generate_text(prompt=prompt, gen_length=GEN_LENGTH, steps=GEN_STEPS)
                    
                    # Generate without watermark
                    generator.watermark = None
                    generated_text_no_wm = generator.generate_text(prompt=prompt, gen_length=GEN_LENGTH, steps=GEN_STEPS)

                    # Evaluate
                    perplexity_wm = calculate_perplexity(generated_text_wm, perplexity_model, perplexity_tokenizer)
                    perplexity_no_wm = calculate_perplexity(generated_text_no_wm, perplexity_model, perplexity_tokenizer)
                    embedding_wm, embedding_no_wm = similarity_model.encode([generated_text_wm, generated_text_no_wm])
                    cos_sim = F.cosine_similarity(torch.tensor(embedding_wm), torch.tensor(embedding_no_wm), dim=0).item()
                    
                    # Use the new z-score based detection
                    detection_wm_result = watermark.detect(text=generated_text_wm, tokenizer=generator.tokenizer, z_threshold=Z_THRESHOLD)
                    detection_no_wm_result = watermark.detect(text=generated_text_no_wm, tokenizer=generator.tokenizer, z_threshold=Z_THRESHOLD)

                    result_entry = {
                        "prompt": prompt,
                        "generated_text_wm": generated_text_wm,
                        "generated_text_no_wm": generated_text_no_wm,
                        "perplexity_wm": perplexity_wm,
                        "perplexity_no_wm": perplexity_no_wm,
                        "cosine_similarity": cos_sim,
                        "detection_wm": detection_wm_result,
                        "detection_no_wm": detection_no_wm_result,
                    }
                    f.write(json.dumps(result_entry) + "\n")

                    count += 1
                    if count % 10 == 0:
                        print(f"  ... processed {count}/{NUM_SAMPLES} samples")

                except Exception as e:
                    print(f"Error during generation/detection for prompt: {prompt}\nError: {e}", file=sys.stderr)

        print("Test run complete.")

        if results_file_path.exists() and count > 0:
            print("Analyzing results...")
            analyze_jsonl(results_file_path, report_txt_path)
        else:
            print("Test run finished, but no valid results were generated.")

    except Exception as e:
        print(f"A fatal error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
