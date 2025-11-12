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

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.llada import LLaDAGenerator
import importlib.util
project_root = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location(
    "init_unmasking_watermark", 
    str(project_root / "watermarks" / "11_02_init_unmasking_watermark.py")
)
init_unmasking_watermark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(init_unmasking_watermark)
SimpleRemaskWatermark = init_unmasking_watermark.SimpleRemaskWatermark


SECRET_KEY = "xooks"
WATERMARK_DELTA = 1.0
NUM_SAMPLES = 50
OUTPUT_DIR = Path("eli5_recovery_test")
RUN_NAME = "llada_8b_c4_wm_test_delta_8"
SEED = 42
GEN_STEPS = 128
GEN_LENGTH = 128


# calculate ppl and cosine sim
perplexity_model = GPT2LMHeadModel.from_pretrained('gpt2')
perplexity_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
perplexity_model.eval()
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')


from scripts.text_quality import calculate_perplexity, evaluate_text_quality


from scripts.analysis import analyze_jsonl


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{RUN_NAME}_{timestamp}"
    run_dir = OUTPUT_DIR / run_dir_name
    run_dir.mkdir()
    results_file_path = run_dir / "results.jsonl"
    report_txt_path = run_dir / "summary_report.txt"

    print(f"Starting C4 watermark recovery test...")
    print(f"Results will be saved in: {run_dir}")

    try:
        set_global_seed(SEED)
        print(f"Seed set to {SEED}")

        print("Initializing watermark...")
        watermark = SimpleRemaskWatermark(
            key=SECRET_KEY,
            delta=WATERMARK_DELTA
        )

        print("Loading LLaDA model...")
        generator = LLaDAGenerator(
            config=str(project_root / "configs" / "llada_8b_instruct.yaml"),
            watermark=watermark,
            device_map="auto"
        )
        print(f"Model {generator.model_path} loaded on device: {generator.device}")

        ds = load_dataset(
            "allenai/c4",
            "en.noblocklist",
            split="validation",
            streaming=True
        )

        print(f"Running generation and detection for {NUM_SAMPLES} samples...")
        
        all_results = []
        count = 0
        
        with open(results_file_path, 'w') as f:
            for ex in ds:
                if count >= NUM_SAMPLES:
                    break
                
                text = ex.get("question")
                if text is None:
                    continue
                
                prompt = text.strip()
                
                if not (50 <= len(prompt) <= 1024):
                    continue
                full_model_prompt = f"""Question: Your task is to be a text completion engine. Provide the most logical and natural continuation or answer for the following text. 
                Do not add any other commentary, preamble, or explanation. Respond only with the text that should follow.
                Text to complete: "{prompt}"

                Answer:"""
                prompt_snippet = " ".join(prompt.split()[:10])
                
                try:
                    start_time_wm = time.time()
                    generated_text_wm = generator.generate_text(
                        prompt=full_model_prompt,
                        gen_length=GEN_LENGTH,
                        steps=GEN_STEPS
                    )
                    gen_time_wm = time.time() - start_time_wm
                    generator.watermark = None
                    start_time_no_wm = time.time()
                    generated_text_no_wm = generator.generate_text(
                        prompt=prompt_snippet,
                        gen_length=GEN_LENGTH,
                        steps=GEN_STEPS
                    )
                    gen_time_no_wm = time.time() - start_time_no_wm
                    generator.watermark = watermark
                    perplexity_wm = calculate_perplexity(generated_text_wm, perplexity_model, perplexity_tokenizer)
                    perplexity_no_wm = calculate_perplexity(generated_text_no_wm, perplexity_model, perplexity_tokenizer)
                    embeddings = similarity_model.encode([generated_text_wm, generated_text_no_wm])
                    cos_sim = F.cosine_similarity(torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0).item()
                    detection_wm_result = watermark.detect(
                        text=generated_text_wm,
                        tokenizer=generator.tokenizer,
                        num_timestamps=GEN_STEPS,
                        unmasked_tokens_per_timestamp=GEN_LENGTH / GEN_STEPS
                    )
                    detection_no_wm_result = watermark.detect(
                        text=generated_text_no_wm,
                        tokenizer=generator.tokenizer,
                        num_timestamps=GEN_STEPS,
                        unmasked_tokens_per_timestamp=GEN_LENGTH / GEN_STEPS
                    )
                    
                    result_entry = {
                        "prompt": prompt_snippet,
                        "generated_text_wm": generated_text_wm,
                        "generated_text_no_wm": generated_text_no_wm,
                        "gen_time_wm": gen_time_wm,
                        "gen_time_no_wm": gen_time_no_wm,
                        "perplexity_wm": perplexity_wm,
                        "perplexity_no_wm": perplexity_no_wm,
                        "cosine_similarity": cos_sim,
                        "detection_wm": detection_wm_result,
                        "detection_no_wm": detection_no_wm_result,
                    }
                    
                    f.write(json.dumps(result_entry) + "\n")
                    all_results.append(result_entry)
                    
                    count += 1
                    if count % 10 == 0:
                        print(f"  ... processed {count}/{NUM_SAMPLES} samples")

                except Exception as e:
                    print(f"Error during generation/detection for prompt: {prompt_snippet}")
                    print(f"Error: {e}")

        print("Test run complete.")
        
        if results_file_path.exists():
            analyze_jsonl(results_file_path, report_txt_path)
        else:
            print("Test run finished, but no 'results.jsonl' file was found.")

    except FileNotFoundError as e:
        print(f"Error: A required file was not found.")
        print(f"Details: {e}")
        print("Please ensure 'configs/llada_8b_instruct.yaml' exists and that your python path is correct.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

