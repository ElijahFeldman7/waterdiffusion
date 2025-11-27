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
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.llada import LLaDAGenerator
import importlib.util
sys.path.insert(0, str(Path(__file__).parent.parent / "zurich" / "src"))

project_root = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location(
    "init_unmasking_watermark",
    str(project_root / "watermarks" / "11_02_init_unmasking_watermark.py")
)
init_unmasking_watermark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(init_unmasking_watermark)
SimpleRemaskWatermark = init_unmasking_watermark.SimpleRemaskWatermark


SECRET_KEY = "xooks"
WATERMARK_DELTA = 2.0
NUM_SAMPLES = 10000
SEED = 42
GEN_STEPS = 400
GEN_LENGTH = 400


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    try:
        set_global_seed(SEED)
        print(f"Seed set to {SEED}")

        print("Initializing watermark detector...")
        watermark_detector = SimpleRemaskWatermark(
            key=SECRET_KEY,
            delta=WATERMARK_DELTA
        )

        print("Loading LLaDA model...")
        generator = LLaDAGenerator(
            config=str(project_root / "configs" / "llada_8b_instruct.yaml"),
            watermark=None,
            device_map="auto"
        )
        print(f"Model {generator.model_path} loaded on device: {generator.device}")

        print("Loading eli5 dataset stream...")
        ds = load_dataset(
            "sentence-transformers/eli5",
            split="train",
            streaming=True
        )

        print(f"Running generation and detection for {NUM_SAMPLES} samples...")
        
        satisfaction_rates = []
        count = 0
        
        for ex in ds:
            if count >= NUM_SAMPLES:
                break
            
            text = ex.get("question")
            if text is None:
                continue
            
            prompt = text.strip()
            
            if not (50 <= len(prompt) <= 1024):
                continue

            try:
                generated_text_no_wm = generator.generate_text(
                    prompt=prompt,
                    gen_length=GEN_LENGTH,
                    steps=GEN_STEPS,
                    batch_size=40
                )
                
                detection_result = watermark_detector.detect(
                    text=generated_text_no_wm,
                    tokenizer=generator.tokenizer,
                    num_timestamps=GEN_STEPS,
                    unmasked_tokens_per_timestamp=GEN_LENGTH / GEN_STEPS
                )
                
                if 'satisfaction_rate' in detection_result:
                    satisfaction_rates.append(detection_result['satisfaction_rate'])
                else:
                    print(f"Warning: 'satisfaction_rate' not found in detection result for sample {count+1}.")

                count += 1
                if count % 100 == 0:
                    print(f"  ... processed {count}/{NUM_SAMPLES} samples")

            except Exception as e:
                print(f"Error during generation/detection for prompt: {prompt}")
                print(f"Error: {e}")

        print("Test run complete.")
        
        if satisfaction_rates:
            average_satisfaction_rate = np.mean(satisfaction_rates)
            print("\n--- Null Hypothesis Satisfaction Rate ---")
            print(f"Average Satisfaction Rate over {len(satisfaction_rates)} samples: {average_satisfaction_rate:.4f}")
        else:
            print("\nCould not calculate average satisfaction rate. No values were collected.")

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