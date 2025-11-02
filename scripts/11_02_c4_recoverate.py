import subprocess
import json
import sys
from pathlib import Path
import random
from datetime import datetime
import torch
import numpy as np
from datasets import load_dataset
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.llada import LLaDAGenerator
import importlib.util
spec = importlib.util.spec_from_file_location(
    "init_unmasking_watermark", 
    str(Path(__file__).parent.parent / "watermarks" / "11_02_init_unmasking_watermark.py")
)
init_unmasking_watermark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(init_unmasking_watermark)
SimpleRemaskWatermark = init_unmasking_watermark.SimpleRemaskWatermark


SECRET_KEY = "xooks"
WATERMARK_DELTA = 4.0
NUM_SAMPLES = 500
OUTPUT_DIR = Path("c4_recovery_test")
RUN_NAME = "llada_8b_c4_wm_test"
SEED = 42

def analyze_jsonl(file_path: Path):
    total_samples = 0
    detected_samples = 0
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    result = json.loads(line)
                    total_samples += 1
                    
                    if 'detection' in result and result['detection'].get('is_watermarked', False):
                        detected_samples += 1
                        
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line: {line.strip()}")

        if total_samples == 0:
            print("Error: No valid results found in the file.")
            return

        recovery_rate = (detected_samples / total_samples) * 100
        
        print("\n--- Watermark Recovery Analysis ---")
        print(f"Total samples processed: {total_samples}")
        print(f"Samples detected: {detected_samples}")
        print(f"Recoverability Rate: {recovery_rate:.2f}%")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


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

        print("Loading C4 dataset stream...")
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
                
                text = ex.get("text")
                if text is None:
                    continue
                
                prompt = text.strip()
                
                if not (50 <= len(prompt) <= 1024):
                    continue

                prompt_snippet = " ".join(prompt.split()[:10])
                
                try:
                    generated_text = generator.generate_text(
                        prompt=prompt_snippet,
                        gen_length=128,
                        steps=128
                    )
                    
                    detection_result = watermark.detect(
                        text=generated_text,
                        tokenizer=generator.tokenizer
                    )
                    
                    result_entry = {
                        "prompt": prompt_snippet,
                        "generated_text": generated_text,
                        "detection": detection_result
                    }
                    
                    f.write(json.dumps(result_entry) + "\n")
                    all_results.append(result_entry)
                    
                    count += 1
                    if count % 25 == 0:
                        print(f"  ... processed {count}/{NUM_SAMPLES} samples")

                except Exception as e:
                    print(f"Error during generation/detection for prompt: {prompt_snippet}")
                    print(f"Error: {e}")

        print("Test run complete.")
        
        if results_file_path.exists():
            analyze_jsonl(results_file_path)
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

