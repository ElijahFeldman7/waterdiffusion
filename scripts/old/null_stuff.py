import json
import math
import random
import sys
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
import matplotlib.pyplot as plt
import importlib.util
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from models.llada import LLaDAGenerator

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = project_root / "runs" / f"null_run_{run_id}"
run_dir.mkdir(parents=True, exist_ok=True)

watermark_path = project_root / "watermarks" / "11_02_init_unmasking_watermark.py"
spec = importlib.util.spec_from_file_location("init_unmasking_watermark", str(watermark_path))
init_unmasking_watermark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(init_unmasking_watermark)
SimpleRemaskWatermark = init_unmasking_watermark.SimpleRemaskWatermark

SECRET_KEY = "xooks"
WATERMARK_DELTA = 2.0
NUM_SAMPLES = 5000 
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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Run Directory: {run_dir}")

        watermark_detector = SimpleRemaskWatermark(
            key=SECRET_KEY,
            delta=WATERMARK_DELTA
        )

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading LLaDA model...")
        generator = LLaDAGenerator(
            config=str(project_root / "configs" / "llada_8b_instruct.yaml"),
            watermark=None, 
            device_map="auto"
        )

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading eli5 dataset...")
        ds = load_dataset("sentence-transformers/eli5", split="train", streaming=True)

        jsonl_file = run_dir / "null_results.jsonl"
        null_data = []
        count = 0
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting generation for {NUM_SAMPLES} samples...")

        with open(jsonl_file, "a", encoding="utf-8") as f_out:
            for ex in ds:
                if count >= NUM_SAMPLES:
                    break
                
                prompt = ex.get("question", "").strip()
                if not (50 <= len(prompt) <= 1024):
                    continue

                try:
                    output_text = generator.generate_text(
                        prompt=prompt,
                        gen_length=GEN_LENGTH,
                        steps=GEN_STEPS,
                        block_length=40
                    )
                    
                    res = watermark_detector.detect(output_text, generator.tokenizer)
                    
                    n = res['total_pairs_checked']
                    k = res['satisfied_pairs']
                    
                    if n > 0:
                        mu = n * 0.5
                        sigma = math.sqrt(n * 0.25)
                        z = (k - mu) / sigma
                        rate = k / n
                    else:
                        z, rate = 0.0, 0.0

                    sample_result = {
                        "n": n,
                        "rate": float(rate),
                        "z": float(z),
                        "timestamp": datetime.now().isoformat()
                    }

                    f_out.write(json.dumps(sample_result) + "\n")
                    f_out.flush()
                    
                    null_data.append(sample_result)
                    count += 1

                    if count % 100 == 0:
                        current_avg_z = np.mean([d['z'] for d in null_data])
                        print(f"  ... {count}/{NUM_SAMPLES} processed. Avg Z (running): {current_avg_z:.4f}")

                except Exception as e:
                    print(f"Error on sample {count}: {e}")
                    continue

        if null_data:
            z_scores = [d['z'] for d in null_data]
            empirical_z_threshold = np.percentile(z_scores, 99)
            
            print("\n" + "="*50)
            print("NULL HYPOTHESIS FINAL ANALYSIS")
            print("="*50)
            print(f"Total Samples: {len(z_scores)}")
            print(f"Mean Z-score:  {np.mean(z_scores):.4f}")
            print(f"EMPIRICAL 1% FPR THRESHOLD (Z): {empirical_z_threshold:.4f}")
            print("="*50)

            metadata = {
                "empirical_z_threshold": float(empirical_z_threshold),
                "num_samples": len(z_scores),
                "gen_length_target": GEN_LENGTH,
                "gen_steps": GEN_STEPS,
                "mean_z": float(np.mean(z_scores)),
                "std_z": float(np.std(z_scores))
            }
            with open(run_dir / "null_metadata.json", "w") as f_meta:
                json.dump(metadata, f_meta, indent=4)
            
            plt.figure(figsize=(10, 6))
            plt.hist(z_scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            plt.axvline(empirical_z_threshold, color='red', linestyle='--', linewidth=2, 
                        label=f'1% FPR Threshold (Z={empirical_z_threshold:.2f})')
            plt.title(f'Z-Score Distribution (Unwatermarked Text, N={len(z_scores)})')
            plt.xlabel('Z-Score')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.savefig(run_dir / "null_z_distribution.png")
            
            print(f"Success! Files saved in: {run_dir}")

    except Exception as e:
        print(f"Fatal script error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()