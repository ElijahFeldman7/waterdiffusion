import json
import math
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from datasets import load_dataset

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
WATERMARK_DELTA = 8.0
NUM_SAMPLES = 50
OUTPUT_DIR = Path("c4_recovery_test")
RUN_NAME = "llada_8b_c4_wm_test_delta_8"
SEED = 42
GEN_STEPS = 128
GEN_LENGTH = 128


def count_syllables(word: str) -> int:
    word = word.lower()
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 0
    vowels = "aeiouy"
    syllables = 0
    prev_char_was_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_char_was_vowel:
            syllables += 1
        prev_char_was_vowel = is_vowel
    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return syllables if syllables > 0 else 1

#flesch stuff
def evaluate_text_quality(text: str) -> Dict[str, float]:
    text = text.strip()
    if not text:
        return {
            "avg_sentence_length": float("nan"),
            "avg_word_length": float("nan"),
            "lexical_diversity": float("nan"),
            "flesch_reading_ease": float("nan"),
            "sequence_length": float("nan"),
        }
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"\b\w+\b", text.lower())
    num_sentences = max(len(sentences), 1)
    num_words = max(len(words), 1)
    total_chars = sum(len(word) for word in words)
    total_syllables = sum(count_syllables(word) for word in words)
    unique_tokens = len(set(words))

    avg_sentence_length = len(words) / num_sentences
    avg_word_length = total_chars / num_words
    lexical_diversity = unique_tokens / num_words
    avg_syllables = total_syllables / num_words
    flesch_reading_ease = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables

    return {
        "avg_sentence_length": float(avg_sentence_length),
        "avg_word_length": float(avg_word_length),
        "lexical_diversity": float(lexical_diversity),
        "flesch_reading_ease": float(flesch_reading_ease),
        "sequence_length": float(len(words)),
    }


def analyze_jsonl(file_path: Path):
    total_samples = 0
    detected_samples = 0
    strong_detection_count = 0
    very_strong_detection_count = 0
    watermark_strength_stats = {
        "p_values": [],
        "z_scores": [],
        "satisfaction_rates": [],
        "num_timestamps": [],
        "unmasked_tokens_per_timestamp": [],
        "detection_margins": [],
    }
    quality_totals = {
        "avg_sentence_length": 0.0,
        "avg_word_length": 0.0,
        "lexical_diversity": 0.0,
        "flesch_reading_ease": 0.0,
        "sequence_length": 0.0,
    }
    quality_sample_count = 0
    
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
                    
                    detection = result.get('detection', {})
                    if detection:
                        p_value = detection.get('score')
                        threshold = detection.get('threshold')
                        z_score = detection.get('z_score')
                        satisfaction_rate = detection.get('satisfaction_rate')
                        num_timestamps = detection.get('num_timestamps')
                        unmasked_tokens_per_timestamp = detection.get('unmasked_tokens_per_timestamp')
                        
                        if p_value is not None:
                            watermark_strength_stats['p_values'].append(p_value)
                        if z_score is not None:
                            watermark_strength_stats['z_scores'].append(z_score)
                            if z_score >= 2.326:  # 99% one-tailed confidence, cf. Gloaguen et al. (2025)
                                strong_detection_count += 1
                            if z_score >= 3.090:  # 99.9% one-tailed confidence
                                very_strong_detection_count += 1
                        if satisfaction_rate is not None:
                            watermark_strength_stats['satisfaction_rates'].append(satisfaction_rate)
                        if num_timestamps is not None:
                            watermark_strength_stats['num_timestamps'].append(num_timestamps)
                        if unmasked_tokens_per_timestamp is not None:
                            watermark_strength_stats['unmasked_tokens_per_timestamp'].append(unmasked_tokens_per_timestamp)
                        if p_value is not None and threshold is not None:
                            watermark_strength_stats['detection_margins'].append(threshold - p_value)
                    
                    generated_text = result.get('generated_text', '')
                    if generated_text:
                        quality_metrics = evaluate_text_quality(generated_text)
                        quality_sample_count += 1
                        for key in quality_totals:
                            value = quality_metrics.get(key)
                            if value is not None and not math.isnan(value):
                                quality_totals[key] += value
                        
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
        
        if watermark_strength_stats['z_scores']:
            mean_z = float(np.mean(watermark_strength_stats['z_scores']))
            median_z = float(np.median(watermark_strength_stats['z_scores']))
            std_z = float(np.std(watermark_strength_stats['z_scores']))
            mean_p = float(np.mean(watermark_strength_stats['p_values'])) if watermark_strength_stats['p_values'] else float('nan')
            mean_satisfaction = float(np.mean(watermark_strength_stats['satisfaction_rates'])) if watermark_strength_stats['satisfaction_rates'] else float('nan')
            avg_margin = float(np.mean(watermark_strength_stats['detection_margins'])) if watermark_strength_stats['detection_margins'] else float('nan')
            avg_timestamps = float(np.mean(watermark_strength_stats['num_timestamps'])) if watermark_strength_stats['num_timestamps'] else float('nan')
            avg_unmasked = float(np.mean(watermark_strength_stats['unmasked_tokens_per_timestamp'])) if watermark_strength_stats['unmasked_tokens_per_timestamp'] else float('nan')
            
            print("\n--- Watermark Strength Metrics ---")
            print(f"Mean z-score: {mean_z:.4f}")
            print(f"Median z-score: {median_z:.4f}")
            print(f"Std. dev. z-score: {std_z:.4f}")
            print(f"Mean p-value: {mean_p:.4g}")
            print(f"Mean satisfaction rate: {mean_satisfaction:.4f}")
            print(f"Average detection margin (threshold - p): {avg_margin:.4f}")
            print(f"Avg. diffusion timestamps: {avg_timestamps:.2f}")
            print(f"Avg. unmasked tokens per timestamp: {avg_unmasked:.2f}")
            
            strong_rate = (strong_detection_count / total_samples) * 100
            very_strong_rate = (very_strong_detection_count / total_samples) * 100
            print("\n--- Robustness Indicators ---")
            print(f"Sequences with ≥99% confidence (z ≥ 2.33): {strong_rate:.2f}%")
            print(f"Sequences with ≥99.9% confidence (z ≥ 3.09): {very_strong_rate:.2f}%")
        
        if quality_sample_count > 0:
            print("\n--- Text Quality Metrics (averages) ---")
            for key, total in quality_totals.items():
                avg_value = total / quality_sample_count
                print(f"{key.replace('_', ' ').title()}: {avg_value:.4f}")
            print(f"Samples evaluated for quality: {quality_sample_count}")

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
                        gen_length=GEN_LENGTH,
                        steps=GEN_STEPS
                    )
                    
                    detection_result = watermark.detect(
                        text=generated_text,
                        tokenizer=generator.tokenizer,
                        num_timestamps=GEN_STEPS,
                        unmasked_tokens_per_timestamp=GEN_LENGTH / GEN_STEPS
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

