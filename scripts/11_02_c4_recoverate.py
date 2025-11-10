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
WATERMARK_DELTA = 0.1
NUM_SAMPLES = 50
OUTPUT_DIR = Path("c4_recovery_test")
RUN_NAME = "llada_8b_c4_wm_test_delta_8"
SEED = 42
GEN_STEPS = 128
GEN_LENGTH = 128


# calculate ppl and cosine sim
perplexity_model = GPT2LMHeadModel.from_pretrained('gpt2')
perplexity_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
perplexity_model.eval()
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')


def calculate_perplexity(text: str, model, tokenizer) -> float:
    if not text.strip():
        return float("nan")
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids
    if input_ids.size(1) > 1024:
        input_ids = input_ids[:, :1024]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss
    perplexity = torch.exp(neg_log_likelihood)
    return perplexity.item()
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
        "p_values": [], "z_scores": [], "satisfaction_rates": [],
        "num_timestamps": [], "unmasked_tokens_per_timestamp": [], "detection_margins": [],
    }
    
    quality_stats = {
        "gen_time_wm": [], "gen_time_no_wm": [], "perplexity_wm": [],
        "perplexity_no_wm": [], "cosine_similarity": [],
    }

    quality_totals_wm = {k: 0.0 for k in ["avg_sentence_length", "avg_word_length", "lexical_diversity", "flesch_reading_ease", "sequence_length"]}
    quality_totals_no_wm = {k: 0.0 for k in quality_totals_wm}
    quality_sample_count = 0

    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                
                try:
                    result = json.loads(line)
                    total_samples += 1
                    
                    if 'detection' in result and result['detection'].get('is_watermarked', False):
                        detected_samples += 1
                    
                    detection = result.get('detection', {})
                    if detection:
                        if (p_value := detection.get('score')) is not None: watermark_strength_stats['p_values'].append(p_value)
                        if (z_score := detection.get('z_score')) is not None:
                            watermark_strength_stats['z_scores'].append(z_score)
                            if z_score >= 2.326: strong_detection_count += 1
                            if z_score >= 3.090: very_strong_detection_count += 1
                        if (sr := detection.get('satisfaction_rate')) is not None: watermark_strength_stats['satisfaction_rates'].append(sr)
                        if (nt := detection.get('num_timestamps')) is not None: watermark_strength_stats['num_timestamps'].append(nt)
                        if (ut := detection.get('unmasked_tokens_per_timestamp')) is not None: watermark_strength_stats['unmasked_tokens_per_timestamp'].append(ut)
                        if p_value is not None and (thresh := detection.get('threshold')) is not None:
                            watermark_strength_stats['detection_margins'].append(thresh - p_value)

                    for key in quality_stats:
                        if (val := result.get(key)) is not None and not math.isnan(val):
                            quality_stats[key].append(val)

                    if (text_wm := result.get('generated_text_wm')) and (text_no_wm := result.get('generated_text_no_wm')):
                        quality_metrics_wm = evaluate_text_quality(text_wm)
                        quality_metrics_no_wm = evaluate_text_quality(text_no_wm)
                        quality_sample_count += 1
                        for key in quality_totals_wm:
                            if (val := quality_metrics_wm.get(key)) is not None and not math.isnan(val): quality_totals_wm[key] += val
                            if (val := quality_metrics_no_wm.get(key)) is not None and not math.isnan(val): quality_totals_no_wm[key] += val

                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line: {line.strip()}")

        if total_samples == 0:
            print("Error: No valid results found in the file.")
            return

        final_results = {"summary": {}, "watermark_strength": {}, "performance_impact": {}, "quality_impact": {}, "detailed_quality_metrics": {}}

        recovery_rate = (detected_samples / total_samples) * 100 if total_samples > 0 else 0
        final_results["summary"] = {"total_samples": total_samples, "detected_samples": detected_samples, "recovery_rate_percent": recovery_rate}
        print("\n--- Watermark Recovery Analysis ---")
        print(f"Total samples processed: {total_samples}")
        print(f"Samples detected: {detected_samples}")
        print(f"Recoverability Rate: {recovery_rate:.2f}%")
        
        if watermark_strength_stats['z_scores']:
            mean_z = float(np.mean(watermark_strength_stats['z_scores']))
            std_z = float(np.std(watermark_strength_stats['z_scores']))
            strong_rate = (strong_detection_count / total_samples) * 100
            very_strong_rate = (very_strong_detection_count / total_samples) * 100
            final_results["watermark_strength"] = {
                "mean_z_score": mean_z, "std_dev_z_score": std_z,
                "strong_detection_rate_percent": strong_rate, "very_strong_detection_rate_percent": very_strong_rate
            }
            print(f"Mean z-score: {mean_z:.4f} | Std. dev: {std_z:.4f}")
            print(f"Detection Confidence: ≥99% (z≥2.33): {strong_rate:.2f}% | ≥99.9% (z≥3.09): {very_strong_rate:.2f}%")

        if quality_stats["gen_time_wm"]:
            mean_gen_time_wm = float(np.mean(quality_stats["gen_time_wm"]))
            mean_gen_time_no_wm = float(np.mean(quality_stats["gen_time_no_wm"]))
            time_diff = mean_gen_time_wm - mean_gen_time_no_wm
            time_overhead = (time_diff / mean_gen_time_no_wm) * 100 if mean_gen_time_no_wm > 0 else 0
            final_results["performance_impact"] = {
                "mean_gen_time_wm_s": mean_gen_time_wm, "mean_gen_time_no_wm_s": mean_gen_time_no_wm,
                "mean_time_overhead_s": time_diff, "mean_time_overhead_percent": time_overhead
            }
            print(f"Generation Time (WM): {mean_gen_time_wm:.4f}s")
            print(f"Generation Time (No WM): {mean_gen_time_no_wm:.4f}s")
            print(f"Mean Time Overhead: {time_diff:.4f}s ({time_overhead:.2f}%)")

        if quality_stats["perplexity_wm"]:
            mean_ppl_wm = float(np.mean(quality_stats["perplexity_wm"]))
            mean_ppl_no_wm = float(np.mean(quality_stats["perplexity_no_wm"]))
            ppl_diff = mean_ppl_wm - mean_ppl_no_wm
            ppl_increase = (ppl_diff / mean_ppl_no_wm) * 100 if mean_ppl_no_wm > 0 else 0
            mean_cos_sim = float(np.mean(quality_stats["cosine_similarity"]))
            final_results["quality_impact"] = {
                "mean_perplexity_wm": mean_ppl_wm, "mean_perplexity_no_wm": mean_ppl_no_wm,
                "mean_perplexity_increase": ppl_diff, "mean_perplexity_increase_percent": ppl_increase,
                "mean_cosine_similarity": mean_cos_sim
            }
            print(f"\nPerplexity (WM): {mean_ppl_wm:.4f}")
            print(f"Perplexity (No WM): {mean_ppl_no_wm:.4f}")
            print(f"Mean Perplexity Increase: {ppl_diff:.4f} ({ppl_increase:.2f}%)")
            print(f"\nCosine Similarity (WM vs No WM): {mean_cos_sim:.4f}")

        if quality_sample_count > 0:
            print("\n--- Detailed Text Quality Metrics (averages) ---")
            print(f"{'Metric':<25} | {'Watermarked':<15} | {'No Watermark':<15}")
            print("-" * 60)
            detailed_wm = {}
            detailed_no_wm = {}
            for key in quality_totals_wm:
                avg_wm = quality_totals_wm[key] / quality_sample_count
                avg_no_wm = quality_totals_no_wm[key] / quality_sample_count
                detailed_wm[key] = avg_wm
                detailed_no_wm[key] = avg_no_wm
                print(f"{key.replace('_', ' ').title():<25} | {avg_wm:<15.4f} | {avg_no_wm:<15.4f}")
            final_results["detailed_quality_metrics"] = {"watermarked": detailed_wm, "no_watermark": detailed_no_wm}
            print(f"\nSamples evaluated for quality: {quality_sample_count}")

        results_json_path = file_path.parent / "results.json"
        with open(results_json_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"\nAggregated results saved to: {results_json_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
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
                    start_time_wm = time.time()
                    generated_text_wm = generator.generate_text(
                        prompt=prompt_snippet,
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
                    # --- Detections ---
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

