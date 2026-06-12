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

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.llada import LLaDAGenerator

import importlib.util
project_root = Path(__file__).parent.parent

spec = importlib.util.spec_from_file_location(
    "init_unmasking_watermark",
    str(project_root / "watermarks" / "1_28_nonvectorized.py")
)
init_unmasking_watermark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(init_unmasking_watermark)
SimpleRemaskWatermark = init_unmasking_watermark.SimpleRemaskWatermark

sys.path.insert(0, str(project_root / "zurich" / "src"))
from dlm_watermark.configs import MainConfiguration
from dlm_watermark.watermarks.watermark_factory import load_watermark_from_config
import yaml


SECRET_KEY = "xooks"
WATERMARK_DELTA = 6.0
NUM_SAMPLES = 100
OUTPUT_DIR = "runs"
RUN_NAME = "llada_8b_eli5_wm_test"
SEED = 42
GEN_STEPS = 400
GEN_LENGTH = 400
P_VALUE_THRESHOLD = 0.01
BLOCK_LENGTH = 40


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


def save_run_desc(run_dir: Path, gen_length: int, p_value_threshold: float):
    config_data = {
        "TIMESTAMP": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "RUN_NAME": RUN_NAME,
        "SEED": SEED,
        "OUTPUT_DIR": str(run_dir),
        "NUM_PROMPTS": NUM_SAMPLES,
        "WATERMARK_KEY": SECRET_KEY,
        "WATERMARK_DELTA": WATERMARK_DELTA,
        "GENERATION_LENGTH": gen_length,
        "GENERATION_STEPS": GEN_STEPS,
        "P_VALUE_THRESHOLD": p_value_threshold,
        "LLADA_CONFIG_PATH": str(project_root / "configs" / "llada_8b_instruct.yaml"),
        "PERPLEXITY_MODEL": "GPT2",
        "SIMILARITY_MODEL": "all-MiniLM-L6-v2",
        "BLOCK_LENGTH": BLOCK_LENGTH
    }
    desc_path = run_dir / "desc.txt"
    with open(desc_path, 'w') as f:
        f.write("RUN CONFIG \n")
        for key, value in config_data.items():
            f.write(f"{key}: {value}\n")
    print(f"run config saved to {desc_path}")


def parse_summary_report(report_path: Path) -> dict:
    metrics = {
        'tpr': None, 'fpr': None, 'perplexity_wm': None, 'perplexity_no_wm': None,
    }
    if not report_path.exists():
        print(f"Warning: Report file not found at {report_path}")
        return metrics

    with open(report_path, 'r') as f:
        content = f.read()

        tpr_match = re.search(r"True Positive Rate \(TPR/Recall\):\s+([\d.]+)", content)
        if tpr_match: metrics['tpr'] = float(tpr_match.group(1))

        fpr_match = re.search(r"False Positive Rate \(FPR\):\s+([\d.]+)", content)
        if fpr_match: metrics['fpr'] = float(fpr_match.group(1))

        ppl_wm_match = re.search(r"Perplexity \(WM\):\s+([\d.]+)", content)
        if ppl_wm_match: metrics['perplexity_wm'] = float(ppl_wm_match.group(1))

        ppl_no_wm_match = re.search(r"Perplexity \(No WM\):\s+([\d.]+)", content)
        if ppl_no_wm_match: metrics['perplexity_no_wm'] = float(ppl_no_wm_match.group(1))

    return metrics


def main():
    base_output_dir = project_root / OUTPUT_DIR
    date_str = datetime.utcnow().strftime("%m_%d_%H_%M")
    run_dir = base_output_dir / f"{date_str}_{RUN_NAME}"
    run_dir.mkdir(parents=True, exist_ok=True)

    length_specific_dir = run_dir / f"gen_length_{GEN_LENGTH}"
    length_specific_dir.mkdir(parents=True, exist_ok=True)

    results_file_path = length_specific_dir / "results.jsonl"
    report_txt_path = length_specific_dir / "summary_report.txt"

    print(f"Results will be saved in: {length_specific_dir}")


    try:
        set_global_seed(SEED)
        print(f"Seed set to {SEED}")

        save_run_desc(run_dir, GEN_LENGTH, P_VALUE_THRESHOLD)

        print("Initializing our watermark...")
        our_watermark = SimpleRemaskWatermark(
            key=SECRET_KEY,
            delta=WATERMARK_DELTA
        )

        print("Initializing zurich watermark...")
        config_path = project_root / "zurich" / "configs" / "main" / "Llada" / "eli5_test.yaml"
        config = MainConfiguration(**yaml.safe_load(open(config_path, "r")))
        zurich_watermark = load_watermark_from_config(config=config.watermark_config, tokenizer=None, watermark_type=config.watermark_type)

        print("Loading LLaDA model...")
        generator = LLaDAGenerator(
            config=str(project_root / "configs" / "llada_8b_instruct.yaml"),
            watermark=our_watermark,
            device_map="auto"
        )
        print(f"Model {generator.model_path} loaded on device: {generator.device}")

        if hasattr(zurich_watermark, 'tokenizer') and zurich_watermark.tokenizer is None:
            zurich_watermark.tokenizer = generator.tokenizer


        waterbench_path = project_root / "zurich" / "data" / "WaterBench" / "2-1_longform_qa.jsonl"
        print(f"Loading WaterBench data from {waterbench_path}")
        with open(waterbench_path, "r") as datafile:
            lines = datafile.readlines()

        print(f"Running generation and detection for {NUM_SAMPLES} WaterBench samples...")
        count = 0
        with open(results_file_path, 'w') as f:
            for line in lines:
                if count >= NUM_SAMPLES:
                    break
                ex = json.loads(line)
                context = ex.get("context", "")
                input_text = ex.get("input", "")
                prompt = f"You are a helpful assistant, please answer the following question within 300 words:\n{context}\n{input_text}"
                if not (50 <= len(prompt) <= 1024):
                    continue
                try:
                    generator.watermark = our_watermark
                    start_time_our_wm = time.time()
                    generated_text_our_wm = generator.generate_text(
                        prompt=prompt,
                        gen_length=GEN_LENGTH,
                        steps=GEN_STEPS,
                        block_length=BLOCK_LENGTH
                    )
                    gen_time_our_wm = time.time() - start_time_our_wm

                    generator.watermark = zurich_watermark
                    start_time_zurich_wm = time.time()
                    generated_text_zurich_wm = generator.generate_text(
                        prompt=prompt,
                        gen_length=GEN_LENGTH,
                        steps=GEN_STEPS,
                        block_length=BLOCK_LENGTH
                    )
                    gen_time_zurich_wm = time.time() - start_time_zurich_wm

                    generator.watermark = None
                    start_time_no_wm = time.time()
                    generated_text_no_wm = generator.generate_text(
                        prompt=prompt,
                        gen_length=GEN_LENGTH,
                        steps=GEN_STEPS,
                        block_length=BLOCK_LENGTH
                    )
                    gen_time_no_wm = time.time() - start_time_no_wm

                    perplexity_our_wm = calculate_perplexity(generated_text_our_wm, perplexity_model, perplexity_tokenizer)
                    perplexity_zurich_wm = calculate_perplexity(generated_text_zurich_wm, perplexity_model, perplexity_tokenizer)
                    perplexity_no_wm = calculate_perplexity(generated_text_no_wm, perplexity_model, perplexity_tokenizer)

                    embeddings = similarity_model.encode([generated_text_our_wm, generated_text_zurich_wm, generated_text_no_wm])
                    cos_sim_our_vs_no = F.cosine_similarity(torch.tensor(embeddings[0]), torch.tensor(embeddings[2]), dim=0).item()
                    cos_sim_zurich_vs_no = F.cosine_similarity(torch.tensor(embeddings[1]), torch.tensor(embeddings[2]), dim=0).item()

                    detection_our_wm_result = our_watermark.detect(
                        text=generated_text_our_wm,
                        tokenizer=generator.tokenizer,
                        num_timestamps=GEN_STEPS,
                        unmasked_tokens_per_timestamp=GEN_LENGTH / GEN_STEPS,
                        p_value_threshold=P_VALUE_THRESHOLD
                    )
                    tokenized_text_zurich = generator.tokenizer(generated_text_zurich_wm, return_tensors="pt")
                    detection_zurich_wm_result = zurich_watermark.detect(
                        input_ids=tokenized_text_zurich['input_ids'].to(generator.device),
                        attention_mask=tokenized_text_zurich['attention_mask'].to(generator.device)
                    )
                    detection_no_wm_result = our_watermark.detect(
                        text=generated_text_no_wm,
                        tokenizer=generator.tokenizer,
                        num_timestamps=GEN_STEPS,
                        unmasked_tokens_per_timestamp=GEN_LENGTH / GEN_STEPS,
                        p_value_threshold=P_VALUE_THRESHOLD
                    )

                    result_entry = {
                        "prompt": prompt,
                        "generated_text_our_wm": generated_text_our_wm,
                        "generated_text_zurich_wm": generated_text_zurich_wm,
                        "generated_text_no_wm": generated_text_no_wm,
                        "gen_time_our_wm": gen_time_our_wm,
                        "gen_time_zurich_wm": gen_time_zurich_wm,
                        "gen_time_no_wm": gen_time_no_wm,
                        "perplexity_our_wm": perplexity_our_wm,
                        "perplexity_zurich_wm": perplexity_zurich_wm,
                        "perplexity_no_wm": perplexity_no_wm,
                        "cosine_similarity_our_vs_no": cos_sim_our_vs_no,
                        "cosine_similarity_zurich_vs_no": cos_sim_zurich_vs_no,
                        "detection_our_wm": detection_our_wm_result,
                        "detection_zurich_wm": detection_zurich_wm_result,
                        "detection_no_wm": detection_no_wm_result,
                    }

                    f.write(json.dumps(result_entry) + "\n")

                    count += 1
                    if count % 10 == 0:
                        print(f"  ... processed {count}/{NUM_SAMPLES} samples")

                except Exception as e:
                    print(f"Error during generation/detection for prompt: {prompt}")
                    print(f"Error: {e}")

        print("Test run complete.")

        if results_file_path.exists():
            analyze_jsonl(results_file_path, report_txt_path)
        else:
            print("Test run finished, but no 'results.jsonl' file was found.")

        metrics = parse_summary_report(report_txt_path)
        if all(val is not None for val in metrics.values()):
            print("Generating plots...")
            plt.style.use('seaborn-v0_8-whitegrid')

            fig1, ax1 = plt.subplots()
            labels = ['TPR', 'FPR']
            values = [metrics['tpr'], metrics['fpr']]
            ax1.bar(labels, values, color=['green', 'red'])
            ax1.set_ylabel("Rate")
            ax1.set_title(f"Watermark Detection Performance (Length: {GEN_LENGTH})")
            ax1.set_ylim(0, 1.05)
            for i, v in enumerate(values):
                if isinstance(v, (int, float)):
                    ax1.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
            perf_plot_path = run_dir / "detection_performance.png"
            fig1.savefig(perf_plot_path, dpi=300)
            print(f"Saved detection performance plot to: {perf_plot_path}")

            fig2, ax2 = plt.subplots()
            labels_quality = ['Perplexity (Ours)', 'Perplexity (Zurich)', 'Perplexity (No WM)']
            values_quality = [metrics['perplexity_our_wm'], metrics['perplexity_zurich_wm'], metrics['perplexity_no_wm']]
            ax2.bar(labels_quality, values_quality, color=['blue', 'purple', 'orange'])
            ax2.set_ylabel("Perplexity")
            ax2.set_title(f"Text Quality (Perplexity) (Length: {GEN_LENGTH})")
            for i, v in enumerate(values_quality):
                if isinstance(v, (int, float)):
                    ax2.text(i, v + 0.5, f"{v:.2f}", ha='center', fontweight='bold')
            quality_plot_path = run_dir / "text_quality_impact.png"
            fig2.savefig(quality_plot_path, dpi=300)
            print(f"Saved text quality plot to: {quality_plot_path}")
        else:
            print("Could not find any valid results to plot.")

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