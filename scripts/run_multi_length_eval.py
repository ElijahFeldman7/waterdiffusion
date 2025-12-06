import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.llada import LLaDAGenerator
from scripts.analysis import analyze_jsonl
from scripts.text_quality import calculate_perplexity

project_root = Path(__file__).parent.parent
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "init_unmasking_watermark",
        str(project_root / "watermarks" / "11_02_init_unmasking_watermark.py")
    )
    init_unmasking_watermark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(init_unmasking_watermark)
    SimpleRemaskWatermark = init_unmasking_watermark.SimpleRemaskWatermark
except ImportError:
    print("Could not import the watermark module. Please ensure the path is correct.")
    sys.exit(1)

NUM_PROMPTS = 100
RUN_NAME = "eli5_multi_length"
SEED = 42
OUTPUT_DIR = "runs"
SECRET_KEY = "xooks"
WATERMARK_DELTA = 4.0


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def run_evaluation_for_length(
    generator: LLaDAGenerator,
    watermark: SimpleRemaskWatermark,
    prompts: list[str],
    gen_length: int,
    output_dir: Path,
    perplexity_model,
    perplexity_tokenizer,
    similarity_model
):    
    print(f"\n----- Starting Evaluation for Generation Length: {gen_length} -----")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file_path = output_dir / "results.jsonl"
    report_txt_path = output_dir / "summary_report.txt"

    num_samples = len(prompts)
    count = 0

    with open(results_file_path, 'w') as f:
        for prompt in prompts:
            try:
                print(f"  [Len {gen_length}] Generating watermarked text for prompt: {prompt[:30]}...")
                generator.watermark = watermark
                generated_text_wm = generator.generate_text(
                    prompt=prompt, gen_length=gen_length, steps=gen_length
                )
                
                print(f"  [Len {gen_length}] Generating non-watermarked text...")
                generator.watermark = None
                generated_text_no_wm = generator.generate_text(
                    prompt=prompt, gen_length=gen_length, steps=gen_length
                )
                
                print(f"  [Len {gen_length}] Calculating perplexity...")
                perplexity_wm = calculate_perplexity(generated_text_wm, perplexity_model, perplexity_tokenizer)
                perplexity_no_wm = calculate_perplexity(generated_text_no_wm, perplexity_model, perplexity_tokenizer)
                
                print(f"  [Len {gen_length}] Calculating similarity...")
                embeddings = similarity_model.encode([generated_text_wm, generated_text_no_wm])
                cos_sim = F.cosine_similarity(torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0).item()
                
                print(f"  [Len {gen_length}] Detecting watermark in watermarked text...")
                detection_wm_result = watermark.detect(
                    text=generated_text_wm, tokenizer=generator.tokenizer, num_timestamps=gen_length,
                    unmasked_tokens_per_timestamp=1, p_value_threshold=0.05
                )
                
                print(f"  [Len {gen_length}] Detecting watermark in non-watermarked text...")
                detection_no_wm_result = watermark.detect(
                    text=generated_text_no_wm, tokenizer=generator.tokenizer, num_timestamps=gen_length,
                    unmasked_tokens_per_timestamp=1, p_value_threshold=0.05
                )
                
                result_entry = {
                    "prompt": prompt,
                    "gen_length": gen_length,
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
                    print(f"  ... processed {count}/{num_samples} prompts for length {gen_length}")

            except Exception as e:
                print(f"Error during generation/detection for prompt (len {gen_length}): {prompt[:80]}...")
                print(f"Error: {e}")

    print(f"Evaluation for length {gen_length} complete. Results saved to {results_file_path}")
    
    if results_file_path.exists():
        analyze_jsonl(results_file_path, report_txt_path)
    else:
        print(f"No results file found for length {gen_length}. Skipping report generation.")

def save_run_desc(run_dir: Path, gen_lengths: list[int]):
    config_data = {
        "TIMESTAMP": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "RUN_NAME": RUN_NAME,
        "SEED": SEED,
        "OUTPUT_DIR": str(run_dir),
        "NUM_PROMPTS": NUM_PROMPTS,
        "WATERMARK_KEY": SECRET_KEY,
        "WATERMARK_DELTA": WATERMARK_DELTA,
        "GENERATION_LENGTHS_TESTED": gen_lengths,
        "LLADA_CONFIG_PATH": str(project_root / "configs" / "llada_8b_instruct.yaml"),
        "PERPLEXITY_MODEL": "GPT2",
        "SIMILARITY_MODEL": "all-MiniLM-L6-v2"
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
    date_str = datetime.utcnow().strftime("%m_%d")
    run_dir = base_output_dir / f"{date_str}_{RUN_NAME}"
    print(f"Overall results will be saved in subdirectories of: {run_dir}")
    try:
        set_global_seed(SEED)
        print(f"Global random seed set to {SEED}")
        run_dir.mkdir(parents=True, exist_ok=True)
        gen_lengths_to_test = [100, 200, 300, 400]
        save_run_desc(run_dir, gen_lengths_to_test)

        ds = load_dataset("sentence-transformers/eli5", split="train", streaming=True)
        prompts = []
        for ex in ds:
            if len(prompts) >= NUM_PROMPTS:
                break
            text = ex.get("question")
            if text:
                prompt = text.strip()
                prompts.append(prompt)  
        if len(prompts) < NUM_PROMPTS:
            raise ValueError(f"Could not find enough prompts. Found {len(prompts)}, but need {NUM_PROMPTS}.")
        watermark = SimpleRemaskWatermark(key=SECRET_KEY, delta=WATERMARK_DELTA)
        generator = LLaDAGenerator(
            config=str(project_root / "configs" / "llada_8b_instruct.yaml"),
            watermark=watermark,
            device_map="auto"
        )
        print(f"Model {generator.model_path} loaded on device: {generator.device}")
        perplexity_model = GPT2LMHeadModel.from_pretrained('gpt2')
        perplexity_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        perplexity_model.eval()
        similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        plot_data = {'gen_lengths': [], 'tpr': [], 'fpr': [], 'perplexity_wm': [], 'perplexity_no_wm': []}

        for length in gen_lengths_to_test:
            length_specific_dir = run_dir / f"gen_length_{length}"
            run_evaluation_for_length(
                generator=generator, watermark=watermark, prompts=prompts, gen_length=length,
                output_dir=length_specific_dir, perplexity_model=perplexity_model,
                perplexity_tokenizer=perplexity_tokenizer, similarity_model=similarity_model
            )
            report_path = length_specific_dir / "summary_report.txt"
            metrics = parse_summary_report(report_path)
            if all(val is not None for val in metrics.values()):
                plot_data['gen_lengths'].append(length)
                plot_data['tpr'].append(metrics['tpr'])
                plot_data['fpr'].append(metrics['fpr'])
                plot_data['perplexity_wm'].append(metrics['perplexity_wm'])
                plot_data['perplexity_no_wm'].append(metrics['perplexity_no_wm'])
        
        print("\n----- All evaluations complete! -----")

        if not plot_data['gen_lengths']:
            print("Could not find any valid results to plot.")
            return

        print("Generating plots...")
        plt.style.use('seaborn-v0_8-whitegrid')

        fig1, ax1 = plt.subplots()
        ax1.plot(plot_data['gen_lengths'], plot_data['tpr'], marker='o', linestyle='-', label='TPR (Detection Rate)')
        ax1.plot(plot_data['gen_lengths'], plot_data['fpr'], marker='s', linestyle='--', label='FPR (False Positives)')
        ax1.set_xlabel("Generation Length (Tokens)"); ax1.set_ylabel("Rate")
        ax1.set_title("Watermark Detection Performance vs. Generation Length"); ax1.legend(); ax1.set_ylim(0, 1.05)
        ax1.set_xticks(plot_data['gen_lengths'])
        perf_plot_path = run_dir / "detection_performance.png"
        fig1.savefig(perf_plot_path, dpi=300)
        print(f"Saved detection performance plot to: {perf_plot_path}")
        fig2, ax2 = plt.subplots()
        ax2.plot(plot_data['gen_lengths'], plot_data['perplexity_wm'], marker='o', linestyle='-', label='Perplexity (Watermarked)')
        ax2.plot(plot_data['gen_lengths'], plot_data['perplexity_no_wm'], marker='s', linestyle='--', label='Perplexity (Original)')
        ax2.set_xlabel("Generation Length (Tokens)"); ax2.set_ylabel("Perplexity")
        ax2.set_title("Text Quality (Perplexity) vs. Generation Length"); ax2.legend()
        ax2.set_xticks(plot_data['gen_lengths'])
        quality_plot_path = run_dir / "text_quality_impact.png"
        fig2.savefig(quality_plot_path, dpi=300)
        print(f"Saved text quality plot to: {quality_plot_path}")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
