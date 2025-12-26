import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer
import torch
import yaml
import importlib.util
from models.llada import LLaDAGenerator
from datetime import datetime

INPUT_FILE = "runs/12_20_04_08_llada_8b_eli5_wm_multidelta/results.jsonl"
OUTPUT_DIR = "runs"
RUN_NAME = "robustness_eval"
WATERMARK_KEY = "xooks"
LLADA_CONFIG = "configs/llada_8b_instruct.yaml"
P_VALUE_THRESHOLD = 0.05
ATTACK_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]
DO_SUBSTITUTION = True
DO_CA_SUBSTITUTION = True
BERT_MODEL_NAME = 'google-bert/bert-base-uncased'

def import_from_path(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

class WordSubstitution:
    def __init__(self, ratio=0.1):
        self.ratio = ratio
        try:
            nltk.data.find('corpora/wordnet')
        except nltk.downloader.DownloadError:
            nltk.download('wordnet')
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            nltk.download('punkt')

    def edit(self, text):
        words = nltk.word_tokenize(text)
        n_words_to_substitute = int(len(words) * self.ratio)
        if n_words_to_substitute == 0: return text
        
        indices_to_substitute = np.random.choice(len(words), n_words_to_substitute, replace=False)

        for i in indices_to_substitute:
            synonyms = get_synonyms(words[i])
            if synonyms:
                words[i] = np.random.choice(synonyms)
        
        return " ".join(words)

class ContextAwareWordSubstitution:
    def __init__(self, ratio=0.1, model_name=BERT_MODEL_NAME, local_files_only=True):
        self.ratio = ratio
        self.tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = BertForMaskedLM.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.eval()

    def edit(self, text):
        words = text.split()
        if not words:
            return ""
            
        n_words_to_substitute = int(len(words) * self.ratio)
        if n_words_to_substitute == 0:
            return text

        indices_to_substitute = np.random.choice(len(words), n_words_to_substitute, replace=False)

        for i in indices_to_substitute:
            original_word = words[i]
            words[i] = self.tokenizer.mask_token
            masked_text = " ".join(words)

            inputs = self.tokenizer(masked_text, return_tensors='pt', truncation=True, max_length=512)
            masked_indices = torch.where(inputs['input_ids'][0] == self.tokenizer.mask_token_id)[0]
            
            if masked_indices.nelement() > 0:
                with torch.no_grad():
                    outputs = self.model(**inputs)
                predictions = outputs.logits
                masked_index = masked_indices[0] 
                predicted_index = torch.argmax(predictions[0, masked_index]).item()
                predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])[0]
                words[i] = predicted_token
            else:
                words[i] = original_word

        return " ".join(words)

def save_run_desc(run_dir: Path):
    config_data = {
        "TIMESTAMP": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "RUN_NAME": RUN_NAME,
        "INPUT_FILE": INPUT_FILE,
        "WATERMARK_KEY": WATERMARK_KEY,
        "LLADA_CONFIG": LLADA_CONFIG,
        "ATTACK_RATIOS": ATTACK_RATIOS,
        "DO_SUBSTITUTION": DO_SUBSTITUTION,
        "DO_CA_SUBSTITUTION": DO_CA_SUBSTITUTION,
        "P_VALUE_THRESHOLD": P_VALUE_THRESHOLD
    }
    desc_path = run_dir / "desc.txt"
    with open(desc_path, 'w') as f:
        f.write("RUN CONFIG \n")
        for key, value in config_data.items():
            f.write(f"{key}: {value}\n")
    print(f"Run config saved to {desc_path}")

def main():
    parser = argparse.ArgumentParser(description="Run robustness evaluation on watermarked text.")
    parser.add_argument('--online', action='store_true', help="Enable online mode to download models if not cached.")
    args = parser.parse_args()
    local_files_only = not args.online

    project_root = Path(__file__).parent.parent
    base_output_dir = project_root / OUTPUT_DIR
    date_str = datetime.utcnow().strftime("%m_%d_%H_%M")
    run_dir = base_output_dir / f"{date_str}_{RUN_NAME}"
    run_dir.mkdir(parents=True, exist_ok=True)

    output_path = run_dir / "results.jsonl"
    save_run_desc(run_dir)

    try:
        if DO_CA_SUBSTITUTION:
            print(f"Checking {BERT_MODEL_NAME}...")
            BertTokenizer.from_pretrained(BERT_MODEL_NAME, local_files_only=local_files_only)
            BertForMaskedLM.from_pretrained(BERT_MODEL_NAME, local_files_only=local_files_only)
        
        with open(project_root / LLADA_CONFIG, 'r') as f:
            config_data = yaml.safe_load(f)
            model_path = config_data.get('pretrained_model_name_or_path')
            if model_path:
                print(f"Checking tokenizer for LLaDA ({model_path})...")
                AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
    except Exception as e:
        print(f"Warning during cache check: {e}")
        if local_files_only:
            print("Continuing, but script may fail if files are missing and offline.")
        else:
            print("Attempting to download missing files...")

    print(f"Loading tokenizer from config: {LLADA_CONFIG}")
    generator = LLaDAGenerator(config=str(project_root / LLADA_CONFIG), local_files_only=local_files_only)
    tokenizer = generator.tokenizer

    print("Loading watermark...")
    watermark_module = import_from_path('watermarks/11_02_init_unmasking_watermark.py', 'watermark')
    SimpleRemaskWatermark = watermark_module.SimpleRemaskWatermark
    watermark = SimpleRemaskWatermark(key=WATERMARK_KEY)

    print(f"Reading input file: {INPUT_FILE}")
    df = pd.read_json(project_root / INPUT_FILE, lines=True)
    
    results = []
    attacks = {}
    if DO_SUBSTITUTION:
        attacks['substitution'] = WordSubstitution
    if DO_CA_SUBSTITUTION:
        attacks['ca_substitution'] = ContextAwareWordSubstitution

    for attack_name, AttackClass in attacks.items():
        for ratio in ATTACK_RATIOS:
            print(f"Running {attack_name} with ratio {ratio}...")
            if attack_name == 'ca_substitution':
                attack = AttackClass(ratio=ratio, local_files_only=local_files_only)
            else:
                attack = AttackClass(ratio=ratio)
            
            for i, row in df.iterrows():
                generated_text_wm = row['generated_text_wm']
                
                original_detection = watermark.detect(generated_text_wm, tokenizer)
                original_p_value = original_detection['score']

                edited_completion = attack.edit(generated_text_wm)
                edited_detection = watermark.detect(edited_completion, tokenizer)
                edited_p_value = edited_detection['score']

                result_row = {
                    'original_completion': generated_text_wm,
                    'edited_completion': edited_completion,
                    'attack': attack_name,
                    'ratio': ratio,
                    'original_p_value': original_p_value,
                    'edited_p_value': edited_p_value
                }
                for col in df.columns:
                    if col not in ['generated_text_wm']:
                         result_row[col] = row[col]

                results.append(result_row)

    results_df = pd.DataFrame(results)
    results_df.to_json(output_path, orient='records', lines=True)
    print(f"Results saved to {output_path}")

    print("\nAnalyzing results...")
    summary_path = run_dir / "summary_report.txt"
    analyze_robustness_results(output_path, summary_path, P_VALUE_THRESHOLD)


def analyze_robustness_results(jsonl_path: Path, summary_path: Path, threshold: float):
    df = pd.read_json(jsonl_path, lines=True)
    
    summary_data = []

    for (attack, ratio), group in df.groupby(['attack', 'ratio']):
        total = len(group)
        successful_attacks = group[(group['original_p_value'] <= threshold) & (group['edited_p_value'] > threshold)]
        success_rate = len(successful_attacks) / total if total > 0 else 0
        
        summary_data.append({
            'attack': attack,
            'ratio': ratio,
            'total_samples': total,
            'successful_attacks': len(successful_attacks),
            'success_rate': success_rate
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    report_lines = ["--- Robustness Evaluation Summary ---"]
    report_lines.append(f"P-value threshold for detection: {threshold}\n")
    report_lines.append(summary_df.to_string())
    
    report_content = "\n".join(report_lines)
    print(report_content)
    with open(summary_path, 'w') as f:
        f.write(report_content)
    print(f"\nSummary report saved to {summary_path}")


if __name__ == "__main__":
    main()