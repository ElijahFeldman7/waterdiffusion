import argparse
import importlib.util
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer
from tqdm.auto import tqdm

PROJECT_ROOT = Path("/scratch/efeldma5/waterdiffusion/waterdiffusion")
DEFAULT_WATERMARK_PATH = PROJECT_ROOT / "watermarks" / "proposal_1_ckas.py"
WATERBENCH_DIR = PROJECT_ROOT / "zurich" / "data" / "WaterBench"
LLADA_CONFIG_PATH = PROJECT_ROOT / "configs" / "llada_8b_instruct.yaml"

sys.path.insert(0, str(PROJECT_ROOT))
from models.llada import LLaDAGenerator


@dataclass
class EvalConfig:
    run_dir: str
    watermark_path: str
    secret_key: str = "xooks"
    delta: float = 4.0
    context_window: int = 2
    
    num_samples: int = 300
    gen_length: int = 300
    gen_steps: int = 300
    block_length: int = 25
    
    p_value_threshold: float = 0.01
    skip_if_exists: bool = True
    
    shard: int = 0
    num_shards: int = 1
    
    run_attacks: bool = True
    skip_truncation: bool = False
    skip_deletion: bool = False
    skip_insertion: bool = False
    skip_substitution: bool = False
    skip_paraphrase: bool = False
    
    system_prompt: str = "You are a helpful assistant, please answer the following question within 300 words:"

    @property
    def shard_save_path(self) -> str:
        return str(Path(self.run_dir) / f"shard_{self.shard}_of_{self.num_shards}.jsonl")
    @property
    def merged_save_path(self) -> str:
        return str(Path(self.run_dir) / "results_merged.jsonl")
    @property
    def summary_save_path(self) -> str:
        return str(Path(self.run_dir) / "summary_report.txt")

def compute_seq_rep_n(text: str, tokenizer: Any, n: int = 2) -> float:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < n: return 0.0
    ngrams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)

def load_waterbench_all(waterbench_dir: str, max_scan_per_file: int = 5000) -> List[Dict]:
    jsonl_files = sorted(Path(waterbench_dir).glob("*.jsonl"))
    if not jsonl_files: raise FileNotFoundError(f"No .jsonl files found in {waterbench_dir}")
    per_file = []
    for fpath in jsonl_files:
        records = []
        with open(fpath, "r") as f:
            for i, line in enumerate(f):
                if i >= max_scan_per_file: break
                ex = json.loads(line)
                ex["_source_file"] = fpath.name
                records.append(ex)
        per_file.append(records)
    interleaved = []
    max_len = max(len(r) for r in per_file)
    for i in range(max_len):
        for records in per_file:
            if i < len(records): interleaved.append(records[i])
    return interleaved

def build_prompt(example: Dict, system_prompt: str) -> Optional[str]:
    prompt = f"{system_prompt}\n{example.get('context', '')}\n{example.get('input', '')}".strip()
    return prompt if 50 <= len(prompt) <= 1024 else None

def get_shard_pairs(all_pairs: List[Tuple[str, str]], num_samples: int, shard: int, num_shards: int) -> List[Tuple[str, str]]:
    pool = all_pairs[:num_samples]
    chunk_size = len(pool) // num_shards
    start = shard * chunk_size
    end = start + chunk_size if shard < num_shards - 1 else len(pool)
    return pool[start:end]

class DipperParaphraser:
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl"):
        print(f"Loading DIPPER model ({model})... ")
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(
            model, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.model.eval()

    @torch.no_grad()
    def paraphrase(self, text: str, lex_div: int, order_div: int) -> str:
        prompt = f"lexical = {lex_div}, order = {order_div} {text}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_length=512, do_sample=True, top_p=0.75, top_k=None)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class TextAttacker:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.crop_lengths = [25, 50, 100, 200]
        self.fillers = ["actually", "very", "furthermore", "and", "so", "just", "well", "but"]
        self.dipper = DipperParaphraser() if not config.skip_paraphrase and config.run_attacks else None

    def run_all_structural(self, text: str) -> Dict[str, str]:
        attacks = {}
        words = text.split()
        
        if not self.config.skip_truncation:
            for L in self.crop_lengths:
                if len(words) <= L:
                    attacks[f"trunc_{L}"] = text
                else:
                    start = random.randint(0, len(words) - L)
                    attacks[f"trunc_{L}"] = " ".join(words[start : start + L])

        if not self.config.skip_deletion:
            for r in self.ratios:
                kept = [w for w in words if random.random() > r]
                attacks[f"del_{r}"] = " ".join(kept) if kept else text

        if not self.config.skip_insertion:
            for r in self.ratios:
                out = []
                for w in words:
                    out.append(w)
                    if random.random() < r: out.append(random.choice(self.fillers))
                attacks[f"ins_{r}"] = " ".join(out)

        if not self.config.skip_substitution:
            vocab = list(set(words))
            for r in self.ratios:
                out = [random.choice(vocab) if random.random() < r and len(w) > 3 else w for w in words]
                attacks[f"sub_{r}"] = " ".join(out)

        if not self.config.skip_paraphrase and self.dipper:
            try:
                attacks["para_L20_O0"] = self.dipper.paraphrase(text, 20, 0)
                attacks["para_L40_O20"] = self.dipper.paraphrase(text, 40, 20)
                attacks["para_L60_O40"] = self.dipper.paraphrase(text, 60, 40)
            except Exception as e:
                print(f"DIPPER Paraphrase Failed: {e}")

        return attacks

class WatermarkEvaluator:
    def __init__(self, config: EvalConfig):
        self.config = config
        Path(config.run_dir).mkdir(parents=True, exist_ok=True)
        
        spec = importlib.util.spec_from_file_location("wm", config.watermark_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        wm_class = next(obj for name, obj in vars(mod).items() if isinstance(obj, type) and name.endswith("Watermark"))
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.generator = LLaDAGenerator(config=str(LLADA_CONFIG_PATH), watermark=None, device_map=self.device)
        self.watermark = wm_class(key=config.secret_key, delta=config.delta, context_window=config.context_window, vocab_size=len(self.generator.tokenizer), device=self.device)
        self.ppl_model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device).eval()
        self.ppl_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.sim_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.attacker = TextAttacker(config)

    def _generate_valid(self, prompt: str, use_wm: bool) -> Tuple[str, float, float]:
        self.generator.watermark = self.watermark if use_wm else None
        
        for attempt in range(10):
            t0 = time.time()
            text = self.generator.generate_text(prompt=prompt, gen_length=self.config.gen_length, steps=self.config.gen_steps, block_length=self.config.block_length)
            t_total = time.time() - t0
            
            too_short = len(text) < 0.75 * self.config.gen_length
            rep = compute_seq_rep_n(text, self.generator.tokenizer, n=2)
            
            if not too_short and rep <= 0.5:
                tok_sec = len(self.generator.tokenizer.encode(text)) / max(t_total, 0.001)
                return text, t_total, tok_sec
            print(f"  Re-generating (attempt {attempt + 1}): too_short={too_short}, rep={rep:.3f}")
        
        return text, t_total, len(self.generator.tokenizer.encode(text)) / max(t_total, 0.001)

    def evaluate_sample(self, prompt: str, source_file: str) -> Dict:
        txt_wm, time_wm, tps_wm = self._generate_valid(prompt, True)
        txt_nw, time_nw, tps_nw = self._generate_valid(prompt, False)
        
        enc_wm = self.ppl_tokenizer(txt_wm, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        enc_nw = self.ppl_tokenizer(txt_nw, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            ppl_wm = float(torch.exp(self.ppl_model(enc_wm.input_ids, labels=enc_wm.input_ids).loss).item()) if enc_wm.input_ids.shape[1] > 1 else float('inf')
            ppl_nw = float(torch.exp(self.ppl_model(enc_nw.input_ids, labels=enc_nw.input_ids).loss).item()) if enc_nw.input_ids.shape[1] > 1 else float('inf')
            
        sim = float(F.cosine_similarity(torch.tensor(self.sim_model.encode(txt_wm)).unsqueeze(0), torch.tensor(self.sim_model.encode(txt_nw)).unsqueeze(0)).item())
        rep_wm = compute_seq_rep_n(txt_wm, self.generator.tokenizer)
        rep_nw = compute_seq_rep_n(txt_nw, self.generator.tokenizer)

        det_wm = self.watermark.detect(txt_wm, self.generator.tokenizer, self.config.p_value_threshold)
        det_nw = self.watermark.detect(txt_nw, self.generator.tokenizer, self.config.p_value_threshold)

        res = {
            "prompt": prompt, "source_file": source_file, "shard": self.config.shard,
            "completion_wm": txt_wm, "completion_nw": txt_nw,
            "time_wm": time_wm, "time_nw": time_nw,
            "tok_sec_wm": tps_wm, "tok_sec_nw": tps_nw,
            "ppl_wm": ppl_wm, "ppl_nw": ppl_nw,
            "seq_rep_2_wm": rep_wm, "seq_rep_2_no_wm": rep_nw, 
            "similarity": sim,
            "base_det_wm": det_wm["is_watermarked"], "base_det_nw": det_nw["is_watermarked"],
        }

        if self.config.run_attacks:
            res["attacks"] = {}
            for atk_name, atk_text in self.attacker.run_all_structural(txt_wm).items():
                res["attacks"][f"wm_{atk_name}"] = self.watermark.detect(atk_text, self.generator.tokenizer)["is_watermarked"]
            for atk_name, atk_text in self.attacker.run_all_structural(txt_nw).items():
                res["attacks"][f"nw_{atk_name}"] = self.watermark.detect(atk_text, self.generator.tokenizer)["is_watermarked"]

        return res

    def summarize(self, results_path: str):
        df = pd.read_json(results_path, lines=True)
        if df.empty: return
        n = len(df)
        tpr = df["base_det_wm"].mean()
        fpr = df["base_det_nw"].mean()
        
        lines = [
            "="*60, "WATERMARK EVALUATION & ROBUSTNESS REPORT", "="*60,
            f"Config: Delta={self.config.delta}, Key={self.config.secret_key}, Watermark={Path(self.config.watermark_path).name}",
            f"Total Samples Processed: {n}",
            "\n--- 1. GENERATION OVERHEAD ---",
            f"Avg Time (WM) : {df['time_wm'].mean():.2f} sec | Tokens/sec: {df['tok_sec_wm'].mean():.2f}",
            f"Avg Time (NW) : {df['time_nw'].mean():.2f} sec | Tokens/sec: {df['tok_sec_nw'].mean():.2f}",
            f"Time Penalty  : +{(df['time_wm'].mean() / df['time_nw'].mean() - 1)*100:.1f}%",
            "\n--- 2. QUALITY DEGRADATION ---",
            f"Perplexity (WM): {df['ppl_wm'].mean():.2f} | (NW): {df['ppl_nw'].mean():.2f}",
            f"Repetition (WM): {df['seq_rep_2_wm'].mean():.3f} | (NW): {df['seq_rep_2_no_wm'].mean():.3f}",
            f"Cosine Sim     : {df['similarity'].mean():.3f}",
            "\n--- 3. BASELINE DETECTION ---",
            f"TPR: {tpr:.3f} | FPR: {fpr:.3f} | FNR: {1-tpr:.3f} | TNR: {1-fpr:.3f}",
        ]

        if self.config.run_attacks and "attacks" in df.columns:
            lines.append("\n--- 4. ROBUSTNESS PROFILE (ATTACKS TPR / FPR) ---")
            attack_keys = [k.replace('wm_', '') for k in df.iloc[0]["attacks"].keys() if k.startswith('wm_')]
            
            for atk in sorted(attack_keys):
                atk_tpr = df["attacks"].apply(lambda x: x.get(f"wm_{atk}", False)).mean()
                atk_fpr = df["attacks"].apply(lambda x: x.get(f"nw_{atk}", False)).mean()
                lines.append(f"Attack [{atk.ljust(12)}]: TPR = {atk_tpr:.3f} | FPR = {atk_fpr:.3f}")

        lines.append("\n" + "="*60 + "\n--- GENERATION SAMPLES ---")
        for i, row in df.iterrows():
            lines.append(f"\n[Sample {i+1}] Prompt: {row['prompt']}")
            lines.append(f"[Watermarked] {row['completion_wm']}")
            lines.append(f"[Original]    {row['completion_nw']}")
            lines.append("-" * 40)

        report = "\n".join(lines)
        print(f"Summary generated ({n} samples).")
        with open(self.config.summary_save_path, "w") as f: f.write(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--num_shards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    parser.add_argument("--run_dir", type=str, default=str(PROJECT_ROOT / f"runs/eval_{datetime.utcnow().strftime('%m%d_%H%M')}"))
    parser.add_argument("--merge_only", action="store_true")
    
    parser.add_argument("--watermark_path", type=str, default=str(DEFAULT_WATERMARK_PATH))
    parser.add_argument("--delta", type=float, default=4.0)
    parser.add_argument("--num_samples", type=int, default=600)
    
    parser.add_argument("--no_attacks", action="store_true", help="Skip all robustness attacks")
    parser.add_argument("--skip_truncation", action="store_true")
    parser.add_argument("--skip_deletion", action="store_true")
    parser.add_argument("--skip_insertion", action="store_true")
    parser.add_argument("--skip_substitution", action="store_true")
    parser.add_argument("--skip_paraphrase", action="store_true", help="Skips loading 11B DIPPER model")
    
    args = parser.parse_args()

    SEED = 42 + args.shard
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    cfg = EvalConfig(
        run_dir=args.run_dir,
        watermark_path=args.watermark_path,
        delta=args.delta,
        num_samples=args.num_samples,
        shard=args.shard,
        num_shards=args.num_shards,
        run_attacks=not args.no_attacks,
        skip_truncation=args.skip_truncation,
        skip_deletion=args.skip_deletion,
        skip_insertion=args.skip_insertion,
        skip_substitution=args.skip_substitution,
        skip_paraphrase=args.skip_paraphrase
    )

    if args.merge_only:
        shard_files = sorted(Path(cfg.run_dir).glob("shard_*_of_*.jsonl"))
        with open(cfg.merged_save_path, "w") as out_f:
            for sf in shard_files:
                with open(sf, "r") as in_f:
                    for line in in_f: out_f.write(line)
        print(f"Merged output saved to {cfg.merged_save_path}")
        WatermarkEvaluator(cfg).summarize(cfg.merged_save_path)
    
    else:
        all_examples = load_waterbench_all(WATERBENCH_DIR)
        all_pairs = [(build_prompt(ex, cfg.system_prompt), ex.get("_source_file", "unknown")) for ex in all_examples if build_prompt(ex, cfg.system_prompt)]
        shard_pairs = get_shard_pairs(all_pairs, cfg.num_samples, cfg.shard, cfg.num_shards)

        existing = set()
        if os.path.exists(cfg.shard_save_path):
            with open(cfg.shard_save_path, "r") as f:
                for line in f:
                    try: existing.add(json.loads(line)["prompt"])
                    except: pass

        todo = [(p, s) for p, s in shard_pairs if p not in existing]
        print(f"[Shard {cfg.shard}] Running {len(todo)} samples.")

        evaluator = WatermarkEvaluator(cfg)
        with open(cfg.shard_save_path, "a") as out_f:
            for prompt, source in tqdm(todo, desc=f"Shard {cfg.shard}"):
                try:
                    res = evaluator.evaluate_sample(prompt, source)
                    out_f.write(json.dumps(res) + "\n")
                    out_f.flush()
                except Exception as e:
                    print(f"Error on prompt: {e}")
                    traceback.print_exc()

        if cfg.num_shards == 1:
            evaluator.summarize(cfg.shard_save_path)