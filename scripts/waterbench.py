import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import binom
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

PROJECT_ROOT = Path("/scratch/efeldma5/waterdiffusion/waterdiffusion")
WATERMARK_PATH = PROJECT_ROOT / "watermarks" / "1_28_unmasking_algo.py"
WATERBENCH_PATH = PROJECT_ROOT / "zurich" / "data" / "WaterBench" / "2-1_longform_qa.jsonl"
LLADA_CONFIG_PATH = PROJECT_ROOT / "configs" / "llada_8b_instruct.yaml"

sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util
_spec = importlib.util.spec_from_file_location("remask_watermark", str(WATERMARK_PATH))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SimpleRemaskWatermark = _mod.SimpleRemaskWatermark

from models.llada import LLaDAGenerator



@dataclass
class RemaskEvalConfig:
    gen_length: int = 400
    gen_steps: int = 400
    block_length: int = 40

    secret_key: str = "xooks"
    delta: float = 2.0
    context_window: List[int] = field(default_factory=lambda: [-2, -1, 1, 2])
    vocab_size: int = 32000        

    num_samples: int = 100
    p_value_threshold: float = 0.01
    batch_size: int = 1              
    skip_if_exists: bool = True
    tqdm: bool = True

    waterbench_path: str = str(WATERBENCH_PATH)
    save_path: str = str(
        PROJECT_ROOT / "runs" / f"remask_eval_{datetime.utcnow().strftime('%m_%d_%H_%M')}.jsonl"
    )
    ppl_model_name: str = "gpt2"

    system_prompt: str = (
        "You are a helpful assistant, please answer the following question within 300 words:"
    )



def compute_ppl(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    text: str,
    max_length: int = 1024,
) -> float:
    """GPT-2 perplexity for a single string."""
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encodings.input_ids
    if input_ids.shape[1] < 2:
        return float("inf")
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss
    return float(torch.exp(loss).item())


def compute_seq_rep_n(text: str, tokenizer: Any, n: int = 2) -> float:
    """Fraction of repeated n-grams — high value → repetitive output."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < n:
        return 0.0
    ngrams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        F.cosine_similarity(
            torch.tensor(a).unsqueeze(0),
            torch.tensor(b).unsqueeze(0),
        ).item()
    )



def load_waterbench(path: str, num_samples: int) -> List[Dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            if len(records) >= num_samples:
                break
            ex = json.loads(line)
            records.append(ex)
    return records


def build_prompt(example: Dict, system_prompt: str) -> Optional[str]:
    context = example.get("context", "")
    question = example.get("input", "")
    prompt = f"{system_prompt}\n{context}\n{question}".strip()
    if not (50 <= len(prompt) <= 1024):
        return None
    return prompt



class RemaskEvaluator:
    def __init__(self, config: RemaskEvalConfig):
        self.config = config
        self.watermark = SimpleRemaskWatermark(
            key=config.secret_key,
            delta=config.delta,
            context_window=config.context_window,
            vocab_size=config.vocab_size,
        )

        print("Loading LLaDA generator...")
        self.generator = LLaDAGenerator(
            config=str(LLADA_CONFIG_PATH),
            watermark=self.watermark,
            device_map="auto",
        )
        print(f"  Model loaded on: {self.generator.device}")

        print("Loading PPL model (GPT-2)...")
        self.ppl_model = GPT2LMHeadModel.from_pretrained(config.ppl_model_name)
        self.ppl_tokenizer = GPT2Tokenizer.from_pretrained(config.ppl_model_name)
        self.ppl_model.eval()

        print("Loading similarity model...")
        self.sim_model = SentenceTransformer("all-MiniLM-L6-v2")

        os.makedirs(os.path.dirname(config.save_path), exist_ok=True)


    def _count_existing(self, prompt: str) -> int:
        """Return 1 if this prompt already has a result row, else 0."""
        if not self.config.skip_if_exists:
            return 0
        df = self._load_results()
        if df.empty or "prompt" not in df.columns:
            return 0
        return int((df["prompt"] == prompt).sum())

    def _load_results(self) -> pd.DataFrame:
        if not os.path.exists(self.config.save_path):
            return pd.DataFrame()
        return pd.read_json(self.config.save_path, lines=True)


    def _generate_valid(self, prompt: str, watermark_mode: Optional[str]) -> str:
        cfg = self.config
        if watermark_mode == "wm":
            self.generator.watermark = self.watermark
        else:
            self.generator.watermark = None

        for attempt in range(10):
            text = self.generator.generate_text(
                prompt=prompt,
                gen_length=cfg.gen_length,
                steps=cfg.gen_steps,
                block_length=cfg.block_length,
            )
            too_short = len(text) < 0.75 * cfg.gen_length
            rep = compute_seq_rep_n(text, self.generator.tokenizer, n=2)
            if not too_short and rep <= 0.5:
                return text
            print(f"  Re-generating (attempt {attempt + 1}): too_short={too_short}, rep={rep:.3f}")

        return text


    def _detect(self, text: str) -> Dict:
        return self.watermark.detect(
            text=text,
            tokenizer=self.generator.tokenizer,
            p_value_threshold=self.config.p_value_threshold,
        )


    def _evaluate_sample(self, prompt: str) -> Dict:
        cfg = self.config

        t0 = time.time()
        text_wm = self._generate_valid(prompt, watermark_mode="wm")
        time_wm = time.time() - t0

        t0 = time.time()
        text_no_wm = self._generate_valid(prompt, watermark_mode=None)
        time_no_wm = time.time() - t0

        ppl_wm = compute_ppl(self.ppl_model, self.ppl_tokenizer, text_wm)
        ppl_no_wm = compute_ppl(self.ppl_model, self.ppl_tokenizer, text_no_wm)

        rep_wm = compute_seq_rep_n(text_wm, self.generator.tokenizer, n=2)
        rep_no_wm = compute_seq_rep_n(text_no_wm, self.generator.tokenizer, n=2)

        embeddings = self.sim_model.encode([text_wm, text_no_wm])
        cos_sim = cosine_similarity(embeddings[0], embeddings[1])

        detection_wm = self._detect(text_wm)
        detection_no_wm = self._detect(text_no_wm)

        return {
            "prompt": prompt,
            "completion_wm": text_wm,
            "completion_no_wm": text_no_wm,
            "gen_time_wm": time_wm,
            "gen_time_no_wm": time_no_wm,
            "ppl_wm": ppl_wm,
            "ppl_no_wm": ppl_no_wm,
            "seq_rep_2_wm": rep_wm,
            "seq_rep_2_no_wm": rep_no_wm,
            "cosine_similarity_wm_vs_no_wm": cos_sim,
            "det_wm_is_watermarked": detection_wm["is_watermarked"],
            "det_wm_score": detection_wm["score"],
            "det_wm_satisfied_pairs": detection_wm["satisfied_pairs"],
            "det_wm_total_pairs": detection_wm["total_pairs_checked"],
            "det_wm_satisfaction_rate": detection_wm["satisfaction_rate"],
            "det_no_wm_is_watermarked": detection_no_wm["is_watermarked"],
            "det_no_wm_score": detection_no_wm["score"],
            "det_no_wm_satisfied_pairs": detection_no_wm["satisfied_pairs"],
            "det_no_wm_total_pairs": detection_no_wm["total_pairs_checked"],
            "det_no_wm_satisfaction_rate": detection_no_wm["satisfaction_rate"],
            "watermark_key": cfg.secret_key,
            "delta": cfg.delta,
            "context_window": cfg.context_window,
            "gen_length": cfg.gen_length,
            "gen_steps": cfg.gen_steps,
            "block_length": cfg.block_length,
            "p_value_threshold": cfg.p_value_threshold,
        }


    def evaluate(self):
        cfg = self.config
        examples = load_waterbench(cfg.waterbench_path, cfg.num_samples * 3)

        processed = 0
        skipped = 0

        with open(cfg.save_path, "a") as out_f:
            for ex in tqdm(examples, desc="WaterBench samples", disable=not cfg.tqdm):
                if processed >= cfg.num_samples:
                    break

                prompt = build_prompt(ex, cfg.system_prompt)
                if prompt is None:
                    continue

                if self._count_existing(prompt) > 0:
                    skipped += 1
                    processed += 1
                    continue

                try:
                    result = self._evaluate_sample(prompt)
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                    processed += 1

                    if processed % 10 == 0:
                        print(f"  ... {processed}/{cfg.num_samples} samples done")

                except Exception as e:
                    print(f"  Error on sample {processed}: {e}")
                    continue

        print(f"\nDone. {processed} samples processed ({skipped} skipped). Results: {cfg.save_path}")


    def summarize(self):
        df = self._load_results()
        if df.empty:
            print("No results to summarize.")
            return

        n = len(df)
        tpr = df["det_wm_is_watermarked"].mean()
        fpr = df["det_no_wm_is_watermarked"].mean()

        report_lines = [
            "=" * 60,
            "REMASK WATERMARK EVALUATION SUMMARY",
            f"Samples evaluated       : {n}",
            f"Generation length       : {self.config.gen_length}",
            f"Delta                   : {self.config.delta}",
            f"Context window          : {self.config.context_window}",
            f"P-value threshold       : {self.config.p_value_threshold}",
            "-" * 60,
            "Detection",
            f"  True Positive Rate (TPR/Recall) : {tpr:.4f}",
            f"  False Positive Rate (FPR)       : {fpr:.4f}",
            f"  Avg score (WM)                  : {df['det_wm_score'].mean():.4f}",
            f"  Avg score (No WM)               : {df['det_no_wm_score'].mean():.4f}",
            f"  Avg satisfaction rate (WM)      : {df['det_wm_satisfaction_rate'].mean():.4f}",
            f"  Avg satisfaction rate (No WM)   : {df['det_no_wm_satisfaction_rate'].mean():.4f}",
            "-" * 60,
            "Text Quality",
            f"  Perplexity (WM)   : {df['ppl_wm'].mean():.2f}",
            f"  Perplexity (No WM): {df['ppl_no_wm'].mean():.2f}",
            f"  Seq-Rep-2 (WM)    : {df['seq_rep_2_wm'].mean():.4f}",
            f"  Seq-Rep-2 (No WM) : {df['seq_rep_2_no_wm'].mean():.4f}",
            f"  Cosine sim WM vs No-WM: {df['cosine_similarity_wm_vs_no_wm'].mean():.4f}",
            "-" * 60,
            "Timing",
            f"  Avg gen time (WM)   : {df['gen_time_wm'].mean():.1f}s",
            f"  Avg gen time (No WM): {df['gen_time_no_wm'].mean():.1f}s",
            "=" * 60,
        ]

        report = "\n".join(report_lines)
        print(report)

        report_path = Path(self.config.save_path).with_suffix(".summary.txt")
        with open(report_path, "w") as f:
            f.write(report + "\n")
        print(f"Summary saved to: {report_path}")

        return {
            "tpr": tpr,
            "fpr": fpr,
            "ppl_wm": df["ppl_wm"].mean(),
            "ppl_no_wm": df["ppl_no_wm"].mean(),
        }


if __name__ == "__main__":
    import random

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    config = RemaskEvalConfig(
        num_samples=100,
        gen_length=400,
        gen_steps=400,
        block_length=40,
        delta=2.0,
        secret_key="xooks",
        p_value_threshold=0.01,
        skip_if_exists=True,
    )

    evaluator = RemaskEvaluator(config)
    evaluator.evaluate()
    evaluator.summarize()