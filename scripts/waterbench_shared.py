import argparse
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WATERMARK_PATH = PROJECT_ROOT / "watermarks" / "3_16_hash.py"
WATERBENCH_DIR = PROJECT_ROOT / "zurich" / "data" / "WaterBench"
LLADA_CONFIG_PATH = PROJECT_ROOT / "configs" / "llada_8b_instruct.yaml"

sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location("remask_watermark", str(WATERMARK_PATH))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SimpleRemaskWatermark = _mod.SimpleRemaskWatermark

from models.llada import LLaDAGenerator


@dataclass
class RemaskEvalConfig:
    gen_length: int = 300
    gen_steps: int = 100
    block_length: int = 25

    secret_key: str = "xooks"
    delta: float = 4.0
    context_window: List[int] = field(default_factory=lambda: [-2, -1, 1, 2])

    num_samples: int = 600
    p_value_threshold: float = 0.01
    skip_if_exists: bool = True
    tqdm: bool = True

    shard: int = 0
    num_shards: int = 1

    run_dir: str = str(PROJECT_ROOT / "runs" / f"remask_eval_{datetime.utcnow().strftime('%m_%d_%H_%M')}")
    waterbench_dir: str = str(WATERBENCH_DIR)
    ppl_model_name: str = "gpt2"
    system_prompt: str = (
        "You are a helpful assistant, please answer the following question within 300 words:"
    )

    @property
    def shard_save_path(self) -> str:
        return str(Path(self.run_dir) / f"shard_{self.shard}_of_{self.num_shards}.jsonl")

    @property
    def merged_save_path(self) -> str:
        return str(Path(self.run_dir) / "results_merged.jsonl")

    @property
    def summary_save_path(self) -> str:
        return str(Path(self.run_dir) / "summary.txt")


def compute_ppl(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    text: str,
    max_length: int = 1024,
) -> float:
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids
    if input_ids.shape[1] < 2:
        return float("inf")
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss
    return float(torch.exp(loss).item())


def compute_seq_rep_n(text: str, tokenizer: Any, n: int = 2) -> float:
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


def load_waterbench_all(waterbench_dir: str, max_scan_per_file: int = 5000) -> List[Dict]:
    jsonl_files = sorted(Path(waterbench_dir).glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {waterbench_dir}")

    print(f"Found {len(jsonl_files)} WaterBench files:")
    per_file: List[List[Dict]] = []
    for fpath in jsonl_files:
        records = []
        with open(fpath, "r") as f:
            for i, line in enumerate(f):
                if i >= max_scan_per_file:
                    break
                ex = json.loads(line)
                ex["_source_file"] = fpath.name
                records.append(ex)
        per_file.append(records)
        print(f"  {fpath.name}: {len(records)} records")

    interleaved = []
    max_len = max(len(r) for r in per_file)
    for i in range(max_len):
        for records in per_file:
            if i < len(records):
                interleaved.append(records[i])

    print(f"  Total after interleaving: {len(interleaved)}")
    return interleaved


def build_prompt(example: Dict, system_prompt: str) -> Optional[str]:
    context = example.get("context", "")
    question = example.get("input", "")
    prompt = f"{system_prompt}\n{context}\n{question}".strip()
    if not (50 <= len(prompt) <= 1024):
        return None
    return prompt


def load_existing_prompts(save_path: str) -> Set[str]:
    if not os.path.exists(save_path):
        return set()
    seen = set()
    with open(save_path, "r") as f:
        for line in f:
            try:
                row = json.loads(line)
                if "prompt" in row:
                    seen.add(row["prompt"])
            except json.JSONDecodeError:
                continue
    return seen


def get_shard_pairs(
    all_pairs: List[Tuple[str, str]],
    num_samples: int,
    shard: int,
    num_shards: int,
) -> List[Tuple[str, str]]:
    pool = all_pairs[:num_samples]
    if len(pool) < num_samples:
        raise ValueError(
            f"Only {len(pool)} valid prompts found, need {num_samples}. "
            "Increase max_scan_per_file in load_waterbench_all."
        )
    chunk_size = num_samples // num_shards
    start = shard * chunk_size
    end = start + chunk_size if shard < num_shards - 1 else num_samples
    return pool[start:end]


def merge_shards(config: "RemaskEvalConfig") -> str:
    run_dir = Path(config.run_dir)
    shard_files = sorted(run_dir.glob("shard_*_of_*.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {run_dir}")

    merged_path = config.merged_save_path
    total = 0
    with open(merged_path, "w") as out_f:
        for shard_file in shard_files:
            with open(shard_file, "r") as in_f:
                for line in in_f:
                    out_f.write(line)
                    total += 1
            print(f"  Merged {shard_file.name}")

    print(f"Merged {total} total rows → {merged_path}")
    return merged_path


class RemaskEvaluator:
    def __init__(self, config: RemaskEvalConfig):
        self.config = config

        Path(config.run_dir).mkdir(parents=True, exist_ok=True)
        print(f"  Run directory: {config.run_dir}")

        self.watermark = SimpleRemaskWatermark(
            key=config.secret_key,
            delta=config.delta,
            context_window=config.context_window,
        )

        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[Shard {config.shard}/{config.num_shards}] Loading LLaDA on {device_str}...")
        self.generator = LLaDAGenerator(
            config=str(LLADA_CONFIG_PATH),
            watermark=self.watermark,
            device_map=device_str,
        )
        print(f"  Model loaded on: {self.generator.device}")

        print("  Loading PPL model (GPT-2)...")
        self.ppl_model = GPT2LMHeadModel.from_pretrained(config.ppl_model_name)
        self.ppl_tokenizer = GPT2Tokenizer.from_pretrained(config.ppl_model_name)
        self.ppl_model.eval()

        print("  Loading similarity model...")
        self.sim_model = SentenceTransformer("all-MiniLM-L6-v2")

        self._save_run_desc()

    def _save_run_desc(self):
        desc_path = Path(self.config.run_dir) / "run_desc.json"
        if desc_path.exists():
            return 

        cfg = self.config
        desc = {
            "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "run_dir": cfg.run_dir,
            "watermark": {
                "class": type(self.watermark).__name__,
                "file": str(WATERMARK_PATH),
                "key": cfg.secret_key,
                "delta": cfg.delta,
                "context_window": cfg.context_window,
            },
            "model": {
                "config": str(LLADA_CONFIG_PATH),
                "path": getattr(self.generator, "model_path", "unknown"),
                "device": str(self.generator.device),
            },
            "generation": {
                "gen_length": cfg.gen_length,
                "gen_steps": cfg.gen_steps,
                "block_length": cfg.block_length,
                "system_prompt": cfg.system_prompt,
            },
            "evaluation": {
                "num_samples": cfg.num_samples,
                "num_shards": cfg.num_shards,
                "p_value_threshold": cfg.p_value_threshold,
                "ppl_model": cfg.ppl_model_name,
                "similarity_model": "all-MiniLM-L6-v2",
                "waterbench_dir": cfg.waterbench_dir,
            },
            "environment": {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
                "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", "local"),
                "hostname": os.uname().nodename,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
                "python": sys.version,
            },
        }

        with open(desc_path, "w") as f:
            json.dump(desc, f, indent=2)
        print(f"  Run desc saved to: {desc_path}")

    def _load_results(self, path: Optional[str] = None) -> pd.DataFrame:
        p = path or self.config.shard_save_path
        if not os.path.exists(p):
            return pd.DataFrame()
        return pd.read_json(p, lines=True)

    def _generate_valid(self, prompt: str, use_watermark: bool) -> str:
        cfg = self.config
        self.generator.watermark = self.watermark if use_watermark else None

        text = ""
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

    def _evaluate_sample(self, prompt: str, source_file: str) -> Dict:
        cfg = self.config

        t0 = time.time()
        text_wm = self._generate_valid(prompt, use_watermark=True)
        time_wm = time.time() - t0

        t0 = time.time()
        text_no_wm = self._generate_valid(prompt, use_watermark=False)
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
            "source_file": source_file,
            "shard": cfg.shard,
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
        all_examples = load_waterbench_all(cfg.waterbench_dir)

        all_pairs: List[Tuple[str, str]] = []
        for ex in all_examples:
            prompt = build_prompt(ex, cfg.system_prompt)
            if prompt is not None:
                all_pairs.append((prompt, ex.get("_source_file", "unknown")))

        shard_pairs = get_shard_pairs(all_pairs, cfg.num_samples, cfg.shard, cfg.num_shards)

        existing_prompts: Set[str] = set()
        if cfg.skip_if_exists:
            existing_prompts = load_existing_prompts(cfg.shard_save_path)
            print(f"[Shard {cfg.shard}] {len(existing_prompts)} already done.")

        todo = [(p, s) for p, s in shard_pairs if p not in existing_prompts]
        print(f"[Shard {cfg.shard}] Running {len(todo)} samples (assigned: {len(shard_pairs)}).")

        processed = 0
        with open(cfg.shard_save_path, "a") as out_f:
            for prompt, source_file in tqdm(
                todo,
                desc=f"Shard {cfg.shard}",
                disable=not cfg.tqdm,
            ):
                try:
                    result = self._evaluate_sample(prompt, source_file)
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                    processed += 1
                    if processed % 10 == 0:
                        print(f"  [Shard {cfg.shard}] {processed}/{len(todo)} done")
                except Exception as e:
                    print(f"  [Shard {cfg.shard}] Error (prompt[:60]={prompt[:60]!r}): {e}")
                    continue

        print(f"[Shard {cfg.shard}] Done. {processed} samples written to {cfg.shard_save_path}")

    def summarize(self, results_path: Optional[str] = None):
        df = self._load_results(results_path)
        if df.empty:
            print("No results to summarize.")
            return

        n = len(df)
        tpr = df["det_wm_is_watermarked"].mean()
        fpr = df["det_no_wm_is_watermarked"].mean()

        lines = [
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

        if "source_file" in df.columns and df["source_file"].nunique() > 1:
            lines.append("\nPer-split TPR / FPR")
            lines.append("-" * 60)
            for src, grp in df.groupby("source_file"):
                split_tpr = grp["det_wm_is_watermarked"].mean()
                split_fpr = grp["det_no_wm_is_watermarked"].mean()
                lines.append(f"  {src:<40} TPR={split_tpr:.4f}  FPR={split_fpr:.4f}  n={len(grp)}")

        report = "\n".join(lines)
        print(report)

        report_path = self.config.summary_save_path
        with open(report_path, "w") as f:
            f.write(report + "\n")
        print(f"Summary saved to: {report_path}")

        return {"tpr": tpr, "fpr": fpr, "ppl_wm": df["ppl_wm"].mean(), "ppl_no_wm": df["ppl_no_wm"].mean()}


def parse_args() -> argparse.Namespace:
    default_shard = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    default_num_shards = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=default_shard)
    parser.add_argument("--num_shards", type=int, default=default_num_shards)
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Shared run directory. Required for --merge_only.")
    parser.add_argument("--merge_only", action="store_true",
                        help="Skip generation; just merge existing shard files and summarize.")

    args = parser.parse_args()
    print(f"Shard config: shard={args.shard}, num_shards={args.num_shards} "
          f"(SLURM_ARRAY_TASK_ID={os.environ.get('SLURM_ARRAY_TASK_ID', 'not set')}, "
          f"SLURM_ARRAY_TASK_COUNT={os.environ.get('SLURM_ARRAY_TASK_COUNT', 'not set')})")
    return args


if __name__ == "__main__":
    args = parse_args()

    SEED = 42 + args.shard
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    run_dir = args.run_dir or str(
        PROJECT_ROOT / "runs" / f"remask_eval_{datetime.utcnow().strftime('%m_%d_%H_%M')}"
    )

    config = RemaskEvalConfig(
        num_samples=600,
        gen_length=300,
        gen_steps=100,
        block_length=25,
        delta=4.0,
        secret_key="xooks",
        p_value_threshold=0.01,
        skip_if_exists=True,
        shard=args.shard,
        num_shards=args.num_shards,
        run_dir=run_dir,
    )

    if args.merge_only:
        merged_path = merge_shards(config)
        evaluator = RemaskEvaluator.__new__(RemaskEvaluator)
        evaluator.config = config
        evaluator.summarize(results_path=merged_path)
    else:
        evaluator = RemaskEvaluator(config)
        evaluator.evaluate()

        if args.num_shards == 1:
            evaluator.summarize()