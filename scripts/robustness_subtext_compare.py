import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


def resolve_project_root() -> Path:
    cluster_root = Path("/scratch/efeldma5/waterdiffusion/waterdiffusion")
    if cluster_root.exists():
        return cluster_root
    return Path(__file__).resolve().parent.parent


def import_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DirectZurichDetector:

    def __init__(self, core, device: str):
        self.core = core
        self.device = device

    def detect(self, text: str, tokenizer, p_value_threshold: float = 0.01):
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        out = self.core.detect(input_ids=input_ids, attention_mask=attention_mask)
        z = float(out.get("z_score", 0.0))
        p = float(out.get("p_value", 1.0))
        return {
            "is_watermarked": bool(p < p_value_threshold),
            "score": z,
            "z_score": z,
            "p_value": p,
            "token_color": out.get("token_color", []),
            "mask": out.get("mask", []),
        }


def load_texts(jsonl_path: Path, text_column: str) -> List[str]:
    df = pd.read_json(jsonl_path, lines=True)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {jsonl_path}")
    texts = [t for t in df[text_column].fillna("").tolist() if isinstance(t, str) and t.strip()]
    return texts


def parse_ratios(ratio_str: str) -> List[float]:
    vals = [float(x.strip()) for x in ratio_str.split(",") if x.strip()]
    vals = [round(v, 6) for v in vals]
    for v in vals:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Attack ratio must be in [0,1], got {v}")
    return vals


def iter_windows(length: int, window: int, stride: int):
    if window >= length:
        yield 0
        return

    last = length - window
    start = 0
    while start <= last:
        yield start
        start += stride

    if (last % stride) != 0:
        yield last


def run_dataset(
    dataset_name: str,
    texts: List[str],
    tokenizer,
    detector,
    detector_name: str,
    ratios: List[float],
    stride: int,
    p_value_threshold: float,
    max_texts: int,
) -> List[Dict]:
    rows: List[Dict] = []

    effective_texts = texts[:max_texts] if max_texts > 0 else texts

    for text_idx, text in enumerate(tqdm(effective_texts, desc=f"{dataset_name} texts")):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        seq_len = len(token_ids)
        if seq_len < 10:
            continue

        for ratio in ratios:
            keep_len = max(10, int(round((1.0 - ratio) * seq_len)))
            keep_len = min(keep_len, seq_len)

            for win_start in iter_windows(seq_len, keep_len, stride):
                win_end = win_start + keep_len
                sub_ids = token_ids[win_start:win_end]
                sub_text = tokenizer.decode(sub_ids, skip_special_tokens=True)

                detection = detector.detect(
                    sub_text,
                    tokenizer,
                    p_value_threshold=p_value_threshold,
                )

                rows.append(
                    {
                        "dataset": dataset_name,
                        "detector": detector_name,
                        "text_idx": text_idx,
                        "attack_ratio": ratio,
                        "orig_token_len": seq_len,
                        "sub_token_len": keep_len,
                        "window_start": win_start,
                        "window_end": win_end,
                        "p_value": float(detection.get("p_value", 1.0)),
                        "z_score": float(detection.get("z_score", 0.0)),
                        "score": float(detection.get("score", 0.0)),
                        "is_watermarked": bool(detection.get("is_watermarked", False)),
                    }
                )

    return rows


def summarize(rows_df: pd.DataFrame, p_value_threshold: float) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "attack_ratio",
                "num_subtexts",
                "mean_p_value",
                "median_p_value",
                "frac_p_below_threshold",
                "mean_z_score",
            ]
        )

    grouped = rows_df.groupby(["dataset", "detector", "attack_ratio"], as_index=False).agg(
        num_subtexts=("p_value", "size"),
        mean_p_value=("p_value", "mean"),
        median_p_value=("p_value", "median"),
        mean_z_score=("z_score", "mean"),
    )

    frac = (
        rows_df.assign(p_below=lambda x: x["p_value"] < p_value_threshold)
        .groupby(["dataset", "detector", "attack_ratio"], as_index=False)["p_below"]
        .mean()
        .rename(columns={"p_below": "frac_p_below_threshold"})
    )

    out = grouped.merge(frac, on=["dataset", "detector", "attack_ratio"], how="left")
    return out.sort_values(["dataset", "detector", "attack_ratio"]).reset_index(drop=True)


def main():
    project_root = resolve_project_root()

    parser = argparse.ArgumentParser(description="Subtext robustness p-value comparison")
    parser.add_argument(
        "--project_root",
        type=Path,
        default=project_root,
        help="Project root (cluster default: /scratch/efeldma5/waterdiffusion/waterdiffusion)",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="subtext_robustness_compare",
        help="Prefix for output folder under runs/<run_id>_<date>/",
    )
    parser.add_argument(
        "--theirs_jsonl",
        type=Path,
        default=Path("runs/zurich_eval_same_prompts_7006629/delta_4.0/results_with_ppl.jsonl"),
        help="Path to 'theirs' results (relative to project_root unless absolute)",
    )
    parser.add_argument(
        "--ours_jsonl",
        type=Path,
        default=Path("runs/remask_eval_6946085/results_merged.jsonl"),
        help="Path to 'ours' results (relative to project_root unless absolute)",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="completion_wm",
        help="Text column to evaluate",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Tokenizer used for detection tokenization",
    )
    parser.add_argument("--key", type=str, default="xooks", help="Watermark key")
    parser.add_argument("--theirs_delta", type=float, default=4.0, help="Delta for theirs Zurich detector")
    parser.add_argument("--ours_delta", type=float, default=16.0, help="Delta for ours DWT detector")
    parser.add_argument(
        "--p_value_threshold",
        type=float,
        default=0.01,
        help="Detection threshold used in summary fraction",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated attack ratios",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Subtext sliding-window stride (1 means all contiguous subtexts)",
    )
    parser.add_argument(
        "--max_texts",
        type=int,
        default=0,
        help="Optional limit for quick runs; 0 means all texts",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow downloading tokenizer files if missing from cache",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    theirs_path = args.theirs_jsonl if args.theirs_jsonl.is_absolute() else (root / args.theirs_jsonl)
    ours_path = args.ours_jsonl if args.ours_jsonl.is_absolute() else (root / args.ours_jsonl)
    theirs_watermark_path = root / "zurich/src/dlm_watermark/watermarks/diffusion_watermark.py"
    ours_watermark_path = root / "watermarks/3_29_DWT_watermark.py"

    if not theirs_path.exists():
        raise FileNotFoundError(f"Missing theirs_jsonl: {theirs_path}")
    if not ours_path.exists():
        raise FileNotFoundError(f"Missing ours_jsonl: {ours_path}")
    if not theirs_watermark_path.exists():
        raise FileNotFoundError(f"Missing theirs detector file: {theirs_watermark_path}")
    if not ours_watermark_path.exists():
        raise FileNotFoundError(f"Missing ours detector file: {ours_watermark_path}")

    ratios = parse_ratios(args.ratios)

    run_stamp = datetime.utcnow().strftime("%m_%d_%H_%M")
    run_name = f"{args.run_id}_{run_stamp}"
    out_dir = root / "runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    local_files_only = not args.online
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    detector_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    zurich_src = root / "zurich" / "src"
    if str(zurich_src) not in sys.path:
        sys.path.insert(0, str(zurich_src))

    from dlm_watermark.watermarks.diffusion_watermark import OurWatermark

    theirs_core = OurWatermark(
        delta=args.theirs_delta,
        enforce_kl=False,
        convolution_kernel=[-1],
        greenlist_type="bernoulli",
        greenlist_params={"gamma": 0.25},
        topk=50,
        n_iter=1,
        seeding_scheme="sumhash",
        tokenizer=tokenizer,
        device=detector_device,
    )
    theirs_detector = DirectZurichDetector(theirs_core, detector_device)

    ours_module = import_from_path(ours_watermark_path, "wm_mod_subtext_ours")
    OursDWTWatermark = getattr(ours_module, "DWTWatermark")
    ours_detector = OursDWTWatermark(key=args.key, delta=args.ours_delta)

    theirs_texts = load_texts(theirs_path, args.text_column)
    ours_texts = load_texts(ours_path, args.text_column)

    all_rows: List[Dict] = []
    all_rows.extend(
        run_dataset(
            dataset_name="theirs",
            texts=theirs_texts,
            tokenizer=tokenizer,
            detector=theirs_detector,
            detector_name="ZurichOurWatermark(direct)",
            ratios=ratios,
            stride=args.stride,
            p_value_threshold=args.p_value_threshold,
            max_texts=args.max_texts,
        )
    )
    all_rows.extend(
        run_dataset(
            dataset_name="ours",
            texts=ours_texts,
            tokenizer=tokenizer,
            detector=ours_detector,
            detector_name="DWTWatermark(3_29)",
            ratios=ratios,
            stride=args.stride,
            p_value_threshold=args.p_value_threshold,
            max_texts=args.max_texts,
        )
    )

    rows_df = pd.DataFrame(all_rows)
    details_path = out_dir / "subtext_pvalues_details.jsonl"
    if not rows_df.empty:
        rows_df.to_json(details_path, orient="records", lines=True)
    else:
        details_path.write_text("")

    summary_df = summarize(rows_df, args.p_value_threshold)
    summary_csv_path = out_dir / "subtext_pvalues_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    summary_txt_path = out_dir / "summary_report.txt"
    with summary_txt_path.open("w") as f:
        f.write("--- Subtext Robustness P-Value Summary ---\n")
        f.write(f"project_root: {root}\n")
        f.write(f"theirs_jsonl: {theirs_path}\n")
        f.write(f"ours_jsonl: {ours_path}\n")
        f.write(f"text_column: {args.text_column}\n")
        f.write(f"theirs_detector_path: {theirs_watermark_path}\n")
        f.write(f"ours_detector_path: {ours_watermark_path}\n")
        f.write(f"tokenizer_name: {args.tokenizer_name}\n")
        f.write(f"key: {args.key}\n")
        f.write(f"theirs_delta: {args.theirs_delta}\n")
        f.write(f"ours_delta: {args.ours_delta}\n")
        f.write(f"ratios: {ratios}\n")
        f.write(f"stride: {args.stride}\n")
        f.write(f"max_texts: {args.max_texts}\n")
        f.write(f"p_value_threshold: {args.p_value_threshold}\n")
        f.write("\n")
        if summary_df.empty:
            f.write("No rows produced.\n")
        else:
            f.write(summary_df.to_string(index=False))
            f.write("\n")

    config_path = out_dir / "run_config.json"
    config = {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "project_root": str(root),
        "theirs_jsonl": str(theirs_path),
        "ours_jsonl": str(ours_path),
        "text_column": args.text_column,
        "theirs_detector_path": str(theirs_watermark_path),
        "ours_detector_path": str(ours_watermark_path),
        "tokenizer_name": args.tokenizer_name,
        "key": args.key,
        "theirs_delta": args.theirs_delta,
        "ours_delta": args.ours_delta,
        "ratios": ratios,
        "stride": args.stride,
        "max_texts": args.max_texts,
        "p_value_threshold": args.p_value_threshold,
        "details_jsonl": str(details_path),
        "summary_csv": str(summary_csv_path),
        "summary_report": str(summary_txt_path),
    }
    config_path.write_text(json.dumps(config, indent=2))

    print(f"Saved details: {details_path}")
    print(f"Saved summary csv: {summary_csv_path}")
    print(f"Saved summary report: {summary_txt_path}")


if __name__ == "__main__":
    main()
