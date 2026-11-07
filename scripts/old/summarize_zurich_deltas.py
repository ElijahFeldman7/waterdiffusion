#!/usr/bin/env python3
"""Generate per-delta summary reports (with GPT-2 PPL) for Zurich eval outputs."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm


def compute_ppl(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    text: str,
    device: torch.device,
    max_length: int = 1024,
) -> float:
    encodings = tokenizer(
        text if isinstance(text, str) else "",
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encodings.input_ids.to(device)
    if input_ids.shape[1] < 2:
        return float("inf")
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss
    return float(torch.exp(loss).item())


def finite_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def build_detection_flags(df: pd.DataFrame, p_value_threshold: float) -> pd.DataFrame:
    df = df.copy()

    if "det_wm_is_watermarked" not in df.columns:
        if "det_wm_p_value" in df.columns:
            det_wm_p = pd.to_numeric(df["det_wm_p_value"], errors="coerce")
            df["det_wm_is_watermarked"] = det_wm_p < p_value_threshold
        elif "det_wm_z_score" in df.columns:
            det_wm_z = pd.to_numeric(df["det_wm_z_score"], errors="coerce")
            df["det_wm_is_watermarked"] = det_wm_z > 0
        else:
            df["det_wm_is_watermarked"] = False

    if "det_no_wm_is_watermarked" not in df.columns:
        if "det_no_wm_p_value" in df.columns:
            det_no_wm_p = pd.to_numeric(df["det_no_wm_p_value"], errors="coerce")
            df["det_no_wm_is_watermarked"] = det_no_wm_p < p_value_threshold
        elif "det_no_wm_z_score" in df.columns:
            det_no_wm_z = pd.to_numeric(df["det_no_wm_z_score"], errors="coerce")
            df["det_no_wm_is_watermarked"] = det_no_wm_z > 0
        else:
            df["det_no_wm_is_watermarked"] = False

    return df


def summarize_delta(df: pd.DataFrame, delta: float, output_path: Path, p_value_threshold: float) -> None:
    n = len(df)
    if n == 0:
        output_path.write_text("No results for this delta.\n")
        return

    tp = int(df["det_wm_is_watermarked"].sum())
    fn = int(n - tp)
    fp = int(df["det_no_wm_is_watermarked"].sum())
    tn = int(n - fp)

    tpr = tp / n if n else float("nan")
    fpr = fp / n if n else float("nan")

    lines = [
        "--- Watermark Detection Analysis ---",
        f"Delta: {delta}",
        f"Total samples processed: {n}",
        f"True Positives (TP): {tp}",
        f"False Negatives (FN): {fn}",
        f"True Negatives (TN): {tn}",
        f"False Positives (FP): {fp}",
        "",
        "--- Detection Rates ---",
        f"True Positive Rate (TPR/Recall): {tpr:.4f}",
        f"False Positive Rate (FPR): {fpr:.4f}",
        f"False Negative Rate (FNR): {1-tpr:.4f}",
        f"True Negative Rate (TNR/Specificity): {1-fpr:.4f}",
        f"P-value threshold used for detection flags: {p_value_threshold}",
        "",
        "--- Score Statistics ---",
        f"det_wm_z_score mean/std: {df['det_wm_z_score'].mean():.6f} / {df['det_wm_z_score'].std(ddof=1):.6f}",
        f"det_no_wm_z_score mean/std: {df['det_no_wm_z_score'].mean():.6f} / {df['det_no_wm_z_score'].std(ddof=1):.6f}",
        f"det_wm_p_value mean/std: {df['det_wm_p_value'].mean():.6f} / {df['det_wm_p_value'].std(ddof=1):.6f}",
        f"det_no_wm_p_value mean/std: {df['det_no_wm_p_value'].mean():.6f} / {df['det_no_wm_p_value'].std(ddof=1):.6f}",
        "",
        "--- Quality Impact ---",
        f"Perplexity (WM): {finite_mean(df['ppl_wm']):.4f}",
        f"Perplexity (No WM): {finite_mean(df['ppl_no_wm']):.4f}",
        "",
        "--- Dataset Breakdown ---",
        str(df.groupby("source_file").size()),
        "",
        "--- Existing Eval Metric ---",
        f"tpr_at_1pct (first row): {df['tpr_at_1pct'].iloc[0] if 'tpr_at_1pct' in df.columns else 'N/A'}",
        "",
        "=" * 60,
        "--- Generation Samples ---",
    ]

    for i, row in df.reset_index(drop=True).iterrows():
        lines.extend(
            [
                f"--- Sample {i + 1} ---",
                f"Source: {row.get('source_file', 'unknown')}",
                f"Prompt: {row.get('prompt', '')}",
                f"Perplexity (WM): {row.get('ppl_wm', float('nan')):.4f} | "
                f"Perplexity (No WM): {row.get('ppl_no_wm', float('nan')):.4f}",
                f"Watermarked Text Detection: "
                f"{'True Positive' if bool(row.get('det_wm_is_watermarked', False)) else 'False Negative'}",
                f"Non-Watermarked Text Detection: "
                f"{'False Positive' if bool(row.get('det_no_wm_is_watermarked', False)) else 'True Negative'}",
                "",
                "[Watermarked Text]",
                row.get("completion_wm", ""),
                "",
                "[Original Text]",
                row.get("completion_no_wm", ""),
                "",
                "-" * 40,
                "",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-delta summary reports for Zurich eval runs.")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Run directory containing watermark_results_*.jsonl",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="watermark_results_waterbench_mixed_300samples.jsonl",
        help="Input JSONL filename inside run_dir",
    )
    parser.add_argument(
        "--p_value_threshold",
        type=float,
        default=0.01,
        help="Threshold to derive detection booleans from p-values.",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    input_path = run_dir / args.input_file
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading results from: {input_path}")
    df = pd.read_json(input_path, lines=True)
    if df.empty:
        raise ValueError("Input file has no rows")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading GPT-2 for PPL on device: {device}")
    ppl_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    ppl_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print("Computing per-sample PPL...")
    ppl_wm = []
    ppl_no_wm = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="PPL"):
        ppl_wm.append(compute_ppl(ppl_model, ppl_tokenizer, row.get("completion_wm", ""), device))
        ppl_no_wm.append(compute_ppl(ppl_model, ppl_tokenizer, row.get("completion_no_wm", ""), device))

    df = df.copy()
    df["ppl_wm"] = ppl_wm
    df["ppl_no_wm"] = ppl_no_wm
    df = build_detection_flags(df, args.p_value_threshold)

    deltas = sorted(df["delta"].dropna().unique().tolist())
    print(f"Found deltas: {deltas}")

    for delta in deltas:
        delta_df = df[df["delta"] == delta].copy()
        delta_dir = run_dir / f"delta_{delta}"
        delta_dir.mkdir(parents=True, exist_ok=True)

        delta_jsonl_path = delta_dir / "results_with_ppl.jsonl"
        with delta_jsonl_path.open("w") as f:
            for record in delta_df.to_dict(orient="records"):
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

        summary_path = delta_dir / "summary_report.txt"
        summarize_delta(delta_df, delta, summary_path, args.p_value_threshold)

        print(f"Wrote: {delta_jsonl_path}")
        print(f"Wrote: {summary_path}")

    print("Done.")


if __name__ == "__main__":
    main()
