import json
import math
import os
import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_base_points(runs_dir: Path):
    skip = {"remask_eval_6927558", "remask_eval_6930413", "prompt1_8519605", "prompt1_8519420", "prompt1_8519375"}
    base = []

    if not runs_dir.exists():
        return base

    for folder in sorted(os.listdir(runs_dir)):
        if folder == "old" or folder in skip:
            continue

        desc_path = runs_dir / folder / "run_desc.json"
        if not desc_path.exists():
            continue

        try:
            with open(desc_path, 'r') as f:
                desc = json.load(f)
        except:
            continue
            
        wm_file = desc.get("watermark", {}).get("file", "")
        if "3_29_DWT_watermark.py" not in wm_file:
            continue

        tpr_path = runs_dir / folder / "tpr.txt"
        if not tpr_path.exists():
            continue

        try:
            txt = tpr_path.read_text()
            m_mean = re.search(r"mean PPL \(WM\):\s+([\d.]+)", txt)
            m_tpr = re.search(r"global TPR:\s+([\d.]+)", txt)
            
            if m_mean and m_tpr:
                mean_val = float(m_mean.group(1))
                log_ppl = math.log(mean_val)
                base.append((folder, log_ppl, float(m_tpr.group(1))))
            else:
                m_log = re.search(r"log PPL \(WM\):\s+([\d.]+)", txt)
                if m_log and m_tpr:
                    base.append((folder, float(m_log.group(1)), float(m_tpr.group(1))))
        except:
            continue

    base.sort(key=lambda x: x[1])
    return base

def calculate_tpr_at_1fpr(df: pd.DataFrame) -> float:
    if "det_no_wm_p_value" not in df.columns or "det_wm_p_value" not in df.columns:
        return 0.0
    
    no_wm_pvals = df["det_no_wm_p_value"].dropna().values
    wm_pvals = df["det_wm_p_value"].dropna().values
    
    if len(no_wm_pvals) == 0 or len(wm_pvals) == 0:
        return 0.0
    
    thresh = np.percentile(no_wm_pvals, 1.0)
    tpr = np.mean(wm_pvals <= thresh)
    return float(tpr)
def load_run_points(run_dir: Path, label: str):
    points = []
    if not run_dir.exists():
        print(f"Warning: run_dir {run_dir} does not exist.")
        return points

    delta_folders = list(run_dir.glob("delta_*"))
    try:
        delta_folders.sort(key=lambda p: float(p.name.split("_", 1)[1]))
    except:
        delta_folders.sort()

    for d in delta_folders:
        p_merged = d / "results_merged.jsonl"
        p_with_ppl = d / "results_with_ppl.jsonl"
        p = p_merged if p_merged.exists() else p_with_ppl

        if not p.exists():
            continue

        try:
            delta_str = d.name.split("_", 1)[1]
            delta = float(delta_str)
        except:
            delta = 0.0

        # Skip deltas 1.0 and 2.0 as requested
        if delta in [1.0, 2.0]:
            print(f"  Skipping delta {delta} for {label} (requested)")
            continue

        try:
            df = pd.read_json(p, lines=True)
            if "ppl_wm" not in df.columns:
                continue

            # Use log(mean(PPL)) to match summary_report.txt and user expectations
            mean_ppl = df["ppl_wm"].mean()
            if mean_ppl <= 0:
                continue
            log_ppl = math.log(mean_ppl)

            tpr = calculate_tpr_at_1fpr(df)
            points.append((delta, log_ppl, tpr))
            print(f"Loaded {label} delta {delta}: log_ppl={log_ppl:.4f} (mean_ppl={mean_ppl:.2f}), tpr={tpr:.4f}")
        except Exception as e:
            print(f"Error loading {p}: {e}")
            continue

    return points


def main():
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    default_runs_dir = project_root / "runs"
    
    prompt1_run_dir = default_runs_dir / "prompt1_8519605"
    zurich_run_dir = default_runs_dir / "zurich_eval_same_prompts_7006629"

    parser = argparse.ArgumentParser(description="Plot TPR vs log(PPL) comparison.")
    parser.add_argument("--runs-dir", type=Path, default=default_runs_dir)
    parser.add_argument("--prompt1-dir", type=Path, default=prompt1_run_dir)
    parser.add_argument("--zurich-dir", type=Path, default=zurich_run_dir)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    print(f"Loading baseline from: {args.runs_dir}")
    base = load_base_points(args.runs_dir)
    
    print(f"Loading prompt1 from: {args.prompt1_dir}")
    prompt1 = load_run_points(args.prompt1_dir, "prompt1")
    
    print(f"Loading zurich from: {args.zurich_dir}")
    zurich = load_run_points(args.zurich_dir, "zurich")

    fig, ax = plt.subplots(figsize=(10, 7))

    if base:
        ax.plot([x[1] for x in base], [x[2] for x in base], marker="o", linewidth=1.5, markersize=5, color="gray", label="Baseline (DWT)", alpha=0.4, linestyle="--")

    if zurich:
        ax.plot([x[1] for x in zurich], [x[2] for x in zurich], marker="^", linewidth=3, markersize=9, color="#059669", label="Zurich Eval", alpha=1.0)
        for delta, log_ppl, tpr in zurich:
            ax.annotate(f"{delta}", (log_ppl, tpr), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color="#059669", fontweight='bold')

    if prompt1:
        ax.plot([x[1] for x in prompt1], [x[2] for x in prompt1], marker="s", linewidth=3, markersize=9, color="#D97706", label="Prompt1", alpha=1.0)
        for delta, log_ppl, tpr in prompt1:
            ax.annotate(f"{delta}", (log_ppl, tpr), textcoords="offset points", xytext=(0,-18), ha='center', fontsize=9, color="#D97706", fontweight='bold')

    ax.set_xlabel("log(PPL)", fontsize=13)
    ax.set_ylabel("TPR @ 1% FPR", fontsize=13)
    ax.set_ylim(-0.02, 1.05)
    
    all_points = base + prompt1 + zurich
    if all_points:
        ax.set_xlim(left=min([x[1] for x in all_points]) * 0.98, right=max([x[1] for x in all_points]) * 1.02)
        
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_title("TPR vs log(PPL): Zurich vs Prompt1", fontsize=15)

    out = args.out if args.out is not None else (args.prompt1_dir / "zurich_vs_prompt1.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"\nSaved plot to: {out}")

if __name__ == "__main__":
    main()
