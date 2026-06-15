import json
import math
import os
import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def calculate_tpr_at_1fpr(df: pd.DataFrame) -> float:
    """
    Calculates TPR at 1% FPR. 
    Uses det_no_wm_p_value to find the 1% threshold, then applies to det_wm_p_value.
    Matches logic in calc_1fpr.py.
    """
    if "det_no_wm_p_value" not in df.columns or "det_wm_p_value" not in df.columns:
        # Fallback for Zurich files that already have it
        if "tpr_at_1pct" in df.columns:
            return float(df["tpr_at_1pct"].iloc[0])
        return 0.0
    
    no_wm_pvals = df["det_no_wm_p_value"].dropna().values
    wm_pvals = df["det_wm_p_value"].dropna().values
    
    if len(no_wm_pvals) == 0 or len(wm_pvals) == 0:
        return 0.0
    
    # Threshold is the 1st percentile of no-WM p-values
    thresh = np.percentile(no_wm_pvals, 1.0)
    tpr = np.mean(wm_pvals <= thresh)
    return float(tpr)

def load_run_points(run_dir: Path, label: str, exclude_deltas=None):
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

        if exclude_deltas and delta in exclude_deltas:
            continue

        try:
            df = pd.read_json(p, lines=True)
            if "ppl_wm" not in df.columns:
                continue

            # Use log(mean(PPL))
            mean_ppl = df["ppl_wm"].mean()
            if mean_ppl <= 0:
                continue
            log_ppl = math.log(mean_ppl)

            tpr = calculate_tpr_at_1fpr(df)
            points.append((delta, log_ppl, tpr))
            print(f"Loaded {label} delta {delta}: log_ppl={log_ppl:.4f}, tpr={tpr:.4f}")
        except Exception as e:
            print(f"Error loading {p}: {e}")
            continue

    return points

def main():
    root = Path(__file__).resolve().parent.parent.parent
    runs = root / "runs"
    
    # Run IDs provided by user/discovery
    configs = [
        {"dir": runs / "prompt1_8519605", "label": "Prop 1 (CKAS)", "color": "#D97706", "marker": "s", "exclude": [1.0, 2.0]},
        {"dir": runs / "prompt2_8522189", "label": "Prop 2 (ARC)", "color": "#3B82F6", "marker": "o", "exclude": [1.0, 2.0]},
        {"dir": runs / "prompt3_8522188", "label": "Prop 3 (KLB)", "color": "#8B5CF6", "marker": "D", "exclude": []},
        # Prop 4 excluded as requested
        {"dir": runs / "zurich_eval_same_prompts_7006629", "label": "Zurich", "color": "#059669", "marker": "^", "exclude": [1.0, 2.0]},
    ]

    plt.figure(figsize=(11, 7))
    
    all_x = []
    
    for conf in configs:
        points = load_run_points(conf["dir"], conf["label"], exclude_deltas=conf["exclude"])
        if not points:
            continue
            
        x = [p[1] for p in points]
        y = [p[2] for p in points]
        all_x.extend(x)
        
        plt.plot(x, y, label=conf["label"], color=conf["color"], marker=conf["marker"], linewidth=2.5, markersize=8, alpha=0.9)
        
        # Delta labels removed as requested

    plt.xlabel("log(mean(PPL))", fontsize=13)
    plt.ylabel("TPR @ 1% FPR", fontsize=13)
    plt.title("Comparison of Watermarking Proposals\nTPR vs log(PPL)", fontsize=15)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=11)
    plt.ylim(-0.05, 1.05)
    
    if all_x:
        plt.xlim(left=min(all_x)*0.98, right=max(all_x)*1.02)

    out_path = root / "runs" / "all_proposals_compare.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"\nSaved comparison plot to: {out_path}")

if __name__ == "__main__":
    main()
