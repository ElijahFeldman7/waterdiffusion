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
    if "det_no_wm_p_value" not in df.columns or "det_wm_p_value" not in df.columns:
        return 0.0
    no_wm_pvals = df["det_no_wm_p_value"].dropna().values
    wm_pvals = df["det_wm_p_value"].dropna().values
    if len(no_wm_pvals) == 0 or len(wm_pvals) == 0:
        return 0.0
    thresh = np.percentile(no_wm_pvals, 1.0)
    return float(np.mean(wm_pvals <= thresh))

def load_kl_points(run_dir: Path, label: str, exclude_deltas=None):
    points = []
    if not run_dir.exists():
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
            delta = float(d.name.split("_", 1)[1])
            if exclude_deltas and delta in exclude_deltas:
                continue

            df = pd.read_json(p, lines=True)
            if "ppl_wm" not in df.columns or "ppl_no_wm" not in df.columns:
                continue

            mean_ppl_wm = df["ppl_wm"].mean()
            mean_ppl_no_wm = df["ppl_no_wm"].mean()
            
            # Empirical KL estimation via log-PPL difference
            kl_emp = math.log(mean_ppl_wm) - math.log(mean_ppl_no_wm)
            
            tpr = calculate_tpr_at_1fpr(df)
            points.append((delta, kl_emp, tpr))
            print(f"Loaded {label} delta {delta}: KL_emp={kl_emp:.4f}, tpr={tpr:.4f}")
        except Exception as e:
            print(f"Error loading {p}: {e}")
            continue

    return points

def main():
    root = Path(__file__).resolve().parent.parent.parent
    runs_dir = root / "runs"
    
    prompt1_dir = runs_dir / "prompt1_8519605"
    zurich_dir = runs_dir / "zurich_eval_same_prompts_7006629"

    print("Loading Prompt1 (ckas) points...")
    p1_points = load_kl_points(prompt1_dir, "Prompt1")
    
    print("\nLoading Zurich points...")
    z_points = load_kl_points(zurich_dir, "Zurich")

    fig, ax = plt.subplots(figsize=(10, 7))

    if p1_points:
        ax.plot(
            [x[1] for x in p1_points],
            [x[2] for x in p1_points],
            marker="s",
            linewidth=2.5,
            markersize=8,
            color="#D97706",
            label="Prompt1 (ckas)",
            alpha=0.9
        )
        for delta, kl, tpr in p1_points:
            ax.annotate(f"d={delta}", (kl, tpr), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color="#D97706")

    if z_points:
        ax.plot(
            [x[1] for x in z_points],
            [x[2] for x in z_points],
            marker="^",
            linewidth=2.5,
            markersize=8,
            color="#059669",
            label="Zurich",
            alpha=0.9
        )
        for delta, kl, tpr in z_points:
            ax.annotate(f"d={delta}", (kl, tpr), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color="#059669")

    ax.set_xlabel("Empirical KL Divergence (Δ log PPL)", fontsize=13)
    ax.set_ylabel("TPR @ 1% FPR", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=12)
    ax.set_title("Detection Power vs. Information Cost (Empirical KL)", fontsize=15)

    out = prompt1_dir / "tpr_kl_compare.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"\nSaved plot to: {out}")

if __name__ == "__main__":
    main()
