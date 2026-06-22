"""
Plot all proposals: TPR @ 1% FPR vs KL divergence (Δ log PPL).

This is the KL divergence counterpart of plot_all_proposals.py, which plots
TPR vs log(PPL).  Here the x-axis is the empirical KL divergence, estimated
as log(mean_ppl_wm) - log(mean_ppl_no_wm), which measures the *information
cost* of the watermark rather than the absolute quality.

Usage:
    python scripts/graphs/plot_all_proposals_kl.py
    python scripts/graphs/plot_all_proposals_kl.py --use-qwen   # use Qwen PPL columns
"""

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


def load_kl_points(run_dir: Path, label: str, ppl_wm_key: str, ppl_no_wm_key: str,
                   exclude_deltas=None):
    """Load (delta, kl_divergence, tpr) points from a run directory."""
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
        # Try qwen-reeval file first, then merged, then with_ppl
        p_qwen = d / "results_merged_qwen_ppl.jsonl"
        p_qwen2 = d / "results_with_ppl_qwen_ppl.jsonl"
        p_merged = d / "results_merged.jsonl"
        p_with_ppl = d / "results_with_ppl.jsonl"

        # Choose file: prefer qwen-reeval if using qwen keys
        if ppl_wm_key == "ppl_wm_qwen":
            candidates = [p_qwen, p_qwen2, p_merged, p_with_ppl]
        else:
            candidates = [p_merged, p_with_ppl]

        p = None
        for c in candidates:
            if c.exists():
                p = c
                break

        if p is None:
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

            if ppl_wm_key not in df.columns or ppl_no_wm_key not in df.columns:
                # Fall back to standard columns if qwen columns not available
                if "ppl_wm" in df.columns and "ppl_no_wm" in df.columns:
                    ppl_wm_key_eff = "ppl_wm"
                    ppl_no_wm_key_eff = "ppl_no_wm"
                else:
                    print(f"  Skipping {d.name}: missing PPL columns")
                    continue
            else:
                ppl_wm_key_eff = ppl_wm_key
                ppl_no_wm_key_eff = ppl_no_wm_key

            mean_ppl_wm = df[ppl_wm_key_eff].mean()
            mean_ppl_no_wm = df[ppl_no_wm_key_eff].mean()

            if mean_ppl_wm <= 0 or mean_ppl_no_wm <= 0:
                continue

            # Empirical KL estimation: log(PPL_wm) - log(PPL_no_wm)
            kl_emp = math.log(mean_ppl_wm) - math.log(mean_ppl_no_wm)

            tpr = calculate_tpr_at_1fpr(df)
            points.append((delta, kl_emp, tpr))
            print(f"Loaded {label} delta {delta}: KL_emp={kl_emp:.4f}, tpr={tpr:.4f} (from {p.name})")
        except Exception as e:
            print(f"Error loading {p}: {e}")
            continue

    return points


def main():
    parser = argparse.ArgumentParser(
        description="Plot all proposals: TPR vs KL divergence."
    )
    parser.add_argument(
        "--use-qwen", action="store_true",
        help="Use ppl_wm_qwen / ppl_no_wm_qwen columns (from Qwen-32B re-evaluation)."
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output PNG path. Defaults to runs/all_proposals_kl_divergence.png"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    runs = root / "runs"

    # PPL column keys
    if args.use_qwen:
        ppl_wm_key = "ppl_wm_qwen"
        ppl_no_wm_key = "ppl_no_wm_qwen"
        ppl_label = "Qwen-32B"
    else:
        ppl_wm_key = "ppl_wm"
        ppl_no_wm_key = "ppl_no_wm"
        ppl_label = "GPT-2"

    configs = [
        {"dir": runs / "prompt1_8519605", "label": "Prop 1 (CKAS)", "color": "#D97706", "marker": "s", "exclude": []},
        {"dir": runs / "prompt2_8522189", "label": "Prop 2 (ARC)",  "color": "#3B82F6", "marker": "o", "exclude": []},
        {"dir": runs / "prompt3_8522188", "label": "Prop 3 (KLB)",  "color": "#8B5CF6", "marker": "D", "exclude": []},
        {"dir": runs / "prompt4_8522187", "label": "Prop 4",        "color": "#EF4444", "marker": "P", "exclude": []},
        {"dir": runs / "prompt5_8525479", "label": "Prop 5", "color": "#72B572", "marker": "s", "exclude": []},
        {"dir": runs / "zurich_eval_same_prompts_7006629", "label": "Zurich", "color": "#059669", "marker": "^", "exclude": []},
    ]

    plt.figure(figsize=(11, 7))

    all_x = []

    for conf in configs:
        points = load_kl_points(
            conf["dir"], conf["label"],
            ppl_wm_key, ppl_no_wm_key,
            exclude_deltas=conf["exclude"]
        )
        if not points:
            print(f"  No points for {conf['label']}")
            continue

        x = [p[1] for p in points]
        y = [p[2] for p in points]
        all_x.extend(x)

        plt.plot(
            x, y,
            label=conf["label"],
            color=conf["color"],
            marker=conf["marker"],
            linewidth=2.5,
            markersize=8,
            alpha=0.9
        )

        # Add delta annotations
        for delta, kl, tpr in points:
            plt.annotate(
                f"δ={delta}",
                (kl, tpr),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=7,
                color=conf["color"],
                alpha=0.7,
            )

    plt.xlabel("Empirical KL Divergence (Δ log PPL)", fontsize=13)
    plt.ylabel("TPR @ 1% FPR", fontsize=13)
    plt.title(
        f"Comparison of Watermarking Proposals\nTPR vs KL Divergence ({ppl_label})",
        fontsize=15
    )
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=11)
    plt.ylim(-0.05, 1.05)

    if all_x:
        margin = 0.05 * (max(all_x) - min(all_x)) if max(all_x) != min(all_x) else 0.1
        plt.xlim(left=min(all_x) - margin, right=max(all_x) + margin)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = runs / "all_proposals_kl_divergence.png"

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"\nSaved KL divergence comparison plot to: {out_path}")


if __name__ == "__main__":
    main()
