import json
import math
import os
import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def load_base_points(runs_dir: Path):
    skip = {"remask_eval_6927558", "remask_eval_6930413", "prompt1_8519605", "prompt1_8519420", "prompt1_8519375"}
    base = []
    if not runs_dir.exists(): return base
    for folder in sorted(os.listdir(runs_dir)):
        if folder == "old" or folder in skip: continue
        tpr_path = runs_dir / folder / "tpr.txt"
        if not tpr_path.exists(): continue
        try:
            txt = tpr_path.read_text()
            m_mean = re.search(r"mean PPL \(WM\):\s+([\d.]+)", txt)
            m_tpr = re.search(r"global TPR:\s+([\d.]+)", txt)
            if m_mean and m_tpr:
                base.append((math.log(float(m_mean.group(1))), float(m_tpr.group(1))))
        except: continue
    return sorted(base)

def calculate_tpr_at_1fpr(df: pd.DataFrame) -> float:
    if "det_no_wm_p_value" not in df.columns or "det_wm_p_value" not in df.columns: return 0.0
    no_wm_pvals = df["det_no_wm_p_value"].dropna().values
    wm_pvals = df["det_wm_p_value"].dropna().values
    if len(no_wm_pvals) == 0 or len(wm_pvals) == 0: return 0.0
    thresh = np.percentile(no_wm_pvals, 1.0)
    return float(np.mean(wm_pvals <= thresh))

def load_run_points(run_dir: Path, exclude_deltas=None):
    points = []
    if not run_dir.exists(): return points
    for d in sorted(run_dir.glob("delta_*"), key=lambda p: float(p.name.split("_")[1])):
        delta = float(d.name.split("_")[1])
        if exclude_deltas and delta in exclude_deltas: continue
        p = d / "results_merged.jsonl" if (d / "results_merged.jsonl").exists() else d / "results_with_ppl.jsonl"
        if not p.exists(): continue
        try:
            df = pd.read_json(p, lines=True)
            log_ppl = math.log(df["ppl_wm"].mean())
            tpr = calculate_tpr_at_1fpr(df)
            points.append((log_ppl, tpr))
        except: continue
    return sorted(points)

def get_tpr_at_ppl(points, target_log_ppl):
    if not points or len(points) < 2: return None
    ppls = [p[0] for p in points]
    tprs = [p[1] for p in points]
    if target_log_ppl < min(ppls) or target_log_ppl > max(ppls): return None
    f = interp1d(ppls, tprs, kind='linear')
    return float(f(target_log_ppl))

def main():
    root = Path(__file__).resolve().parent.parent.parent
    runs = root / "runs"
    
    base = load_base_points(runs)
    prompt1 = load_run_points(runs / "prompt1_8519605", exclude_deltas=[1.0, 2.0])
    zurich = load_run_points(runs / "zurich_eval_same_prompts_7006629", exclude_deltas=[1.0, 2.0])

    targets = [2.2, 2.3, 2.4, 2.5, 2.7, 3.0]
    
    print(f"{'log(PPL)':<10} | {'Baseline':<12} | {'Prompt1':<12} | {'Zurich':<12}")
    print("-" * 55)
    
    for t in targets:
        v_base = get_tpr_at_ppl(base, t)
        v_p1 = get_tpr_at_ppl(prompt1, t)
        v_z = get_tpr_at_ppl(zurich, t)
        
        s_base = f"{v_base:.4f}" if v_base is not None else "N/A"
        s_p1 = f"{v_p1:.4f}" if v_p1 is not None else "N/A"
        s_z = f"{v_z:.4f}" if v_z is not None else "N/A"
        
        print(f"{t:<10.1f} | {s_base:<12} | {s_p1:<12} | {s_z:<12}")

if __name__ == "__main__":
    main()
