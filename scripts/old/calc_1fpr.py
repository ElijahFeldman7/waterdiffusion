import json
import numpy as np
from collections import defaultdict
import os

RUNS_DIR = "/Users/eli/research/waterdiffusion/runs"
TARGET_FPR = 0.01

jsonl_paths = []
for folder in os.listdir(RUNS_DIR):
    if folder == "old":
        continue
    path = os.path.join(RUNS_DIR, folder, "results_merged.jsonl")
    if os.path.exists(path):
        jsonl_paths.append((folder, path))

jsonl_paths.sort()

for folder_name, INPUT_PATH in jsonl_paths:
    OUTPUT_PATH = os.path.join(RUNS_DIR, folder_name, "tpr.txt")

    records = []
    with open(INPUT_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records or "det_no_wm_p_value" not in records[0]:
        print(f"  SKIPPING {folder_name} — missing expected fields")
        continue

    print(f"Processing {folder_name} — {len(records)} records")

    bdelta = defaultdict(list)
    for r in records:
        bdelta[r["delta"]].append(r)

    no_wm_pval = [r["det_no_wm_p_value"] for r in records if "det_no_wm_p_value" in r]
    if not no_wm_pval:
        print(f"  SKIPPING {folder_name} — no det_no_wm_p_value in any record")
        continue

    thresh = np.percentile(no_wm_pval, TARGET_FPR * 100)

    results = []
    for delta in sorted(bdelta.keys()):
        recs = bdelta[delta]

        wm_pvals    = np.array([r["det_wm_p_value"]    for r in recs if "det_wm_p_value"    in r])
        no_wm_pvals = np.array([r["det_no_wm_p_value"] for r in recs if "det_no_wm_p_value" in r])
        wm_ppls     = np.array([r["ppl_wm"]             for r in recs])
        no_wm_ppls  = np.array([r["ppl_no_wm"]          for r in recs])

        per_delta_thresh    = np.percentile(no_wm_pvals, TARGET_FPR * 100)
        global_tpr          = np.mean(wm_pvals <= thresh)
        fpr_check           = np.mean(no_wm_pvals <= thresh)
        tpr_per_delta       = np.mean(wm_pvals <= per_delta_thresh)
        fpr_check_per_delta = np.mean(no_wm_pvals <= per_delta_thresh)
        mean_ppl_wm         = np.mean(wm_ppls)
        mean_ppl_no_wm      = np.mean(no_wm_ppls)
        log_pplwm           = np.mean(np.log(wm_ppls))
        log_pplnwm          = np.mean(np.log(no_wm_ppls))
        delta_log_ppl       = log_pplwm - log_pplnwm

        results.append({
            "delta":               delta,
            "n":                   len(recs),
            "mean_ppl_wm":         mean_ppl_wm,
            "mean_ppl_no_wm":      mean_ppl_no_wm,
            "log_pplwm":           log_pplwm,
            "log_pplnwm":          log_pplnwm,
            "delta_log_ppl":       delta_log_ppl,
            "thresh":              thresh,
            "global_tpr":          global_tpr,
            "fpr_check":           fpr_check,
            "per_delta_thresh":    per_delta_thresh,
            "tpr_per_delta":       tpr_per_delta,
            "fpr_check_per_delta": fpr_check_per_delta,
        })

    with open(OUTPUT_PATH, "w") as f:
        f.write(f"source: {INPUT_PATH}\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(f"delta:              {r['delta']}\n")
            f.write(f"n:                  {r['n']}\n\n")
            f.write(f"mean PPL (WM):      {r['mean_ppl_wm']:.4f}\n")
            f.write(f"mean PPL (no WM):   {r['mean_ppl_no_wm']:.4f}\n\n")
            f.write(f"log PPL (WM):       {r['log_pplwm']:.4f}\n")
            f.write(f"log PPL (no WM):    {r['log_pplnwm']:.4f}\n")
            f.write(f"delta log PPL:      {r['delta_log_ppl']:.4f}\n\n")
            f.write(f"threshold:          {r['thresh']:.6e}\n")
            f.write(f"global TPR:         {r['global_tpr']:.4f}\n")
            f.write(f"FPR check:          {r['fpr_check']:.4f}\n\n")
            f.write(f"per-delta thresh:   {r['per_delta_thresh']:.6e}\n")
            f.write(f"per-delta TPR:      {r['tpr_per_delta']:.4f}\n")
            f.write(f"per-delta FPR:      {r['fpr_check_per_delta']:.4f}\n")
            f.write("-" * 60 + "\n\n")

    print(f"  -> wrote {OUTPUT_PATH}")

print("\nAll done.")