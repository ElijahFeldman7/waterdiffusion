import os
import matplotlib.pyplot as plt
import re
import json 
RUNS_DIR = "/Users/eli/research/waterdiffusion/runs"
SKIP_FOLDERS = {"remask_eval_6927558", "remask_eval_6930413"}
points = [] 

for folder in sorted(os.listdir(RUNS_DIR)):
    if folder == "old" or folder in SKIP_FOLDERS:
        continue

    desc_path = os.path.join(RUNS_DIR, folder, "run_desc.json")
    if not os.path.exists(desc_path):
        continue

    with open(desc_path) as f:
        desc = json.load(f)

    wm_file = desc.get("watermark", {}).get("file", "")
    if "3_29_DWT_watermark.py" not in wm_file:
        continue
    tpr_path = os.path.join(RUNS_DIR, folder, "tpr.txt")
    if not os.path.exists(tpr_path):
        continue

    with open(tpr_path, "r") as f:
        content = f.read()

    log_ppl_wm = re.search(r"log PPL \(WM\):\s+([\d.]+)", content)
    global_tpr = re.search(r"global TPR:\s+([\d.]+)", content)

    if log_ppl_wm and global_tpr:
        points.append((folder, float(log_ppl_wm.group(1)), float(global_tpr.group(1))))
        print(f"  {folder}: log_ppl_wm={log_ppl_wm.group(1)}, tpr={global_tpr.group(1)}")
    else:
        print(f"  SKIPPING {folder} — could not parse values")

if not points:
    print("No data points found.")
    exit()

points.sort(key=lambda x: x[1])

labels    = [p[0] for p in points]
log_ppls  = [p[1] for p in points]
tprs      = [p[2] for p in points]

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(log_ppls, tprs, marker="o", linewidth=2, markersize=7, color="#1D5BBF")


ax.set_xlabel("log(PPL)", fontsize=12)
ax.set_ylabel("TPR @ 1 FPR", fontsize=12)
ax.set_ylim(0, 1.05)
ax.grid(True, linestyle="--", alpha=0.5)

OUTPUT = os.path.join(RUNS_DIR, "tpr_ppl.png")
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150)
