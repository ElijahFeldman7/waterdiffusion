import json
import math
import os
import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_base_points(runs_dir: Path):
    skip = {"remask_eval_6927558", "remask_eval_6930413"}
    base = []

    for folder in sorted(os.listdir(runs_dir)):
        if folder == "old" or folder in skip:
            continue

        desc_path = runs_dir / folder / "run_desc.json"
        if not desc_path.exists():
            continue

        desc = json.loads(desc_path.read_text())
        wm_file = desc.get("watermark", {}).get("file", "")
        if "3_29_DWT_watermark.py" not in wm_file:
            continue

        tpr_path = runs_dir / folder / "tpr.txt"
        if not tpr_path.exists():
            continue

        txt = tpr_path.read_text()
        m1 = re.search(r"log PPL \(WM\):\s+([\d.]+)", txt)
        m2 = re.search(r"global TPR:\s+([\d.]+)", txt)
        if m1 and m2:
            base.append((folder, float(m1.group(1)), float(m2.group(1))))

    base.sort(key=lambda x: x[1])
    return base


def load_ours_points(run_dir: Path):
    ours = []
    for d in sorted(run_dir.glob("delta_*"), key=lambda p: float(p.name.split("_", 1)[1])):
        p = d / "results_with_ppl.jsonl"
        if not p.exists():
            continue

        delta = float(d.name.split("_", 1)[1])
        if delta == 1.0:
            continue

        df = pd.read_json(p, lines=True)
        log_ppl = math.log(float(df["ppl_wm"].mean()))
        tpr = float(df["tpr_at_1pct"].iloc[0])
        ours.append((delta, log_ppl, tpr))

    return ours


def main():
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    default_runs_dir = project_root / "runs"
    default_run_dir = default_runs_dir / "zurich_eval_same_prompts_7006629"

    parser = argparse.ArgumentParser(description="Plot TPR vs log(PPL) comparison.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=default_runs_dir,
        help="Path to the top-level runs directory.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=default_run_dir,
        help="Path to the run directory containing delta_* folders.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output PNG path. Defaults to <run-dir>/tpr_ppl_compare.png.",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir
    run_dir = args.run_dir

    if not runs_dir.exists():
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    base = load_base_points(runs_dir)
    ours = load_ours_points(run_dir)

    fig, ax = plt.subplots(figsize=(9, 6))

    if base:
        ax.plot(
            [x[1] for x in base],
            [x[2] for x in base],
            marker="o",
            linewidth=2,
            markersize=6,
            color="#1D5BBF",
            label="Our watermark",
        )

    if ours:
        ax.plot(
            [x[1] for x in ours],
            [x[2] for x in ours],
            marker="s",
            linewidth=2,
            markersize=6,
            color="#D97706",
            label=f"zurich",
        )

    ax.set_xlabel("log(PPL)", fontsize=12)
    ax.set_ylabel("TPR @ 1 FPR", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    out = args.out if args.out is not None else (run_dir / "tpr_ppl_compare.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)

    print(f"saved {out}")
    print(f"base_points {len(base)} ours_points {len(ours)}")


if __name__ == "__main__":
    main()
