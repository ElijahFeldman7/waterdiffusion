import argparse
import json
import os
import csv
from datetime import datetime
from collections import defaultdict
import matplotlib
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import gaussian_filter1d

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def compute_roc(run_dir: str):
    input_file = os.path.join(run_dir, 'results.jsonl')
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found")

    lines_by_length = defaultdict(list)
    with open(input_file) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip():
                try:
                    d = json.loads(line)
                    if i < 1000:
                        gen_len = 100
                    elif i < 2000:
                        gen_len = 200
                    elif i < 3000:
                        gen_len = 300
                    else:
                        gen_len = 400
                    lines_by_length[gen_len].append(d)
                except json.JSONDecodeError:
                    continue

    if not lines_by_length:
        raise ValueError("No data lines found in results file")

    plt.figure(figsize=(8, 8), dpi=200)
    plt.gca().set_facecolor('#f0f0f0')

    csv_path = os.path.join(run_dir, 'roc_points.csv')
    all_aucs = {}

    sorted_gen_lengths = sorted(lines_by_length.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_gen_lengths)))

    with open(csv_path, 'w', newline='') as csvf:
        w = csv.writer(csvf)
        w.writerow(['gen_length', 'fpr', 'tpr', 'threshold'])

        for i, gen_len in enumerate(sorted_gen_lengths):
            lines = lines_by_length[gen_len]
            pos_scores = np.array([-float(d['detection_wm']['score']) for d in lines if d.get('detection_wm') and 'score' in d['detection_wm']])
            neg_scores = np.array([-float(d['detection_no_wm']['score']) for d in lines if d.get('detection_no_wm') and 'score' in d['detection_no_wm']])

            if len(pos_scores) == 0 or len(neg_scores) == 0:
                print(f"Skipping gen_len {gen_len} due to insufficient samples: wm={len(pos_scores)}, no_wm={len(neg_scores)}")
                continue

            y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
            y_scores = np.concatenate([pos_scores, neg_scores])

            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            all_aucs[gen_len] = roc_auc

            for f, t, th in zip(fpr, tpr, thresholds):
                w.writerow([gen_len, f, t, th])

            label = f'Gen Length = {gen_len} (AUC={roc_auc:.4f})'
            if len(tpr) > 10:
                tpr_smooth = gaussian_filter1d(tpr, sigma=1.5)
                tpr_smooth = np.maximum.accumulate(tpr_smooth)
                tpr_smooth[-1] = 1.0
                plt.plot(fpr, tpr_smooth, label=label, color=colors[i], lw=2.5)
            else:
                plt.plot(fpr, tpr, label=label, color=colors[i], lw=2.5)
    plt.xscale('log')

    plt.xlim([1e-3, 1.0])
    plt.ylim([0.0, 1.05])
    log_range = np.logspace(-3, 0, 100)
    plt.plot(log_range, log_range, color='#ff6666', lw=1.5, ls='--', label='No watermark')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC curve by Generation Length - {os.path.basename(run_dir)}', fontsize=14)
    plt.grid(True, which="major", color='white', linewidth=1.2)
    plt.grid(True, which="minor", color='white', linewidth=0.6, alpha=0.5)
    plt.legend(frameon=False, loc='lower right')
    plt.tight_layout()

    png_path = os.path.join(run_dir, 'roc_curve.png')
    plt.savefig(png_path)
    plt.close()

    summary_path = os.path.join(run_dir, 'summary_report.txt')
    with open(summary_path, 'a') as sf:
        sf.write(f"\nROC Analysis by Generation Length - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        for gen_len, roc_auc in all_aucs.items():
            sf.write(f"  Gen Length {gen_len}: AUC = {roc_auc:.6f}\n")
        sf.write(f"CSV: {csv_path}\nPNG: {png_path}\n")

    return csv_path, png_path, all_aucs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute ROC on a run folder containing results.jsonl, grouped by generation length.')
    parser.add_argument('--run_dir', required=True, help='Run folder containing results.jsonl')
    args = parser.parse_args()

    csv_path, png_path, auc_vals = compute_roc(args.run_dir)
    print(f'Wrote ROC CSV: {csv_path}, PNG: {png_path}')
    for gen_len, auc_val in auc_vals.items():
        print(f'  Gen Length {gen_len}: AUC = {auc_val:.6f}')
