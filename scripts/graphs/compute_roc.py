import argparse
import json
import os
import csv
from datetime import datetime
from collections import defaultdict
import matplotlib
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 15,
    'font.weight': 'medium',
    'axes.labelweight': 'bold',
    'figure.facecolor': 'white'
})

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

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    colors = ['#1A85FF', '#D41159', '#FFC20A', '#004D40'] 
    
    csv_path = os.path.join(run_dir, 'roc_points.csv')
    all_aucs = {}
    sorted_gen_lengths = sorted(lines_by_length.keys())

    with open(csv_path, 'w', newline='') as csvf:
        w = csv.writer(csvf)
        w.writerow(['gen_length', 'fpr', 'tpr', 'threshold'])

        for i, gen_len in enumerate(sorted_gen_lengths):
            lines = lines_by_length[gen_len]
            pos_scores = np.array([-float(d['detection_wm']['score']) for d in lines if d.get('detection_wm') and 'score' in d['detection_wm']])
            neg_scores = np.array([-float(d['detection_no_wm']['score']) for d in lines if d.get('detection_no_wm') and 'score' in d['detection_no_wm']])

            if len(pos_scores) == 0 or len(neg_scores) == 0:
                continue

            y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
            y_scores = np.concatenate([pos_scores, neg_scores])

            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            
            fpr = np.concatenate([[0], fpr])
            tpr = np.concatenate([[0], tpr])
            
            roc_auc = auc(fpr, tpr)
            all_aucs[gen_len] = roc_auc

            for f, t, th in zip(fpr, tpr, thresholds):
                w.writerow([gen_len, f, t, th])

            color = colors[i % len(colors)]
            label = f'Len {gen_len} (AUC={roc_auc:.4f})'
            
            if len(tpr) > 10:
                tpr_smooth = gaussian_filter1d(tpr, sigma=1.0)
                tpr_smooth = np.maximum.accumulate(tpr_smooth)
                tpr_smooth[-1] = 1.0
                ax.plot(fpr, tpr_smooth, label=label, color=color, lw=3.5, zorder=5)
            else:
                ax.plot(fpr, tpr, label=label, color=color, lw=3.5, zorder=5)
            
            ax.fill_between(fpr, tpr, color=color, alpha=0.06)

    ax.set_xscale('log')
    ax.set_xlim([1e-3, 1.0])
    ax.set_ylim([0.0, 1.02])

    log_range = np.logspace(-3, 0, 100)
    ax.plot(log_range, log_range, color='#555555', lw=1.5, ls='--', label='Random Guess', alpha=0.7)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    ax.grid(True, which="major", color='#DDDDDD', linestyle='-', linewidth=0.8)
    ax.grid(True, which="minor", color='#EEEEEE', linestyle=':', linewidth=0.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(frameon=True, loc='lower right', facecolor='white', framealpha=1.0, edgecolor='#CCCCCC')
    
    plt.tight_layout()
    png_path = os.path.join(run_dir, 'roc_curve.png')
    plt.savefig(png_path)
    plt.close()

    summary_path = os.path.join(run_dir, 'summary_report.txt')
    with open(summary_path, 'a') as sf:
        sf.write(f"\nROC Analysis - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        for gen_len, roc_auc in all_aucs.items():
            sf.write(f"  Gen Length {gen_len}: AUC = {roc_auc:.6f}\n")

    return csv_path, png_path, all_aucs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    args = parser.parse_args()
    compute_roc(args.run_dir)