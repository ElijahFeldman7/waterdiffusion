import argparse
import json
import math
import csv
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def compute_roc(run_dir: str, points: int = 101):
    input_file = os.path.join(run_dir, 'results.jsonl')
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found")

    lines = []
    with open(input_file) as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    if not lines:
        raise ValueError("No data lines found in results file")

    no_wm_scores = [d.get('detection_no_wm', {}).get('score')
                    for d in lines if d.get('detection_no_wm') is not None and 'score' in d.get('detection_no_wm')]
    num_no_wm = len(no_wm_scores)
    num_wm = sum(1 for d in lines if d.get('detection_wm') is not None and 'score' in d.get('detection_wm'))

    if num_no_wm == 0 or num_wm == 0:
        raise ValueError(f"Insufficient samples: no_wm={num_no_wm}, wm={num_wm}")

    no_wm_sorted = sorted([s for s in no_wm_scores if s is not None])
    min_score = no_wm_sorted[0]
    max_score = no_wm_sorted[-1]

    fpr_targets = [i / (points - 1) for i in range(points)]

    rows = []
    for fpr_t in fpr_targets:
        k = int(math.ceil(fpr_t * num_no_wm))
        if k == 0:
            threshold = min_score - 1e-12
        elif k >= num_no_wm:
            threshold = max_score + 1e-12
        else:
            threshold = no_wm_sorted[k - 1]

        fp = sum(1 for d in lines if d.get('detection_no_wm') is not None and 'score' in d['detection_no_wm'] and d['detection_no_wm']['score'] < threshold)
        tp = sum(1 for d in lines if d.get('detection_wm') is not None and 'score' in d['detection_wm'] and d['detection_wm']['score'] < threshold)
        fn = num_wm - tp
        tn = num_no_wm - fp

        fpr_actual = fp / num_no_wm
        tpr = tp / num_wm

        rows.append({
            'fpr_target': fpr_t,
            'fpr_actual': fpr_actual,
            'tpr': tpr,
            'threshold': threshold,
            'tp': tp,
            'fn': fn,
            'fp': fp,
            'tn': tn,
        })
    csv_path = os.path.join(run_dir, 'roc_points.csv')
    with open(csv_path, 'w', newline='') as csvf:
        w = csv.writer(csvf)
        w.writerow(['fpr_target', 'fpr_actual', 'tpr', 'threshold', 'tp', 'fn', 'fp', 'tn'])
        for r in rows:
            w.writerow([r['fpr_target'], r['fpr_actual'], r['tpr'], r['threshold'], r['tp'], r['fn'], r['fp'], r['tn']])
    fprs = np.array([r['fpr_actual'] for r in rows])
    tprs = np.array([r['tpr'] for r in rows])
    auc = np.trapz(tprs, fprs)

    plt.figure(figsize=(6, 6))
    dense_x = None
    dense_y = None
    try:
        smooth_flag = smooth 
    except NameError:
        smooth_flag = False
    try:
        marker_step = marker_step_arg
    except NameError:
        marker_step = 1
    try:
        no_markers = no_markers_flag
    except NameError:
        no_markers = False

    if smooth_flag:
        dense_x = np.linspace(0.0, 1.0, 1000)
        dense_y = np.interp(dense_x, fprs, tprs)
        plt.plot(dense_x, dense_y, '-', linewidth=1.5, label=f'AUC={auc:.4f}')
        if not no_markers:
            if marker_step <= 1:
                plt.plot(fprs, tprs, 'o', markersize=4)
            else:
                idxs = list(range(0, len(fprs), marker_step))
                plt.plot(fprs[idxs], tprs[idxs], 'o', markersize=4)
    else:
        if no_markers:
            plt.plot(fprs, tprs, '-', label=f'AUC={auc:.4f}')
        else:
            plt.plot(fprs, tprs, marker='.', label=f'AUC={auc:.4f}')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve - {os.path.basename(run_dir)}')
    plt.legend()
    plt.grid(True)

    png_path = os.path.join(run_dir, 'roc_curve.png')
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()

    summary_path = os.path.join(run_dir, 'summary_report.txt')
    with open(summary_path, 'a') as sf:
        sf.write(f"\nROC Analysis (sweep {points} points) - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        sf.write(f"AUC (trapezoid): {auc:.6f}\n")
        sf.write(f"CSV: {csv_path}\nPNG: {png_path}\n")

    return csv_path, png_path, auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute ROC on a run folder containing results.jsonl')
    parser.add_argument('--run_dir', required=True, help='Run folder containing results.jsonl')
    parser.add_argument('--points', type=int, default=101, help='Number of equally spaced FPR points (default: 101)')
    parser.add_argument('--smooth', action='store_true', help='Interpolate a dense smooth ROC curve for plotting')
    parser.add_argument('--marker-step', type=int, default=10, help='Plot markers every N points when smoothing (default: 10)')
    parser.add_argument('--no-markers', action='store_true', help='Do not plot marker dots; plot only lines')
    args = parser.parse_args()
    globals()['smooth'] = args.smooth
    globals()['marker_step_arg'] = args.marker_step
    globals()['no_markers_flag'] = args.no_markers

    csv_path, png_path, auc = compute_roc(args.run_dir, args.points)
    print(f'Wrote ROC CSV: {csv_path}, PNG: {png_path}, AUC: {auc:.6f}')
