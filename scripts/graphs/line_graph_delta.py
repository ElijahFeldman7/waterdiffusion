import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import roc_curve

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Roboto Condensed', 'Arial']

def plot(run_dir: str, target_fpr=0.01):
    input_file = os.path.join(run_dir, 'results.jsonl')
    delta_groups = defaultdict(list)

    with open(input_file) as f:
        for line in f:
            if line.strip():
                try:
                    d = json.loads(line)
                    delta = d.get('delta')
                    if delta is not None:
                        delta_groups[float(delta)].append(d)
                except (json.JSONDecodeError, ValueError):
                    continue

    results = []
    sorted_deltas = sorted(delta_groups.keys())
    for delta in sorted_deltas:
        lines = delta_groups[delta]
        pos_scores = np.array([-float(d['detection_wm']['score']) for d in lines if 'detection_wm' in d])
        neg_scores = np.array([-float(d['detection_no_wm']['score']) for d in lines if 'detection_no_wm' in d])
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            continue
        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_scores = np.concatenate([pos_scores, neg_scores])
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        tpr_at_target = np.interp(target_fpr, fpr, tpr)
        results.append((delta, tpr_at_target))

    if not results:
        print('no data to plot')
        return

    deltas, tprs = zip(*results)
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    plt.plot(deltas, tprs, marker='o', linestyle='-', color='#2980b9', 
             linewidth=3, markersize=10, label=f'TPR @ {target_fpr*100:.0f}% FPR')

    sns.despine(ax=ax, top=True, right=True)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

    plt.xlabel(r'Delta ($\delta$)', fontsize=22, fontweight='bold', labelpad=15)
    plt.ylabel(f'Detection Rate (TPR @ {target_fpr*100:.0f}% FPR)', fontsize=22, fontweight='bold', labelpad=15)
    
    plt.xticks(deltas, fontsize=16)
    plt.yticks(np.arange(0.6, 1.1, 0.1), fontsize=16)
    ax.tick_params(width=2.0, length=8)

    plt.ylim(0.6, 1.05)
    
    plt.tight_layout()

    out_path = os.path.join(run_dir, 'line_graph.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    args = parser.parse_args()
    plot(args.run_dir)