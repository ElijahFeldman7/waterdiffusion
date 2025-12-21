import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_curve

def plot(run_dir: str, target_fpr=0.01):
    input_file = os.path.join(run_dir, 'results.jsonl')
    delta_groups =defaultdict(list)

    with open(input_file) as f:
        for line in f:
            if line.strip():
                try:
                    d=json.loads(line)
                    delta = d.get('delta')
                    if delta is not None:
                        delta_groups[float(delta)].append(d)
                except (json.JSONDecodeError, ValueError):
                    continue
    results =[]
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
        tpr_at_target = np.interp(target_fpr,fpr,tpr)
        results.append((delta, tpr_at_target))
    if not results:
        print('no data to plot')
        return
    deltas, tprs = zip(*results)
    plt.figure(figsize=(10,6), dpi=150)
    plt.plot(deltas, tprs, marker='o', linestyle='-', color='#007bff', linewidth=2.5, markersize=8, label=f'TPR @ {target_fpr*100:.0f}% FPR')
    plt.xlabel(r'Delta ($\delta$)',fontweight='bold')
    plt.ylabel('TPR @ 1% FPR', fontweight='bold')
    plt.title('plot at gen len 400')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0.3, 1.05)
    plt.yticks(np.arange(0.3,1.1,0.1))
    plt.xticks(deltas)
    plt.tight_layout()
    out_path = os.path.join(run_dir, 'line_graph.png')
    plt.savefig(out_path)
    plt.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    args = parser.parse_args()
    plot(args.run_dir)