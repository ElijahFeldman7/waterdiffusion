import argparse
import json
import os
import matplotlib
from collections import defaultdict
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import gaussian_filter1d


matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 28,
    'font.weight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
})

def compute_professional_roc(run_dir: str):
    input_file = os.path.join(run_dir, 'results.jsonl')
    delta_groups = defaultdict(list)
    
    with open(input_file) as f:
        for line in f:
            if line.strip():
                try:
                    d = json.loads(line)
                    delta_val = d.get('delta', 'unknown')
                    if str(delta_val) in ['1.0', '2.0', '3.0', '4.0', 1.0, 2.0, 3.0, 4.0]:
                        delta_groups[delta_val].append(d)
                except json.JSONDecodeError:
                    continue

    plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.gca()
    
    sorted_deltas = sorted(delta_groups.keys(), key=lambda x: float(x))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sorted_deltas)))
    
    for i, delta in enumerate(sorted_deltas):
        lines = delta_groups[delta]
        pos_scores = np.array([-float(d['detection_wm']['score']) for d in lines if 'detection_wm' in d])
        neg_scores = np.array([-float(d['detection_no_wm']['score']) for d in lines if 'detection_no_wm' in d])
        
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            continue

        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_scores = np.concatenate([pos_scores, neg_scores])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        if len(tpr) > 10:
            tpr = gaussian_filter1d(tpr, sigma=1.5)
            tpr = np.maximum.accumulate(tpr)
            tpr[-1] = 1.0

        plt.plot(fpr, tpr, label=rf'$\delta = {delta}$ (AUC={roc_auc:.4f})', color=colors[i], lw=3.5) # Increased line width

    plt.xscale('log')
    plt.xlim([1e-2, 1.0])
    plt.ylim([0.0, 1.0])
    
    log_range = np.logspace(-2, 0, 100)
    plt.plot(log_range, log_range, color='#ff6666', lw=2.5, ls='--', label='No watermark')    
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title(f'ROC Curves by Delta', pad=20) 
    
    plt.grid(True, which="major", color='gray', linestyle=':', linewidth=0.8, alpha=0.7) 
    plt.grid(True, which="minor", color='gray', linestyle=':', linewidth=0.4, alpha=0.4) 
    
    plt.legend(frameon=False, loc='lower right')
    
    ax.set_box_aspect(1) 
    
    plt.tight_layout()
    out = os.path.join(run_dir, 'roc.png')
    plt.savefig(out)
    plt.close()
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    args = parser.parse_args()
    compute_professional_roc(args.run_dir)