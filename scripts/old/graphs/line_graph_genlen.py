import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 10,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def plot(run_dir: str):
    input_file = os.path.join(run_dir, 'results.jsonl')
    if not os.path.exists(input_file):
        return

    data = []
    with open(input_file) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    lens = [100, 200, 300, 400]
    chunksize = 1000
    results = []
    for i, length in enumerate(lens):
        start = i * chunksize
        end = (i + 1) * chunksize
        subset = data[start:end]
        if not subset:
            continue
        pos_scores = np.array([-float(d['detection_wm']['score']) for d in subset if 'detection_wm' in d])
        neg_scores = np.array([-float(d['detection_no_wm']['score']) for d in subset if 'detection_no_wm' in d])
        
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            continue
        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_scores = np.concatenate([pos_scores, neg_scores])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        tpr_at_target = np.interp(0.01, fpr, tpr)
        results.append((length, tpr_at_target))

    if not results:
        return
    lengths, tprs = zip(*results)

    fig, ax = plt.subplots(figsize=(4.875, 3.75), dpi=300)
    
    plt.plot(lengths, tprs, marker='o', linestyle='-', color='#2980b9', 
             linewidth=1.5, markersize=5, label=r'TPR @ 1% FPR')

    sns.despine(ax=ax)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0, linewidth=0.5)

    plt.xlabel('Generation Length (Tokens)', labelpad=6, color='black')
    plt.ylabel(r'TPR @ 1% FPR', labelpad=6, color='black')
    
    plt.xticks(lengths, color='black')
    plt.yticks(np.arange(0.4, 1.1, 0.1), color='black')
    ax.tick_params(axis='both', colors='black', width=0.8, length=4)

    plt.ylim(0.4, 1.05)
    plt.tight_layout(pad=0.5)

    plt.savefig(os.path.join(run_dir, 'length_performance_graph.png'), bbox_inches='tight')
    plt.savefig(os.path.join(run_dir, 'length_performance_graph.pdf'), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    args = parser.parse_args()
    plot(args.run_dir)