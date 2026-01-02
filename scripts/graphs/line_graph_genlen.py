import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Roboto Condensed', 'Arial']

def plot(run_dir: str):
    input_file = os.path.join(run_dir, 'results.jsonl')
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
        print('no data to plot')
        return
    lengths, tprs = zip(*results)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    plt.plot(lengths, tprs, marker='o', linestyle='-', color='#2980b9', 
             linewidth=3, markersize=10, label='TPR @ 1% FPR')

    sns.despine(ax=ax, top=True, right=True)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

    plt.xlabel('Generation Length (Tokens)', fontsize=22, fontweight='bold', labelpad=15)
    plt.ylabel('TPR @ 1% FPR', fontsize=22, fontweight='bold', labelpad=15)
    
    plt.xticks(lengths, fontsize=16)
    plt.yticks(np.arange(0.4, 1.1, 0.1), fontsize=16)
    ax.tick_params(width=2.0, length=8)

    plt.ylim(0.4, 1.05)
    
    plt.tight_layout()

    out_path = os.path.join(run_dir, 'length_performance_graph.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    args = parser.parse_args()
    plot(args.run_dir)