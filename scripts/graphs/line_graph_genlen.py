import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

def plot(run_dir: str):
    input_file = os.path.join(run_dir, 'results.jsonl')
    data =[]
    with open(input_file) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    lens =[100,200,300,400]
    chunksize = 1000
    results=[]
    for i,length in enumerate(lens):
        start=i*chunksize
        end=(i+1)*chunksize
        subset =data[start:end]
        if not subset:
            continue
        pos_scores = np.array([-float(d['detection_wm']['score']) for d in subset if 'detection_wm' in d])
        neg_scores = np.array([-float(d['detection_no_wm']['score']) for d in subset if 'detection_no_wm' in d])
        
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            continue
        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_scores = np.concatenate([pos_scores, neg_scores])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        tpr_at_target= np.interp(0.01, fpr,tpr)
        results.append((length,tpr_at_target))

    if not results:
        print('no data to plot')
        return
    lengths, tprs = zip(*results)

    plt.figure(figsize=(8,6), dpi=300)
    plt.plot(lengths, tprs, marker='o', linestyle='-', color='#007bff', linewidth=2.5, markersize=8, label=f'TPR @ 1% FPR')
    plt.xlabel('Lengths',fontweight='bold', fontsize='12')
    plt.ylabel('TPR @ 1% FPR', fontweight='bold', fontsize='12')
    plt.title('tpr over gen lens')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0.4, 1.05)
    plt.yticks(np.arange(0.4, 1.1, 0.1))
    plt.xticks(lengths)
    plt.tight_layout()
    out_path = os.path.join(run_dir, 'line_graph.png')
    plt.savefig(out_path)
    plt.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    args = parser.parse_args()
    plot(args.run_dir)