import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Roboto Condensed', 'Arial']

INPUT_FILE = "runs/12_27_20_02_robustness_eval/results.jsonl"
OUTPUT_DIR = "runs/12_27_20_02_robustness_eval"

def calculate_tpr_at_fpr(y_true, y_scores, target_fpr=0.01):
    fpr, tpr, thresholds = roc_curve(y_true, -np.array(y_scores))
    if np.all(fpr > target_fpr):
        return 0.0
    idx = np.searchsorted(fpr, target_fpr, side='right')
    if idx > 0 and fpr[idx] > target_fpr:
        tpr_at_target = np.interp(target_fpr, fpr[idx-1:idx+1], tpr[idx-1:idx+1])
    else:
        tpr_at_target = tpr[idx -1] if idx > 0 else 0.0
    return tpr_at_target

def main():
    original_positives = []
    original_negatives = []
    
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            row = json.loads(line)
            original_positives.append(row['original_p_value'])
            original_negatives.append(row['detection_no_wm']['score'])

    y_true_orig = np.concatenate([np.ones(len(original_positives)), np.zeros(len(original_negatives))])
    y_scores_orig = np.concatenate([original_positives, original_negatives])
    tpr_orig = calculate_tpr_at_fpr(y_true_orig, y_scores_orig, target_fpr=0.01)

    data = {}
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            row = json.loads(line)
            attack = row['attack']
            ratio = row['ratio']
            if (attack, ratio) not in data:
                data[(attack, ratio)] = {'positives': [], 'negatives': []}
            data[(attack, ratio)]['positives'].append(row['edited_p_value'])
            data[(attack, ratio)]['negatives'].append(row['detection_no_wm']['score'])

    results = []
    for (attack, ratio), p_values in data.items():
        y_true = np.concatenate([np.ones(len(p_values['positives'])), np.zeros(len(p_values['negatives']))])
        y_scores = np.concatenate([p_values['positives'], p_values['negatives']])
        tpr = calculate_tpr_at_fpr(y_true, y_scores, target_fpr=0.01)
        results.append({'attack': attack, 'ratio': ratio, 'tpr_at_1_fpr': tpr})

    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    
    styles = {
        'substitution': {'color': '#e74c3c', 'marker': 's', 'linestyle': '--', 'linewidth': 3, 'markersize': 10},
        'ca_substitution': {'color': '#2980b9', 'marker': 'o', 'linestyle': '-', 'linewidth': 3, 'markersize': 10}
    }
    
    for attack_type in sorted(df['attack'].unique()):
        subset = df[df['attack'] == attack_type].sort_values('ratio')
        ratios = np.concatenate([[0.0], subset['ratio']])
        tprs = np.concatenate([[tpr_orig], subset['tpr_at_1_fpr']])
        style = styles.get(attack_type, {'color': 'gray'})
        plt.plot(ratios, tprs, label=attack_type.replace('_', ' ').title(), **style)

    import seaborn as sns
    sns.despine(ax=ax, top=True, right=True)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

    plt.xlabel("Attack Ratio", fontsize=22, fontweight='bold', labelpad=15)
    plt.ylabel("TPR @ 1% FPR", fontsize=22, fontweight='bold', labelpad=15)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.tick_params(width=2.0, length=8)

    plt.legend(fontsize=16, frameon=True, loc='best', framealpha=0.9)
    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.01, max(df['ratio']) + 0.05)
    plt.tight_layout()
    
    output_path = Path(OUTPUT_DIR) / "robustness_tpr_plot.png"
    plt.savefig(output_path)

if __name__ == "__main__":
    main()