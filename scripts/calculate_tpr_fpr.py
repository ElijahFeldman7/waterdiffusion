
import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from pathlib import Path

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
    print(f"TPR at 1% FPR for original data: {tpr_orig:.4f}")

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
        positives = p_values['positives']
        negatives = p_values['negatives']
        
        y_true = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
        y_scores = np.concatenate([positives, negatives])

        tpr = calculate_tpr_at_fpr(y_true, y_scores, target_fpr=0.01)
        
        results.append({
            'attack': attack,
            'ratio': ratio,
            'tpr_at_1_fpr': tpr
        })

    df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))

    styles = {
        'substitution': {'color': 'blue', 'marker': 's', 'linestyle': 'dotted'},
        'ca_substitution': {'color': 'blue', 'marker': 'o', 'linestyle': 'solid'}
    }
    
    for attack_type in sorted(df['attack'].unique()):
        subset = df[df['attack'] == attack_type].sort_values('ratio')
        
        ratios = np.concatenate([[0.0], subset['ratio']])
        tprs = np.concatenate([[tpr_orig], subset['tpr_at_1_fpr']])
        
        style = styles.get(attack_type, {})
        plt.plot(ratios, tprs, label=attack_type, **style)

    plt.xlabel("Attack Ratio")
    plt.ylabel("TPR at 1% FPR")
    plt.title("TPR at 1% FPR vs. Attack Ratio")
    plt.legend()
    plt.grid(True)
    
    output_path = Path(OUTPUT_DIR) / "tpr_fpr_scatter.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
