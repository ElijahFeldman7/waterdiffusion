import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Roboto Condensed', 'Arial']

data_dir = Path("runs/12_17_16_55_llada_8b_eli5_wm_test/gen_length_400")
file_path = data_dir / "results.jsonl"

data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

df['sat_rate'] = df['detection_wm'].apply(lambda x: x['satisfaction_rate'])
df['n_trials'] = df['detection_wm'].apply(lambda x: x['total_pairs_checked'])

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
sns.set_style("white")

sns.scatterplot(data=df, x='cosine_similarity', y='sat_rate', 
                alpha=0.2, color='#34495e', s=25, label="Individual Samples")

sns.kdeplot(data=df, x='cosine_similarity', y='sat_rate', thresh=0.05,
            levels=[0.5, 0.9, 0.99], color="red", alpha=0.6, linewidths=1.5, gridsize=200)
sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
ax.spines['left'].set_linewidth(2.0)
ax.spines['bottom'].set_linewidth(2.0)
ax.spines['left'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')
ax.grid(True, linestyle='--', alpha=0.3, color='gray', zorder=0)
gamma = 0.50 
critical_sat_rate = 0.529

plt.axhline(y=critical_sat_rate, color='crimson', linestyle='--', label=f'Significance Threshold ($\hat{{p}} > {critical_sat_rate}$)')
plt.axvline(x=0.70, color='gray', linestyle=':', label='Quality Threshold ($cos > 0.7$)')

plt.xlabel("Cosine Similarity", fontsize=22, fontweight='bold')
plt.ylabel("Satisfaction rate", fontsize=22, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.fill_between([0.7, 1.0], critical_sat_rate, 1.0, color='green', alpha=0.05, label="Reliable Detection Zone")

plt.xlim(0.4, 1.0)
plt.ylim(0.45, 0.8)
plt.legend(loc='upper right', frameon=True, fontsize=16)
plt.tight_layout()

save_path = data_dir / "binomial_signal_analysis.pdf"
plt.savefig(save_path)
print(f"Graph saved to {save_path}")