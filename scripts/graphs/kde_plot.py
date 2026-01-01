no on this import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path("runs/12_17_16_55_llada_8b_eli5_wm_test/gen_length_400")
file_path = data_dir / "results.jsonl"

data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

df['sat_rate'] = df['detection_wm'].apply(lambda x: x['satisfaction_rate'])
df['n_trials'] = df['detection_wm'].apply(lambda x: x['total_pairs_checked'])

plt.figure(figsize=(10, 8), dpi=300)
sns.set_style("white")

sns.scatterplot(data=df, x='cosine_similarity', y='sat_rate', 
                alpha=0.3, color='#34495e', s=25, label="Individual Samples")

sns.kdeplot(data=df, x='cosine_similarity', y='sat_rate', 
            levels=5, color="red", alpha=0.4, linewidths=1)

gamma = 0.50 
critical_sat_rate = 0.55

plt.axhline(y=critical_sat_rate, color='crimson', linestyle='--', label=f'Significance Threshold ($\hat{{p}} > {critical_sat_rate}$)')
plt.axvline(x=0.70, color='gray', linestyle=':', label='Quality Threshold ($cos > 0.7$)')

plt.title("strength vs", fontsize=15)
plt.xlabel("Cosine Similarity", fontsize=12)
plt.ylabel("Satisfaction rate", fontsize=12)

plt.fill_between([0.7, 1.0], critical_sat_rate, 1.0, color='green', alpha=0.05, label="Reliable Detection Zone")

plt.legend(loc='lower left', frameon=True)
plt.tight_layout()

save_path = data_dir / "binomial_signal_analysis.png"
plt.savefig(save_path)
print(f"Graph saved to {save_path}")