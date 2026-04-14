import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

data_dir = Path("runs/12_17_16_55_llada_8b_eli5_wm_test/gen_length_400")
file_path = data_dir / "results.jsonl"

data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)

df['neg_log_p'] = -np.log10(df['detection_wm'].apply(lambda x: x['score']) + 1e-20)

plt.figure(figsize=(10, 7), dpi=300)
sns.set_style("white")

sns.scatterplot(data=df, x='cosine_similarity', y='neg_log_p', 
                alpha=0.4, color='#2980b9', s=30, label='Samples')

plt.axhline(y=1.3, color='#e74c3c', linestyle='--', linewidth=2, label='Detection Limit (o=0.05)')

plt.axvspan(0.8, 1.0, color='#2ecc71', alpha=0.1, label='Target Quality Zone')

plt.xlabel("Cosine Similarity", fontsize=12)
plt.ylabel("($-\log(p)$)", fontsize=12)


plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper left')

save_path = data_dir / "p_value_vs_quality.png"
plt.savefig(save_path, bbox_inches='tight')
